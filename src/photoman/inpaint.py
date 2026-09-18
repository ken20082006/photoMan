"""Inpainting 引擎（見 docs/design.md §7.4.1）。

**本地是預設路徑，不是退路。** 三個理由：

1. **天生遮罩原生。** 遮罩外保證不變，不是訓練出來的性質。
2. **沒有縮小倍率的問題。** §5.3.4 講的頻寬損失只發生在 API 模型——
   本地是原解析度進出，所以那個失效模式根本不存在。
3. 免費，而且離線可用。

API 只保留給本地做得不好的情況：大面積、複雜材質、
需要**語意替換**（不只是移除，而是要變成別的東西）。

引擎藏在 Protocol 之後，所以換引擎不會影響合成器——保證是合成器的性質。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class Inpainter(Protocol):
    """把遮罩內的內容重新生成。"""

    name: str

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """``image`` 與 ``mask`` 都是**裁切圖**（原解析度、uint8 RGB）。

        回傳同尺寸的 uint8 RGB。**遮罩以外應該盡量不動**，
        但這不是契約——契約在合成器那邊：無論這個函數回傳甚麼
        （哪怕整張圖都改了），合成器都只會取遮罩內的像素。
        """
        ...


class TeleaInpainter:
    """OpenCV 的傳統 inpainting。

    **適合薄刮痕、細小物件**——因為要填的面積很小，沒有多少東西需要「編」
    出來。而且它是**確定性的**：同一輸入永遠同一輸出，可以寫成測試。

    ⚠️ **不適合大面積或複雜材質。** 它是擴散式的填充，大面積會變成
    一片模糊。移除一個人這種工作它做不好——那是 LaMa 或 API 模型的範圍。
    """

    def __init__(self, radius: int = 3, *, flags: int = cv2.INPAINT_TELEA) -> None:
        self.radius = radius
        self.flags = flags

    @property
    def name(self) -> str:
        return "telea" if self.flags == cv2.INPAINT_TELEA else "ns"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if not binary.any():
            return image.copy()
        return cv2.inpaint(image, binary, self.radius, self.flags)


class LamaInpainter:
    """LaMa（Large Mask Inpainting）。

    授權 **Apache-2.0**（`opencv/inpainting_lama` 的官方 ONNX 封裝）。
    這是本地路徑的主力引擎——它會產生與周圍一致的内容，
    而不是像 Telea 那樣平均出一片平滑。

    ⚠️ **這個 ONNX 的空間維度固定是 512×512**（``batch`` 是動態的，
    空間不是）。所以：

    - 裁切圖小於 512 → 沒有損失，這是最理想的情況
    - 裁切圖大於 512 → 縮小進去，輸出再放大回來。
      **放大回來的軟化只影響填補區**（我們只取遮罩內），
      而填補區本來就是平滑的生成內容，所以代價比帳面上小。
    - ⚠️ 真正大的裁切圖（遠超 512）應該切塊分別處理再融合——見 §5.3.4
    """

    # 模型固定的空間尺寸。不是可調參數——ONNX 的形狀就是這樣匯出的。
    INPUT_SIZE = 512

    def __init__(self, model_path: str | Path, *, session=None) -> None:
        self.model_path = Path(model_path)
        self._session = session

    @property
    def session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - 環境相依
                raise ImportError(
                    'LaMa 需要 onnxruntime。安裝：pip install -e ".[models]"'
                ) from exc
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"找不到 LaMa 模型：{self.model_path}\n見 docs/PROGRESS.md 的下載說明。"
                )
            ort.set_default_logger_severity(3)  # 這個匯出會吐大量無用的警告
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        return self._session

    @property
    def name(self) -> str:
        return "lama"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        size = self.INPUT_SIZE
        height, width = image.shape[:2]
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if not binary.any():
            return image.copy()

        # 非等比縮放——這是上游參考實作的做法。等比加黑邊會令模型把
        # 容量浪費在邊距上，而縮放的失真會在放大回去時還原。
        small_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(binary, (size, size), interpolation=cv2.INTER_NEAREST)

        image_blob = small_image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        mask_blob = (small_mask > 0).astype(np.float32)[None, None]

        output = self.session.run(None, {"image": image_blob, "mask": mask_blob})[0]

        # 這個匯出的輸出已經是 0–255，不是 0–1。
        filled = np.clip(output[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        return cv2.resize(filled, (width, height), interpolation=cv2.INTER_LANCZOS4)


class ApiInpainter:
    """經由雲端模型做**語意編輯**——不只是移除，而是聽指令改變內容。

    「移除這個人」「把她的動作改成揮手」「在這裡加一隻小狗」——
    這一類工作在本地 inpainting 做不到（LaMa 只會由邊界往內填，
    它沒有「理解」的能力），必須交給生成模型。

    ⚠️ **API 沒有 mask 參數**，所以做不到真正的遮罩式 inpainting。
    這裡的做法是把選取範圍**畫在第二張參考圖上**，並在指令裡說明
    ——不然模型不知道「加一隻小狗」是要加在哪裡。

    安全性由合成器保證：模型就算把整張裁切圖都改爛，
    合成器也只會取遮罩內的像素，遮罩外仍然逐位元組不變。
    """

    def __init__(
        self,
        model: str,
        prompt: str,
        *,
        resolution: str | None = None,
        client=None,
        show_selection: bool = True,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.resolution = resolution
        self.show_selection = show_selection
        self._client = client
        self.last_cost: float | None = None
        self.last_seconds: float | None = None

    @property
    def name(self) -> str:
        return f"api:{self.model}"

    def _client_or_default(self):
        if self._client is None:
            from photoman.providers import client_from_config

            self._client = client_from_config()
        return self._client

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from photoman.providers import ProviderError

        binary = np.asarray(mask) > 0
        if not binary.any():
            return image.copy()

        references = [image]
        instruction = self.prompt.strip()
        if self.show_selection:
            references.append(_highlight(image, binary))
            instruction = (
                f"{instruction}\n\n"
                "The second image marks the area to change with a red tint. "
                "Only change what is inside the marked area. "
                "Leave everything outside the marked area exactly as it is."
            )

        try:
            response = self._client_or_default().edit(
                image,
                instruction,
                model=self.model,
                resolution=self.resolution,
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - 統一轉成看得懂的訊息
            raise ProviderError(f"呼叫 {self.model} 失敗：{exc}") from exc

        self.last_cost = response.cost_usd
        self.last_seconds = response.seconds
        return response.image


def _highlight(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把選取範圍畫成紅色半透明——用來告訴模型要改哪裡。"""
    tinted = image.astype(np.float32)
    red = np.zeros_like(tinted)
    red[..., 0] = 255.0
    tinted[mask] = tinted[mask] * (1.0 - alpha) + red[mask] * alpha
    return np.clip(tinted, 0, 255).astype(np.uint8)


def get_inpainter(method: str, **options) -> Inpainter:
    """由名稱取得引擎。

    命名採用 ``family`` 或 ``family:variant``——``api:<model-id>``
    留給日後的遠端模型。
    """
    family = method.split(":", 1)[0]
    if family == "telea":
        return TeleaInpainter(**options)
    if family == "ns":
        return TeleaInpainter(flags=cv2.INPAINT_NS, **options)
    if family == "lama":
        from photoman.paths import lama_model_path

        options.setdefault("model_path", lama_model_path())
        return LamaInpainter(**options)
    if family == "api":
        # 命名是 api:<model-id>，例如 api:bytedance-seed/seedream-5-0-lite
        _, _, model = method.partition(":")
        if not model:
            raise ValueError("api: 後面要接模型代號，例如 api:bytedance-seed/seedream-5-0-lite")
        return ApiInpainter(model=model, **options)
    raise ValueError(
        f"未知的 inpainting 方法：{method}\n"
        "支援 telea、ns、lama，以及 api:<model-id>。"
    )
