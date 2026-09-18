"""點擊分割——「指一下物件就選中」（見 docs/design.md §7.2）。

**這是「很難仔細地叫 AI 修那一部分」的技術答案。**
不是更好的提示詞，是**用滑鼠取代文字**：文字是低頻寬的空間描述語言，
「把左邊那個人手上的膠袋拿走」要求模型先猜「膠袋」指的是哪一團像素。
空間問題要用空間通道解決。

用 MobileSAM：授權 Apache-2.0、官方 ONNX、CPU 上 0.7–0.9 秒。

⚠️ **一個驅動整個設計的架構事實：SAM 系列內部一律縮到 1024。**
所以**選取的時間與原圖大小幾乎無關**——42 MP 與 12 MP 一樣快。
這是少數不受原圖大小影響的環節。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

Point = tuple[int, int]
Box = tuple[int, int, int, int]  # (x0, y0, x1, y1)

# SAM 的圖像編碼器固定的工作尺寸。位置編碼是為這個尺寸訓練的，
# 不可以改——這是模型的事實，不是可調參數。
ENCODER_SIZE = 1024

# 遮罩輸出的低解析度網格。解碼器先在這個尺寸產生，再放大到目標尺寸。
LOW_RES = 256


class Segmenter(Protocol):
    """由使用者的指示取出物件的遮罩。"""

    def segment(
        self,
        image: np.ndarray,
        *,
        point: Point | None = None,
        box: Box | None = None,
        negative_points: Sequence[Point] = (),
    ) -> np.ndarray:
        """回傳 bool 遮罩，**原圖座標、與 ``image`` 同尺寸**。

        ``point`` 是「這個就是了」，``negative_points`` 是「這些不是」。
        兩者可以併用：先點一下物件，若選多了就再點一下要排除的地方。
        """
        ...


class MobileSamSegmenter:
    """MobileSAM——點一下物件就取出它的遮罩。

    **嵌入會快取。** 圖像編碼器是慢的那一半（CPU 上約 0.7–0.9 秒），
    解碼器很快。所以同一個圖像上的重複點擊只需要付一次編碼成本——
    這對互動式使用是必要的：使用者按一下要立即有反應。
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        *,
        encoder_session=None,
        decoder_session=None,
    ) -> None:
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self._encoder = encoder_session
        self._decoder = decoder_session
        self._cache_key: tuple | None = None
        self._cache: tuple[np.ndarray, np.ndarray, float] | None = None

    # ── 工作階段 ────────────────────────────────────────────────

    def _sessions(self):
        if self._encoder is None or self._decoder is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - 環境相依
                raise ImportError(
                    'MobileSAM 需要 onnxruntime。安裝：pip install -e ".[models]"'
                ) from exc
            ort.set_default_logger_severity(3)
            for path in (self.encoder_path, self.decoder_path):
                if not path.exists():
                    raise FileNotFoundError(
                        f"找不到 MobileSAM 模型：{path}\n見 docs/PROGRESS.md 的下載說明。"
                    )
            providers = ["CPUExecutionProvider"]
            if self._encoder is None:
                self._encoder = ort.InferenceSession(str(self.encoder_path), providers=providers)
            if self._decoder is None:
                self._decoder = ort.InferenceSession(str(self.decoder_path), providers=providers)
        return self._encoder, self._decoder

    def _embed(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """計算（或取出快取的）圖像嵌入。

        快取以「形狀 + 粗略內容摘要」為鍵。用粗略摘要而不是完整雜湊，
        是因為完整雜湊 42 MP 要几百毫秒——而那個成本本身就違背了快取的目的。
        """
        key = (image.shape, image[::64, ::64].tobytes())
        if self._cache_key == key and self._cache is not None:
            return self._cache

        encoder, _ = self._sessions()
        height, width = image.shape[:2]

        # 縮到長邊 1024。模型內部會再補邊到 1024×1024，
        # 所以像素不會被進一步拉伸——比例保持原樣。
        scale = ENCODER_SIZE / max(height, width)
        if scale >= 1.0:
            # 已經比 1024 小——原樣送，模型內部補邊即可。
            # 放大再送只會增加成本而不增加資訊。
            small = image
        else:
            small = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        embeddings = encoder.run(None, {"input_image": small.astype(np.float32)})[0]
        self._cache_key = key
        self._cache = (embeddings, small, scale)
        return self._cache

    def embed(self, image: np.ndarray) -> None:
        """預先計算嵌入。介面可以在載入圖片之後、使用者點擊之前呼叫它。

        這樣第一次點擊就不必等編碼——使用者只會感覺到「按下去就有反應」。
        """
        self._embed(image)

    # ── 分割 ────────────────────────────────────────────────────

    def segment(
        self,
        image: np.ndarray,
        *,
        point: Point | None = None,
        box: Box | None = None,
        negative_points: Sequence[Point] = (),
    ) -> np.ndarray:
        if point is None and box is None:
            raise ValueError("需要一個 point 或一個 box——沒有的話不知道要選甚麼")

        _, decoder = self._sessions()
        embeddings, small, scale = self._embed(image)
        small_height, small_width = small.shape[:2]
        height, width = image.shape[:2]

        # 座標要換到編碼器的工作空間（縮圖）。模型內部補邊在右下，
        # 左上對齊，所以縮圖座標就是工作空間座標。
        coordinates: list[list[float]] = []
        labels: list[float] = []

        # box 在 SAM 裡是兩個點：左上（標籤 2）與右下（標籤 3）。
        if box is not None:
            x0, y0, x1, y1 = box
            coordinates += [
                [x0 * scale, y0 * scale],
                [x1 * scale, y1 * scale],
            ]
            labels += [2.0, 3.0]
        if point is not None:
            coordinates.append([point[0] * scale, point[1] * scale])
            labels.append(1.0)
        for negative in negative_points:
            coordinates.append([negative[0] * scale, negative[1] * scale])
            labels.append(0.0)

        # orig_im_size 直接給**原圖**尺寸，所以遮罩一次就回到原圖解析度，
        # 不必再做第二次縮放（多一次縮放就多一次邊緣失真）。
        masks, scores, _ = decoder.run(
            None,
            {
                "image_embeddings": embeddings,
                "point_coords": np.array([coordinates], dtype=np.float32),
                "point_labels": np.array([labels], dtype=np.float32),
                "mask_input": np.zeros((1, 1, LOW_RES, LOW_RES), dtype=np.float32),
                "has_mask_input": np.array([0], dtype=np.float32),
                "orig_im_size": np.array([height, width], dtype=np.float32),
            },
        )

        # 多個候選取分數最高的一個。分數是模型的自我評估，
        # 但這裡沒有其他訊號可用——而使用者看得到結果，錯了再點一次就好。
        best = int(np.argmax(scores[0]))
        return masks[0][best] > 0


def get_segmenter(model_dir: str | Path | None = None) -> Segmenter:
    """取得預設的分割引擎。"""
    from photoman.paths import MODELS_DIR

    directory = Path(model_dir) if model_dir is not None else MODELS_DIR
    return MobileSamSegmenter(
        directory / "mobile_sam_encoder.onnx",
        directory / "mobile_sam_decoder.onnx",
    )
