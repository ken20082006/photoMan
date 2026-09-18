"""專案與圖層的資料模型（見 docs/design.md §6）。

**這個模組的核心職責是「重跑不可以悄悄覆蓋人工修改過的內容」**
——mangaMan 為此吃過一次苦（見其 `PROGRESS.md` 第 7 節）。
三個機制：

1. **永久 ID。** 圖層一旦建立，ID 永不改變。不可以「重新載入 = 重新派 ID」，
   這樣做第二天那個專案就廢了，因為所有人工調整都會對不上。
2. **``locked`` 旗標。** 人工修改過的層，重跑不准覆蓋。
3. **快取鍵是一條鏈。** 第 N 層的鍵包含前面所有層的鍵，
   所以上游改了，下游的快取自然失效——不需要手動傳播失效。
"""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


def new_layer_id() -> str:
    """派一個新的永久 ID。

    六個十六進位字元（約 1600 萬種）對單一專案的圖層數目而言足夠，
    而且短得可以印在介面上給人看。
    """
    return f"L_{secrets.token_hex(3)}"


class SourceRef(BaseModel):
    """對原檔的引用——**只記路徑與雜湊，不複製檔案**。

    匯出時需要的 ICC 與 EXIF 位元組，是回到原檔重新讀取的。
    這樣做有兩個好處：專案檔不會被 base64 撐大，而且原檔永遠是唯一真相。

    ⚠️ **代價：原檔被移動或改動，專案就失效。** 所以記下 ``sha256``，
    載入時核對並明確報錯，而不是默默用一張不同的圖繼續跑。
    """

    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    format: str
    width: int
    height: int
    bit_depth: int
    icc_description: str | None = None
    had_orientation_tag: bool = False


class OutputSettings(BaseModel):
    """幾何設定——**不是圖層**（§6.1a）。

    裁剪、旋轉、拉直只是匯出時的取景。這個決定令遮罩永遠存在原圖座標，
    於是「幾何變更令下游遮罩失效」這整個問題類別不存在。
    """

    crop: tuple[int, int, int, int] | None = None
    rotate_deg: float = 0.0
    straighten: float | None = None
    format: str = "png"
    bit_depth: int = 8


class Checksum(BaseModel):
    """校驗環量度——模型在「被告知不要改」的區域改了多少（§5.2）。"""

    model_config = ConfigDict(frozen=True)

    max_abs_diff: float
    at: str


class LayerBase(BaseModel):
    """所有圖層共有的部分。"""

    id: str = Field(default_factory=new_layer_id)
    enabled: bool = True
    locked: bool = False

    def signature(self) -> dict[str, Any]:
        """影響輸出的參數——用來算快取鍵。

        子類別必須覆寫。**只有真正影響像素的參數可以進來**：
        多放一個會令快取無效（白花錢），少放一個會令快取回錯結果（更糟）。
        """
        raise NotImplementedError


class DeterministicLayer(LayerBase):
    """確定性調整：曝光、白平衡、曲線、對比（§4）。

    契約是**無損**，所以這一層不經過任何模型。

    逐點運算與裁切可交換，所以 N 個這一類的層可以合成成單一函數、
    一次套用——沒有中間量化損失（§6.1c）。
    """

    type: Literal["deterministic"] = "deterministic"
    params: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> dict[str, Any]:
        return {"type": self.type, "params": self.params}


class GenerativeLayer(LayerBase):
    """局部生成：移除物件、修補、擴展畫布（§4）。

    契約是 **``M_blend`` 羽化帶以外逐 bit 不變**。

    ``mask_sha256`` 要進簽名——使用者改了遮罩，快取必須失效。
    漏了這一項的話，改了遮罩卻拿到舊結果，而且沒有任何跡象。
    """

    type: Literal["generative"] = "generative"
    method: str  # "telea" | "lama" | "api:<model-id>"
    crop: tuple[int, int, int, int]
    mask_sha256: str
    prompt: str | None = None
    feather_px: int = 12
    scale: float | None = None

    # ── 跑完之後才填的 ──
    result_file: str | None = None
    checksum: Checksum | None = None
    cache_key: str | None = None

    def signature(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "method": self.method,
            "prompt": self.prompt,
            "crop": list(self.crop),
            "feather_px": self.feather_px,
            "mask": self.mask_sha256,
        }


Layer = Annotated[DeterministicLayer | GenerativeLayer, Field(discriminator="type")]


class Project(BaseModel):
    """一個專案：一張原圖 + 一串圖層 + 匯出設定。"""

    schema_version: int = SCHEMA_VERSION
    name: str
    source: SourceRef
    output: OutputSettings = Field(default_factory=OutputSettings)
    layers: list[Layer] = Field(default_factory=list)

    def enabled_layers(self) -> list[Layer]:
        """只有啟用中的層參與運算。停用不是刪除——使用者可以再打開。"""
        return [layer for layer in self.layers if layer.enabled]

    def layer_by_id(self, layer_id: str) -> Layer | None:
        return next((layer for layer in self.layers if layer.id == layer_id), None)

    def cache_keys(self) -> dict[str, str]:
        """為每一層算出快取鍵——**這是一條鏈**。

        第 N 層的鍵包含第 N-1 層的鍵，所以上游任何改動都會令下游失效。
        這是刻意的：手動傳播失效一定會漏，而漏掉的後果是拿到錯的結果
        卻以為是對的。
        """
        keys: dict[str, str] = {}
        previous = self.source.sha256
        for layer in self.enabled_layers():
            payload = json.dumps(
                {"upstream": previous, **layer.signature()},
                sort_keys=True,
                ensure_ascii=False,
            )
            previous = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            keys[layer.id] = previous
        return keys
