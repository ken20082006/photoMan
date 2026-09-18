"""一次局部生成編輯的完整流程（見 docs/design.md §5.3.2）。

這是把各塊接起來的地方：

```
遮罩前處理（兩個遮罩）→ 規劃裁切 → 取出裁切圖 → 交給引擎
   → 量度校驗環 → 合成（只寫入遮罩內）
```

**合成是最後一步，而且它無論如何都會守住保證。** 校驗環是用來判斷
「這張結果值不值得用」的品質閘門，不是安全閘門——就算引擎把整張裁切圖
都改爛了，合成器也只會取遮罩內的像素。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photoman.composite import CropBox, composite_patch_into_bytes, plan_crop, ring_delta
from photoman.inpaint import get_inpainter
from photoman.mask import DEFAULT_DILATE_PX, DEFAULT_FEATHER_PX, prepare_masks
from photoman.match import match_colour, match_grain
from photoman.store import mask_digest

# 校驗環的參考門檻（見 §5.4）。
#
# ⚠️ **實測推翻了「一個門檻通用」的想法**（2026-09-18）：
#
#   引擎      校驗環   填補品質
#   Telea      0.0     差（平滑多邊形灰塊）
#   LaMa      24–27    好（幾乎看不出接縫）
#
# LaMa 是遮罩原生的模型，卻把「被告知不要改」的區域改了 24–27 級——
# 因為它整張重繪，加上 512 往返的重新取樣。**這正是 §5.6 的實證：
# 保證只能來自合成器，不能來自模型。**
#
# 但也因此，**校驗環的大小與填補品質無關**。它是診斷數字，不是品質閘門。
# 對本地引擎尤其如此——它們本來就會重繪整個裁切圖，而我們從來不採用
# 遮罩外的部分，所以那些漂移對我們無害。
#
# → 所以 :meth:`EditResult.is_trustworthy` 現在**要明確給門檻才會判斷**。
# 只有在「模型聲稱會保留遮罩外」時（例如 API 模型）才有意義。
API_THRESHOLD = 2.0


@dataclass(frozen=True)
class EditResult:
    """一次編輯的結果。"""

    image: np.ndarray  # uint8 sRGB，合成後的完整影像
    crop: CropBox
    mask_sha256: str
    checksum: float  # 校驗環量度；nan 代表沒有校驗環可用。**診斷用，不是品質分數**
    method: str

    def is_trustworthy(self, threshold: float | None = None) -> bool:
        """這張結果是否落在指定的校驗環門檻內。

        **不給門檻就回傳 ``True``**——因為校驗環的大小與填補品質無關
        （見上方的實測表），憑它自動否決會錯殺。

        門檻只在「這個模型聲稱會保留遮罩外」時才有意義，
        例如 API 模型：``≤ 2`` 是無害的 VAE 量化，``> 2`` 是外滲。
        """
        if threshold is None or np.isnan(self.checksum):
            return True
        return self.checksum <= threshold


def apply_generative_edit(
    base: np.ndarray,
    mask: np.ndarray,
    *,
    method: str = "telea",
    dilate_px: int = DEFAULT_DILATE_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
    margin_ratio: float = 0.35,
    min_margin_px: int = 64,
    checksum_inset_px: int = 0,
    match_texture: bool = True,
    **engine_options,
) -> EditResult:
    """對 ``base`` 套用一次局部生成編輯。

    ``base``
        底圖，uint8 sRGB、**原圖座標**（§6.1c）。
    ``mask``
        使用者圈選的範圍，bool、原圖座標。

    回傳的 ``image`` 是新的陣列；``base`` 不會被修改，
    而且 ``mask`` 對應的混合範圍以外，位元組與 ``base`` 完全相同。
    """
    if base.dtype != np.uint8:
        raise ValueError(f"底圖必須是 uint8 sRGB，得到 {base.dtype}")
    if mask.shape[:2] != base.shape[:2]:
        raise ValueError(f"遮罩的尺寸 {mask.shape[:2]} 不等於底圖的尺寸 {base.shape[:2]}")
    if not (mask > 0).any():
        raise ValueError("遮罩是空的——沒有東西要改")

    denoise, alpha = prepare_masks(mask, dilate_px=dilate_px, feather_px=feather_px)
    crop = plan_crop(denoise, margin_ratio=margin_ratio, min_margin_px=min_margin_px)

    x, y, width, height = crop
    base_crop = base[y : y + height, x : x + width].copy()
    denoise_crop = denoise[y : y + height, x : x + width]

    engine = get_inpainter(method, **engine_options)
    patch = engine.inpaint(base_crop, denoise_crop)

    # 校驗環必須在合成之前量度——而且要在色彩與顆粒校正之前，
    # 因為它量度的是**模型的行為**，不是我們加工之後的結果（§5.2）。
    delta = ring_delta(base_crop, patch, denoise, crop, inset_px=checksum_inset_px)

    if match_texture:
        # 第 9、10 步：把生成塊融入周圍。**這是令填補消失的關鍵。**
        # 少了這兩步，任何引擎——本地或 API——都會留下一個
        # 與遮罩同形狀的平滑色塊。
        patch = match_colour(patch, base_crop, denoise_crop)
        patch = match_grain(patch, base_crop, denoise_crop)

    image = composite_patch_into_bytes(base, patch, crop, alpha)

    return EditResult(
        image=image,
        crop=crop,
        mask_sha256=mask_digest(mask),
        checksum=delta,
        method=method,
    )
