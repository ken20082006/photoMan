"""可重播的編輯圖層。

**一層只需要三個東西就能精確重播：裁切框、遮罩、貼片。**

為甚麼是重播而不是把成品存下來：42 MP 的成品是 127 MB，三層就是 381 MB。
而貼片只有零點幾 MB。重播本身是「複製 + 幾次貼上」，毫秒級——
所以復原與重做不需要任何額外的儲存，成品也不必進記憶體。

**為甚麼這樣仍然精確。** ``composite_patch_into_bytes`` 是純函數，
而它需要的 alpha 是 ``blend_alpha_for(遮罩, feather_px)`` 的純函數
——**與 ``dilate_px`` 無關**（實測：由 8 改成 40，alpha 逐位元相同），
所以重播不必知道它。同一組輸入必然得到同一組 alpha
（morphology 與 distanceTransform 都是確定性的），
所以重播的結果與當初那一次**逐位元組相同**。已用真實的 LaMa 實測過。

**保證由此升級。** 每一步都在 alpha 為 0 的地方做字面上的位元組複製，
所以每一步都保持「這一層的遮罩之外等於它的輸入」。歸納下來就是
**「所有圖層遮罩的聯集之外，位元組等於原檔」**——比單層的版本更強，
而且同樣是架構的性質，不是關於任何模型的聲明。

**遮罩只留裁切區那一小塊。** ``plan_crop`` 保證裁切框完整包住遮罩
（框是取前處理後遮罩的外接矩形再加邊距），所以框外的遮罩必然是空的，
展開回原圖大小是精確的。整張遮罩在 42 MP 是 42 MB／層，裁切區約 0.5 MB／層。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from photoman.composite import composite_patch_into_bytes
from photoman.mask import blend_alpha_for
from photoman.project import GenerativeLayer


@dataclass(frozen=True)
class EditLayer:
    """一層已接受的編輯：持久化的描述 + 重播所需的像素。"""

    layer: GenerativeLayer
    mask: np.ndarray  # 裁切區 bool；框外一定是空的
    patch: np.ndarray  # 裁切區 uint8，已完成色彩與顆粒對齊

    @classmethod
    def from_edit(
        cls,
        layer: GenerativeLayer,
        mask: np.ndarray,
        patch: np.ndarray,
    ) -> EditLayer:
        """由一次編輯的產物建立一層。

        遮罩超出裁切框就**在這裡**報錯——此時還握有完整的遮罩。
        留到重播才發現的話，我們手上只剩一張被裁掉一角的遮罩，
        既看不出少了甚麼，也無從還原。
        """
        x, y, width, height = layer.crop
        if patch.shape[:2] != (height, width):
            raise ValueError(f"貼片的尺寸 {patch.shape[:2]} 不等於裁切框的尺寸 {(height, width)}")

        whole = np.asarray(mask) > 0
        region = whole[y : y + height, x : x + width]
        if int(np.count_nonzero(region)) != int(np.count_nonzero(whole)):
            raise ValueError(
                "遮罩超出了裁切框——裁切框必須完整包住遮罩，否則重播時框外那部分不會被改到"
            )
        return cls(layer=layer, mask=region.copy(), patch=patch)

    def full_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """把裁切區的遮罩展開回原圖座標。

        復原時放回介面的也是這一個——使用者看到的就是自己當初圈的東西。
        """
        whole = np.zeros(shape, dtype=bool)
        x, y, width, height = self.layer.crop
        whole[y : y + height, x : x + width] = self.mask
        return whole


def render(source: np.ndarray, layers: Sequence[EditLayer]) -> np.ndarray:
    """由原檔位元組重播所有圖層，回傳 uint8 sRGB。

    空清單回傳 ``source`` 的複本（不是 ``source`` 本身——
    呼叫方常常會把結果當成「目前的成品」，不該有機會改到原檔）。

    **每次都整個重算，不做增量合成。** 重算是複製加幾次貼上，
    而復原本來就需要它。這樣「成品永遠等於 render(source, layers)」
    是一個不變式，不是一個要小心維護的但書。
    """
    working = source.copy()
    shape = source.shape[:2]
    for edit in layers:
        alpha = blend_alpha_for(
            edit.full_mask(shape),
            feather_px=edit.layer.feather_px,
        )
        working = composite_patch_into_bytes(working, edit.patch, edit.layer.crop, alpha)
    return working
