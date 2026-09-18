"""合成器——本案的核心保證所在（見 docs/design.md §5）。

**這是整個專案唯一不可妥協的部分。**

鐵律：文字提示是請求，遮罩是約束。而「遮罩外不變」這條保證，
只有在**本地像素空間合成**時才成立。模型廠商自己都承認會漂移——
FLUX.1-Fill-dev 的官方模型卡寫明填充區以外有輕微色偏，
diffusers 的維護者說明 inpaint 權重是**刻意**訓練成跨接縫融合的。
所以保證必須是這個檔案的性質，不是關於任何模型的聲明。

模組內所有座標都是**原圖座標**（§6.1a：幾何是最後的輸出設定，
遮罩永遠不需要變換）。
"""

from __future__ import annotations

import numpy as np

from photoman.color import linear_to_srgb, srgb_to_linear

CropBox = tuple[int, int, int, int]  # (x, y, w, h)


def plan_crop(
    mask: np.ndarray,
    *,
    margin_ratio: float = 0.35,
    min_margin_px: int = 64,
) -> CropBox:
    """由遮罩算出要送給模型的裁切框。

    ``margin_ratio`` **按遮罩外接框的比例取，不是固定像素數**（§5.3.4）。
    寫死的話，小遮罩會把大量預算浪費在邊距上，大遮罩則完全沒有上下文。

    邊距決定**縮小倍率**，而縮小倍率決定畫質：放大回原生解析度那一步
    是有損的，它把有效頻率上限砍半。羽化救不了它——那是頻寬問題。
    """
    rows = np.flatnonzero(np.any(mask > 0, axis=1))
    cols = np.flatnonzero(np.any(mask > 0, axis=0))
    if rows.size == 0 or cols.size == 0:
        raise ValueError("遮罩是空的，無法規劃裁切框")

    height, width = mask.shape[:2]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    margin = max(min_margin_px, int(round(max(bottom - top, right - left) * margin_ratio)))

    x0 = max(0, left - margin)
    y0 = max(0, top - margin)
    x1 = min(width, right + margin)
    y1 = min(height, bottom + margin)
    return (x0, y0, x1 - x0, y1 - y0)


def downscale_factor(crop: CropBox, model_size: int) -> float:
    """裁切圖送進模型時的縮小倍率。

    ``≤ 1.5`` 沒問題；``~2`` 是邊緣；``> 2–3`` 明顯比周圍低頻；``> 4`` 一眼看出。
    """
    _, _, width, height = crop
    return max(width, height) / float(model_size)


def composite_patch(
    original: np.ndarray,
    patch: np.ndarray,
    crop: CropBox,
    alpha: np.ndarray,
) -> np.ndarray:
    """把生成的一塊貼回原圖，**只寫入 ``alpha > 0`` 的像素**。

    ``original``
        原圖（線性光 float32）。不會被修改。
    ``patch``
        模型生成的一塊，尺寸必須等於 ``crop`` 的闊高。
    ``crop``
        ``patch`` 在原圖中的位置。
    ``alpha``
        混合遮罩，**與原圖同尺寸**，核心為 1、邊界漸變到 0。
        由 :func:`photoman.mask.make_blend_alpha` 產生。

    回傳新的陣列。``alpha == 0`` 的像素與 ``original`` **逐 bit 相同**。

    實作上刻意用「複製原圖再只寫入內部」，而不是對整張圖做
    ``original*(1-a) + patch*a``。後者在 ``a == 0`` 時雖然數學上等於原值，
    但浮點運算會令 ``-0.0`` 變成 ``+0.0``——位元就不同了，而契約要求的是位元相同。
    """
    x, y, width, height = crop
    if patch.shape[:2] != (height, width):
        raise ValueError(f"生成塊的尺寸 {patch.shape[:2]} 不等於裁切框的尺寸 {(height, width)}")
    if alpha.shape[:2] != original.shape[:2]:
        raise ValueError(f"混合遮罩的尺寸 {alpha.shape[:2]} 不等於原圖的尺寸 {original.shape[:2]}")

    result = original.copy()
    region_alpha = alpha[y : y + height, x : x + width]
    inside = region_alpha > 0.0
    if not inside.any():
        return result

    target = result[y : y + height, x : x + width]
    weight = region_alpha[inside]
    if target.ndim == 3:
        weight = weight[:, None]

    # 右邊先求值，所以就地賦值不會污染計算。
    target[inside] = target[inside] * (1.0 - weight) + patch[inside] * weight
    return result


def verify_unchanged(
    original: np.ndarray,
    result: np.ndarray,
    alpha: np.ndarray,
) -> float:
    """量度 ``alpha == 0`` 區域的最大絕對差。

    ★ **契約要求這個數字恰為 0**，不是「接近零」。
    這是可以寫成斷言的性質，也是 §5.4 的測試一。
    """
    untouched = alpha <= 0.0
    if not untouched.any():
        return 0.0
    return float(np.abs(result[untouched] - original[untouched]).max())


def ring_delta(
    original: np.ndarray,
    returned: np.ndarray,
    denoise_mask: np.ndarray,
    crop: CropBox,
    *,
    inset_px: int = 0,
) -> float:
    """量度模型在「被告知不要改」的區域改了多少——即校驗環量度。

    裁切圖上留一圈校驗環，它同時做三件事（§5.2）：
    給模型上下文、我們丟棄它的渲染結果、以及**當作校驗和**。
    模型若守規矩，回傳的校驗環應該與送出的一模一樣。

    **必須在合成之前量度**：偏離過大就拒絕這張結果，不合成。
    這樣就不是用提示詞拜託模型守規矩，而是用幾何驗證它有沒有守。

    ``original`` 與 ``returned`` 都應該是 **8-bit sRGB**（送出去與收回來的
    那個空間），因為門檻是在那個尺度上定的：
    ``0`` = 逐位元；``≤ 2`` = 無害的 VAE 量化；``> 2`` = 外滲。

    ``inset_px`` 排除緊貼遮罩邊界的一圈。模型在邊界附近的少許改動是
    預期之內的（那正是它在做的工作），要量的是**遠處有沒有被動**。

    回傳 ``nan`` 表示沒有校驗環可用（遮罩填滿了整個裁切圖）。
    """
    if original.shape != returned.shape:
        raise ValueError(
            f"送出的裁切圖 {original.shape} 與收回的 {returned.shape} 尺寸不符——"
            "無法對位，應該拒絕這個結果"
        )

    x, y, width, height = crop
    mask_region = denoise_mask[y : y + height, x : x + width] > 0

    ring = ~mask_region
    if inset_px > 0:
        # 把遮罩膨脹 inset_px 之後再反轉，等於「離遮罩至少 inset_px」。
        from photoman.mask import dilate

        ring = ~dilate(mask_region, inset_px)

    if not ring.any():
        return float("nan")

    difference = np.abs(returned[ring].astype(np.int16) - original[ring].astype(np.int16))
    return float(difference.max())


def composite_patch_into_bytes(
    base: np.ndarray,
    patch: np.ndarray,
    crop: CropBox,
    alpha: np.ndarray,
) -> np.ndarray:
    """在 **uint8 sRGB** 層合成——底圖存原檔位元組，這是實際使用的那一個。

    與 :func:`composite_patch` 的分工：

    - :func:`composite_patch` 是**數學核心**（線性光 float32），
      負責混合本身，也是所有數值測試的對象。
    - 本函數是**儲存層的入口**，負責 uint8 ↔ 線性光的轉換，
      以及把「遮罩外不變」落實成**字面上的位元組複製**。

    為甚麼要在 uint8 層做（§6.1c）：底圖以原檔編碼儲存（42 MP 是 127 MB，
    而不是 float32 的 506 MB）。合成時複製一份原圖位元組，只在遮罩內寫入——
    **遮罩外的像素根本沒有經過任何浮點運算，所以不可能有往返誤差。**

    混合本身仍然在線性光做（§5.5）——在 gamma 編碼空間混合會出現暗邊或亮邊。

    回傳新的 uint8 陣列。``alpha == 0`` 的位元組與 ``base`` 完全相同。
    """
    if base.dtype != np.uint8 or patch.dtype != np.uint8:
        raise ValueError(f"這個函數只處理 uint8，收到 {base.dtype} 與 {patch.dtype}")

    x, y, width, height = crop
    if patch.shape[:2] != (height, width):
        raise ValueError(f"生成塊的尺寸 {patch.shape[:2]} 不等於裁切框的尺寸 {(height, width)}")

    region_alpha = alpha[y : y + height, x : x + width]
    inside = region_alpha > 0.0
    if not inside.any():
        return base.copy()

    # 只有裁切區需要線性化——這是這個設計省記憶體的地方。
    base_region = srgb_to_linear(base[y : y + height, x : x + width].astype(np.float32) / 255.0)
    patch_region = srgb_to_linear(patch.astype(np.float32) / 255.0)

    weight = region_alpha[inside]
    if base_region.ndim == 3:
        weight = weight[:, None]
    blended = base_region[inside] * (1.0 - weight) + patch_region[inside] * weight

    result_region = base_region.copy()
    result_region[inside] = blended

    encoded = np.floor(np.clip(linear_to_srgb(result_region), 0.0, 1.0) * 255.0 + 0.5).astype(
        np.uint8
    )

    # ★ 只寫入遮罩內的像素。
    #
    # 刻意**不**寫回整個裁切區——那樣做會令遮罩外的像素經過
    # uint8 → 線性光 → uint8 的往返，於是保證就建立在「往返精確」
    # 這個假設上。實測那個往返確實精確，但保證不應該依賴它：
    # 只寫入內部，遮罩外的位元組就是真正未被碰過的。
    result = base.copy()
    target = result[y : y + height, x : x + width]
    target[inside] = encoded[inside]
    return result
