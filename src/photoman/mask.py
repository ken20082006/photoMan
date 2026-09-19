"""遮罩前處理，以及兩個遮罩的衍生（見 docs/design.md §5.3.1、§5.3.3）。

**兩個遮罩，不是一個：**

- 去雜訊遮罩 ``M_denoise``：交給模型。接近二值、稍微膨脹。
  **不可以是寬闊的柔和漸變**——PELC（CVPR 2026）證明在潛空間用軟遮罩做
  線性混合不等於像素空間混合，會產生無效的潛向量。
- 混合遮罩 ``M_blend``：交給合成器。羽化在這裡。

**填內洞不是可選項。** SAM 產生的遮罩是有洞的、不連續的；
不填的話，模型會在物件內部看到「保留」的指令，結果是物件沒有被完全移除。
"""

from __future__ import annotations

import cv2
import numpy as np

# 模型的洞在**寫入範圍之外**再擴張多少。
#
# ⚠️ **這不是一個小數字，而且理由是實測出來的**（2026-09-18）：
# 物件在遮罩邊界之外還有柔邊、反鋸齒與陰影。若模型的洞剛好貼著物件，
# 它會看到那些殘留的暗色並**往洞裡延伸**——實測 LaMa 把骰子周圍的洞
# 填成亮度 77，而周圍的布是 114。
#
# 這也解釋了一個反直覺的現象：**精確的遮罩不是正確的輸入。**
# 用點擊分割取到的遮罩貼得極緊，反而比手繪的粗略橢圓效果差。
DEFAULT_DILATE_PX = 16

# 混合羽化。**下限是 8 像素**——小於潛空間壓縮倍數的羽化對模型等於不存在。
# 有用的範圍是 16–32 像素。注意這個羽化是我們自己套用的，
# 不是送去當模型的軟條件。見 §5.3.1。
DEFAULT_FEATHER_PX = 12


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """填滿遮罩的內部孔洞。

    做法：先補一圈背景邊界，然後由角落灌水。灌不到的補圖區域
    就是被封在前景裡面的內洞。

    先補邊界是必要的——若遮罩貼著圖片角落，背景會斷成幾塊，
    直接由 (0,0) 灌水會漏掉其餘的背景區域，然後把它們誤判成內洞。
    """
    mask = (binary > 0).astype(np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    inverse = (1 - padded).astype(np.uint8)
    flood = inverse.copy()
    cv2.floodFill(flood, np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8), (0, 0), 2)
    holes = (flood != 2) & (inverse == 1)
    return (padded.astype(bool) | holes)[1:-1, 1:-1]


def remove_specks(binary: np.ndarray, min_area: int) -> np.ndarray:
    """移除面積小於 ``min_area`` 的連通分量。

    分割模型會在背景紋理上產生零星的小塊；它們不是使用者選的東西，
    而且會令遮罩的邊界變得崎嶇。
    """
    mask = (binary > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def dilate(binary: np.ndarray, radius_px: int) -> np.ndarray:
    """以圓形結構元素膨脹。

    用圓形而不是方形，因為方形的角會令膨脹後的邊界出現四個尖角，
    羽化之後仍然看得出痕跡。
    """
    if radius_px <= 0:
        return binary.astype(bool)
    size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate((binary > 0).astype(np.uint8), kernel).astype(bool)


def make_denoise_mask(
    mask: np.ndarray,
    *,
    dilate_px: int = DEFAULT_DILATE_PX,
    specks_min_area: int = 64,
) -> np.ndarray:
    """產出交給模型的那一個遮罩：填洞 → 去斑 → 膨脹 → 二值化。

    **膨脹要在羽化之前**（§5.3.3）——這樣模糊會向外延伸進重生區，
    而不是向內侵入要保留的區域。

    回傳二值 bool 陣列。不套用任何模糊：模型要的是接近二值的輸入。
    """
    prepared = fill_holes(mask)
    prepared = remove_specks(prepared, specks_min_area)
    return dilate(prepared, dilate_px)


def make_blend_alpha(
    mask: np.ndarray,
    *,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> np.ndarray:
    """產出合成用的 alpha：核心為 1，向邊界漸變到 0。

    ★ **遮罩以外恰為 0.0。** 這是逐 bit 保證的基礎——
    合成器只寫入 ``alpha > 0`` 的像素，所以這個歸零必須是精確的，
    不是「接近零」。

    輸入應該是**已經膨脹過的**遮罩（``M_blend`` 的範圍），
    這樣 alpha 漸變到 0 的位置才會落在背景上，而不是落在物件的邊緣——
    否則羽化帶會把物件的邊緣混合進生成內容，留下鬼影。
    """
    binary = (mask > 0).astype(bool)
    if feather_px <= 0:
        return binary.astype(np.float32)

    # distanceTransform 對每個前景像素給出「到最近背景像素的距離」，
    # 所以它天然就是一條由邊界向內的漸變。
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(feather_px), 0.0, 1.0).astype(np.float32)
    alpha[~binary] = 0.0
    return alpha


def blend_alpha_for(
    mask: np.ndarray,
    *,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> np.ndarray:
    """由使用者的選取直接算出合成 alpha——**重播時走的就是這一條**。

    與 :func:`prepare_masks` 的關係：那一個會順便多膨脹一次給模型用，
    但**那一層膨脹不影響 alpha**（實測：``dilate_px`` 由 8 改成 40，
    alpha 逐位元相同）。所以重播不需要知道 ``dilate_px``，
    也不必付出那一次膨脹與它 42 MP 的記憶體。

    兩邊共用這一個實作，所以重播不可能與原本那一次算出不同的 alpha。
    """
    return make_blend_alpha(_blend_base(mask, feather_px), feather_px=feather_px)


def prepare_masks(
    mask: np.ndarray,
    *,
    dilate_px: int = DEFAULT_DILATE_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """由使用者的選取一次產出兩個遮罩。

    回傳 ``(denoise_mask, blend_alpha)``。兩層膨脹，各有各的理由：

    ```
    物件 →[+羽化半径]→ 寫入範圍 →[+dilate_px]→ 模型的洞
    ```

    **第一層膨脹量等於羽化半徑**，這樣 alpha 才會在使用者圈選的邊界上
    剛好到達 1.0。少於羽化半徑的話，羽化帶會橫跨物件的外緣——
    那一圈只被部分替換，原物件的深色邊緣會透出來，形成鬼影。

    **第二層是給模型的額外邊距**，讓它看不到物件的柔邊與陰影
    （見 ``DEFAULT_DILATE_PX`` 的說明）。
    """
    blend_base = _blend_base(mask, feather_px)
    denoise = dilate(blend_base, dilate_px)

    return denoise, make_blend_alpha(blend_base, feather_px=feather_px)


def _blend_base(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """寫入範圍：填洞 → 去斑 → 膨脹羽化半徑。

    這是 alpha 與模型遮罩**共同**的起點，所以只有一份實作。
    """
    return dilate(remove_specks(fill_holes(mask), 64), feather_px)
