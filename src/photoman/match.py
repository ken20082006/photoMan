"""色彩與顆粒對齊——把生成塊融入周圍（見 docs/design.md §5.3.2 第 9、10 步）。

**為甚麼需要這一層（實測得出的結論）：**

移除三顆骰子的實驗顯示，**每一個引擎**——本地的 LaMa、Seedream、
Gemini、FLUX、Qwen——都留下同一個形狀的色塊，而那形狀就是**遮罩的形狀**。

共同的原因不是模型，是這兩步沒有做：

| 量度 | 數值 |
|---|---|
| 周圍布料的顆粒能量 | 52.2 |
| LaMa 填補區 | 0.29 倍 |
| seedream-5-0-lite | 0.27 倍 |

填補是**平滑的**，貼在有顆粒的布上，所以輪廓一眼看出。
這是經典的 inpainting 痕跡，而它與模型的好壞無關——
再好的模型也不知道「這張照片的顆粒長甚麼樣」。

**校正必須整塊套用，不可以局部混合。** 局部混合會在填補區內部
製造第二條接縫，等於用一個問題換另一個問題。
"""

from __future__ import annotations

import cv2
import numpy as np

# 計算參考統計時，離遮罩邊界多遠才取樣。
# 太近會取到遮罩邊緣的殘留（那正是我們要擺脫的東西），
# 太遠則可能取到不同的材質區域。
REFERENCE_INSET_PX = 6

# 「模型大致保留了這裡的原內容」的判定門檻（0–255，取三通道最大差）。
#
# 這是 ``intent="edit"`` 用來量模型色偏的依據：**在模型自己保留的內容上量**，
# 因為那裡我們知道真值（原圖）。見 :func:`match_colour` 的說明。
#
# 為甚麼是 8：實測模型的描繪誤差是 1–5 級，而使用者要的改動通常大得多
# （改髮型那一次是 50 級）。門檻放大到 20 會把「使用者要的細微改動」也
# 算成誤差而抹掉——實測偏移量會由 +0.4 漲到 +2.9，也就是愈修愈偏。
PRESERVED_DIFF = 8

# 保留區至少要佔遮罩的這個比例，否則樣本不足，退回用遮罩外的環量。
MIN_PRESERVED_FRACTION = 0.1


def _reference_region(mask: np.ndarray, inset_px: int) -> np.ndarray:
    """遮罩以外、且離邊界至少 ``inset_px`` 的像素。"""
    from photoman.mask import dilate

    return ~dilate(mask, inset_px)


def _clean_shift(mask: np.ndarray) -> tuple[int, int] | None:
    """找一個平移量，令遮罩區取樣到**離物件夠遠**的乾淨背景。

    ⚠️ **``np.roll`` 會繞回，這是一個很容易踩的陷阱。** 第一版用
    「遮罩尺寸的兩倍」當平移量：在 568 像素高的裁切圖裡，遮罩高 272，
    平移量 544——但 568 − 544 = 24，所以實際上只移了 24 像素，
    取到的仍然是**骰子自己的陰影**，顆粒搬運等於沒有做。

    所以不能用「平移多少」思考，要用「平移之後落在哪裡」思考：
    逐一試候選平移量，取第一個與遮罩（再外擴一圈）不重疊的。

    找不到就回傳 ``None``——遮罩太大時確實沒有乾淨的位置可取。
    """
    height, width = mask.shape[:2]
    guard = _reference_region(mask, 4)

    for fraction_y, fraction_x in (
        (0.5, 0.5),
        (0.5, -0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, 0.0),
        (0.0, 0.5),
        (-0.5, 0.0),
        (0.0, -0.5),
        (0.34, 0.34),
        (-0.34, -0.34),
    ):
        offset = (int(height * fraction_y), int(width * fraction_x))
        if offset == (0, 0):
            continue
        if not (np.roll(mask, offset, axis=(0, 1)) & guard).any():
            return offset
    return None


def _edit_offset(
    patch: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """編輯意圖要扣掉的偏移——**在模型自己保留的內容上量**。

    遮罩內有些像素模型是「照抄」原圖的（臉、衣服……），有些是它新畫的
    （頭髮、新加的物件）。前者我們知道真值，所以在那裡量得到模型真正的
    描繪誤差；後者量不到（我們不知道它「應該」是甚麼）。

    在遮罩外的環上量是錯的——那裡的內容與遮罩內不同，模型的誤差也不同。
    實測：背景偏 −10.8 而臉只偏 −5，把背景的偏量套到臉上會令臉提亮 8.9 級，
    那正是使用者看到的色差。

    遮罩內幾乎沒有保留區時（模型把整塊換掉了），退回用環量。
    """
    difference = np.abs(patch.astype(np.int16) - original.astype(np.int16)).max(axis=2)
    preserved = mask & (difference < PRESERVED_DIFF)

    lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_original = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)

    if preserved.sum() < MIN_PRESERVED_FRACTION * mask.sum():
        return lab_original[reference].mean(axis=0) - lab_patch[reference].mean(axis=0)
    return lab_original[preserved].mean(axis=0) - lab_patch[preserved].mean(axis=0)


def match_colour(
    patch: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    *,
    intent: str = "fill",
    inset_px: int = REFERENCE_INSET_PX,
) -> np.ndarray:
    """把生成塊的**整體色調**校正到與周圍一致（LAB 空間的位移）。

    在 LAB 而不是 RGB 做，是因為 LAB 的 L 通道大致對應感知亮度——
    在 RGB 做會令色相跟著偏移。

    ⚠️ **只修均值，不修標準差。這一點是實測修正的**（2026-09-18）。

    第一版做了完整的 Reinhard 轉移（同時縮放標準差）。結果是一片
    **彩色斑塊**：生成塊很平滑（標準差小），周圍的布有顆粒（標準差大），
    所以倍率被放得極高——它把填補裡極微小的變化**放大**成可見的色斑。

    標準差是**顆粒**的事，而顆粒由 :func:`match_grain` 用搬運法處理，
    那條路才是對的（搬來的顆粒是真實的顆粒，不是放大的雜訊）。

    只修均值等於修正曝光與色偏——那正是模型真正會出錯的地方，
    而且這個操作不會放大任何東西。

    ★ **``intent`` 決定「均值要對到甚麼」，這是 2026-09-19 才分開的。**

    ``"fill"``（填補，本機引擎）
        把生成塊遮罩區的均值對到**周圍**。填出來的東西本來就應該像周圍，
        所以這是對的。

    ``"edit"``（編輯，雲端模型）
        **只扣掉模型的色偏，而且是在「模型自己保留的內容」上量它。**

        理由是一個實測到的嚴重失敗：``fill`` 的做法會把使用者**要求**的
        顏色改動抹掉。實測「把衣服改成紅色」：周圍是綠草、模型畫了
        (170, 40, 40) 的紅，經過 ``fill`` 之後變成 **(89, 129, 60)**，
        也就是周圍的綠色——**叫它改紅色，它抹回綠色**。

        ⚠️ **2026-09-20 修正：量的地方不可以是遮罩外的環。**

        第一版用遮罩外的環去量「模型把整張圖偏移了多少」，再套到遮罩上。
        那個假設是「模型的色偏在全圖一致」——**實測不成立**：

        | 改髮型那一次（遮罩內含臉與頭髮） | |
        |---|---|
        | 背景（環）的偏移 | −10.8 |
        | 臉的偏移 | **−5** |
        | 扣掉環的偏移之後，臉的殘餘差 | **+8.9** |

        也就是說，把背景的偏移套到臉上會**過頭**——那正是使用者回報
        「換成・加入還是有點色差」的來源。

        → 現在改為：在遮罩內挑出**模型大致保留了原內容**的像素
        （``|模型 − 原圖| < PRESERVED_DIFF``），在那裡量偏移。
        那裡我們知道真值，所以量到的是模型真正的描繪誤差。

        實測（使用者三個真實圖層、三個不同模型）——「模型保留區」的殘餘差：

        | 案例 | 不修正 | 用環量（舊） | 用保留區量（新） |
        |---|---|---|---|
        | 移除人物 ①（gemini） | −3.0 | −3.5 | **−0.3** |
        | 移除人物 ②（grok） | −1.8 | −1.3 | **−0.3** |
        | 改髮型（gpt-5.4） | −3.5 | +8.9 | **−0.3** |

        遮罩內幾乎沒有保留區時（模型把整塊換掉了，例如純粹的移除），
        退回用遮罩外的環量——那時沒有真值可以對。

    整塊校正，不做局部混合。回傳新的陣列。
    """
    reference = _reference_region(mask, inset_px)
    if not reference.any() or not mask.any():
        return patch

    lab_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_original = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)

    if intent == "edit":
        offset = _edit_offset(patch, original, mask, reference)
    else:
        offset = lab_original[reference].mean(axis=0) - lab_patch[mask].mean(axis=0)

    corrected = np.clip(lab_patch + offset, 0.0, 255.0).astype(np.uint8)

    result = patch.copy()
    result[mask] = cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)[mask]
    return result


def match_grain(
    patch: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = 1.2,
) -> np.ndarray:
    """把生成塊的顆粒換成周圍的顆粒。

    **這是令填補消失的關鍵一步。** 生成模型產生的內容在低頻（顏色、
    光影）上是對的，但它沒有重現這張照片的顆粒——而人眼對「這裡比較滑」
    極其敏感，即使色調完全正確也會看出輪廓。

    ⚠️ **做法是「搬運」而不是「合成」，這一點是實測修正的**（2026-09-18）。
    第一版用合成雜訊：量度周圍與生成塊的高頻能量差，補上等量的隨機雜訊。
    結果是一大片**彩色雜訊**。兩個原因：

    1. 隨機雜訊逐通道獨立，看起來是彩色噪點；**真實顆粒是亮度相關的**。
    2. 參考區包含了物件的柔邊與陰影，把高頻能量估得遠高於真實顆粒。

    搬運法沒有這兩個問題：布的顆粒是統計平穩的，所以**同一張圖另一處的
    顆粒就是有效的樣本**——而且是真實的顆粒，色彩相關性天然正確，
    也不需要任何校正常數。

    做法：算出整張裁切圖的高頻殘差，把遮罩區的殘差**由平移後的位置取用**
    （平移量足以把取樣點移到乾淨的背景上）。
    """
    if not mask.any():
        return patch

    grey = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    # 用雙邊濾波而不是高斯：高斯會把物件的邊緣留在殘差裡，
    # 而我們要的是顆粒，不是邊緣。
    smooth = cv2.bilateralFilter(grey, 5, 20, 20)
    residual = (grey.astype(np.float32) - smooth.astype(np.float32))[:, :, None]

    shift = _clean_shift(mask)
    if shift is None:
        # 找不到離物件夠遠的乾淨位置（遮罩太大）。此時搬過去的顆粒
        # 會混到物件自己的殘留——寧可完全不搬，也不要加進受污染的顆粒。
        return patch
    shifted = np.roll(residual, shift, axis=(0, 1))

    # ★ 替換，不是疊加。
    #
    # 只加的話會**超出**：生成塊自己已經有一點紋理，再疊上完整的顆粒
    # 就變成 1.67 倍（實測），比周圍粗糙。正確做法是先取生成塊的
    # 低頻內容（顏色與光影——那正是模型做對的部分），再把顆粒換成
    # 搬過來的那一份。
    low_frequency = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), sigma)

    result = patch.astype(np.float32)
    # 顆粒是亮度現象，所以三個通道用同一個量——這正是搬運法自動做到的。
    result[mask] = low_frequency[mask] + shifted[mask]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)
