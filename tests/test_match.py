"""色彩與顆粒對齊。

★ 這裡最重要的一條是**意圖**：填補與編輯要對到不同的東西。
用錯了會把使用者要求的顏色改動直接抹掉——而那個失敗是靜默的，
成品看起來「很自然」，只是不是你要的顏色。
"""

from __future__ import annotations

import numpy as np

from photoman.match import match_colour


def _grass(shape=(200, 300)) -> np.ndarray:
    """有顆粒的綠草地。顆粒是刻意的——純色會令統計量退化。"""
    rng = np.random.default_rng(1)
    base = np.zeros((*shape, 3), np.uint8)
    base[:, :] = (90, 130, 60)
    return np.clip(
        base.astype(np.int16) + rng.integers(-6, 6, base.shape, dtype=np.int16), 0, 255
    ).astype(np.uint8)


def _mask(shape=(200, 300), box=(slice(60, 140), slice(100, 200))) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[box] = True
    return mask


def _mean(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    return image[region].mean(axis=0)


class TestEditIntentKeepsTheRequestedChange:
    """★ 這是那個 bug 的回歸測試。

    使用者叫模型「把衣服改成紅色」，模型照做了。我們的後處理
    不可以把它改回周圍的顏色。
    """

    def test_the_red_stays_red(self) -> None:
        base = _grass()
        mask = _mask()
        patch = base.copy()
        patch[mask] = (170, 40, 40)  # 模型照指令畫的紅色

        result = match_colour(patch, base, mask, intent="edit")

        red = _mean(result, mask)
        assert red[0] > 140, f"紅色被拉回周圍了：{red.round(0)}"
        assert red[0] - red[1] > 80, "紅色的色相被改變了"

    def test_fill_intent_would_destroy_it(self) -> None:
        """把上面那個失敗寫成測試——它是真的會發生，不是假想。

        填補意圖會把生成塊的均值對到周圍，那對「填一個洞」是對的，
        對「換一件衣服的顏色」是災難。
        """
        base = _grass()
        mask = _mask()
        patch = base.copy()
        patch[mask] = (170, 40, 40)

        result = match_colour(patch, base, mask, intent="fill")

        assert np.abs(_mean(result, mask) - _mean(base, ~mask)).max() < 3, (
            "填補意圖應該把均值對到周圍——這是它的定義"
        )


class TestEditIntentRemovesTheDrift:
    """對齊只寫遮罩內，所以要看**遮罩那一塊**有沒有被修正回原圖。

    這兩條的貼片都是「原圖 + 整體漂移」——沒有要改的內容，
    所以修正之後遮罩內應該回到原圖的樣子。
    """

    def test_a_global_brightening_is_removed(self) -> None:
        """雲端模型常把整張回傳的圖調亮（實測 +12.7 到 +25.6 級）。

        那會令成品偏亮偏灰，而它是**模型的漂移**，不是使用者要的改動，
        所以要把整個生成塊一起平移回去。
        """
        base = _grass()
        mask = _mask()
        patch = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)

        result = match_colour(patch, base, mask, intent="edit")

        assert np.abs(_mean(result, mask) - _mean(base, mask)).max() < 4, (
            f"漂移沒有被扣掉：{_mean(result, mask).round(1)} vs {_mean(base, mask).round(1)}"
        )

    def test_a_colour_change_is_removed_too(self) -> None:
        """不只是亮度——整體的色偏也一樣要扣掉。"""
        base = _grass()
        mask = _mask()
        patch = np.clip(base.astype(np.int16) + np.array([15, 0, -15]), 0, 255).astype(np.uint8)

        result = match_colour(patch, base, mask, intent="edit")

        assert np.abs(_mean(result, mask) - _mean(base, mask)).max() < 4

    def test_fill_intent_does_not_remove_the_drift_the_same_way(self) -> None:
        """★ 兩者的分別要在這裡看得出來。

        填補意圖把遮罩區的均值對到**周圍**；編輯意圖對到**原圖同一塊**。
        這裡的貼片是原圖加漂移，所以兩者剛好都不難看——差別在
        遮罩區的內容本來就與周圍不同時（上一個測試類別）。
        """
        base = _grass()
        mask = _mask()
        patch = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)

        filled = match_colour(patch, base, mask, intent="fill")
        edited = match_colour(patch, base, mask, intent="edit")

        # 填補：遮罩區的均值 == 周圍的均值
        assert np.abs(_mean(filled, mask) - _mean(base, ~mask)).max() < 4
        # 編輯：遮罩區的均值 == 原圖同一塊的均值
        assert np.abs(_mean(edited, mask) - _mean(base, mask)).max() < 4


class TestTheContract:
    def test_nothing_outside_the_mask_changes(self) -> None:
        """對齊只可以動遮罩內——這是合成器的契約。"""
        base = _grass()
        mask = _mask()
        patch = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)

        for intent in ("fill", "edit"):
            result = match_colour(patch, base, mask, intent=intent)
            np.testing.assert_array_equal(result[~mask], patch[~mask])

    def test_an_empty_mask_is_a_no_op(self) -> None:
        base = _grass()
        patch = base.copy()

        result = match_colour(patch, base, np.zeros(base.shape[:2], bool), intent="edit")

        np.testing.assert_array_equal(result, patch)

    def test_a_full_mask_is_a_no_op(self) -> None:
        """整張都是遮罩時，沒有「周圍」可以參考——那就不要動它。"""
        base = _grass()
        patch = base.copy()

        result = match_colour(patch, base, np.ones(base.shape[:2], bool), intent="edit")

        np.testing.assert_array_equal(result, patch)

    def test_the_input_patch_is_not_mutated(self) -> None:
        base = _grass()
        mask = _mask()
        patch = np.clip(base.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        before = patch.copy()

        match_colour(patch, base, mask, intent="edit")

        np.testing.assert_array_equal(patch, before)


class TestEditMeasuresOnPreservedContent:
    """★ 修正：量的地方不可以是遮罩外的環。

    第一版用遮罩外的環去量「模型把整張圖偏移了多少」，再套到遮罩上。
    那個假設是「模型的色偏在全圖一致」——**實測不成立**。

    使用者的真實案例（改髮型，遮罩是框住整個頭的大方框）：

    | | |
    |---|---|
    | 背景（環）的偏移 | −10.8 |
    | 臉的偏移 | −5 |
    | 扣掉環的偏移之後，臉的殘餘差 | **+8.9** |

    背景的偏移套到臉上會過頭——那就是使用者回報的「還是有點色差」。
    """

    def _scene(self):
        """底色＋質感。遮罩內一半是「模型保留的內容」，一半是它新畫的。"""
        rng = np.random.default_rng(4)
        original = np.zeros((160, 200, 3), np.uint8)
        original[:, :] = (120, 130, 140)
        original += rng.integers(-4, 4, original.shape, dtype=np.int16).astype(np.uint8)
        mask = np.zeros((160, 200), bool)
        mask[40:120, 40:160] = True
        return original, mask

    def test_the_error_is_measured_on_the_kept_content(self) -> None:
        original, mask = self._scene()
        patch = original.copy()
        # 模型把「它保留的內容」畫得偏暗 6 級，同時又真的改了另一半
        patch[mask] = np.clip(patch[mask].astype(np.int16) - 6, 0, 255).astype(np.uint8)
        patch[40:80, 40:160] = (200, 180, 60)  # 使用者要的改動（頭髮）
        # 遮罩外的環偏得更兇——那是背景，與遮罩內的內容無關
        patch[~mask] = np.clip(patch[~mask].astype(np.int16) - 30, 0, 255).astype(np.uint8)

        fixed = match_colour(patch, original, mask, intent="edit")

        kept = mask & (np.abs(patch.astype(np.int16) - original.astype(np.int16)).max(axis=2) < 20)
        residue = fixed[kept].astype(float).mean(axis=0) - original[kept].astype(float).mean(axis=0)
        assert np.abs(residue).max() < 3, f"保留的內容沒有被修正回來：{residue.round(1)}"

        # 而它不可以因為背景偏了 30 就把使用者要的改動也拉回去
        changed = fixed[40:80, 40:160].astype(float).mean(axis=(0, 1))
        assert changed[0] > 150, f"使用者要的改動被抹掉了：{changed.round(0)}"

    def test_it_falls_back_to_the_ring_when_nothing_was_kept(self) -> None:
        """模型把整塊換掉時（純粹的移除），遮罩內沒有真值可以對。"""
        original, mask = self._scene()
        patch = np.clip(original.astype(np.int16) - 25, 0, 255).astype(np.uint8)

        fixed = match_colour(patch, original, mask, intent="edit")

        assert (
            np.abs(
                fixed[mask].astype(float).mean(axis=0) - original[mask].astype(float).mean(axis=0)
            ).max()
            < 4
        ), "退回用環量的話，應該把整體偏移扣掉"

    def test_a_subtle_requested_change_survives(self) -> None:
        """★ 門檻不可以太大。

        「把這裡調暗 15 級」是使用者要的，不是模型的誤差。門檻若放到 20，
        它會被當成誤差而**被抹掉**——實測偏移量會由 +0.4 漲到 +2.9。
        """
        original, mask = self._scene()
        patch = original.copy()
        patch[mask] = np.clip(patch[mask].astype(np.int16) - 15, 0, 255).astype(np.uint8)

        fixed = match_colour(patch, original, mask, intent="edit")

        delta = fixed[mask].astype(float).mean(axis=0) - original[mask].astype(float).mean(axis=0)
        assert delta.mean() < -10, f"使用者要的 −15 被修掉了：{delta.round(1)}"
