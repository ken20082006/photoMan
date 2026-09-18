"""遮罩前處理的測試。

最關鍵的一項是 ``make_blend_alpha`` 在遮罩外**恰為 0.0**——
那是合成器逐 bit 保證的基礎，不能只是「接近零」。
"""

from __future__ import annotations

import numpy as np

from photoman.mask import (
    dilate,
    fill_holes,
    make_blend_alpha,
    make_denoise_mask,
    prepare_masks,
    remove_specks,
)


class TestFillHoles:
    def test_fills_an_interior_hole(self) -> None:
        mask = np.ones((20, 20), dtype=bool)
        mask[8:12, 8:12] = False

        filled = fill_holes(mask)

        assert filled.all()

    def test_leaves_a_genuine_gap_alone(self) -> None:
        """由一邊開口的凹槽不是內洞，不應該被填。"""
        mask = np.ones((20, 20), dtype=bool)
        mask[8:12, 12:] = False  # 通到右邊界

        filled = fill_holes(mask)

        assert not filled[8:12, 12:].any()

    def test_mask_touching_a_corner_still_works(self) -> None:
        """遮罩貼著角落時，背景會斷成幾塊。

        若沒有先補一圈背景邊界，由 (0,0) 灌水會漏掉其餘的背景區域，
        然後把它們誤判成內洞——整張圖會被填滿。
        """
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:5, 0:5] = True
        mask[8:12, 8:12] = True

        filled = fill_holes(mask)

        assert filled[0:5, 0:5].all()
        assert filled[8:12, 8:12].all()
        assert not filled[15:, 15:].any(), "背景被誤判成內洞"

    def test_empty_mask_stays_empty(self) -> None:
        assert not fill_holes(np.zeros((10, 10), dtype=bool)).any()


class TestRemoveSpecks:
    def test_drops_small_components(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True  # 面積 400
        mask[35, 35] = True  # 面積 1

        cleaned = remove_specks(mask, min_area=64)

        assert cleaned[10:30, 10:30].all()
        assert not cleaned[35, 35]

    def test_keeps_components_at_the_threshold(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:18, 10:18] = True  # 面積 64

        assert remove_specks(mask, min_area=64)[10:18, 10:18].all()


class TestDilate:
    def test_grows_by_the_radius(self) -> None:
        mask = np.zeros((41, 41), dtype=bool)
        mask[20, 20] = True

        grown = dilate(mask, 5)

        assert grown[20, 25], "右邊應該長了 5 像素"
        assert not grown[20, 26], "不應該長超過 5 像素"

    def test_zero_radius_is_a_no_op(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True

        np.testing.assert_array_equal(dilate(mask, 0), mask)


class TestBlendAlpha:
    """合成器的契約由此開始。"""

    def test_outside_is_exactly_zero(self) -> None:
        """★ 這是逐 bit 保證的基礎——必須精確，不是「接近零」。"""
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True

        alpha = make_blend_alpha(mask, feather_px=8)

        outside = ~mask
        assert np.all(alpha[outside] == 0.0)
        # 逐位元檢查，不只是數值相等——-0.0 也不算數
        assert not np.any(np.signbit(alpha[outside]))

    def test_core_reaches_full_strength(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True

        alpha = make_blend_alpha(mask, feather_px=8)

        assert alpha[30, 30] == 1.0, "離邊界夠遠的內部應該到全強度"

    def test_ramps_towards_the_boundary(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[10:50, 10:50] = True

        alpha = make_blend_alpha(mask, feather_px=16)

        assert 0.0 < alpha[10, 30] < alpha[14, 30] < alpha[24, 30]

    def test_zero_feather_gives_a_hard_mask(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True

        alpha = make_blend_alpha(mask, feather_px=0)

        np.testing.assert_array_equal(alpha, mask.astype(np.float32))


class TestBothMasks:
    def test_denoise_covers_the_blend_region(self) -> None:
        """模型的遮罩要比混合範圍稍大。

        否則模型的生成內容沒有空間去覆蓋羽化帶的整個寬度，
        羽化帶的外緣就會露出一截原圖。
        """
        mask = np.zeros((80, 80), dtype=bool)
        mask[30:50, 30:50] = True

        denoise, alpha = prepare_masks(mask, dilate_px=8, feather_px=8)

        written = alpha > 0
        assert denoise[written].all(), "混合範圍超出了模型的遮罩"
        assert denoise.sum() > written.sum(), "模型的遮罩應該比混合範圍大"

    def test_empty_mask_produces_no_write_region(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)

        _, alpha = prepare_masks(mask)

        assert not (alpha > 0).any()


class TestDenoiseMask:
    def test_is_binary(self) -> None:
        """模型要的是接近二值的輸入。

        PELC（CVPR 2026）證明在潛空間用軟遮罩做線性混合不等於像素空間混合，
        會產生無效的潛向量——所以不可以在這裡出現中間值。
        """
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True

        denoise = make_denoise_mask(mask)

        assert denoise.dtype == np.bool_
        assert set(np.unique(denoise)).issubset({False, True})

    def test_fills_holes_before_dilating(self) -> None:
        """SAM 的遮罩是有洞的。

        不填的話，模型會在物件內部看到「保留」的指令，
        結果是物件沒有被完全移除。
        """
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        mask[18:22, 18:22] = False  # SAM 留下的洞

        denoise = make_denoise_mask(mask)

        assert denoise[19, 19], "內洞沒有被填起來，模型會保留那一塊"
