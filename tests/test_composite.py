"""合成器的測試——本案的核心保證。

這些測試比這個專案的其他任何測試都重要。§5.4 的兩個自動化測試，
第一個就在這裡：**遮罩外逐 bit 不變**。
"""

from __future__ import annotations

import numpy as np
import pytest

from photoman.composite import (
    composite_patch,
    downscale_factor,
    plan_crop,
    ring_delta,
    verify_unchanged,
)


def _image(height: int = 32, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(20260918)
    return rng.random((height, width, 3), dtype=np.float32)


def _alpha(height: int = 32, width: int = 32, *, core=slice(8, 24)) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[core, core] = 1.0
    return alpha


class TestByteIdenticalGuarantee:
    """契約：``alpha == 0`` 的像素與原圖逐 bit 相同。"""

    def test_outside_is_exactly_unchanged(self) -> None:
        original = _image()
        patch = np.ones((16, 16, 3), dtype=np.float32)
        alpha = _alpha()

        result = composite_patch(original, patch, (8, 8, 16, 16), alpha)

        assert verify_unchanged(original, result, alpha) == 0.0

    def test_inside_is_replaced(self) -> None:
        original = _image()
        patch = np.ones((16, 16, 3), dtype=np.float32)
        alpha = _alpha()

        result = composite_patch(original, patch, (8, 8, 16, 16), alpha)

        np.testing.assert_array_equal(result[8:24, 8:24], 1.0)

    def test_original_is_not_mutated(self) -> None:
        original = _image()
        before = original.copy()
        alpha = _alpha()

        composite_patch(original, np.ones((16, 16, 3), dtype=np.float32), (8, 8, 16, 16), alpha)

        np.testing.assert_array_equal(original, before)

    def test_empty_alpha_returns_an_exact_copy(self) -> None:
        original = _image()
        alpha = np.zeros((32, 32), dtype=np.float32)

        result = composite_patch(
            original, np.ones((4, 4, 3), dtype=np.float32), (0, 0, 4, 4), alpha
        )

        np.testing.assert_array_equal(result, original)

    def test_negative_zero_is_preserved(self) -> None:
        """``-0.0`` 與 ``+0.0`` 數學上相等，位元不同。

        這就是為甚麼合成器用「複製再只寫入內部」，而不是對整張圖做
        ``original*(1-a) + patch*a``——後者在 ``a == 0`` 時會把 ``-0.0``
        變成 ``+0.0``，位元就不同了。
        """
        original = np.full((8, 8, 3), -0.0, dtype=np.float32)
        alpha = np.zeros((8, 8), dtype=np.float32)
        alpha[2:6, 2:6] = 1.0
        patch = np.full((4, 4, 3), 0.5, dtype=np.float32)

        result = composite_patch(original, patch, (2, 2, 4, 4), alpha)

        outside = np.ones((8, 8), dtype=bool)
        outside[2:6, 2:6] = False
        assert np.all(np.signbit(result[outside])), "遮罩外的 -0.0 被改成了 +0.0"

    def test_feathered_edge_blends(self) -> None:
        """羽化帶內應該是混合值，不是硬切。"""
        original = np.zeros((16, 16, 3), dtype=np.float32)
        patch = np.ones((8, 8, 3), dtype=np.float32)
        alpha = np.zeros((16, 16), dtype=np.float32)
        alpha[4:12, 4:12] = 0.25  # 外圈：羽化帶
        alpha[6:10, 6:10] = 1.0  # 內核：全強度

        result = composite_patch(original, patch, (4, 4, 8, 8), alpha)

        assert result[8, 8, 0] == pytest.approx(1.0)  # 內核
        assert result[4, 4, 0] == pytest.approx(0.25)  # 羽化帶是混合值，不是硬切
        assert result[0, 0, 0] == pytest.approx(0.0)  # 裁切框以外未被寫入


class TestContractEnforcement:
    def test_patch_size_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="尺寸"):
            composite_patch(_image(), np.ones((5, 5, 3), dtype=np.float32), (0, 0, 4, 4), _alpha())

    def test_alpha_size_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="尺寸"):
            composite_patch(
                _image(),
                np.ones((4, 4, 3), dtype=np.float32),
                (0, 0, 4, 4),
                np.zeros((8, 8), dtype=np.float32),
            )


class TestPlanCrop:
    def test_margin_is_proportional_not_fixed(self) -> None:
        """邊距要按遮罩大小取比例（§5.3.4）。

        寫死的話，小遮罩會浪費預算在邊距上，大遮罩則完全沒有上下文。
        """
        small = np.zeros((400, 400), dtype=bool)
        small[100:110, 100:110] = True

        large = np.zeros((400, 400), dtype=bool)
        large[100:300, 100:300] = True

        small_crop = plan_crop(small, margin_ratio=0.5, min_margin_px=1)
        large_crop = plan_crop(large, margin_ratio=0.5, min_margin_px=1)

        # 小遮罩 10px → 邊距 5px；大遮罩 200px → 邊距 100px。
        assert small_crop == (95, 95, 20, 20)
        assert large_crop == (0, 0, 400, 400)

    def test_minimum_margin_applies_to_tiny_masks(self) -> None:
        mask = np.zeros((400, 400), dtype=bool)
        mask[200:202, 200:202] = True

        crop = plan_crop(mask, margin_ratio=0.1, min_margin_px=64)

        assert crop == (136, 136, 130, 130)

    def test_crop_is_clipped_to_the_image(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:5, 0:5] = True

        x, y, width, height = plan_crop(mask, margin_ratio=1.0, min_margin_px=64)

        assert x == 0 and y == 0
        assert width <= 100 and height <= 100

    def test_empty_mask_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="空的"):
            plan_crop(np.zeros((10, 10), dtype=bool))


class TestDownscaleFactor:
    def test_reports_the_quality_determining_ratio(self) -> None:
        # 送 2048 給 1024 的模型 = 縮小 2 倍，落在「邊緣」那一格。
        assert downscale_factor((0, 0, 2048, 2048), 1024) == pytest.approx(2.0)
        assert downscale_factor((0, 0, 1024, 1024), 1024) == pytest.approx(1.0)


class TestRingDelta:
    """校驗環：模型若守規矩，回傳的校驗環應該與送出的一模一樣。"""

    def _fixture(self):
        original = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[24:40, 24:40] = True
        return original, mask, (0, 0, 64, 64)

    def test_identical_ring_scores_zero(self) -> None:
        original, mask, crop = self._fixture()

        assert ring_delta(original, original.copy(), mask, crop) == 0.0

    def test_drift_is_detected(self) -> None:
        """模型在不該動的地方動了手——要量得出來。"""
        original, mask, crop = self._fixture()
        returned = original.copy()
        returned[2, 2, 0] = 103  # 校驗環上的一點漂移，差 3 > 門檻 2

        assert ring_delta(original, returned, mask, crop) == 3.0

    def test_changes_inside_the_mask_are_ignored(self) -> None:
        """遮罩內的改動正是模型的工作，不應該算進校驗。"""
        original, mask, crop = self._fixture()
        returned = original.copy()
        returned[32, 32] = 255

        assert ring_delta(original, returned, mask, crop) == 0.0

    def test_inset_excludes_the_boundary_band(self) -> None:
        """緊貼遮罩邊界的少許改動是預期之內的。"""
        original, mask, crop = self._fixture()
        returned = original.copy()
        returned[23, 32, 0] = 120  # 就在遮罩外一格

        assert ring_delta(original, returned, mask, crop) == 20.0
        assert ring_delta(original, returned, mask, crop, inset_px=2) == 0.0

    def test_size_mismatch_is_rejected(self) -> None:
        """尺寸不符就無法對位，應該拒絕而不是硬套。"""
        original, mask, crop = self._fixture()
        with pytest.raises(ValueError, match="尺寸不符"):
            ring_delta(original, original[:32, :32], mask, crop)

    def test_full_mask_has_no_ring(self) -> None:
        original = np.full((8, 8, 3), 100, dtype=np.uint8)
        mask = np.ones((8, 8), dtype=bool)

        assert np.isnan(ring_delta(original, original.copy(), mask, (0, 0, 8, 8)))
