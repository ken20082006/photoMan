"""色彩空間轉換的測試。"""

from __future__ import annotations

import numpy as np
import pytest

from photoman.color import linear_to_srgb, srgb_to_linear


class TestRoundTrip:
    def test_round_trip_is_stable(self) -> None:
        values = np.linspace(0.0, 1.0, 4096, dtype=np.float32)

        restored = linear_to_srgb(srgb_to_linear(values))

        np.testing.assert_allclose(restored, values, atol=1e-6)

    def test_round_trip_on_an_image(self) -> None:
        rng = np.random.default_rng(20260918)
        image = rng.random((64, 64, 3), dtype=np.float32)

        restored = linear_to_srgb(srgb_to_linear(image))

        np.testing.assert_allclose(restored, image, atol=1e-6)


class TestKnownValues:
    def test_endpoints_are_fixed(self) -> None:
        assert srgb_to_linear(np.float32(0.0)) == 0.0
        assert srgb_to_linear(np.float32(1.0)) == 1.0

    def test_mid_grey_is_not_half_in_linear_light(self) -> None:
        """sRGB 的中灰（0.5）在線性光是 0.21 左右，不是 0.5。

        這正是為甚麼要在線性光做混合——在 sRGB 空間取平均值
        會得到比感知上暗的結果，邊界就出現暗邊。
        """
        assert srgb_to_linear(np.float32(0.5)) == pytest.approx(0.2140, abs=1e-4)

    def test_is_monotonic(self) -> None:
        values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        linear = srgb_to_linear(values)

        assert np.all(np.diff(linear) > 0)


class TestNegativeValues:
    """調整層可以在線性空間產生負值。

    負數的 2.4 次方在實數域無定義——直接代入會得到 NaN，
    然後 NaN 會沿著 pipeline 擴散到整張圖，症狀是匯出時整張圖變黑。
    """

    def test_negatives_do_not_produce_nan(self) -> None:
        values = np.array([-0.5, -0.01, 0.0, 0.01, 0.5], dtype=np.float32)

        assert not np.isnan(srgb_to_linear(values)).any()
        assert not np.isnan(linear_to_srgb(values)).any()

    def test_negatives_survive_a_round_trip_with_their_sign(self) -> None:
        values = np.array([-0.5, -0.1, -0.001], dtype=np.float32)

        restored = linear_to_srgb(srgb_to_linear(values))

        np.testing.assert_allclose(restored, values, atol=1e-6)
        assert np.all(np.signbit(restored))
