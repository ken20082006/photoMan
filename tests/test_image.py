"""保真匯入匯出的測試。

最重要的一項是**往返精確**：``uint8 → 線性光 → uint8`` 必須逐值相同。
若不精確，「遮罩外逐 bit 不變」這條保證就會斷在匯出那一步——
而那個失敗是靜默的，整張圖每個像素都差一兩級，看起來「差不多」。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from photoman.color import linear_to_srgb, srgb_to_linear
from photoman.image import load, load_srgb, save

_EXIF_ORIENTATION = 274


def _write_png(path, array: np.ndarray, **params) -> None:
    Image.fromarray(array, mode="RGB").save(path, **params)


class TestRoundTripIsExact:
    """保證鏈：原檔 → 載入 → 匯出，遮罩外的像素必須逐 bit 相同。"""

    def test_every_grey_value_survives(self) -> None:
        values = np.arange(256, dtype=np.float32) / 255.0

        back = np.clip(linear_to_srgb(srgb_to_linear(values)), 0.0, 1.0)
        round_tripped = np.floor(back * 255.0 + 0.5).astype(np.uint8)

        np.testing.assert_array_equal(round_tripped, np.arange(256, dtype=np.uint8))

    def test_a_random_image_survives(self, tmp_path) -> None:
        rng = np.random.default_rng(20260918)
        original = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        source = tmp_path / "in.png"
        _write_png(source, original)

        loaded = load(source)
        out = tmp_path / "out.png"
        save(loaded.linear, out, info=loaded.info)

        np.testing.assert_array_equal(np.asarray(Image.open(out).convert("RGB")), original)

    def test_extreme_values_survive(self, tmp_path) -> None:
        """全黑、全白與中間值——這幾個是最容易在量化時出錯的。"""
        original = np.zeros((4, 4, 3), dtype=np.uint8)
        original[0, :] = 255
        original[1, :] = 1
        original[2, :] = 128
        source = tmp_path / "in.png"
        _write_png(source, original)

        loaded = load(source)
        out = tmp_path / "out.png"
        save(loaded.linear, out, info=loaded.info)

        np.testing.assert_array_equal(np.asarray(Image.open(out).convert("RGB")), original)


class TestMetadata:
    def test_source_info_is_captured(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _write_png(source, np.zeros((16, 32, 3), dtype=np.uint8))

        loaded = load(source)

        assert loaded.info.width == 32
        assert loaded.info.height == 16
        assert loaded.info.format == "PNG"
        assert len(loaded.info.sha256) == 64

    def test_sha256_distinguishes_files(self, tmp_path) -> None:
        first = tmp_path / "a.png"
        second = tmp_path / "b.png"
        _write_png(first, np.zeros((8, 8, 3), dtype=np.uint8))
        _write_png(second, np.ones((8, 8, 3), dtype=np.uint8))

        assert load(first).info.sha256 != load(second).info.sha256

    def test_missing_file_is_reported_clearly(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="找不到檔案"):
            load(tmp_path / "does-not-exist.png")


class TestExifOrientation:
    """方向標籤不套用的話，直拍的照片會躺平。

    而使用者只會覺得「這個工具壞了」，不會想到是 EXIF。
    """

    def _write_rotated(self, path, orientation: int) -> None:
        image = Image.new("RGB", (40, 20), (200, 100, 50))
        exif = Image.Exif()
        exif[_EXIF_ORIENTATION] = orientation
        image.save(path, exif=exif)

    def test_orientation_is_applied_on_load(self, tmp_path) -> None:
        source = tmp_path / "rotated.jpg"
        self._write_rotated(source, 6)  # 順時針轉 90 度

        loaded = load(source)

        assert (loaded.info.width, loaded.info.height) == (40, 20), "原始尺寸應該照檔案記錄"
        assert loaded.linear.shape[:2] == (40, 20), "像素應該已經轉正，所以是 40 高 20 闊"

    def test_orientation_tag_is_stripped_on_save(self, tmp_path) -> None:
        """像素已轉正，留著標籤會令看圖軟體再轉一次。"""
        source = tmp_path / "rotated.jpg"
        self._write_rotated(source, 6)
        loaded = load(source)

        out = tmp_path / "out.jpg"
        save(loaded.linear, out, info=loaded.info)

        assert _EXIF_ORIENTATION not in Image.open(out).getexif()

    def test_upright_image_is_untouched(self, tmp_path) -> None:
        source = tmp_path / "upright.jpg"
        self._write_rotated(source, 1)

        loaded = load(source)

        assert loaded.linear.shape[:2] == (20, 40)


class TestSrgbBase:
    """底圖以原檔位元組儲存（§6.1c）——42 MP 由 506 MB 降到 127 MB。

    而且合成時遮罩外是**字面上的位元組複製**，不是「浮點運算後剛好相等」。
    """

    def test_returns_uint8(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _write_png(source, np.full((8, 8, 3), 128, dtype=np.uint8))

        loaded = load_srgb(source)

        assert loaded.srgb.dtype == np.uint8

    def test_agrees_with_the_linear_loader(self, tmp_path) -> None:
        """兩個載入器必須描述同一張圖，差別只在表示方式。"""
        rng = np.random.default_rng(20260918)
        original = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        source = tmp_path / "in.png"
        _write_png(source, original)

        srgb = load_srgb(source)
        linear = load(source)

        np.testing.assert_array_equal(srgb.srgb, original)
        # linear.linear 是線性光，要先轉回 sRGB 才能與位元組比較。
        back = np.clip(linear_to_srgb(linear.linear), 0.0, 1.0)
        np.testing.assert_array_equal(srgb.srgb, np.floor(back * 255.0 + 0.5).astype(np.uint8))

    def test_captures_the_same_metadata(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _write_png(source, np.zeros((8, 8, 3), dtype=np.uint8))

        assert load_srgb(source).info.sha256 == load(source).info.sha256

    def test_applies_exif_orientation(self, tmp_path) -> None:
        image = Image.new("RGB", (40, 20), (200, 100, 50))
        exif = Image.Exif()
        exif[_EXIF_ORIENTATION] = 6
        source = tmp_path / "rotated.jpg"
        image.save(source, exif=exif)

        assert load_srgb(source).srgb.shape[:2] == (40, 20)

    def test_missing_file_is_reported_clearly(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="找不到檔案"):
            load_srgb(tmp_path / "does-not-exist.png")


class TestWorkingSpace:
    def test_output_is_linear_float32(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _write_png(source, np.full((8, 8, 3), 128, dtype=np.uint8))

        loaded = load(source)

        assert loaded.linear.dtype == np.float32
        # sRGB 的 128 在線性光是 0.216 左右，不是 0.5
        assert loaded.linear[0, 0, 0] == pytest.approx(0.2158, abs=1e-3)

    def test_black_and_white_are_fixed_points(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        original = np.zeros((4, 8, 3), dtype=np.uint8)
        original[:, 4:] = 255
        _write_png(source, original)

        loaded = load(source)

        assert loaded.linear[0, 0, 0] == 0.0
        assert loaded.linear[0, 7, 0] == 1.0

    def test_unsupported_bit_depth_is_refused_not_silently_wrong(self, tmp_path) -> None:
        """16 位元匯出尚未實作。寧可明確拒絕，也不要悄悄降成 8 位元。"""
        source = tmp_path / "in.png"
        _write_png(source, np.zeros((4, 4, 3), dtype=np.uint8))
        loaded = load(source)

        with pytest.raises(NotImplementedError, match="16"):
            save(loaded.linear, tmp_path / "out.png", bit_depth=16)
