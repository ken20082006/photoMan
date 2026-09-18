"""編輯流程的測試（見 docs/design.md §5.3.2）。

**最強的一條在 ``TestEngineCannotBreakTheGuarantee``：**
就算引擎把整張裁切圖改成隨機雜訊，合成器仍然令遮罩外的位元組完全不變。

那一條成立的話，「換任何模型都不會影響保證」就是結構性的事實，
而不是對某個模型的信任。
"""

from __future__ import annotations

import numpy as np
import pytest

from photoman.edit import apply_generative_edit
from photoman.inpaint import TeleaInpainter, get_inpainter
from photoman.mask import prepare_masks


class _GarbageInpainter:
    """把所有東西都改成隨機雜訊——用來證明引擎破壞不了保證。"""

    name = "garbage"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(0, 256, size=image.shape, dtype=np.uint8)


class _NoopInpainter:
    name = "noop"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return image.copy()


def _base(height: int = 160, width: int = 200) -> np.ndarray:
    rng = np.random.default_rng(20260918)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _mask(height: int = 160, width: int = 200) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[60:100, 80:130] = True
    return mask


class TestEngineCannotBreakTheGuarantee:
    """★ 保證是合成器的性質，不是關於任何模型的聲明。"""

    def test_garbage_engine_still_preserves_everything_outside(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _base()
        mask = _mask()
        monkeypatch.setattr("photoman.edit.get_inpainter", lambda *a, **k: _GarbageInpainter())

        result = apply_generative_edit(base, mask)

        # 精確的契約：被改動的像素必須是「宣告的寫入範圍」的子集。
        # 用同一個函數算出那個範圍，而不是自己猜膨脹與羽化有多大。
        _, alpha = prepare_masks(mask, dilate_px=8, feather_px=12)
        written = alpha > 0
        changed = np.any(result.image != base, axis=2)

        assert not (changed & ~written).any(), "有像素在寫入範圍以外被改動"

    def test_everything_outside_the_write_region_is_bit_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _base()
        mask = _mask()
        monkeypatch.setattr("photoman.edit.get_inpainter", lambda *a, **k: _GarbageInpainter())

        result = apply_generative_edit(base, mask)

        _, alpha = prepare_masks(mask, dilate_px=8, feather_px=12)
        np.testing.assert_array_equal(result.image[alpha == 0], base[alpha == 0])

    def test_noop_engine_leaves_the_image_untouched(self, monkeypatch) -> None:
        base = _base()
        monkeypatch.setattr("photoman.edit.get_inpainter", lambda *a, **k: _NoopInpainter())

        result = apply_generative_edit(base, _mask())

        np.testing.assert_array_equal(result.image, base)

    def test_base_is_never_mutated(self, monkeypatch) -> None:
        base = _base()
        before = base.copy()
        monkeypatch.setattr("photoman.edit.get_inpainter", lambda *a, **k: _GarbageInpainter())

        apply_generative_edit(base, _mask())

        np.testing.assert_array_equal(base, before)


class TestContractEnforcement:
    def test_empty_mask_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="遮罩是空的"):
            apply_generative_edit(_base(), np.zeros((160, 200), dtype=bool))

    def test_float_base_is_rejected(self) -> None:
        """底圖必須是 uint8 sRGB——收到 float 代表呼叫方搞錯了工作空間。"""
        with pytest.raises(ValueError, match="uint8"):
            apply_generative_edit(_base().astype(np.float32), _mask())

    def test_size_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="尺寸"):
            apply_generative_edit(_base(), np.zeros((50, 50), dtype=bool))

    def test_unknown_method_is_reported_clearly(self) -> None:
        with pytest.raises(ValueError, match="未知的 inpainting 方法"):
            get_inpainter("does-not-exist")


class TestResult:
    def test_records_the_geometry(self) -> None:
        result = apply_generative_edit(_base(), _mask(), method="telea")

        x, y, width, height = result.crop
        assert width > 0 and height > 0
        assert x >= 0 and y >= 0
        assert len(result.mask_sha256) == 64
        assert result.method == "telea"

    def test_local_engine_reports_a_zero_checksum(self) -> None:
        """Telea 是確定性的，而且只寫入遮罩內——所以校驗環應該是 0。"""
        result = apply_generative_edit(_base(), _mask(), method="telea")

        assert result.checksum == 0.0

    def test_no_threshold_means_no_gate(self) -> None:
        """校驗環的大小與填補品質無關，所以憑它自動否決會錯殺。

        實測：LaMa 的校驗環是 24–27，而它的填補幾乎看不出接縫；
        Telea 的校驗環是 0，而它一眼看出是貼上去的。
        """
        result = apply_generative_edit(_base(), _mask(), method="telea")

        assert result.is_trustworthy() is True
        assert result.is_trustworthy(threshold=0.0) is True

    def test_a_gate_can_be_applied_when_explicitly_asked_for(self, monkeypatch) -> None:
        monkeypatch.setattr("photoman.edit.get_inpainter", lambda *a, **k: _GarbageInpainter())
        result = apply_generative_edit(_base(), _mask())

        assert result.is_trustworthy() is True  # 不給門檻就不判斷
        assert result.is_trustworthy(threshold=2.0) is False  # 給了就判斷


class TestTeleaEngine:
    def test_fills_the_masked_area(self) -> None:
        image = np.full((60, 60, 3), 200, dtype=np.uint8)
        image[20:40, 20:40] = 0
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True

        filled = TeleaInpainter().inpaint(image, mask)

        assert filled[30, 30].min() > 100, "洞沒有被填成周圍的顏色"

    def test_leaves_the_outside_alone(self) -> None:
        """Telea 只寫入遮罩內——這是它校驗環為 0 的原因。"""
        image = _base(60, 60)
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True

        filled = TeleaInpainter().inpaint(image, mask)

        np.testing.assert_array_equal(filled[~mask], image[~mask])

    def test_empty_mask_is_a_no_op(self) -> None:
        image = _base(40, 40)

        filled = TeleaInpainter().inpaint(image, np.zeros((40, 40), dtype=bool))

        np.testing.assert_array_equal(filled, image)


class TestLamaEngine:
    """LaMa 需要下載模型，所以沒有模型時跳過。

    它是本地路徑的主力引擎——授權 Apache-2.0，而且填補品質
    遠勝 Telea（實測：Telea 是一眼看得出的平滑多邊形灰塊）。
    """

    @pytest.fixture
    def inpainter(self):
        from photoman.paths import lama_model_path

        if not lama_model_path().exists():
            pytest.skip("未下載 LaMa 模型（見 docs/PROGRESS.md）")
        from photoman.inpaint import LamaInpainter

        return LamaInpainter(lama_model_path())

    def test_output_matches_the_input_size(self, inpainter) -> None:
        image = _base(80, 120)

        filled = inpainter.inpaint(image, _mask(80, 120))

        assert filled.shape == image.shape
        assert filled.dtype == np.uint8

    def test_does_not_claim_to_preserve_the_outside(self, inpainter) -> None:
        """★ LaMa 會重繪整張裁切圖——實測校驗環 24–27。

        這不是缺陷，而是為甚麼保證必須在合成器：
        我們從不採用遮罩外的部分，所以那些漂移對我們無害。
        """
        image = _base(80, 120)
        mask = _mask(80, 120)

        filled = inpainter.inpaint(image, mask)

        outside = ~mask
        assert np.array_equal(filled[outside], image[outside]) is False
