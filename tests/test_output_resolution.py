"""輸出不足時先暫停；確認後沿用同一張結果並維持遮罩外保真。"""

import numpy as np
import pytest

from photoman.edit import UpscaleRequired, apply_generative_edit
from photoman.mask import prepare_masks


def scene():
    base = np.random.default_rng(1).integers(0, 180, (160, 240, 3), dtype=np.uint8)
    mask = np.zeros((160, 240), dtype=bool)
    mask[60:100, 80:150] = True
    return base, mask


@pytest.mark.parametrize("size", [(80, 120), (320, 120), (80, 480)])
def test_either_short_edge_requires_confirmation(size):
    base, mask = scene()
    original = base.copy()
    patch = np.full((*size, 3), 220, dtype=np.uint8)
    with pytest.raises(UpscaleRequired) as error:
        apply_generative_edit(base, mask, generated_patch=patch)
    assert error.value.output_size == size[::-1]
    assert error.value.patch is patch
    assert max(error.value.scale) > 1
    assert np.array_equal(base, original)


def test_confirmed_upscale_uses_cached_image_without_an_engine(monkeypatch):
    base, mask = scene()

    def no_engine(*args, **kwargs):
        pytest.fail("確認不能再次呼叫生成引擎")

    monkeypatch.setattr("photoman.edit.get_inpainter", no_engine)
    result = apply_generative_edit(
        base, mask, generated_patch=np.full((80, 120, 3), 220, dtype=np.uint8),
        allow_upscale=True, match_texture=False,
    )
    _, alpha = prepare_masks(mask)
    assert result.image.shape == base.shape
    assert np.array_equal(result.image[alpha == 0], base[alpha == 0])
    assert not np.array_equal(result.image, base)


def test_large_output_can_be_reduced_without_confirmation():
    base, mask = scene()
    result = apply_generative_edit(
        base, mask, generated_patch=np.full((320, 480, 3), 220, dtype=np.uint8),
        match_texture=False,
    )
    assert result.image.shape == base.shape
