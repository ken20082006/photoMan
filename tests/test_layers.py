"""可重播圖層——**這是多步編輯的整個基礎**。

這一組測的是同一個問題：只留下「裁切框 + 遮罩 + 貼片」，
能不能重現當初那一次執行**完全一樣的位元組**。
答不了的話，復原與重做都會悄悄地改變畫面。
"""

from __future__ import annotations

import numpy as np
import pytest

from photoman.edit import apply_generative_edit
from photoman.layers import EditLayer, render
from photoman.mask import blend_alpha_for, prepare_masks
from photoman.project import GenerativeLayer


def _photo(shape=(240, 320)) -> np.ndarray:
    rng = np.random.default_rng(20260919)
    photo = rng.integers(90, 170, size=(*shape, 3), dtype=np.uint8)
    photo[60:120, 80:160] = 25  # 一塊暗色「物件」
    return photo


def _mask(shape=(240, 320), *, box=(slice(50, 130), slice(70, 170))) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[box] = True
    return mask


def _layer_of(result, mask: np.ndarray, method: str = "telea") -> EditLayer:
    layer = GenerativeLayer(
        method=method,
        crop=tuple(result.crop),
        mask_sha256=result.mask_sha256,
        feather_px=result.feather_px,
        dilate_px=result.dilate_px,
    )
    return EditLayer.from_edit(layer, mask, result.patch)


def _edit(base: np.ndarray, mask: np.ndarray, method: str = "telea"):
    return apply_generative_edit(base, mask, method=method)


class TestReplayIsExact:
    """★ 這一組是整個設計的根據。"""

    def test_replay_reproduces_the_original_bytes(self) -> None:
        base = _photo()
        mask = _mask()
        result = _edit(base, mask)

        replayed = render(base, [_layer_of(result, mask)])

        assert np.array_equal(replayed, result.image), "重播出來的位元組與原本那一次不同"

    def test_a_chain_of_layers_replays_the_chain(self) -> None:
        """第二層是在第一層的**成品**上算的——重播要照這個順序。"""
        base = _photo()
        first = _edit(base, _mask())
        second = _edit(first.image, _mask(box=(slice(80, 150), slice(120, 200))))

        replayed = render(
            base,
            [
                _layer_of(first, _mask()),
                _layer_of(second, _mask(box=(slice(80, 150), slice(120, 200)))),
            ],
        )

        assert np.array_equal(replayed, second.image)

    def test_order_matters(self) -> None:
        """把兩層對調會得到不同的結果——所以「順序即堆疊」是真的。"""
        base = _photo()
        first = _edit(base, _mask())
        second = _edit(first.image, _mask(box=(slice(80, 150), slice(120, 200))))

        forward = render(
            base,
            [
                _layer_of(first, _mask()),
                _layer_of(second, _mask(box=(slice(80, 150), slice(120, 200)))),
            ],
        )
        backward = render(
            base,
            [
                _layer_of(second, _mask(box=(slice(80, 150), slice(120, 200)))),
                _layer_of(first, _mask()),
            ],
        )

        assert not np.array_equal(forward, backward)


class TestRenderContract:
    def test_no_layers_returns_a_copy(self) -> None:
        base = _photo()

        result = render(base, [])

        assert np.array_equal(result, base)
        assert result is not base, "回傳原陣列的話，呼叫方一改就會污染原檔"

    def test_the_source_is_never_mutated(self) -> None:
        base = _photo()
        before = base.copy()

        render(base, [_layer_of(_edit(base, _mask()), _mask())])

        assert np.array_equal(base, before)


class TestFromEditRefusesABadLayer:
    def test_a_mask_that_escapes_the_crop_is_refused(self) -> None:
        """遮罩超出裁切框的話，重播會少改一塊而不會報錯——要在這裡擋下來。

        那時還握有完整的遮罩；留到重播才發現，手上只剩一張被裁掉一角的，
        既看不出少了甚麼，也無從還原。
        """
        base = _photo()
        small = _mask(box=(slice(60, 80), slice(90, 110)))
        result = _edit(base, small)

        # 裁切框是「這個小遮罩的外接矩形再加 64 像素邊距」，所以隨便一個
        # 大一點的遮罩未必會超出它——要挑一塊離得夠遠的。
        bigger = small.copy()
        bigger[200:220, 280:300] = True

        with pytest.raises(ValueError, match="超出了裁切框"):
            _layer_of(result, bigger)

    def test_a_patch_of_the_wrong_size_is_refused(self) -> None:
        base = _photo()
        result = _edit(base, _mask())
        wrong = GenerativeLayer(method="telea", crop=(0, 0, 10, 10), mask_sha256=result.mask_sha256)

        with pytest.raises(ValueError, match="貼片的尺寸"):
            EditLayer.from_edit(wrong, _mask(), result.patch)


class TestTheGuaranteeGrowsWithTheLayers:
    """★ 鐵律的多層版本。

    單層是「遮罩外逐 bit 不變」。多層之後是
    **「所有圖層的寫入範圍（alpha > 0 的地方）聯集之外，位元組等於原檔」**。

    注意寫入範圍**不是使用者畫的那個遮罩**：羽化會向外擴一圈
    （``dilate(mask, feather_px)``），所以界線要用 ``blend_alpha_for`` 算，
    不可以自己手寫膨脹——兩次膨脹不等於一次更大的膨脹。
    """

    def test_nothing_changes_outside_the_blend_union(self) -> None:
        base = _photo()
        mask_a = _mask()
        mask_b = _mask(box=(slice(80, 150), slice(120, 200)))
        first = _edit(base, mask_a)
        second = _edit(first.image, mask_b)
        layers = [_layer_of(first, mask_a), _layer_of(second, mask_b)]

        result = render(base, layers)

        allowed = np.zeros(base.shape[:2], dtype=bool)
        for mask in (mask_a, mask_b):
            allowed |= blend_alpha_for(mask) > 0

        changed = (result != base).any(axis=2)
        assert changed.any(), "這兩層應該真的改了東西"
        assert not changed[~allowed].any(), (
            f"寫入範圍之外有 {int(changed[~allowed].sum())} 個像素被動過"
        )


class TestTheCropMaskStorageIsSound:
    """記憶體設計的根據：遮罩一定完整落在裁切框內，所以只存裁切區那一塊。

    42 MP 的整張遮罩是 42 MB／層，裁切區只有零點幾 MB。這個前提不成立的話，
    重播會少改框外那部分——而且是靜默的。
    """

    @pytest.mark.parametrize(
        "box",
        [
            (slice(50, 130), slice(70, 170)),  # 中央
            (slice(0, 40), slice(0, 40)),  # 貼左上角
            (slice(200, 240), slice(280, 320)),  # 貼右下角
            (slice(0, 240), slice(0, 320)),  # 整張圖
        ],
    )
    def test_the_mask_survives_a_round_trip_through_the_crop(self, box) -> None:
        shape = (240, 320)
        mask = _mask(shape, box=box)
        result = _edit(_photo(shape), mask)
        crop = result.crop

        edit = _layer_of(result, mask)

        assert np.array_equal(edit.full_mask(shape), mask), "展開回原圖之後與原遮罩不同"
        x, y, width, height = crop
        assert edit.mask.shape == (height, width), "記憶體裡留的應該是裁切區那一塊"


class TestAlphaDoesNotDependOnDilatePx:
    """重播只帶 ``feather_px``，不帶 ``dilate_px``——這條測試就是它的根據。

    反過來說，如果哪天 alpha 真的開始依賴 ``dilate_px``，
    重播會靜默地算出不同的畫面，而兩邊的程式碼都「看起來很對」。
    """

    def test_the_alpha_is_identical_for_different_dilate_px(self) -> None:
        mask = _mask()

        small, _ = prepare_masks(mask, dilate_px=8)
        large, _ = prepare_masks(mask, dilate_px=40)

        assert not np.array_equal(small, large), "dilate_px 應該有影響——影響的是模型遮罩"
        assert np.array_equal(
            prepare_masks(mask, dilate_px=8)[1], prepare_masks(mask, dilate_px=40)[1]
        ), "alpha 不可以受 dilate_px 影響，否則重播會算錯"

    def test_blend_alpha_for_matches_prepare_masks(self) -> None:
        """兩者必須是同一條路——重播用的那個不可以和原跑的分岔。"""
        mask = _mask()

        assert np.array_equal(
            blend_alpha_for(mask, feather_px=12), prepare_masks(mask, feather_px=12)[1]
        )
