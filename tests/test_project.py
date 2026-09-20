"""專案與圖層的測試。

最重要的一項是**重跑不可以悄悄覆蓋人工修改過的內容**——
mangaMan 為此吃過一次苦，而那個失敗是靜默的：使用者改過的東西
在第二次執行之後消失，而且沒有任何訊息。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from photoman.project import (
    Checksum,
    DeterministicLayer,
    GenerativeLayer,
    Project,
    SourceRef,
    new_layer_id,
)
from photoman.store import (
    ProjectStore,
    SourceChangedError,
    mask_digest,
)


def _source_image(path, size=(32, 32)) -> None:
    rng = np.random.default_rng(20260918)
    array = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _source_ref() -> SourceRef:
    return SourceRef(
        path="C:/photos/a.jpg",
        sha256="0" * 64,
        format="JPEG",
        width=100,
        height=100,
        bit_depth=8,
    )


def _generative(**overrides) -> GenerativeLayer:
    defaults = {
        "method": "telea",
        "crop": (10, 10, 40, 40),
        "mask_sha256": "a" * 64,
    }
    return GenerativeLayer(**{**defaults, **overrides})


class TestPermanentIds:
    def test_ids_are_unique(self) -> None:
        ids = {new_layer_id() for _ in range(500)}

        assert len(ids) == 500

    def test_id_survives_a_save_and_reload(self, tmp_path) -> None:
        """★ 不可以「重新載入 = 重新派 ID」。

        這樣做第二天那個專案就廢了——所有人工調整都會對不上。
        """
        source = tmp_path / "in.png"
        _source_image(source)
        store = ProjectStore.create(tmp_path / "proj", source)
        layer = store.add_layer(_generative())
        original_id = layer.id
        store.save()

        reopened = ProjectStore.open(tmp_path / "proj")

        assert reopened.project.layers[0].id == original_id


class TestCacheKeyChain:
    """快取鍵是一條鏈——上游改了，下游自然失效。

    手動傳播失效一定會漏，而漏掉的後果是拿到錯的結果卻以為是對的。
    """

    def _project(self, layers) -> Project:
        return Project(name="t", source=_source_ref(), layers=layers)

    def test_stable_when_nothing_changes(self) -> None:
        project = self._project([_generative()])

        assert project.cache_keys() == project.cache_keys()

    def test_changes_when_params_change(self) -> None:
        first = self._project([_generative(method="telea")]).cache_keys()
        second = self._project([_generative(method="lama")]).cache_keys()

        assert first != second

    def test_changes_when_dilate_px_changes(self) -> None:
        """``dilate_px`` 改變的是**模型的輸入**，所以生成內容會不同。

        它不影響合成用的 alpha（見 tests/test_layers.py），但一樣要進簽名。
        漏掉它會令「改了邊距卻拿到舊結果」，而且沒有任何跡象。
        """
        first = self._project([_generative(dilate_px=16)]).cache_keys()
        second = self._project([_generative(dilate_px=32)]).cache_keys()

        assert first != second

    def test_changes_when_feather_px_changes(self) -> None:
        first = self._project([_generative(feather_px=12)]).cache_keys()
        second = self._project([_generative(feather_px=24)]).cache_keys()

        assert first != second

    def test_changes_when_the_mask_changes(self) -> None:
        """改了遮罩卻拿到舊結果，而且沒有任何跡象——這是最難察覺的失敗。"""
        first = self._project([_generative(mask_sha256="a" * 64)]).cache_keys()
        second = self._project([_generative(mask_sha256="b" * 64)]).cache_keys()

        assert first != second

    def test_downstream_changes_when_upstream_changes(self) -> None:
        """★ 這是鏈的意義所在。"""
        first = self._project(
            [DeterministicLayer(params={"exposure": 0.1}), _generative()]
        ).cache_keys()
        second = self._project(
            [DeterministicLayer(params={"exposure": 0.2}), _generative()]
        ).cache_keys()

        assert first != second, "上游改了，下游的快取鍵必須跟著變"

    def test_disabled_layers_are_excluded(self) -> None:
        enabled = self._project([_generative(enabled=True)]).cache_keys()
        disabled = self._project([_generative(enabled=False)]).cache_keys()

        assert enabled and not disabled

    def test_prompt_participates(self) -> None:
        first = self._project([_generative(prompt="a cat")]).cache_keys()
        second = self._project([_generative(prompt="a dog")]).cache_keys()

        assert first != second


class TestLockedLayers:
    """鐵律：重跑不可以悄悄覆蓋人工修改過的內容。"""

    def _store_with_layer(self, tmp_path, *, locked: bool) -> ProjectStore:
        source = tmp_path / "in.png"
        _source_image(source)
        store = ProjectStore.create(tmp_path / "proj", source)
        store.add_layer(_generative(locked=locked))
        return store

    def test_locked_layer_is_not_overwritten(self, tmp_path) -> None:
        store = self._store_with_layer(tmp_path, locked=True)
        layer_id = store.project.layers[0].id
        patch = np.full((40, 40, 3), 128, dtype=np.uint8)
        store.write_result(layer_id, patch)

        applied = store.apply_result(
            layer_id,
            result_file="cache/new.png",
            checksum=Checksum(max_abs_diff=0.0, at="2026-09-18T00:00:00"),
            cache_key="new",
        )

        assert applied is False
        assert store.project.layers[0].result_file is None, "locked 的層被覆蓋了"

    def test_unlocked_layer_is_overwritten(self, tmp_path) -> None:
        store = self._store_with_layer(tmp_path, locked=False)
        layer_id = store.project.layers[0].id

        applied = store.apply_result(
            layer_id,
            result_file="cache/new.png",
            checksum=Checksum(max_abs_diff=0.0, at="2026-09-18T00:00:00"),
            cache_key="new",
        )

        assert applied is True
        assert store.project.layers[0].result_file == "cache/new.png"

    def test_locked_is_preserved_across_a_reload(self, tmp_path) -> None:
        store = self._store_with_layer(tmp_path, locked=True)
        store.save()

        assert ProjectStore.open(tmp_path / "proj").project.layers[0].locked is True


class TestMaskDigest:
    def test_is_independent_of_representation(self) -> None:
        """同一個遮罩不論用布林、0/1 還是 0/255 表示，身分都要一樣。

        否則快取會無故失效——使用者沒有改任何東西，卻重新付一次錢。
        """
        base = np.zeros((16, 16), dtype=bool)
        base[4:12, 4:12] = True

        digests = {
            mask_digest(base),
            mask_digest(base.astype(np.uint8)),
            mask_digest(base.astype(np.uint8) * 255),
        }

        assert len(digests) == 1

    def test_distinguishes_different_masks(self) -> None:
        first = np.zeros((16, 16), dtype=bool)
        first[4:12, 4:12] = True
        second = np.zeros((16, 16), dtype=bool)
        second[5:13, 5:13] = True

        assert mask_digest(first) != mask_digest(second)


class TestStore:
    def test_round_trip(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source, size=(24, 40))
        ProjectStore.create(tmp_path / "proj", source, name="測試專案")

        reopened = ProjectStore.open(tmp_path / "proj")

        assert reopened.project.name == "測試專案"
        assert reopened.project.source.width == 40
        assert reopened.project.source.height == 24

    def test_mask_round_trip(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        store = ProjectStore.create(tmp_path / "proj", source)
        layer = store.add_layer(_generative())
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:20, 8:20] = True

        digest = store.write_mask(layer.id, mask)

        np.testing.assert_array_equal(store.read_mask(layer.id), mask)
        assert digest == mask_digest(mask)

    def test_creating_a_project_creates_the_directory_layout(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)

        ProjectStore.create(tmp_path / "proj", source)

        assert (tmp_path / "proj" / "project.json").exists()
        assert (tmp_path / "proj" / "masks").is_dir()
        assert (tmp_path / "proj" / "cache").is_dir()

    def test_project_json_is_readable(self, tmp_path) -> None:
        """專案檔要人能讀——出問題時那是唯一能查的東西。"""
        source = tmp_path / "in.png"
        _source_image(source)
        ProjectStore.create(tmp_path / "proj", source)

        raw = json.loads((tmp_path / "proj" / "project.json").read_text(encoding="utf-8"))

        assert raw["schema_version"] == 1
        assert raw["source"]["width"] == 32


class TestSourceVerification:
    """專案只記路徑不複製檔案，所以原檔改變必須被偵測到。"""

    def test_detects_a_changed_source(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        ProjectStore.create(tmp_path / "proj", source)

        _source_image(source, size=(16, 16))  # 內容換了

        with pytest.raises(SourceChangedError, match="內容與專案記錄不符"):
            ProjectStore.open(tmp_path / "proj")

    def test_detects_a_missing_source(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        ProjectStore.create(tmp_path / "proj", source)
        source.unlink()

        with pytest.raises(SourceChangedError, match="原檔不見了"):
            ProjectStore.open(tmp_path / "proj")

    def test_unchanged_source_opens_fine(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        ProjectStore.create(tmp_path / "proj", source)

        assert ProjectStore.open(tmp_path / "proj") is not None


class TestCachedResult:
    def test_returns_none_when_the_key_does_not_match(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        store = ProjectStore.create(tmp_path / "proj", source)
        layer = store.add_layer(_generative())
        store.write_result(layer.id, np.full((40, 40, 3), 128, dtype=np.uint8))
        store.apply_result(
            layer.id,
            result_file=f"cache/{layer.id}.png",
            checksum=Checksum(max_abs_diff=0.0, at="2026-09-18T00:00:00"),
            cache_key="stale-key",
        )

        assert store.cached_result(layer.id) is None, "快取鍵不符就必須重新執行"

    def test_returns_the_result_when_the_key_matches(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        store = ProjectStore.create(tmp_path / "proj", source)
        layer = store.add_layer(_generative())
        patch = np.full((40, 40, 3), 128, dtype=np.uint8)
        store.write_result(layer.id, patch)
        current = store.project.cache_keys()[layer.id]
        store.apply_result(
            layer.id,
            result_file=f"cache/{layer.id}.png",
            checksum=Checksum(max_abs_diff=0.0, at="2026-09-18T00:00:00"),
            cache_key=current,
        )

        cached = store.cached_result(layer.id)

        assert cached is not None
        np.testing.assert_array_equal(cached, patch)


class TestRender:
    """★ 重開專案要能重播出一模一樣的畫面。

    這是「存檔」這件事的全部意義：專案檔裡只有描述與貼片，
    成品是**算回來的**。算錯的話，使用者第二天打開會看到一張不同的圖，
    而且不會有任何錯誤訊息。
    """

    def _store_with_one_layer(self, tmp_path):
        from photoman.edit import apply_generative_edit

        source = tmp_path / "in.png"
        rng = np.random.default_rng(7)
        photo = rng.integers(90, 170, size=(60, 90, 3), dtype=np.uint8)
        photo[20:40, 30:60] = 25
        Image.fromarray(photo, mode="RGB").save(source)

        store = ProjectStore.create(tmp_path / "proj", source)
        mask = np.zeros((60, 90), dtype=bool)
        mask[18:42, 28:62] = True
        result = apply_generative_edit(photo, mask, method="telea")

        layer = GenerativeLayer(
            method="telea",
            crop=tuple(result.crop),
            mask_sha256=mask_digest(mask),
            feather_px=result.feather_px,
            dilate_px=result.dilate_px,
        )
        store.add_layer(layer)
        store.write_mask(layer.id, mask)
        store.apply_result(
            layer.id,
            result_file=store.write_result(layer.id, result.patch),
            checksum=Checksum(max_abs_diff=0.0, at="2026-09-19T00:00:00"),
            cache_key=store.project.cache_keys()[layer.id],
        )
        store.save()
        return store, result

    def test_render_is_byte_identical_after_a_save_and_reopen(self, tmp_path) -> None:
        store, result = self._store_with_one_layer(tmp_path)

        before = store.render()
        reopened = ProjectStore.open(tmp_path / "proj")
        after = reopened.render()

        assert np.array_equal(before, result.image), "重播與當初那一次不同"
        assert np.array_equal(after, before), "重開之後的重播與存檔前不同"

    def test_render_with_no_layers_is_the_original(self, tmp_path) -> None:
        source = tmp_path / "in.png"
        _source_image(source)
        store = ProjectStore.create(tmp_path / "proj", source)

        rendered = store.render()

        assert rendered.shape == (32, 32, 3)
        np.testing.assert_array_equal(rendered, np.asarray(Image.open(source)))

    def test_remove_last_layer_takes_the_generative_one(self, tmp_path) -> None:
        store, _ = self._store_with_one_layer(tmp_path)

        removed = store.remove_last_layer()

        assert removed is not None
        assert store.project.layers == []
        assert store.remove_last_layer() is None, "沒有了還再拿一次不應該出錯"

    def test_the_mask_file_survives_undo_so_redo_can_use_it(self, tmp_path) -> None:
        """復原**刻意不刪**遮罩與貼片——重做要用它們。"""
        store, _ = self._store_with_one_layer(tmp_path)
        layer_id = store.project.layers[0].id

        store.remove_last_layer()

        assert (tmp_path / "proj" / "masks" / f"{layer_id}.png").exists()
        assert (tmp_path / "proj" / "cache" / f"{layer_id}.png").exists()


class TestRepointSource:
    def test_the_reference_follows_the_content(self, tmp_path) -> None:
        """原檔被搬走之後，同一張照片要接回同一個專案。

        典型情況：上傳的暫存檔被清掉，使用者重新上傳同一張照片。
        """
        first = tmp_path / "first.png"
        _source_image(first)
        store = ProjectStore.create(tmp_path / "proj", first)
        moved = tmp_path / "moved.png"
        moved.write_bytes(first.read_bytes())

        store.repoint_source(moved)
        store.save()

        assert ProjectStore.open(tmp_path / "proj").project.source.path == str(moved.resolve())

    def test_open_without_verification_skips_the_check(self, tmp_path) -> None:
        """內容相符但路徑過期時，要先讀得出來才能把路徑改過去。"""
        first = tmp_path / "first.png"
        _source_image(first)
        ProjectStore.create(tmp_path / "proj", first)
        first.unlink()

        assert ProjectStore.open(tmp_path / "proj", verify=False).project.name
        with pytest.raises(SourceChangedError):
            ProjectStore.open(tmp_path / "proj")
