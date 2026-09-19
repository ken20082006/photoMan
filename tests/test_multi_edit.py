"""★ 多步編輯與復原——「一張照片只能改一次」那個 bug 的回歸測試。

**先前的情況**（實測，`sandbox/two_edits.py`）：先移除 A，再框 B 執行，
A 區域 12000 個像素裡有 **11997 個被重新生成**。三個原因互相加乘：

1. 執行完遮罩沒有清掉，第二次是 A ∪ B 一起送
2. 裁切區因此變大，LaMa 的 512 縮放倍率改變，整塊內容都不一樣
3. 每一次執行都套用在最原始的那張圖上，成品不會回流

全部用 Telea：這裡要驗的是**流程**，不是填補品質。Telea 不需要模型，
所以這一組在任何環境都跑得動。
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photoman import layers as layers_module
from photoman.mask import blend_alpha_for
from photoman.web import server

# 兩個分得很開的物件——第二次編輯不應該碰到第一個。
A = (slice(60, 120), slice(80, 180))
B = (slice(250, 320), slice(380, 500))
BOX_A = {"x0": 80, "y0": 60, "x1": 180, "y1": 120}
BOX_B = {"x0": 380, "y0": 250, "x1": 500, "y1": 320}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每個測試都拿到乾淨的工作階段與乾淨的專案目錄。

    專案目錄要導到 tmp_path：測試用的圖內容都一樣，不導開的話
    它們會共用同一個專案目錄（目錄是以內容雜湊命名的），
    而且會寫進使用者真正的 `%APPDATA%\\photoMan\\projects`。
    """
    monkeypatch.setattr(server, "SESSION", server.Session())
    monkeypatch.setattr(server.config, "projects_dir", lambda: tmp_path / "projects")
    return TestClient(server.app)


@pytest.fixture
def photo(tmp_path):
    rng = np.random.default_rng(11)
    array = rng.integers(110, 160, size=(400, 600, 3), dtype=np.uint8)
    array[A] = 20
    array[B] = 20
    path = tmp_path / "photo.png"
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _open(client, path) -> dict:
    response = client.post("/api/open", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()


def _export(client) -> np.ndarray:
    """匯出的**原解析度**成品。

    注意 ``response.content`` 已經是 PNG 位元組——不是 data URL，
    不要再 base64 解一次。
    """
    response = client.get("/api/result")
    assert response.status_code == 200, response.text
    return np.asarray(Image.open(io.BytesIO(response.content)))


def _mask_of(box: dict, shape=(400, 600)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[box["y0"] : box["y1"], box["x0"] : box["x1"]] = True
    return mask


def _apply(client, box: dict) -> dict:
    client.post("/api/mask/rect", json=box)
    response = client.post("/api/apply", json={"method": "telea"})
    assert response.status_code == 200, response.text
    return response.json()


class TestTheMeasuredBug:
    def test_the_second_edit_does_not_touch_the_first(self, client, photo) -> None:
        """★★ 先前 A 區域 11997 / 12000 個像素被第二次編輯重新生成。"""
        source = np.asarray(Image.open(photo).convert("RGB"))
        _open(client, photo)

        _apply(client, BOX_A)
        after_first = _export(client)
        assert not np.array_equal(after_first[A], source[A]), "A 應該真的被移除了"

        _apply(client, BOX_B)
        after_second = _export(client)

        differing = int((after_second[A] != after_first[A]).any(axis=2).sum())
        assert differing == 0, f"A 區域有 {differing} 個像素被第二次編輯改動了"
        assert not np.array_equal(after_second[B], source[B]), "B 應該被移除了"

    def test_the_mask_is_cleared_after_an_edit(self, client, photo) -> None:
        """留著的話，下一次執行會變成「上一個範圍 ∪ 新範圍」，兩塊一起重算。"""
        _open(client, photo)

        _apply(client, BOX_A)

        assert client.get("/api/state").json()["mask_pixels"] == 0

    def test_the_working_image_always_equals_replaying_the_layers(self, client, photo) -> None:
        """★ 不變式：working == render(source, layers)。

        執行之後 working 直接沿用結果（省一次全圖重播），所以這一條要守著
        ——否則兩條路會慢慢分岔，而分岔是靜默的。
        """
        source = np.asarray(Image.open(photo).convert("RGB"))
        _open(client, photo)

        _apply(client, BOX_A)
        _apply(client, BOX_B)

        assert np.array_equal(_export(client), layers_module.render(source, server.SESSION.layers))


class TestLayers:
    def test_each_edit_becomes_a_layer(self, client, photo) -> None:
        _open(client, photo)

        first = _apply(client, BOX_A)
        second = _apply(client, BOX_B)

        assert first["layers"] == 1
        assert second["layers"] == 2
        state = client.get("/api/state").json()
        assert [layer["method"] for layer in state["layers"]] == ["telea", "telea"]
        assert state["can_undo"] is True

    def test_the_crop_of_the_second_layer_does_not_cover_the_first(self, client, photo) -> None:
        """裁切區變大是 LaMa 品質下滑的原因之一——這裡確認它不會。"""
        _open(client, photo)

        _apply(client, BOX_A)
        _apply(client, BOX_B)

        crops = [layer["crop"] for layer in client.get("/api/state").json()["layers"]]
        x, y, width, height = crops[1]
        assert x > BOX_A["x1"], "第二層的裁切區不應該涵蓋第一層"


class TestUndoRedo:
    def test_undo_restores_the_previous_bytes_exactly(self, client, photo) -> None:
        _open(client, photo)
        _apply(client, BOX_A)
        after_first = _export(client)
        _apply(client, BOX_B)

        client.post("/api/undo")

        assert np.array_equal(_export(client), after_first)

    def test_redo_restores_the_bytes_exactly(self, client, photo) -> None:
        _open(client, photo)
        _apply(client, BOX_A)
        _apply(client, BOX_B)
        after_both = _export(client)
        client.post("/api/undo")

        client.post("/api/redo")

        assert np.array_equal(_export(client), after_both)

    def test_undo_puts_the_mask_back(self, client, photo) -> None:
        """最常見的用法：邊緣還留了一點陰影 → 復原 → 把範圍擴大 → 再執行。"""
        _open(client, photo)
        _apply(client, BOX_A)
        assert client.get("/api/state").json()["mask_pixels"] == 0

        client.post("/api/undo")

        state = client.get("/api/state").json()
        assert state["mask_pixels"] == 100 * 60, "復原要把當初圈的範圍放回來"
        assert state["can_undo"] is False
        assert state["can_redo"] is True

    def test_redo_clears_the_mask_again(self, client, photo) -> None:
        _open(client, photo)
        _apply(client, BOX_A)
        client.post("/api/undo")

        client.post("/api/redo")

        assert client.get("/api/state").json()["mask_pixels"] == 0

    def test_a_new_edit_discards_the_redo_stack(self, client, photo) -> None:
        _open(client, photo)
        _apply(client, BOX_A)
        client.post("/api/undo")
        assert client.get("/api/state").json()["can_redo"] is True

        _apply(client, BOX_B)

        assert client.get("/api/state").json()["can_redo"] is False

    def test_undo_and_redo_are_refused_when_empty(self, client, photo) -> None:
        _open(client, photo)

        assert client.post("/api/undo").status_code == 400
        assert client.post("/api/redo").status_code == 400
        assert "沒有可以" in client.post("/api/undo").json()["detail"]


class TestTheGuaranteeWithSeveralLayers:
    def test_nothing_changes_outside_the_union_of_the_write_regions(self, client, photo) -> None:
        """★★ 鐵律的多層版本。

        界線是**寫入範圍**（羽化會向外擴一圈），不是使用者畫的那條線——
        所以要用 ``blend_alpha_for`` 算，不可以自己手寫膨脹
        （兩次膨脹不等於一次更大的膨脹）。
        """
        source = np.asarray(Image.open(photo).convert("RGB"))
        _open(client, photo)
        for box in (BOX_A, BOX_B):
            _apply(client, box)
            client.post("/api/clear_mask")

        result = _export(client)

        allowed = np.zeros(source.shape[:2], dtype=bool)
        for box in (BOX_A, BOX_B):
            allowed |= blend_alpha_for(_mask_of(box)) > 0

        changed = (result != source).any(axis=2)
        assert changed.any(), "這兩層應該真的改了東西"
        assert not changed[~allowed].any(), (
            f"寫入範圍之外有 {int(changed[~allowed].sum())} 個像素被動過"
        )


class TestPersistence:
    def test_reopening_the_same_photo_restores_the_layers(self, client, photo) -> None:
        """★ 存檔的意義：重開要接回之前的圖層，而且是同一張畫面。"""
        _open(client, photo)
        _apply(client, BOX_A)
        _apply(client, BOX_B)
        before = _export(client)

        state = _open(client, photo)

        assert len(state["layers"]) == 2, "重開應該接回兩層"
        assert np.array_equal(_export(client), before)

    def test_the_project_is_attached_and_reported(self, client, photo) -> None:
        state = _open(client, photo)

        assert state["project_saved"] is True
        assert state["project_note"] is None

    def test_a_photo_that_moved_is_reconnected(self, client, photo, tmp_path) -> None:
        """上傳的暫存檔被清掉、或照片被搬走之後，同一張照片要接回同一個專案。"""
        _open(client, photo)
        _apply(client, BOX_A)
        before = _export(client)

        moved = tmp_path / "moved.png"
        moved.write_bytes(photo.read_bytes())
        photo.unlink()

        state = _open(client, moved)

        assert len(state["layers"]) == 1
        assert np.array_equal(_export(client), before)

    def test_editing_still_works_when_the_project_cannot_be_saved(
        self, client, tmp_path, monkeypatch
    ) -> None:
        """★ 專案掛不上時要降級成純記憶體，而不是不能用。

        **默默不存檔是這裡唯一不能接受的失敗方式**，所以一定要回報。
        """
        blocked = tmp_path / "blocked"
        blocked.write_text("這是一個檔案，不是目錄", encoding="utf-8")
        monkeypatch.setattr(server.config, "projects_dir", lambda: blocked / "projects")

        rng = np.random.default_rng(3)
        array = rng.integers(100, 150, size=(200, 200, 3), dtype=np.uint8)
        array[60:120, 60:140] = 20
        path = tmp_path / "nosave.png"
        Image.fromarray(array, mode="RGB").save(path)

        state = _open(client, path)
        assert state["project_saved"] is False
        assert state["project_note"], "降級一定要說出來"

        _apply(client, {"x0": 60, "y0": 60, "x1": 140, "y1": 120})
        assert client.post("/api/undo").status_code == 200
        assert client.get("/api/state").json()["can_undo"] is False
