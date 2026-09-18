"""本地介面的 API 測試。

用 FastAPI 的測試客戶端，不需要真的開一個伺服器或開瀏覽器。
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photoman.web import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每個測試都拿到乾淨的工作階段。"""
    monkeypatch.setattr(server, "SESSION", server.Session())
    return TestClient(server.app)


@pytest.fixture
def sample_image(tmp_path):
    rng = np.random.default_rng(20260918)
    array = rng.integers(60, 200, size=(160, 240, 3), dtype=np.uint8)
    array[60:100, 80:150] = 20  # 一塊暗色「物件」
    path = tmp_path / "sample.png"
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _open(client, path) -> dict:
    response = client.post("/api/open", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()


def _decode(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)))


class TestPage:
    def test_index_is_served(self, client) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "photoMan" in response.text

    def test_konva_is_vendored(self, client) -> None:
        """介面必須完全離線——Konva 內嵌，不經 CDN。"""
        response = client.get("/static/konva.min.js")

        assert response.status_code == 200
        assert len(response.content) > 100_000

    def test_static_path_traversal_is_refused(self, client) -> None:
        response = client.get("/static/..%2Fserver.py")

        assert response.status_code == 404

    def test_index_has_no_cdn_reference(self, client) -> None:
        """這是「完全離線可用」的前提，破壞它會令工具在沒網絡時失效。"""
        html = client.get("/").text

        assert "http://cdn" not in html and "https://cdn" not in html
        assert "unpkg.com" not in html and "jsdelivr" not in html


class TestStateBeforeOpening:
    def test_reports_nothing_opened(self, client) -> None:
        state = client.get("/api/state").json()

        assert state["opened"] is False
        assert state["has_mask"] is False

    def test_operations_require_an_image(self, client) -> None:
        assert client.get("/api/preview").status_code == 400
        assert client.post("/api/select", json={"x": 1, "y": 1}).status_code == 400
        assert client.post("/api/apply", json={}).status_code == 400


class TestOpening:
    def test_open_reports_the_image(self, client, sample_image) -> None:
        state = _open(client, sample_image)

        assert state["opened"] is True
        assert state["width"] == 240
        assert state["height"] == 160
        assert state["mask_pixels"] == 0

    def test_preview_is_returned(self, client, sample_image) -> None:
        _open(client, sample_image)

        payload = client.get("/api/preview").json()

        assert payload["image"].startswith("data:image/png;base64,")
        assert _decode(payload["image"]).shape[:2] == (160, 240)

    def test_missing_file_is_reported(self, client, tmp_path) -> None:
        response = client.post("/api/open", json={"path": str(tmp_path / "nope.png")})

        assert response.status_code == 404

    def test_large_images_are_downscaled_for_preview(self, client, tmp_path) -> None:
        """42 MP 的圖直接送給瀏覽器既慢又沒有意義。"""
        big = np.zeros((2400, 3200, 3), dtype=np.uint8)
        path = tmp_path / "big.png"
        Image.fromarray(big, mode="RGB").save(path)

        state = _open(client, path)

        assert state["width"] == 3200, "原圖尺寸要如實回報"
        assert state["preview_width"] == server.PREVIEW_MAX, "預覽要縮到上限之內"

    def test_upload_works(self, client, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(server.config, "work_dir", lambda: tmp_path / "work")
        buffer = io.BytesIO()
        Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8), mode="RGB").save(buffer, "PNG")
        buffer.seek(0)

        response = client.post("/api/upload", files={"file": ("photo.png", buffer, "image/png")})

        assert response.status_code == 200, response.text
        assert response.json()["width"] == 60


class TestSelection:
    """選取需要 MobileSAM 模型。沒有模型時跳過。"""

    @pytest.fixture(autouse=True)
    def _require_sam(self):
        if not server._models_ready()["sam"]:
            pytest.skip("未下載 MobileSAM 模型")

    def test_clear_mask_is_a_no_op_on_an_empty_mask(self, client, sample_image) -> None:
        _open(client, sample_image)

        payload = client.post("/api/clear_mask").json()

        assert payload["mask_pixels"] == 0
        assert payload["has_mask"] is False

    def test_click_outside_the_image_is_refused(self, client, sample_image) -> None:
        _open(client, sample_image)

        response = client.post("/api/select", json={"x": 9999, "y": 9999})

        assert response.status_code == 400
        assert "之外" in response.json()["detail"]


class TestFullFlow:
    """端到端：開啟 → 點擊選取 → 移除 → 匯出。

    用 Telea 而不是 LaMa，因為這個測試要快。LaMa 的品質由
    tests/test_edit.py 的量測測試把關。
    """

    @pytest.fixture(autouse=True)
    def _require_sam(self):
        if not server._models_ready()["sam"]:
            pytest.skip("未下載 MobileSAM 模型")

    def test_click_select_apply_export(self, client, sample_image) -> None:
        state = _open(client, sample_image)
        assert state["has_mask"] is False

        # 點一下暗色方塊的中央
        selected = client.post("/api/select", json={"x": 115, "y": 80, "mode": "add"}).json()
        assert selected["has_mask"] is True
        assert selected["mask_pixels"] > 500
        assert selected["overlay"].startswith("data:image/png;base64,")

        # 移除
        applied = client.post("/api/apply", json={"method": "telea"}).json()
        assert applied["image"].startswith("data:image/png;base64,")
        assert client.get("/api/state").json()["has_result"] is True

        # 匯出——**原解析度**，不是預覽。
        export = client.get("/api/result")
        assert export.status_code == 200
        assert _decode("data:image/png;base64," + base64.b64encode(export.content).decode()).shape[
            :2
        ] == (160, 240)

    def test_apply_without_a_selection_is_refused(self, client, sample_image) -> None:
        _open(client, sample_image)

        response = client.post("/api/apply", json={"method": "telea"})

        assert response.status_code == 400
        assert "還沒有選取" in response.json()["detail"]

    def test_clearing_the_mask_disables_apply(self, client, sample_image) -> None:
        _open(client, sample_image)
        client.post("/api/select", json={"x": 115, "y": 80, "mode": "add"})

        cleared = client.post("/api/clear_mask").json()

        assert cleared["mask_pixels"] == 0
        assert client.get("/api/state").json()["has_mask"] is False

    def test_opening_another_image_resets_everything(self, client, sample_image, tmp_path) -> None:
        """換圖之後不應該還看到上一張的選取或成品。

        這是一個很容易漏的地方：狀態留在伺服器上，使用者看到的是
        新圖片疊著舊遮罩。
        """
        _open(client, sample_image)
        client.post("/api/select", json={"x": 115, "y": 80, "mode": "add"})
        client.post("/api/apply", json={"method": "telea"})

        other = tmp_path / "other.png"
        Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8), mode="RGB").save(other)
        state = _open(client, other)

        assert state["has_mask"] is False
        assert state["has_result"] is False
        assert state["width"] == 80


class TestExport:
    def test_export_requires_a_result(self, client, sample_image) -> None:
        _open(client, sample_image)

        response = client.get("/api/result")

        assert response.status_code == 400
        assert "還沒有結果" in response.json()["detail"]


class TestSettings:
    def test_api_key_is_never_returned(self, client, tmp_path, monkeypatch) -> None:
        """★ API key 回傳給瀏覽器時一律遮蔽。"""
        config_file = tmp_path / "config.json"
        monkeypatch.setattr(server.config, "config_path", lambda: config_file)
        monkeypatch.setattr(server.config, "config_dir", lambda: tmp_path)

        client.post("/api/settings", json={"api_key": "sk-or-v1-secret-value"})
        payload = client.get("/api/settings").json()

        assert payload["providers"]["openrouter"]["api_key"] == server.config.MASK
        assert "secret" not in str(payload)

    def test_masked_value_does_not_overwrite_the_stored_key(
        self, client, tmp_path, monkeypatch
    ) -> None:
        """介面把讀到的遮蔽值原樣送回來時，不可以蓋掉真正的 key。

        這是一個很容易寫錯的地方：每次開啟設定再儲存，key 就會變成
        「••••••••」，而使用者不會發現，直到某次呼叫失敗。
        """
        monkeypatch.setattr(server.config, "config_path", lambda: tmp_path / "config.json")
        monkeypatch.setattr(server.config, "config_dir", lambda: tmp_path)

        client.post("/api/settings", json={"api_key": "sk-real"})
        client.post("/api/settings", json={"api_key": server.config.MASK})

        assert server.config.get_api_key() == "sk-real"

    def test_key_is_stored_outside_the_repository(self) -> None:
        """設定檔必須在 %APPDATA%，不在 repo 之內。

        否則 `git add -A` 會把 API key 一併提交——而那個意外一旦發生
        就無法收回，key 已經在 git 歷史裡了。
        """
        from photoman.paths import PROJECT_ROOT

        assert PROJECT_ROOT not in server.config.config_path().parents
