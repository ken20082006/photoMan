"""本地介面的 API 測試。

用 FastAPI 的測試客戶端，不需要真的開一個伺服器或開瀏覽器。
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from photoman.web import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每個測試都拿到乾淨的工作階段。

    ★ 專案目錄也要導到 tmp_path。測試用的圖每一張內容都一樣，
    不導開的話它們會**共用同一個專案目錄**（目錄是以內容雜湊命名的），
    於是後面的測試會看到前面測試留下的圖層，而且會寫進使用者真正的
    `%APPDATA%\\photoMan\\projects`。
    """
    monkeypatch.setattr(server, "SESSION", server.Session())
    monkeypatch.setattr(server.config, "projects_dir", lambda: tmp_path / "projects")
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


class TestManualMask:
    """★ 使用者自己畫的範圍就是遮罩本身。

    這一整組**不需要任何模型**——那正是它的意義：手動圈選是
    「AI 只改我框起來的地方」這條路上唯一不依賴模型判斷的一段。
    """

    def _stroke(self, width: int, height: int, draw) -> str:
        canvas = Image.new("L", (width, height), 0)
        draw(ImageDraw.Draw(canvas))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def test_rect_makes_a_mask_without_any_model(self, client, sample_image) -> None:
        _open(client, sample_image)

        payload = client.post(
            "/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100}
        ).json()

        assert payload["mask_pixels"] == 70 * 40, "矩形本身就是要改的範圍，不多不少"
        assert payload["has_mask"] is True
        assert payload["overlay"].startswith("data:image/png;base64,")

    def test_a_rect_dragged_backwards_still_works(self, client, sample_image) -> None:
        """使用者可以由右下往左上拉。"""
        _open(client, sample_image)

        payload = client.post(
            "/api/mask/rect", json={"x0": 150, "y0": 100, "x1": 80, "y1": 60}
        ).json()

        assert payload["mask_pixels"] == 70 * 40

    def test_rect_is_clipped_to_the_image(self, client, sample_image) -> None:
        """拉到照片外面不應該令伺服器出錯或寫出界。"""
        _open(client, sample_image)

        payload = client.post(
            "/api/mask/rect", json={"x0": 200, "y0": 120, "x1": 9999, "y1": 9999}
        ).json()

        assert payload["mask_pixels"] == (240 - 200) * (160 - 120)

    def test_degenerate_rect_is_refused(self, client, sample_image) -> None:
        _open(client, sample_image)

        response = client.post("/api/mask/rect", json={"x0": 10, "y0": 10, "x1": 10, "y1": 10})

        assert response.status_code == 400
        assert "太小" in response.json()["detail"]

    def test_subtract_removes_the_area(self, client, sample_image) -> None:
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": 40, "y0": 40, "x1": 200, "y1": 120})

        payload = client.post(
            "/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100, "mode": "subtract"}
        ).json()

        assert payload["mask_pixels"] == 160 * 80 - 70 * 40

    def test_subtracting_everything_leaves_an_empty_mask(self, client, sample_image) -> None:
        """擦光之後就沒有東西可以改——不應該留下一個空的結果。"""
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100})

        client.post(
            "/api/mask/rect", json={"x0": 0, "y0": 0, "x1": 240, "y1": 160, "mode": "subtract"}
        )

        assert client.get("/api/state").json()["has_mask"] is False
        assert client.post("/api/apply", json={"method": "telea"}).status_code == 400

    def test_a_brush_stroke_becomes_a_mask(self, client, sample_image) -> None:
        _open(client, sample_image)
        image = self._stroke(240, 160, lambda d: d.ellipse([100, 60, 140, 100], fill=255))

        payload = client.post("/api/mask/stroke", json={"image": image}).json()

        # 直徑 40 的圓面積約 1257 像素；容許邊界上的差異。
        assert 1000 < payload["mask_pixels"] < 1500

    def test_a_stroke_is_scaled_up_to_the_original(self, client, tmp_path) -> None:
        """★ 瀏覽器只知道預覽圖的大小——換算必須在這裡做對。

        42 MP 的圖在瀏覽器上是 1600 像素寬，兩者差 26 倍。換算寫錯的話
        遮罩會落在完全不相干的位置，而畫面上完全看不出來。
        """
        big = np.zeros((2400, 3200, 3), dtype=np.uint8)
        path = tmp_path / "big.png"
        Image.fromarray(big, mode="RGB").save(path)
        state = _open(client, path)
        width, height = state["preview_width"], state["preview_height"]
        assert (width, height) != (3200, 2400), "這張圖本來就應該被縮成預覽"

        empty = self._stroke(width, height, lambda d: None)
        assert client.post("/api/mask/stroke", json={"image": empty}).json()["mask_pixels"] == 0

        upper_half = self._stroke(
            width, height, lambda d: d.rectangle([0, 0, width - 1, height // 2], fill=255)
        )
        payload = client.post("/api/mask/stroke", json={"image": upper_half}).json()

        # 預覽的上半 = 原圖的上半。差幾個邊界像素是可以接受的。
        assert abs(payload["mask_pixels"] - 3200 * 1200) < 3200 * 8

    def test_a_broken_stroke_image_is_refused(self, client, sample_image) -> None:
        _open(client, sample_image)

        response = client.post("/api/mask/stroke", json={"image": "data:image/png;base64,!!!!"})

        assert response.status_code == 400

    def test_manual_mask_endpoints_require_an_image(self, client) -> None:
        rect = client.post("/api/mask/rect", json={"x0": 0, "y0": 0, "x1": 5, "y1": 5})
        stroke = client.post("/api/mask/stroke", json={"image": "data:,"})

        assert rect.status_code == 400
        assert stroke.status_code == 400

    def test_nothing_outside_the_drawn_rect_is_ever_written(self, client, sample_image) -> None:
        """★★ 鐵律的介面版本。

        使用者框起來的地方以外，像素逐 bit 不變。前處理會向外擴張
        （羽化 12 + 模型邊距 16），所以容許一圈邊距——但那一圈之外
        不可以有任何一個位元組被動過。
        """
        x0, y0, x1, y1 = 80, 60, 150, 100
        before = np.asarray(Image.open(sample_image).convert("RGB"))
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": x0, "y0": y0, "x1": x1, "y1": y1})

        client.post("/api/apply", json={"method": "telea"})
        export = client.get("/api/result?format=png")
        after = _decode("data:image/png;base64," + base64.b64encode(export.content).decode())

        changed = (before != after).any(axis=2)
        assert changed.any(), "框起來的地方應該真的被改了"

        margin = 32  # 羽化 12 + 膨脹 16，再留一點餘裕
        allowed = np.zeros(before.shape[:2], dtype=bool)
        allowed[y0 - margin : y1 + margin, x0 - margin : x1 + margin] = True
        assert not changed[~allowed].any(), (
            f"框外有 {int(changed[~allowed].sum())} 個像素被改動——遮罩外必須逐 bit 不變"
        )


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
        export = client.get("/api/result?format=png")
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


class TestDetail:
    """★ 放大時那一塊——由原檔重新取樣，不是預覽被拉大。

    這條路存在的理由：預覽最長邊是 1600，42 MP 的圖在畫布上只有 20%。
    放大超過那個密度之後看到的其實是縮圖被拉大（實測高頻細節只剩一半），
    而「看不清自己修的品質」對一個修圖工具是致命的。
    """

    def test_the_pixels_come_from_the_original(self, client, tmp_path) -> None:
        """★ 回傳的像素必須等於**原檔那一塊**的重新取樣。

        這一條如果錯了，放大之後看到的仍然是另一張圖的放大版——
        而且畫面上完全看不出來。
        """
        rng = np.random.default_rng(5)
        photo = rng.integers(0, 256, size=(600, 900, 3), dtype=np.uint8)
        path = tmp_path / "detail.png"
        Image.fromarray(photo, mode="RGB").save(path)
        _open(client, path)

        payload = client.get(
            "/api/detail?x0=100&y0=200&x1=300&y1=400&w=200&h=200&source=true"
        ).json()

        got = _decode(payload["image"])
        assert (payload["x"], payload["y"]) == (100, 200)
        assert (payload["width"], payload["height"]) == (200, 200)
        assert got.shape[:2] == (200, 200)
        # 原檔那一塊，用同一條路縮放
        expected = np.asarray(
            Image.fromarray(photo[200:400, 100:300]).resize((200, 200), Image.LANCZOS)
        )
        np.testing.assert_array_equal(got[..., :3], expected)

    def test_it_can_be_smaller_than_the_region(self, client, tmp_path) -> None:
        """回傳的是「螢幕要多少像素就給多少」，不是原解析度那一塊。

        放大 1.2 倍時可見範圍是 8 MP——送原解析度既慢又沒有意義。
        """
        rng = np.random.default_rng(6)
        photo = rng.integers(0, 256, size=(400, 400, 3), dtype=np.uint8)
        path = tmp_path / "small.png"
        Image.fromarray(photo, mode="RGB").save(path)
        _open(client, path)

        payload = client.get("/api/detail?x0=0&y0=0&x1=400&y1=400&w=100&h=100&source=true").json()

        assert _decode(payload["image"]).shape[:2] == (100, 100)

    def test_the_region_is_clamped_to_the_image(self, client, sample_image) -> None:
        """畫面可以拖到照片外面，所以請求的範圍也會超出。"""
        _open(client, sample_image)

        payload = client.get(
            "/api/detail?x0=-500&y0=-500&x1=9999&y1=9999&w=300&h=300&source=true"
        ).json()

        assert (payload["x"], payload["y"]) == (0, 0)
        assert (payload["width"], payload["height"]) == (240, 160)

    def test_a_region_entirely_outside_is_refused(self, client, sample_image) -> None:
        _open(client, sample_image)

        response = client.get("/api/detail?x0=5000&y0=5000&x1=6000&y1=6000&w=100&h=100")

        assert response.status_code == 400
        assert "照片之外" in response.json()["detail"]

    def test_the_output_size_is_capped(self, client, sample_image) -> None:
        """擋住畸形的請求——放大時可見範圍很小，這個上限幾乎不會碰到。"""
        _open(client, sample_image)

        payload = client.get(
            "/api/detail?x0=0&y0=0&x1=240&y1=160&w=99999&h=99999&source=true"
        ).json()

        assert max(_decode(payload["image"]).shape[:2]) == server.DETAIL_MAX

    def test_the_overlay_covers_the_same_region(self, client, sample_image) -> None:
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": 0, "y0": 0, "x1": 240, "y1": 80})

        payload = client.get("/api/detail?x0=0&y0=0&x1=240&y1=160&w=240&h=160&source=true").json()

        overlay = _decode(payload["overlay"])
        assert overlay.shape[:2] == (160, 240)
        assert overlay[..., 3].max() > 0, "上半是有選取的，疊圖不可以全透明"

    def test_it_requires_an_image(self, client) -> None:
        response = client.get("/api/detail?x0=0&y0=0&x1=10&y1=10&w=10&h=10")

        assert response.status_code == 400


class TestReferenceImages:
    def test_local_removal_refuses_them(self, client, sample_image) -> None:
        """本機的 LaMa 沒有「參考圖」這個概念——它只會由邊界往內填。"""
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100})

        response = client.post(
            "/api/apply",
            json={"method": "telea", "references": ["data:image/png;base64,AAAA"]},
        )

        assert response.status_code == 400
        assert "參考圖只有雲端模型" in response.json()["detail"]

    def test_too_many_are_refused(self, client, sample_image) -> None:
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100})

        response = client.post(
            "/api/apply",
            json={
                "model": "some/model",
                "prompt": "加一隻狗",
                "references": ["data:,"] * (server.MAX_REFERENCES + 1),
            },
        )

        assert response.status_code == 400
        assert str(server.MAX_REFERENCES) in response.json()["detail"]

    def test_the_model_list_reports_the_limit(self, client, monkeypatch) -> None:
        from photoman.providers import ImageModel

        class _Client:
            def models(self):
                return [ImageModel("a/b", "B", 0.02, "1024", max_references=3)]

        monkeypatch.setattr(server, "SESSION", server.SESSION)
        import photoman.providers as providers

        monkeypatch.setattr(providers, "client_from_config", lambda: _Client())

        payload = client.get("/api/models").json()

        assert payload["models"][0]["max_references"] == 3


class TestEditIntent:
    """顏色對齊的意圖**按引擎決定**，而且是一條固定的產品規則。

    這件事先前試過讓使用者選（介面多兩顆按鈕），但那個選擇本身才是負擔——
    而且使用者第一次用就會遇到「我怎麼知道該選哪個」。現在介面直接寫明
    「只是要移除東西的話用本機」，規則只剩一條。

    實測根據：同一個雲端模型做「移除」時，填補意圖的落差是 2.4／3.9 級、
    編輯意圖是 22.8／38.6 級；但叫它「把衣服改成紅色」時，填補意圖會把
    紅色抹成周圍的綠色（實測 (170,40,40) → (89,129,60)）。
    兩件事都做得到，所以只能選一邊——選的是「雲端是用來換內容的」。
    """

    @pytest.fixture
    def recorded(self, monkeypatch):
        from photoman.edit import EditResult

        seen: dict = {}

        def _fake(base, mask, **kwargs):
            seen.update(kwargs)
            # 裁切框要包得住遮罩——`EditLayer.from_edit` 會檢查這件事，
            # 而那個檢查是刻意的（框不住就代表重播會少改一塊）。
            shape = mask.shape[:2]
            return EditResult(
                image=base.copy(),
                patch=np.zeros((*shape, 3), np.uint8),
                crop=(0, 0, shape[1], shape[0]),
                mask_sha256="a" * 64,
                checksum=0.0,
                method=kwargs.get("method", "telea"),
            )

        monkeypatch.setattr(server, "apply_generative_edit", _fake)
        return seen

    def _select(self, client):
        client.post("/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100})

    def test_a_cloud_model_is_asked_to_edit(self, client, sample_image, recorded) -> None:
        """雲端＝換成別的內容，只扣掉模型的漂移，不動它畫的顏色。"""
        _open(client, sample_image)
        self._select(client)

        client.post("/api/apply", json={"model": "some/model", "prompt": "把衣服改成紅色"})

        assert recorded["intent"] == "edit"

    def test_a_local_engine_fills(self, client, sample_image, recorded) -> None:
        """本機＝填補，內容與周圍一致。它沒有「換成別的內容」的能力。"""
        _open(client, sample_image)
        self._select(client)

        client.post("/api/apply", json={"method": "telea"})

        assert recorded["intent"] == "fill"

    def test_the_client_cannot_override_it(self, client, sample_image, recorded) -> None:
        """介面已經沒有那個選擇了，所以也沒有東西可以覆蓋它。"""
        _open(client, sample_image)
        self._select(client)

        client.post(
            "/api/apply",
            json={"method": "telea", "intent": "edit", "model": "some/model"},
        )

        assert recorded["intent"] == "edit", "雲端本來就是 edit"

        client.post("/api/clear_mask")
        client.post("/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100})
        client.post("/api/apply", json={"method": "telea", "intent": "edit"})

        assert recorded["intent"] == "fill", "本機不受傳進來的值影響"


class TestExportFormat:
    """★ 匯出預設是 WebP，不是 PNG。

    PNG 存照片是浪費——實測同一張 18.4 MP 的圖，PNG 是 17.32 MB
    （比原檔的 JPEG 大 3.4 倍），WebP q95 是 3.49 MB（比原檔還小），
    而平均只差 0.95 級。要完全不失真的人可以要 PNG。
    """

    def _edited(self, client, sample_image):
        _open(client, sample_image)
        client.post("/api/mask/rect", json={"x0": 80, "y0": 60, "x1": 150, "y1": 100})
        client.post("/api/apply", json={"method": "telea"})

    def test_the_default_is_webp(self, client, sample_image) -> None:
        self._edited(client, sample_image)

        response = client.get("/api/result")

        assert response.headers["content-type"] == "image/webp"
        assert ".webp" in response.headers["content-disposition"]

    def test_it_is_a_real_image_of_the_right_size(self, client, sample_image) -> None:
        self._edited(client, sample_image)

        got = np.asarray(Image.open(io.BytesIO(client.get("/api/result").content)))

        assert got.shape[:2] == (160, 240), "匯出必須是原解析度，不是預覽"

    def test_png_can_still_be_asked_for(self, client, sample_image) -> None:
        self._edited(client, sample_image)

        response = client.get("/api/result?format=png")

        assert response.headers["content-type"] == "image/png"
        assert ".png" in response.headers["content-disposition"]
        assert Image.open(io.BytesIO(response.content)).format == "PNG"

    def test_webp_is_smaller_than_png(self, client, tmp_path) -> None:
        """用一張有紋理的圖——純色圖兩者都很小，比不出東西。"""
        rng = np.random.default_rng(3)
        photo = rng.integers(0, 256, size=(600, 900, 3), dtype=np.uint8)
        path = tmp_path / "texture.png"
        Image.fromarray(photo, mode="RGB").save(path)
        self._edited(client, path)

        webp = client.get("/api/result")
        png = client.get("/api/result?format=png")

        assert len(webp.content) < len(png.content)

    def test_an_unknown_format_falls_back_to_webp(self, client, sample_image) -> None:
        """不認識的格式不要變成 500——當作預設值就好。"""
        self._edited(client, sample_image)

        response = client.get("/api/result?format=tiff")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
