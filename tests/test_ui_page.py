"""介面頁面的靜態檢查。

這些測試不需要瀏覽器，但它們守著兩個實際踩過的坑——
兩個都令「開啟圖片」按下去完全沒有反應。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "src" / "photoman" / "web" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css(html: str) -> str:
    return re.search(r"<style>(.*?)</style>", html, re.S).group(1)


@pytest.fixture(scope="module")
def js(html: str) -> str:
    return re.search(r"<script>\n(.*?)\n</script>", html, re.S).group(1)


class TestHiddenAttributeIsNotOverridden:
    """★ 這是一個實際踩過的坑。

    用 JS 設定 ``element.hidden = true`` 只有在該元素沒有更高優先度的
    ``display`` 規則時才有效。``#drop { display: flex }`` 是 id 選擇器，
    優先度高過瀏覽器為 ``[hidden]`` 預設的 ``display: none``——
    少了 ``#drop[hidden] { display: none }``，那層就永遠不會隱藏，
    選完照片之後一直蓋在畫布上，使用者看到的是「按了沒反應」。
    """

    def _ids_with_display(self, css: str) -> set[str]:
        """在 CSS 中對 id 選擇器設定 display 的那些 id。"""
        found = set()
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            if "display" not in body:
                continue
            for part in selector.split(","):
                part = part.strip()
                match = re.fullmatch(r"#([A-Za-z0-9_-]+)", part)
                if match:
                    found.add(match.group(1))
        return found

    def test_every_id_with_display_has_a_hidden_rule(self, css: str) -> None:
        needs_rule = self._ids_with_display(css)
        assert needs_rule, "沒有找到任何設定 display 的 id 規則——選擇器解析可能壞了"

        overridden = {
            element_id
            for element_id in needs_rule
            if not re.search(rf"#{element_id}\[hidden\]\s*\{{[^}}]*display\s*:\s*none", css)
        }

        assert not overridden, (
            f"這些 id 設定了 display，但沒有對應的 [hidden] 規則：{sorted(overridden)}。"
            "用 JS 設定 .hidden 會完全沒有效果。"
        )


class TestKonvaContainerIsExclusive:
    """★ 另一個實際踩過的坑。

    Konva 建立 Stage 時會清空容器的子元素。把其他東西放在同一個容器裡，
    它們會消失，之後 ``getElementById`` 回傳 null——症狀同樣是
    「按了沒反應」，而且只會看到一句 ``Cannot set properties of null``。
    """

    def test_stage_container_is_empty_in_the_markup(self, html: str) -> None:
        match = re.search(r'<div id="stage-wrap">(.*?)</div>', html, re.S)
        assert match, "找不到 stage-wrap"

        assert match.group(1).strip() == "", (
            "stage-wrap 裡面不可以放任何東西——Konva 會把它們刪掉。"
            "版面和覆蓋層要放在外層的 stage-area。"
        )

    def test_overlays_live_in_the_outer_area(self, html: str) -> None:
        area = re.search(r'<div id="stage-area">(.*?)\n\s*<aside', html, re.S)
        assert area, "找不到 stage-area"

        assert 'id="drop"' in area.group(1)
        assert 'id="busy"' in area.group(1)

    def test_layout_code_targets_the_outer_area(self, js: str) -> None:
        """量尺寸要看 stage-area——stage-wrap 由 Konva 擁有。"""
        assert '$("stage-area")' in js
        assert '$("stage-wrap").clientWidth' not in js


class TestFilePickingWorksWithoutJavaScript:
    def test_pick_button_is_a_native_label(self, html: str) -> None:
        """用 ``<label for=...>`` 而不是按鈕加 JS。

        這是 HTML 原生的檔案選擇觸發方式，任何腳本錯誤都影響不到它——
        而腳本錯誤正好是「按了沒反應」的另一個可能成因。
        """
        assert re.search(r'<label[^>]+for="file-input"', html)

    def test_file_input_exists_for_the_label(self, html: str) -> None:
        assert re.search(r'<input[^>]+id="file-input"', html)


class TestBindOrder:
    def test_event_binding_happens_before_the_canvas_is_built(self, js: str) -> None:
        """先綁事件再建畫布。

        反過來的话，Konva 一旦載入失敗，``buildStage()`` 會拋錯，
        綁事件永遠不會執行——所有按鈕都是死的，而且沒有任何訊息。
        """
        main = re.search(r"async function main\(\) \{(.*?)\n\}", js, re.S).group(1)
        # 去掉註解——不然會匹配到註解裡提到的函數名，而不是真正的呼叫。
        # （第一版就是這樣寫的，測試因此誤報。）
        code = "\n".join(line for line in main.splitlines() if not line.strip().startswith("//"))

        assert code.index("bind()") < code.index("buildStage()")


class TestManualSelectionTools:
    """★ 使用者自己圈選——這一組守著「我的手畫的就是遮罩」。"""

    def test_the_manual_tools_are_in_the_markup(self, html: str) -> None:
        for element_id in ("tool-brush", "tool-box", "tool-auto", "mode-add", "mode-sub"):
            assert re.search(rf'id="{element_id}"', html), f"缺少 {element_id}"

    def test_the_brush_size_slider_exists(self, html: str) -> None:
        assert re.search(r'<input[^>]+type="range"[^>]+id="brush-size"', html)

    def test_a_stroke_is_sent_once_on_pointer_up(self, js: str) -> None:
        """筆畫在瀏覽器上畫，放手才送一次。

        反過來（每個 mousemove 都送）會令塗抹嚴重延遲——而延遲在
        塗抹這種直接操作的動作上是不能接受的，使用者會以為滑鼠壞了。
        """
        assert 'stage.on("mousedown touchstart", onPointerDown)' in js
        assert 'stage.on("mousemove touchmove", movePointer)' in js
        assert 'stage.on("mouseup touchend", endPointer)' in js
        assert "/api/mask/stroke" in js and "/api/mask/rect" in js

    def test_releasing_outside_the_canvas_ends_the_stroke(self, js: str) -> None:
        """在照片外面放手也要收筆，否則那一筆會卡住，之後滑鼠一動
        就會由舊的起點長出一條長線。"""
        assert 'window.addEventListener("mouseup", endPointer)' in js

    def test_automatic_selection_only_runs_on_a_click(self, js: str) -> None:
        """筆刷與方框是拖曳的，拖曳結束的 click 不可以又叫一次 SAM。"""
        assert 'if (tool === "auto") clickSelect()' in js

    def test_strokes_are_sent_one_at_a_time(self, js: str) -> None:
        """併發送出的話，先回來的會蓋掉後回的，畫面上看到選取「彈回去」。"""
        assert "maskQueue = maskQueue.then" in js


class TestHistoryControls:
    """多步編輯的介面——每一次執行都直接生效，退路是復原。"""

    def test_the_undo_and_redo_buttons_exist(self, html: str) -> None:
        for element_id in ("btn-undo", "btn-redo", "history-stat", "save-banner"):
            assert re.search(rf'id="{element_id}"', html), f"缺少 {element_id}"

    def test_they_start_disabled(self, html: str) -> None:
        assert re.search(r'id="btn-undo"[^>]*disabled', html)
        assert re.search(r'id="btn-redo"[^>]*disabled', html)

    def test_all_of_them_are_wired_up(self, js: str) -> None:
        assert '$("btn-undo").onclick' in js
        assert '$("btn-redo").onclick' in js
        assert "/api/undo" in js and "/api/redo" in js

    def test_the_buttons_follow_the_server_state(self, js: str) -> None:
        """誰能按全部由 /api/state 決定，不散在各處各自判斷。

        散著判斷很容易出現「按鈕看起來能用、按下去報錯」。
        """
        assert "function renderHistory()" in js
        assert "snapshot.can_undo" in js and "snapshot.can_redo" in js

    def test_the_mask_overlay_is_kept_in_both_views(self, js: str) -> None:
        """看成品時也要看到遮罩——多步編輯是接著圈下一處，
        不應該強迫使用者先切回原圖。
        """
        view = re.search(
            r"async function setView\(next, \{ refit = false \} = \{\}\) \{(.*?)\n\}", js, re.S
        ).group(1)

        assert "setOverlay(maskOverlayURL)" in view


class TestPanAndZoomDetail:
    """拖曳與放大。兩個都是實測到的缺口。"""

    def test_panning_is_on_the_middle_button_not_a_tool(self, js: str) -> None:
        """★ 平移**不佔一個工具按鈕**——它是隨時要用的動作，
        不應該逼使用者在「畫」與「移動」之間切來切去。

        ⚠️ 一開始用右鍵，但右鍵在瀏覽器裡不可靠（選單、拖放、系統手勢
        都會來搶），使用者實測「右鍵移動不行」。所以改用中鍵。
        """
        assert "button === 1" in js, "中鍵要可以拖曳畫面"
        assert "button === 2" not in js, "右鍵不再用來平移——它在瀏覽器裡不可靠"
        assert 'id="tool-move"' not in js

    def test_the_middle_button_auto_scroll_is_suppressed(self, js: str) -> None:
        """中鍵按下去會觸發瀏覽器的自動捲動，要擋掉。"""
        down = re.search(r"function onPointerDown\(event\) \{(.*?)\n\}", js, re.S).group(1)
        assert "preventDefault" in down

    def test_panning_is_implemented_by_hand(self, js: str) -> None:
        """自己實作而不用 Konva 的 drag——不必猜 Konva 的內部狀態，
        而且我們需要一個「視野變了」的鉤子去重抓細節。"""
        assert "let panning = null" in js
        assert "panning.stageX +" in js

    def test_there_is_a_zoom_bar_above_the_canvas(self, html: str) -> None:
        """像市面上的修圖工具：縮放比例與按鈕放在畫布上方。"""
        area = re.search(r'<div id="stage-area">(.*?)\n  </div>', html, re.S).group(1)
        assert 'id="zoom-bar"' in area, "縮放列要在畫布那一區裡"
        for element_id in ("zoom-out", "zoom-in", "zoom-value", "btn-fit"):
            assert re.search(rf'id="{element_id}"', area), f"缺少 {element_id}"

    def test_the_percentage_is_relative_to_the_original(self, js: str) -> None:
        """100% ＝ 原圖一個像素對螢幕一個像素。

        不是相對預覽——那才是使用者心裡的那個數字，而且有了 /api/detail
        之後 100% 是真的做得到。
        """
        body = re.search(r"function zoomPercent\(\) \{(.*?)\n\}", js, re.S).group(1)
        assert "previewScale()" in body
        assert "zoomToOriginal" in js

    def test_switching_the_view_keeps_the_zoom_and_position(self, js: str) -> None:
        """★ 看原圖／看成品切換時要留在原地。

        先前 `setPhoto` **每一次**都呼叫 `fitToWindow()`，所以一切換就被
        重設縮放與位置——而那正是使用者要對比的時候。
        """
        body = re.search(
            r"async function setPhoto\(url, size, \{ refit = false \} = \{\}\) \{(.*?)\n\}",
            js,
            re.S,
        ).group(1)

        assert "if (refit) fitToWindow()" in body, "只有明示要 refit 才重新 fit"
        assert not re.search(r"^\s*fitToWindow\(\);", body, re.M), "不可以無條件 fit"

        # 只有「開一張新圖」要 refit——切換檢視、執行、復原、重做都要留在原地
        app = re.search(r"async function applyState\(state\) \{(.*?)\n\}", js, re.S).group(1)
        assert 'setView("after", { refit: true })' in app

    def test_the_label_follows_every_zoom(self, js: str) -> None:
        """滾輪、按鈕、適合視窗、開圖——每一個都要更新那個數字。"""
        zoom = re.search(r"function zoomBy\(factor, anchor\) \{(.*?)\n\}", js, re.S).group(1)
        assert "updateZoomLabel()" in zoom
        fit = re.search(r"function fitToWindow\(\) \{(.*?)\n\}", js, re.S).group(1)
        assert "updateZoomLabel()" in fit

    def test_a_stroke_cannot_start_outside_the_photo(self, js: str) -> None:
        """★ 實測：在照片外拖曳會沿著邊緣畫出 34,145 個像素，
        因為 beginStroke 用了 clampToImage——那會把界外的點夾到邊緣。"""
        body = re.search(r"function onPointerDown\(event\) \{(.*?)\n\}", js, re.S).group(1)
        # 去掉註解——不然會匹配到註解裡提到的函數名，而不是真正的呼叫。
        # （上面 TestBindOrder 就是這樣誤報過的。）
        code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("//"))

        assert "pointerToImage()" in code
        assert "clampToImage" not in code, "起筆要用 pointerToImage，界外要整個放棄"

    def test_the_detail_layer_exists_and_is_above_the_mask(self, js: str) -> None:
        """放大時的原檔細節要蓋過預覽與它的疊圖，但在筆畫之下。"""
        build = re.search(r"function buildStage\(\) \{(.*?)\n\}", js, re.S).group(1)
        order = [
            build.index(name)
            for name in (
                "stage.add(imageLayer)",
                "stage.add(maskLayer)",
                "stage.add(detailLayer)",
                "stage.add(drawLayer)",
            )
        ]

        assert order == sorted(order)

    def test_detail_is_only_fetched_when_magnifying(self, js: str) -> None:
        assert "DETAIL_ZOOM" in js
        assert "/api/detail?" in js
        body = re.search(r"async function loadDetail\(\) \{(.*?)\n\}", js, re.S).group(1)
        assert "scale <= DETAIL_ZOOM" in body, "縮小時不應該抓細節"

    def test_the_detail_replaces_the_preview_overlay(self, js: str) -> None:
        """兩層疊圖同時顯示的話，紅色會變深。"""
        body = re.search(r"async function showDetail\((.*?)\n\}", js, re.S).group(1)
        assert "maskImage.visible(false)" in body

    def test_the_detail_request_is_debounced(self, js: str) -> None:
        """滾輪會連續觸發——不 debounce 的話每個事件都送一次請求。"""
        assert "function scheduleDetail()" in js
        assert "setTimeout(loadDetail" in js


class TestReferenceImages:
    def test_the_control_exists(self, html: str) -> None:
        assert re.search(r'id="reference-input"', html)
        assert re.search(r'id="reference-list"', html)
        assert re.search(r'<label[^>]+for="reference-input"', html), (
            "要用 label 觸發原生檔案選擇，與「選擇照片」同一個理由"
        )

    def test_they_are_sent_with_the_apply_request(self, js: str) -> None:
        assert "references: references.map" in js

    def test_the_limit_comes_from_the_model(self, js: str) -> None:
        """送出的圖已經有一張是紅色標示，所以要扣一張。"""
        assert "function referenceLimit()" in js
        assert "max_references" in js

    def test_they_are_shrunk_in_the_browser_first(self, js: str) -> None:
        """原檔可能十幾 MB，直接 base64 上傳既慢又沒有意義。"""
        assert "REFERENCE_MAX_PX" in js
        assert "function shrinkToDataURL" in js

    def test_local_removal_refuses_references(self, js: str) -> None:
        assert "參考圖只有雲端模型用得到" in js


class TestExportFormat:
    def test_there_is_a_format_choice(self, html: str) -> None:
        assert re.search(r'id="export-format"', html)
        assert "webp" in html and "png" in html

    def test_the_choice_is_sent_to_the_server(self, js: str) -> None:
        assert "/api/result?format=" in js

    def test_the_difference_is_explained(self, html: str, js: str) -> None:
        """不要讓「檔案小」與「不失真」看起來只是兩個隨機選項。"""
        assert "完全不失真" in html, "選項的文字本身就要講清楚"
        assert '$("export-hint").textContent' in js, "換格式時也要跟著說明改變"


class TestOfflinePromise:
    def test_no_external_resources(self, html: str) -> None:
        """完全離線可用——沒有網絡時這個工具仍然要能用。"""
        for pattern in ("http://", "https://", "unpkg", "jsdelivr", "cdn."):
            assert pattern not in html, f"頁面引用了外部資源：{pattern}"

    def test_konva_is_loaded_locally(self, html: str) -> None:
        assert 'src="/static/konva.min.js"' in html
