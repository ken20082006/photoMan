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
        assert 'stage.on("mousemove touchmove", extendStroke)' in js
        assert 'stage.on("mouseup touchend", endStroke)' in js
        assert "/api/mask/stroke" in js and "/api/mask/rect" in js

    def test_releasing_outside_the_canvas_ends_the_stroke(self, js: str) -> None:
        """在照片外面放手也要收筆，否則那一筆會卡住，之後滑鼠一動
        就會由舊的起點長出一條長線。"""
        assert 'window.addEventListener("mouseup", endStroke)' in js

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
        view = re.search(r"async function setView\(next\) \{(.*?)\n\}", js, re.S).group(1)

        assert "setOverlay(maskOverlayURL)" in view


class TestOfflinePromise:
    def test_no_external_resources(self, html: str) -> None:
        """完全離線可用——沒有網絡時這個工具仍然要能用。"""
        for pattern in ("http://", "https://", "unpkg", "jsdelivr", "cdn."):
            assert pattern not in html, f"頁面引用了外部資源：{pattern}"

    def test_konva_is_loaded_locally(self, html: str) -> None:
        assert 'src="/static/konva.min.js"' in html
