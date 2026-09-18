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


class TestOfflinePromise:
    def test_no_external_resources(self, html: str) -> None:
        """完全離線可用——沒有網絡時這個工具仍然要能用。"""
        for pattern in ("http://", "https://", "unpkg", "jsdelivr", "cdn."):
            assert pattern not in html, f"頁面引用了外部資源：{pattern}"

    def test_konva_is_loaded_locally(self, html: str) -> None:
        assert 'src="/static/konva.min.js"' in html
