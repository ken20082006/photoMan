"""專案目錄的讀寫（見 docs/design.md §6）。

目錄結構：

```
project/
  project.json          主索引
  masks/L_xxxxxx.png    每個生成層的遮罩（原圖座標）
  cache/L_xxxxxx.png    生成結果的快取
```

**遮罩以像素資料的雜湊為身分，不是以 PNG 位元組。** PNG 編碼會隨
編碼器版本改變，用它做快取鍵會令同一張遮罩在不同版本之間無故失效。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from photoman.image import SourceInfo, load_srgb
from photoman.layers import EditLayer
from photoman.layers import render as render_layers
from photoman.project import (
    SCHEMA_VERSION,
    Checksum,
    GenerativeLayer,
    Layer,
    Project,
    SourceRef,
)

PROJECT_FILE = "project.json"
MASK_DIR = "masks"
CACHE_DIR = "cache"


def mask_digest(mask: np.ndarray) -> str:
    """遮罩的身分——以**像素資料**計算，不是以編碼後的位元組。

    先正規化成二值 uint8，這樣同一個遮罩不論用甚麼方式產生
    （布林、0/1、0/255）都會得到同一個雜湊。
    """
    normalized = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()


class SourceChangedError(RuntimeError):
    """原檔與專案記錄不符。

    寧可明確報錯，也不要默默用一張不同的圖繼續跑——
    那會令之前所有的編輯都套用在錯誤的內容上，而且沒有跡象。
    """


class ProjectStore:
    """管理一個專案目錄。"""

    def __init__(self, directory: Path, project: Project) -> None:
        self.directory = Path(directory)
        self.project = project

    # ── 建立與開啟 ──────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        directory: str | Path,
        source_path: str | Path,
        *,
        name: str | None = None,
        loaded: SourceInfo | None = None,
    ) -> ProjectStore:
        """由一張原圖建立新專案。

        ``loaded`` 已經解碼好的原圖可以直接交進來。介面在開啟圖片時
        本來就會解碼一次，42 MP 的圖再解一次要多等好幾秒。
        """
        directory = Path(directory)
        source_path = Path(source_path).resolve()
        loaded = loaded if loaded is not None else load_srgb(source_path)

        project = Project(
            schema_version=SCHEMA_VERSION,
            name=name or source_path.stem,
            source=_to_ref(loaded.info),
        )
        store = cls(directory, project)
        store._ensure_dirs()
        store.save()
        return store

    @classmethod
    def open(cls, directory: str | Path, *, verify: bool = True) -> ProjectStore:
        """開啟既有專案，並核對原檔。

        ``verify=False`` 給「原檔可能已經換了位置」的情況用：那時呼叫方
        已經知道內容相符（專案目錄就是以內容雜湊命名的），要先讀出記錄
        才能把路徑改過去。其餘情況一律核對——默默用一張不同的圖繼續跑
        比報錯糟得多。
        """
        directory = Path(directory)
        path = directory / PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(f"找不到專案檔：{path}")

        project = Project.model_validate_json(path.read_text(encoding="utf-8"))
        store = cls(directory, project)
        if verify:
            store.verify_source()
        return store

    def repoint_source(self, path: str | Path) -> None:
        """把來源引用改到新的路徑——**內容必須相同，否則就是錯的**。

        典型情況：原檔被搬走，或上傳的暫存檔被清掉之後使用者重新上傳
        同一張照片。呼叫方要先確認內容雜湊相符。

        ``SourceRef`` 是 frozen，所以整塊換掉而不是改欄位。
        """
        self.project.source = self.project.source.model_copy(
            update={"path": str(Path(path).resolve())}
        )

    def verify_source(self) -> None:
        """核對原檔仍在，而且內容未變。"""
        source = Path(self.project.source.path)
        if not source.exists():
            raise SourceChangedError(
                f"原檔不見了：{source}\n"
                "專案只記路徑而不複製檔案，所以原檔被移動或刪除之後就無法繼續。"
            )
        actual = _sha256(source)
        if actual != self.project.source.sha256:
            raise SourceChangedError(
                f"原檔的內容與專案記錄不符：{source}\n"
                f"  記錄：{self.project.source.sha256[:16]}…\n"
                f"  實際：{actual[:16]}…\n"
                "拒絕繼續，以免把編輯套用在錯誤的內容上。"
            )

    # ── 儲存 ────────────────────────────────────────────────────

    def save(self) -> None:
        """寫出 project.json。

        **先寫暫存檔再改名。** 直接覆寫的話，若在寫入途中當機，
        專案檔會變成半截的 JSON——而那是使用者的全部工作。
        改名在大多數檔案系統上是原子操作。
        """
        self._ensure_dirs()
        target = self.directory / PROJECT_FILE
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            self.project.model_dump_json(indent=2, exclude_none=False),
            encoding="utf-8",
        )
        temporary.replace(target)

    def _ensure_dirs(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / MASK_DIR).mkdir(exist_ok=True)
        (self.directory / CACHE_DIR).mkdir(exist_ok=True)

    # ── 遮罩與結果 ──────────────────────────────────────────────

    def write_mask(self, layer_id: str, mask: np.ndarray) -> str:
        """寫出遮罩，回傳它的身分（像素雜湊）。"""
        self._ensure_dirs()
        normalized = (np.asarray(mask) > 0).astype(np.uint8) * 255
        Image.fromarray(normalized, mode="L").save(self.directory / MASK_DIR / f"{layer_id}.png")
        return mask_digest(normalized)

    def read_mask(self, layer_id: str) -> np.ndarray:
        path = self.directory / MASK_DIR / f"{layer_id}.png"
        if not path.exists():
            raise FileNotFoundError(f"找不到遮罩：{path}")
        return np.asarray(Image.open(path)) > 0

    def write_result(self, layer_id: str, patch: np.ndarray) -> str:
        """寫出生成結果，回傳相對於專案目錄的路徑。"""
        self._ensure_dirs()
        relative = f"{CACHE_DIR}/{layer_id}.png"
        Image.fromarray(patch).save(self.directory / relative)
        return relative

    def read_result(self, layer_id: str) -> np.ndarray:
        layer = self.project.layer_by_id(layer_id)
        if layer is None or not isinstance(layer, GenerativeLayer) or not layer.result_file:
            raise FileNotFoundError(f"圖層 {layer_id} 沒有快取的結果")
        return np.asarray(Image.open(self.directory / layer.result_file))

    # ── 圖層操作 ────────────────────────────────────────────────

    def add_layer(self, layer: Layer) -> Layer:
        self.project.layers.append(layer)
        return layer

    def remove_last_layer(self) -> GenerativeLayer | None:
        """移除最後一層，回傳它（供復原堆疊使用）；沒有生成層就回傳 ``None``。

        **遮罩與貼片的檔案刻意不刪。** 重做要用它們，而它們只有零點幾 MB。
        孤兒檔案的下場是佔一點磁碟，不是拿到錯的結果——這個交換是值得的。
        """
        for index in range(len(self.project.layers) - 1, -1, -1):
            layer = self.project.layers[index]
            if isinstance(layer, GenerativeLayer):
                del self.project.layers[index]
                return layer
        return None

    def apply_result(
        self,
        layer_id: str,
        *,
        result_file: str,
        checksum: Checksum,
        cache_key: str,
    ) -> bool:
        """把一次執行的結果寫回圖層。**``locked`` 的層不會被覆蓋。**

        回傳是否真的寫入了。被拒絕是正常情況而不是錯誤——
        呼叫方應該據此提示使用者「這一層有你改過的東西，沒有覆蓋」，
        而不是當成失敗。
        """
        layer = self.project.layer_by_id(layer_id)
        if layer is None:
            raise KeyError(f"沒有這個圖層：{layer_id}")
        if layer.locked:
            return False
        if not isinstance(layer, GenerativeLayer):
            raise TypeError(f"圖層 {layer_id} 不是生成層，沒有結果可以寫回")

        layer.result_file = result_file
        layer.checksum = checksum
        layer.cache_key = cache_key
        return True

    # ── 重播 ────────────────────────────────────────────────────

    def render(self, source: np.ndarray | None = None) -> np.ndarray:
        """由磁碟上的遮罩與貼片重播整個專案，回傳 uint8 sRGB。

        **這是重開專案的入口**：它只需要專案目錄裡的東西，
        不必知道當初是怎麼編輯的、更不必重跑任何模型。
        """
        if source is None:
            source = load_srgb(self.project.source.path).srgb

        edits = []
        for layer in self.project.enabled_layers():
            if not isinstance(layer, GenerativeLayer):
                raise NotImplementedError(f"圖層 {layer.id} 的類型還沒有執行引擎，這個專案無法重播")
            edits.append(
                EditLayer.from_edit(layer, self.read_mask(layer.id), self.read_result(layer.id))
            )
        return render_layers(source, edits)

    def cached_result(self, layer_id: str) -> np.ndarray | None:
        """若快取鍵仍然相符，回傳快取的結果；否則回傳 ``None``。

        **這是省錢的地方**：參數沒變就不應該重新呼叫模型。
        """
        layer = self.project.layer_by_id(layer_id)
        if not isinstance(layer, GenerativeLayer) or not layer.result_file:
            return None
        if layer.cache_key != self.project.cache_keys().get(layer_id):
            return None
        return self.read_result(layer_id)


def _to_ref(info: SourceInfo) -> SourceRef:
    return SourceRef(
        path=str(info.path),
        sha256=info.sha256,
        format=info.format,
        width=info.width,
        height=info.height,
        bit_depth=info.bit_depth,
        icc_description=info.icc_description,
        had_orientation_tag=info.had_orientation_tag,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    """讀出原始 JSON——給檢查工具用，正規路徑是 :meth:`ProjectStore.open`。"""
    return json.loads(path.read_text(encoding="utf-8"))
