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
    ) -> ProjectStore:
        """由一張原圖建立新專案。"""
        directory = Path(directory)
        source_path = Path(source_path).resolve()
        loaded = load_srgb(source_path)

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
    def open(cls, directory: str | Path) -> ProjectStore:
        """開啟既有專案，並核對原檔。"""
        directory = Path(directory)
        path = directory / PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(f"找不到專案檔：{path}")

        project = Project.model_validate_json(path.read_text(encoding="utf-8"))
        store = cls(directory, project)
        store.verify_source()
        return store

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
