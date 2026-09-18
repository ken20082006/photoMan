"""本地介面的伺服器（見 docs/design.md §7.2）。

**完全離線。** Konva 內嵌在 `static/`，沒有任何 CDN、沒有建置步驟。
唯一的對外連線是使用者自己選的 API 模型。

**座標約定：瀏覽器一律送「預覽圖的像素座標」，伺服器負責換算成原圖座標。**
這樣瀏覽器不需要知道原圖有多大（42 MP 的圖送給瀏覽器沒有意義），
而換算只有一個地方要做，不容易寫錯。
"""

from __future__ import annotations

import base64
import contextlib
import io
import tempfile
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from photoman import config
from photoman.edit import apply_generative_edit
from photoman.image import load_srgb
from photoman.paths import LAMA_MODEL_FILE, MODELS_DIR, SAM_DECODER_FILE, SAM_ENCODER_FILE
from photoman.segment import get_segmenter

STATIC = Path(__file__).parent / "static"

# 預覽圖的最長邊。原圖可能是 42 MP，直接送給瀏覽器既慢又沒有意義——
# 螢幕上根本看不出分別，而資料量差幾十倍。
PREVIEW_MAX = 1600


def _png_data_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _mask_overlay(mask: np.ndarray, colour=(255, 60, 60), alpha=110) -> np.ndarray:
    """把遮罩畫成 RGBA 疊圖——由瀏覽器疊在原圖上。"""
    height, width = mask.shape[:2]
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask] = (*colour, alpha)
    return overlay


@dataclass
class Session:
    """目前的編輯狀態。單一使用者、一次一張圖。"""

    source_path: Path | None = None
    base: np.ndarray | None = None  # uint8 sRGB，原解析度
    preview: np.ndarray | None = None  # 給瀏覽器看的縮圖
    scale: float = 1.0  # 預覽 / 原圖
    mask: np.ndarray | None = None  # bool，**原圖座標**
    result: np.ndarray | None = None  # uint8 sRGB，原解析度
    checksum: float | None = None
    method: str = "lama"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def require_image(self) -> None:
        if self.base is None:
            raise HTTPException(status_code=400, detail="尚未開啟圖片")

    def to_full(self, x: float, y: float) -> tuple[int, int]:
        """預覽座標 → 原圖座標。"""
        return (int(round(x / self.scale)), int(round(y / self.scale)))

    def snapshot(self) -> dict:
        """目前的狀態——介面用它決定顯示甚麼。"""
        return {
            "opened": self.base is not None,
            "filename": self.source_path.name if self.source_path else None,
            "width": int(self.base.shape[1]) if self.base is not None else 0,
            "height": int(self.base.shape[0]) if self.base is not None else 0,
            "preview_width": int(self.preview.shape[1]) if self.preview is not None else 0,
            "preview_height": int(self.preview.shape[0]) if self.preview is not None else 0,
            "has_mask": bool(self.mask is not None and self.mask.any()),
            "mask_pixels": int(self.mask.sum()) if self.mask is not None else 0,
            "has_result": self.result is not None,
            "checksum": self.checksum,
            "method": self.method,
            "models_ready": _models_ready(),
        }


def _models_ready() -> dict:
    """模型檔案在不在。介面用它決定要不要提示使用者下載。"""
    return {
        "lama": (MODELS_DIR / LAMA_MODEL_FILE).exists(),
        "sam": (MODELS_DIR / SAM_ENCODER_FILE).exists()
        and (MODELS_DIR / SAM_DECODER_FILE).exists(),
    }


SESSION = Session()
_SEGMENTER = None


def _segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = get_segmenter()
    return _SEGMENTER


app = FastAPI(title="photoMan")


# ── 頁面 ────────────────────────────────────────────────────────


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/static/{name}")
def static_file(name: str) -> FileResponse:
    path = STATIC / name
    if not path.exists() or path.parent != STATIC:
        raise HTTPException(status_code=404, detail="找不到檔案")
    return FileResponse(path)


# ── 狀態 ────────────────────────────────────────────────────────


@app.get("/api/state")
def state() -> dict:
    return SESSION.snapshot()


@app.get("/api/preview")
def preview() -> dict:
    SESSION.require_image()
    return {"image": _png_data_url(SESSION.preview)}


@app.get("/api/settings")
def get_settings() -> dict:
    return config.masked()


class SettingsPatch(BaseModel):
    api_key: str


@app.post("/api/settings")
def post_settings(patch: SettingsPatch) -> dict:
    config.update_api_key(patch.api_key)
    return config.masked()


# ── 開啟圖片 ────────────────────────────────────────────────────


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    """由瀏覽器上傳一張圖。

    存到 `%APPDATA%\\photoMan\\work\\` 而不是 repo 之內——
    使用者的照片有版權，不應該有機會被 `git add -A` 掃進去。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="沒有檔案")

    work = config.work_dir()
    work.mkdir(parents=True, exist_ok=True)
    # 保留原本的檔名（使用者認得），但用暫存目錄避免覆蓋同名檔案。
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(dir=work, suffix=suffix, delete=False) as handle:
        handle.write(await file.read())
        target = Path(handle.name)

    return _open(target, display_name=file.filename)


@app.post("/api/open")
def open_path(payload: dict) -> dict:
    """由路徑開啟（本機使用，方便反覆測試同一張圖）。"""
    path = Path(payload.get("path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"找不到檔案：{path}")
    return _open(path, display_name=path.name)


def _open(path: Path, *, display_name: str) -> dict:
    with SESSION.lock:
        loaded = load_srgb(path)
        base = loaded.srgb
        height, width = base.shape[:2]

        scale = min(1.0, PREVIEW_MAX / max(height, width))
        if scale < 1.0:
            preview = np.asarray(
                Image.fromarray(base).resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.LANCZOS,
                )
            )
        else:
            preview = base

        SESSION.source_path = path
        SESSION.base = base
        SESSION.preview = preview
        SESSION.scale = scale
        SESSION.mask = np.zeros((height, width), dtype=bool)
        SESSION.result = None
        SESSION.checksum = None

    # 預先算好嵌入，令第一次點擊就有反應（編碼是慢的那一半）。
    # 沒有 SAM 模型時仍然可以看圖，只是不能點擊選取——所以這裡刻意吞掉錯誤。
    with contextlib.suppress(ImportError, FileNotFoundError):
        _segmenter().embed(base)

    snapshot = SESSION.snapshot()
    snapshot["display_name"] = display_name
    return snapshot


# ── 選取 ────────────────────────────────────────────────────────


class SelectRequest(BaseModel):
    x: float
    y: float
    mode: str = "add"  # add | subtract


@app.post("/api/select")
def select(request: SelectRequest) -> dict:
    SESSION.require_image()
    with SESSION.lock:
        x, y = SESSION.to_full(request.x, request.y)
        height, width = SESSION.base.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise HTTPException(status_code=400, detail="點擊位置在圖片之外")

        try:
            region = _segmenter().segment(SESSION.base, point=(x, y))
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.mode == "subtract":
            SESSION.mask &= ~region
        else:
            SESSION.mask |= region
        SESSION.result = None
        SESSION.checksum = None

        return _mask_payload()


class BoxRequest(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    mode: str = "add"


@app.post("/api/select_box")
def select_box(request: BoxRequest) -> dict:
    """拉一個框選取框內的東西——比逐點擊更適合選一整片。"""
    SESSION.require_image()
    with SESSION.lock:
        x0, y0 = SESSION.to_full(request.x0, request.y0)
        x1, y1 = SESSION.to_full(request.x1, request.y1)
        # 使用者可以由任何方向拉框，所以要先排序。
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        try:
            region = _segmenter().segment(SESSION.base, box=(x0, y0, x1, y1))
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.mode == "subtract":
            SESSION.mask &= ~region
        else:
            SESSION.mask |= region
        SESSION.result = None
        SESSION.checksum = None
        return _mask_payload()


@app.post("/api/clear_mask")
def clear_mask() -> dict:
    SESSION.require_image()
    with SESSION.lock:
        SESSION.mask[:] = False
        SESSION.result = None
        SESSION.checksum = None
        return _mask_payload()


def _mask_payload() -> dict:
    overlay = _mask_overlay(SESSION.mask)
    preview_overlay = np.asarray(
        Image.fromarray(overlay).resize(
            (SESSION.preview.shape[1], SESSION.preview.shape[0]), Image.NEAREST
        )
    )
    return {
        "overlay": _png_data_url(preview_overlay),
        "mask_pixels": int(SESSION.mask.sum()),
        "has_mask": bool(SESSION.mask.any()),
    }


# ── 執行編輯 ────────────────────────────────────────────────────


class ApplyRequest(BaseModel):
    method: str = "lama"
    dilate_px: int = 16
    feather_px: int = 12


@app.post("/api/apply")
def apply(request: ApplyRequest) -> dict:
    SESSION.require_image()
    with SESSION.lock:
        if not SESSION.mask.any():
            raise HTTPException(status_code=400, detail="還沒有選取任何東西")
        try:
            result = apply_generative_edit(
                SESSION.base,
                SESSION.mask,
                method=request.method,
                dilate_px=request.dilate_px,
                feather_px=request.feather_px,
            )
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        SESSION.result = result.image
        SESSION.checksum = float(result.checksum)
        SESSION.method = request.method

        preview = np.asarray(
            Image.fromarray(SESSION.result).resize(
                (SESSION.preview.shape[1], SESSION.preview.shape[0]), Image.LANCZOS
            )
        )
        return {
            "image": _png_data_url(preview),
            "checksum": SESSION.checksum,
            "seconds": None,
        }


@app.get("/api/result")
def download_result() -> Response:
    """匯出成品——**原解析度**，不是預覽。

    用 ``Response`` 而不是 ``FileResponse``：後者要的是檔案路徑，
    而我們在記憶體裡已經有現成的位元組，沒有理由先寫到磁碟再讀回來。
    """
    SESSION.require_image()
    if SESSION.result is None:
        raise HTTPException(status_code=400, detail="還沒有結果可以匯出")
    buffer = io.BytesIO()
    Image.fromarray(SESSION.result).save(buffer, format="PNG")
    name = (SESSION.source_path.stem if SESSION.source_path else "result") + "-photoman.png"
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def run(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    run()
