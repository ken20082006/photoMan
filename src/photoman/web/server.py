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
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from photoman import config
from photoman.edit import API_THRESHOLD, EditResult, apply_generative_edit
from photoman.image import SourceInfo, load_srgb
from photoman.layers import EditLayer
from photoman.layers import render as render_layers
from photoman.paths import LAMA_MODEL_FILE, MODELS_DIR, SAM_DECODER_FILE, SAM_ENCODER_FILE
from photoman.project import Checksum, GenerativeLayer
from photoman.segment import get_segmenter
from photoman.store import PROJECT_FILE, ProjectStore, SourceChangedError

STATIC = Path(__file__).parent / "static"

# 預覽圖的最長邊。原圖可能是 42 MP，直接送給瀏覽器既慢又沒有意義——
# 螢幕上根本看不出分別，而資料量差幾十倍。
PREVIEW_MAX = 1600

# 放大時那一塊最多回傳的邊長。它只在放大時用，而放大時可見的範圍很小，
# 所以這個上限幾乎不會碰到——留著是為了擋住畸形的請求。
DETAIL_MAX = 2400

# 匯出的品質。實測（3213×5712 ＝ 18.4 MP 的真實照片，原檔 5.05 MB）：
#
#   PNG          17.32 MB   逐位元相同
#   WebP 無損    11.85 MB   逐位元相同——但慢十倍，而且只小 32%，不值得
#   WebP q95      3.49 MB   平均差 0.95 級
#
# PNG 比原檔的 JPEG 大 3.4 倍，而 WebP q95 比原檔還小。
# 所以預設用 WebP q95，PNG 留給要完全不失真的人。
WEBP_QUALITY = 95

# 0–6，愈大愈慢愈小。4 是實測的平衡點：18 MP 約 3 秒。
WEBP_METHOD = 4

# 使用者一次最多可以附幾張參考圖。真正的上限由模型決定
# （見 providers.ImageModel.max_references），介面會照那個數字擋；
# 這裡是伺服器端的保險。
MAX_REFERENCES = 4


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


def _downscale(image: np.ndarray) -> np.ndarray:
    """縮成預覽大小。原圖可能是 42 MP，直接送給瀏覽器既慢又沒有意義。

    ★ **預覽只給眼睛看，永遠不可以用來合成。** 它經過 LANCZOS 重新取樣，
    已經不是原檔的位元組了。
    """
    height, width = image.shape[:2]
    scale = min(1.0, PREVIEW_MAX / max(height, width))
    if scale >= 1.0:
        return image
    return np.asarray(
        Image.fromarray(image).resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
        )
    )


@dataclass
class Session:
    """目前的編輯狀態。單一使用者、一次一張圖。

    **``source`` 永遠不變，``working`` 才是編輯的起點。**
    這是整個多步編輯的核心：每一次執行都套用在 ``working`` 上，
    而 ``working`` 是 ``render(source, layers)`` 的結果。
    在這之前，每一次執行都套用在最原始的那張圖上——所以第二次執行
    會把第一次的成果整塊重新生成（實測 12000 個像素裡有 11997 個被改掉）。
    """

    source_path: Path | None = None
    source: np.ndarray | None = None  # uint8 sRGB，原解析度，**永不改動**
    working: np.ndarray | None = None  # uint8 sRGB，= render(source, layers)
    source_preview: np.ndarray | None = None
    preview: np.ndarray | None = None  # **working** 的縮圖
    scale: float = 1.0  # 預覽 / 原圖
    mask: np.ndarray | None = None  # bool，**原圖座標**
    layers: list[EditLayer] = field(default_factory=list)  # 已接受的編輯，順序即堆疊
    redo: list[EditLayer] = field(default_factory=list)  # 復原堆疊
    store: ProjectStore | None = None  # 掛上的專案；掛不上時為 None（仍可編輯）
    project_note: str | None = None
    revision: int = 0  # 每次重算 working 就加一——SAM 的嵌入快取靠它失效
    lock: threading.Lock = field(default_factory=threading.Lock)

    def require_image(self) -> None:
        if self.source is None:
            raise HTTPException(status_code=400, detail="尚未開啟圖片")

    def to_full(self, x: float, y: float) -> tuple[int, int]:
        """預覽座標 → 原圖座標。"""
        return (int(round(x / self.scale)), int(round(y / self.scale)))

    def recompose(self) -> None:
        """由原檔重播所有圖層，重算 working 與預覽。

        **每一次圖層有變動都整個重算**，不做增量。重算是複製加幾次貼上，
        而復原本來就需要它；這樣「working 永遠等於 render(source, layers)」
        是一個不變式，不是一個要小心維護的但書。
        """
        self.working = render_layers(self.source, self.layers)
        self.preview = _downscale(self.working)
        self.revision += 1

    def snapshot(self) -> dict:
        """目前的狀態——介面用它決定顯示甚麼。"""
        return {
            "opened": self.source is not None,
            "filename": self.source_path.name if self.source_path else None,
            "width": int(self.source.shape[1]) if self.source is not None else 0,
            "height": int(self.source.shape[0]) if self.source is not None else 0,
            "preview_width": int(self.preview.shape[1]) if self.preview is not None else 0,
            "preview_height": int(self.preview.shape[0]) if self.preview is not None else 0,
            "has_mask": bool(self.mask is not None and self.mask.any()),
            "mask_pixels": int(self.mask.sum()) if self.mask is not None else 0,
            "has_result": bool(self.layers),
            "layers": [
                {
                    "id": edit.layer.id,
                    "method": edit.layer.method,
                    "prompt": edit.layer.prompt,
                    "crop": list(edit.layer.crop),
                }
                for edit in self.layers
            ],
            "can_undo": bool(self.layers),
            "can_redo": bool(self.redo),
            "project_saved": self.store is not None,
            "project_note": self.project_note,
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
def preview(source: bool = False) -> dict:
    """預覽圖。預設是**目前的成品**（所有已接受的編輯）；``?source=true`` 是原圖。"""
    SESSION.require_image()
    return {"image": _png_data_url(SESSION.source_preview if source else SESSION.preview)}


@app.get("/api/detail")
def detail(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    w: int,
    h: int,
    source: bool = False,
) -> dict:
    """**放大時看的那一塊**——由原檔重新取樣，不是預覽被放大。

    為甚麼需要它：預覽的最長邊是 1600，所以 3213×5712 的圖在畫布上
    只有原來的 28%。放大超過那個密度之後，看到的其實是縮圖被拉大——
    實測高頻細節只剩一半（相鄰像素差的標準差 9.7 → 5.5）。
    對一個修圖工具來說這是致命的：**你沒辦法判斷自己修的品質。**

    ★ **回傳的是「螢幕要多少像素就給多少」，不是原解析度那一塊。**
    送原解析度的話，放大 1.2 倍時就要傳 8 MP；而螢幕根本顯示不了那麼多。
    由原檔裁出來、縮到瀏覽器要的尺寸，資料量就等於視窗大小，
    而畫質是原檔的畫質。

    ``w``/``h`` 是瀏覽器要的輸出尺寸（＝螢幕像素）。
    """
    SESSION.require_image()
    height, width = SESSION.source.shape[:2]

    # 使用者可以把畫面拖到照片外，所以範圍要夾回圖內。
    # ★ 要**先排序再夾**，不是邊排序邊夾：整塊都在圖外的請求
    # （x0=5000, x1=6000）在後者的寫法下會變成 (width, 5000)，
    # 兩個數字相差很遠，於是通過了「範圍太小」的檢查，然後切出一塊空的。
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    left = max(0.0, min(float(width), left))
    right = max(0.0, min(float(width), right))
    top = max(0.0, min(float(height), top))
    bottom = max(0.0, min(float(height), bottom))
    if right - left < 1 or bottom - top < 1:
        raise HTTPException(status_code=400, detail="這一塊在照片之外")

    left, top = int(left), int(top)
    right, bottom = max(left + 1, int(round(right))), max(top + 1, int(round(bottom)))
    target = (
        max(1, min(DETAIL_MAX, w)),
        max(1, min(DETAIL_MAX, h)),
    )

    image = SESSION.source if source else SESSION.working
    region = image[top:bottom, left:right]
    mask_region = SESSION.mask[top:bottom, left:right]

    # 縮小用 BOX（區域平均，幼細的筆畫才不會斷成虛線）、
    # 放大用 BILINEAR（最近鄰會令遮罩邊緣成階梯）。
    shrinking = target[0] < region.shape[1]
    mask_filter = Image.BOX if shrinking else Image.BILINEAR
    image_filter = Image.LANCZOS if shrinking else Image.BICUBIC

    drawn = np.asarray(Image.fromarray(region).resize(target, image_filter))
    small = Image.fromarray((mask_region.astype(np.uint8) * 255), mode="L").resize(
        target, mask_filter
    )
    overlay = _mask_overlay(np.asarray(small) > 127)

    return {
        "image": _png_data_url(drawn),
        "overlay": _png_data_url(overlay),
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
    }


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
        source = loaded.srgb
        height, width = source.shape[:2]

        SESSION.source_path = path
        SESSION.source = source
        SESSION.layers = []
        SESSION.redo = []
        SESSION.mask = np.zeros((height, width), dtype=bool)
        SESSION.store = None
        SESSION.project_note = None
        SESSION.scale = min(1.0, PREVIEW_MAX / max(height, width))
        SESSION.source_preview = _downscale(source)

        _attach_project(SESSION, path, loaded)
        SESSION.recompose()

    # 預先算好嵌入，令第一次點擊就有反應（編碼是慢的那一半）。
    # 沒有 SAM 模型時仍然可以看圖，只是不能點擊選取——所以這裡刻意吞掉錯誤。
    with contextlib.suppress(ImportError, FileNotFoundError):
        _segmenter().embed(SESSION.working)

    snapshot = SESSION.snapshot()
    snapshot["display_name"] = display_name
    return snapshot


# ── 專案 ────────────────────────────────────────────────────────
#
# 專案**依原檔的內容雜湊命名**，所以「同一個目錄」就等於「同一張照片」：
# 照片被搬到別的位置、或者重新上傳同一張，都會接回同一個專案，
# 使用者不必做任何選擇。
#
# 掛不上就降級成純記憶體——編輯、復原、匯出全部照常，只是關掉瀏覽器
# 就沒了。**這件事必須讓使用者看見**（見 project_note），
# 默默不存檔是這裡唯一不能接受的失敗方式。


def _attach_project(session: Session, path: Path, loaded: SourceInfo) -> None:
    directory = config.projects_dir() / loaded.info.sha256
    try:
        if (directory / PROJECT_FILE).exists():
            # 不核對來源：目錄名就是這次開啟的內容雜湊，兩者相符就是同一張照片。
            # 核對的是**記錄裡的那條路徑**，而它可能已經過期（照片被搬走、
            # 或上傳的暫存檔被清掉）——那正是下面要修的情況。
            store = ProjectStore.open(directory, verify=False)
            if Path(store.project.source.path) != path:
                store.repoint_source(path)
                store.save()
        else:
            store = ProjectStore.create(directory, path, name=path.stem, loaded=loaded)
    except (OSError, SourceChangedError, ValueError) as exc:
        session.store = None
        session.project_note = f"這次的修改不會被保存：{exc}"
        return

    session.store = store
    session.layers, session.project_note = _load_layers(store)


def _load_layers(store: ProjectStore) -> tuple[list[EditLayer], str | None]:
    """由專案讀回圖層。讀到壞掉的那一層就停在這裡，保留前面能重播的部分。

    這是刻意的：一個壞掉的貼片不應該令整份工作變成不可開啟。
    """
    edits: list[EditLayer] = []
    for layer in store.project.enabled_layers():
        if not isinstance(layer, GenerativeLayer):
            break
        try:
            edits.append(
                EditLayer.from_edit(layer, store.read_mask(layer.id), store.read_result(layer.id))
            )
        except (OSError, ValueError) as exc:
            return edits, f"第 {len(edits) + 1} 層的材料不完整，只還原到前一層：{exc}"
    return edits, None


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
        height, width = SESSION.source.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise HTTPException(status_code=400, detail="點擊位置在圖片之外")

        try:
            region = _segment(SESSION, point=(x, y))
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return _apply_region(region, request.mode)


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
            region = _segment(SESSION, box=(x0, y0, x1, y1))
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return _apply_region(region, request.mode)


def _segment(session: Session, **hints) -> np.ndarray:
    """在**目前的成品**上做分割，不是在原檔上。

    這一點很重要：編輯過之後，使用者看到的是 working。若拿原檔去分割，
    「自動選取」會選到一個畫面上已經不存在的東西，而且不會有任何錯誤訊息。

    ``revision`` 也要帶進去——嵌入快取的鍵只看縮圖的粗略取樣，
    一個小物件的移除不會改變那些取樣點，於是會拿到編輯前的嵌入。
    """
    return _segmenter().segment(session.working, token=session.revision, **hints)


@app.post("/api/clear_mask")
def clear_mask() -> dict:
    SESSION.require_image()
    with SESSION.lock:
        SESSION.mask[:] = False
        return _mask_payload()


def _apply_region(region: np.ndarray, mode: str) -> dict:
    """把一塊區域併進（或由遮罩移出）目前的選取。"""
    if mode == "subtract":
        SESSION.mask &= ~region
    else:
        SESSION.mask |= region
    return _mask_payload()


def _downscale_mask(mask: np.ndarray) -> np.ndarray:
    """把原圖座標的遮罩縮到預覽解析度。

    用 BOX（區域平均）而不是 NEAREST：1 像素寬的筆畫在縮圖上只有
    零點幾像素，NEAREST 會令它整條消失或斷成虛線，而使用者正在畫的
    那一筆會看起來像「沒有反應」。
    """
    height, width = SESSION.preview.shape[:2]
    if mask.shape[:2] == (height, width):
        return mask
    small = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
        (width, height), Image.BOX
    )
    return np.asarray(small) > 0


def _mask_payload() -> dict:
    # 疊圖只在**預覽解析度**上色。42 MP 的圖若先造全解析度 RGBA 再縮小，
    # 每一筆圈選都要配置近 200 MB，而畫面上看到的結果完全一樣。
    return {
        "overlay": _png_data_url(_mask_overlay(_downscale_mask(SESSION.mask))),
        "mask_pixels": int(np.count_nonzero(SESSION.mask)),
        "has_mask": bool(SESSION.mask.any()),
    }


# ── 手動圈選 ────────────────────────────────────────────────────
#
# 使用者自己畫的範圍**就是遮罩本身**，不經過任何模型判斷。
#
# 這是刻意的：SAM 回答的是「這一團像素是甚麼」，那在物件邊界清楚時很好用；
# 但當使用者心裡已經有一條明確的界線（或者要改的東西根本不是一個物件，
# 例如一片天空、一條裂痕、一段文字），任何自動判斷都只是多一個出錯的地方。
# 空間問題要用空間通道解決——而滑鼠本身就是那個通道。


class RectRequest(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    mode: str = "add"  # add | subtract


@app.post("/api/mask/rect")
def mask_rect(request: RectRequest) -> dict:
    """使用者拉出的矩形直接變成遮罩——不問框裡是甚麼。"""
    SESSION.require_image()
    with SESSION.lock:
        x0, y0 = SESSION.to_full(request.x0, request.y0)
        x1, y1 = SESSION.to_full(request.x1, request.y1)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        height, width = SESSION.source.shape[:2]
        x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
        y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
        if x0 == x1 or y0 == y1:
            raise HTTPException(status_code=400, detail="框太小了——請拉出一個範圍")

        region = np.zeros((height, width), dtype=bool)
        region[y0:y1, x0:x1] = True
        return _apply_region(region, request.mode)


class StrokeRequest(BaseModel):
    # data URL。白色＝使用者塗到的地方，其他顏色一律當作沒塗到。
    image: str
    mode: str = "add"


@app.post("/api/mask/stroke")
def mask_stroke(request: StrokeRequest) -> dict:
    """把一筆筆刷軌跡併進遮罩。

    **由瀏覽器負責畫那筆畫，伺服器只負責換算。** 筆刷的取樣、
    壓感、筆畫重疊都發生在畫布上——在那里它們是免費的，
    送過來只是一張圖。
    """
    SESSION.require_image()
    with SESSION.lock:
        return _apply_region(_decode_mask_png(request.image), request.mode)


def _decode_data_url(data_url: str) -> Image.Image:
    """解開瀏覽器送來的 data URL。格式不對就明確報錯。"""
    payload = data_url.partition(",")[2]
    if not payload:
        raise HTTPException(status_code=400, detail="圖片的格式不對")
    try:
        return Image.open(io.BytesIO(base64.b64decode(payload)))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"圖片無法解讀：{exc}") from exc


def _decode_mask_png(data_url: str) -> np.ndarray:
    """把瀏覽器送來的遮罩圖換算成**原圖座標**的 bool 陣列。

    瀏覽器只知道預覽圖有多大（原圖可能是 42 MP），所以換算只在這裡做一次。

    放大用雙線性再以 0.5 為界，不用最近鄰：最近鄰會令筆畫邊緣在原圖上
    留下階梯，而那個階梯比羽化半徑粗的時候就看得出鋸齒。雙線性給的是
    半像素精度的邊界，正好是筆刷本來就有的不確定度。
    """
    small = np.asarray(_decode_data_url(data_url).convert("L"))

    height, width = SESSION.source.shape[:2]
    if small.shape[:2] != (height, width):
        small = np.asarray(Image.fromarray(small).resize((width, height), Image.BILINEAR))
    return small > 127


# ── 執行編輯 ────────────────────────────────────────────────────


class ApplyRequest(BaseModel):
    # 留空＝移除選中的東西（本機 LaMa）。
    # 有填＝交給雲端模型照指令改，例如「把她的動作改成揮手」「加一隻小狗」。
    prompt: str = ""
    model: str | None = None
    resolution: str | None = None
    method: str = "lama"
    dilate_px: int = 16
    feather_px: int = 12
    # 使用者附的參考圖（data URL）。只有雲端模型用得到——
    # 本機的 LaMa 沒有「參考圖」這個概念，它只會由邊界往內填。
    references: list[str] = []


@app.get("/api/models")
def list_models() -> dict:
    """可用的雲端模型，按價格排序。沒有 API key 時回傳空清單而不是報錯——
    本機功能不需要 key，不應該因為沒有 key 就不能用介面。"""
    from photoman.providers import ProviderError, client_from_config

    try:
        models = client_from_config().models()
    except ProviderError as exc:
        return {"models": [], "note": str(exc)}

    return {
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "price": model.price_per_image,
                "resolution": model.max_resolution,
                # 介面用它決定可以附幾張參考圖。送出的圖已經佔了一張
                # （標了紅色的選取範圍），所以可附的張數要再減一。
                "max_references": model.max_references,
            }
            for model in models
        ],
        "note": None,
    }


@app.post("/api/apply")
def apply(request: ApplyRequest) -> dict:
    SESSION.require_image()
    from photoman.providers import ProviderError

    with SESSION.lock:
        if not SESSION.mask.any():
            raise HTTPException(status_code=400, detail="還沒有選取任何東西")

        wants_instruction = bool(request.prompt.strip())
        if request.model:
            method = f"api:{request.model}"
        elif wants_instruction:
            raise HTTPException(status_code=400, detail="要下指令的話需要先選一個雲端模型。")
        else:
            # 本機路徑。介面固定用 LaMa（品質明顯較好），但引擎可以由
            # 呼叫方指定——測試用它換成 Telea，否則每一個端到端測試都要
            # 付一次 LaMa 的模型載入（實測 12 秒）。
            method = request.method or "lama"

        options: dict = {}
        if method.startswith("api:"):
            options = {
                "prompt": request.prompt,
                "resolution": request.resolution,
                "references": _decode_references(request.references),
            }
        elif request.references:
            raise HTTPException(
                status_code=400,
                detail="參考圖只有雲端模型用得到——本機移除是由邊界往內填，沒有「參考」這個概念。",
            )

        try:
            # ★ 由 **working** 起算，不是 source。這就是多步編輯的關鍵：
            # 之前每一次都套用在最原始的那張圖上，所以第二次執行會把
            # 第一次的成果整塊重新生成。
            result = apply_generative_edit(
                SESSION.working,
                SESSION.mask,
                method=method,
                dilate_px=request.dilate_px,
                feather_px=request.feather_px,
                # ★ 顏色對齊的意圖**按引擎走，而且是一條固定的產品規則**：
                #   本機＝填補（它只會由邊界往內填，內容本來就該像周圍）
                #   雲端＝編輯（它是用來「換成別的內容」的，只扣模型的漂移）
                #
                # 這不是「猜使用者的意圖」——先前試過讓使用者選，但那個選擇
                # 本身才是負擔。現在介面直接寫明「只是要移除東西的話用本機」，
                # 規則就只有一條。代價是用雲端移除東西時會與周圍有色差
                # （實測 22.8／38.6 級），而那正是 `_trust_warning` 會講的事。
                intent="edit" if method.startswith("api:") else "fill",
                **options,
            )
        except (ImportError, FileNotFoundError, ProviderError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _accept(SESSION, result, method=method, prompt=request.prompt)

        return {
            "image": _png_data_url(SESSION.preview),
            "checksum": float(result.checksum),
            "method": method,
            "layers": len(SESSION.layers),
            "warning": _trust_warning(result, method),
        }


def _decode_references(data_urls: list[str]) -> list[np.ndarray]:
    """解開使用者附的參考圖。"""
    if len(data_urls) > MAX_REFERENCES:
        raise HTTPException(
            status_code=400,
            detail=f"最多只可以附 {MAX_REFERENCES} 張參考圖，收到 {len(data_urls)} 張。",
        )
    return [np.asarray(_decode_data_url(url).convert("RGB")) for url in data_urls]


def _trust_warning(result: EditResult, method: str) -> str | None:
    """模型有沒有照遮罩辦事？沒有的話要說出來。

    ★ 這是 ``EditResult.is_trustworthy`` 存在的理由，而它先前**從來沒有
    被呼叫過**。校驗環量度是模型在「被告知不要改」的區域改了多少級；
    本機引擎本來就會重繪整張裁切圖（實測 LaMa 是 24–27），但那無害——
    我們只取遮罩內的像素。**雲端模型聲稱會保留遮罩外，所以它的偏離
    才有意義**（實測 gpt-5.4-image-2 與 seedream-4.5 都是 255，滿級）。

    不否決結果：那是使用者已經付錢的呼叫，而且合成器本來就守住了遮罩外。
    但使用者有權知道自己拿到的是甚麼。
    """
    if not method.startswith("api:") or result.is_trustworthy(API_THRESHOLD):
        return None
    return (
        f"這個模型沒有照遮罩辦事：它在你圈選的範圍之外也改了 "
        f"{result.checksum:.0f} 級（滿級 255），也就是把整張重新畫了一遍。"
        "圈選範圍以外的像素仍然逐位元組沒變，但圈內的顏色是它自己決定的，"
        "可能與周圍不符。框得越緊，這個問題越小；"
        "只是要移除東西的話，用本機移除的顏色會準得多。"
    )


def _accept(
    session: Session,
    result: EditResult,
    *,
    method: str,
    prompt: str = "",
) -> None:
    """把一次執行的結果收下：變成一層、清空遮罩、清掉重做堆疊、寫進專案。

    **執行即生效**（使用者的選擇）。前後對照仍然在——只是發生在執行之後
    「看原圖／看成品」，而退路是復原，它是精確而且即時的。
    """
    layer = GenerativeLayer(
        method=method,
        crop=tuple(result.crop),
        mask_sha256=result.mask_sha256,
        prompt=prompt.strip() or None,
        feather_px=result.feather_px,
        dilate_px=result.dilate_px,
    )
    # from_edit 會在這裡檢查遮罩有沒有超出裁切框——此時還握有完整的遮罩。
    edit = EditLayer.from_edit(layer, session.mask, result.patch)

    session.layers.append(edit)
    # 新的編輯令重做失效：那些層已經不在鏈上了。
    session.redo = []

    # working 直接沿用結果，不必整個重播——它就是由同一個純函數算出來的。
    # （「working 永遠等於 render(source, layers)」由測試守著。）
    session.working = result.image
    session.preview = _downscale(session.working)
    session.revision += 1

    # 存檔要在清遮罩之前——遮罩就是這一層的材料。
    _persist(session, edit, result)

    # ★ 遮罩就地清空。清掉它是修好這個 bug 的第一步：留著的話下一次執行
    # 會變成「上一個範圍 ∪ 新範圍」，兩塊一起重算。
    session.mask[:] = False


def _persist(session: Session, edit: EditLayer, result: EditResult) -> None:
    """把新的一層寫進專案。專案掛不上時只記一則提示，編輯照常。"""
    if session.store is None:
        return
    try:
        layer = edit.layer
        digest = session.store.write_mask(layer.id, session.mask)
        if digest != layer.mask_sha256:
            # 兩邊都是 mask_digest，正常不可能不同。真的不同就代表快取鏈
            # 會對不上——寧可現在報錯，也不要之後拿到錯的結果。
            raise ValueError("遮罩寫入後的雜湊與執行時不符")

        session.store.add_layer(layer)
        session.store.apply_result(
            layer.id,
            result_file=session.store.write_result(layer.id, edit.patch),
            checksum=Checksum(max_abs_diff=_finite(result.checksum), at=_now()),
            # 快取鍵要在圖層加進去之後才算——它是一條鏈，鍵包含上游。
            cache_key=session.store.project.cache_keys()[layer.id],
        )
        session.store.save()
    except (OSError, KeyError, ValueError) as exc:
        session.store = None
        session.project_note = f"寫入專案失敗，之後的修改不會被保存：{exc}"


def _finite(value: float) -> float | None:
    """校驗環量度可能是 nan——那代表裁切區裡沒有校驗環可用（遮罩填滿了它）。

    JSON 沒有 nan，寫進去會令專案檔讀不回來，所以明確轉成「沒有這個數字」。
    """
    return float(value) if np.isfinite(value) else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── 復原與重做 ──────────────────────────────────────────────────


@app.post("/api/undo")
def undo() -> dict:
    """移除最後一層，並**把它的遮罩放回介面**。

    放回遮罩是刻意的：最常見的用法是「邊緣還留了一點陰影 → 復原 →
    把範圍擴大一點 → 再執行一次」。只把成品退回上一步的話，
    使用者得重新圈一次同樣的地方。
    """
    SESSION.require_image()
    with SESSION.lock:
        if not SESSION.layers:
            raise HTTPException(status_code=400, detail="沒有可以復原的修改")
        edit = SESSION.layers.pop()
        SESSION.redo.append(edit)
        if SESSION.store is not None:
            SESSION.store.remove_last_layer()
            SESSION.store.save()
        SESSION.recompose()
        SESSION.mask |= edit.full_mask(SESSION.mask.shape)
        return _undo_payload()


@app.post("/api/redo")
def redo() -> dict:
    """把上一層放回去，並清空遮罩——與當初執行時一致。"""
    SESSION.require_image()
    with SESSION.lock:
        if not SESSION.redo:
            raise HTTPException(status_code=400, detail="沒有可以重做的修改")
        edit = SESSION.redo.pop()
        SESSION.layers.append(edit)
        if SESSION.store is not None:
            SESSION.store.add_layer(edit.layer)
            SESSION.store.save()
        SESSION.recompose()
        SESSION.mask[:] = False
        return _undo_payload()


def _undo_payload() -> dict:
    snapshot = SESSION.snapshot()
    snapshot["image"] = _png_data_url(SESSION.preview)
    snapshot["overlay"] = _mask_payload()["overlay"]
    return snapshot


@app.get("/api/result")
def download_result(format: str = "webp") -> Response:
    """匯出成品——**原解析度**，不是預覽，而且含所有已接受的編輯。

    預設是 WebP。理由很實際：PNG 存照片是浪費——實測同一張 18.4 MP 的圖，
    PNG 是 17.32 MB（比原檔的 JPEG 大 3.4 倍），WebP q95 是 3.49 MB
    （比原檔還小），而平均只差 0.95 級。要完全不失真的人可以要 PNG。

    用 ``Response`` 而不是 ``FileResponse``：後者要的是檔案路徑，
    而我們在記憶體裡已經有現成的位元組，沒有理由先寫到磁碟再讀回來。
    """
    SESSION.require_image()
    if not SESSION.layers:
        # 沒有任何編輯時成品就等於原檔，匯出它沒有意義。
        raise HTTPException(status_code=400, detail="還沒有結果可以匯出")

    image = Image.fromarray(SESSION.working)
    buffer = io.BytesIO()
    if format == "png":
        image.save(buffer, format="PNG")
        extension, media_type = "png", "image/png"
    else:
        image.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
        extension, media_type = "webp", "image/webp"

    stem = SESSION.source_path.stem if SESSION.source_path else "result"
    return Response(
        content=buffer.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{stem}-photoman.{extension}"'},
    )


def run(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    run()
