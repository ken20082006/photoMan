"""保真匯入與匯出（見 docs/design.md §7.1）。

**「保真」在這裡有三個具體的意思，三個都是實測踩出來的：**

1. **HEIC 要讀得到。** 那是 iPhone 的預設格式，而 Pillow 本身讀不到它。
   沒有 ``pillow-heif`` 的話，使用者從手機傳過來的第一張照片就會失敗。
2. **ICC 要處理。** 實測發現 iPhone 的照片是 **Display P3**，不是 sRGB。
   把 P3 的數值直接當成 sRGB 用，飽和色會偏。
3. **EXIF 方向要套用。** 不套用的話，直拍的照片會躺平——
   而使用者只會覺得「這個工具壞了」，不會想到是 EXIF。

**匯入的輸出是內部工作空間：線性光 sRGB、float32**（§6.1b）。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from photoman.color import linear_to_srgb, srgb_to_linear

# HEIC 是 iPhone 的預設格式。註冊之後 Pillow 就能開啟它。
try:  # pragma: no cover - 純粹是環境相依的註冊
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover
    HEIC_SUPPORTED = False

# EXIF 方向標籤。套用之後要重設為 1，否則匯出時會被再轉一次。
_EXIF_ORIENTATION = 274


@dataclass(frozen=True)
class SourceInfo:
    """原圖的完整描述——載入時捕獲，匯出時寫回。

    刻意記下 ``icc_profile`` 的原始位元組與 ``exif``，因為匯出時
    要把它們寫回去。只記「有沒有」是不夠的。
    """

    path: Path
    sha256: str
    format: str
    width: int
    height: int
    bit_depth: int
    icc_profile: bytes | None
    icc_description: str | None
    exif: bytes | None
    had_orientation_tag: bool


@dataclass(frozen=True)
class LoadedImage:
    """已載入的原圖：像素在內部工作空間（線性光 sRGB、float32）。

    ⚠️ 42 MP 的照片這會佔 506 MB。**這是分析用的表示，不是儲存格式。**
    編輯 pipeline 要用 :func:`load_srgb`（§6.1c）。
    """

    info: SourceInfo
    linear: np.ndarray


@dataclass(frozen=True)
class LoadedSrgb:
    """底圖：**uint8 sRGB 位元組**，是編輯 pipeline 的實際儲存格式（§6.1c）。

    42 MP 只佔 127 MB，而且合成時遮罩外是字面上的位元組複製。
    """

    info: SourceInfo
    srgb: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_profile(profile: ImageCms.ImageCmsProfile) -> str:
    try:
        return ImageCms.getProfileDescription(profile).strip()
    except Exception:  # noqa: BLE001 - 描述檔壞掉不應該令載入失敗
        return "（無法讀取描述）"


def _is_srgb(profile: ImageCms.ImageCmsProfile) -> bool:
    """判斷描述檔是否已經等同 sRGB。

    用描述字串判斷是權宜之計，但實務上夠用：若描述裡有 sRGB 就當作是。
    寧可漏判（多做一次恆等轉換，無害）也不要誤判（跳過必要的轉換）。
    """
    return "srgb" in _describe_profile(profile).lower().replace(" ", "")


def _convert_to_srgb(image: Image.Image) -> Image.Image:
    """把影像由它自己的 ICC 描述檔轉到 sRGB。

    沒有描述檔就當作 sRGB——那是 JPEG 的事實標準，也是唯一合理的假設。
    """
    raw = image.info.get("icc_profile")
    if not raw:
        return image
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(raw))
        if _is_srgb(source):
            return image
        # 相對色度意圖：保留色域內的顏色不動，只把色域外的裁剪掉。
        # 感知意圖會連色域內的顏色都一起壓縮，對修圖工具不合適。
        return ImageCms.profileToProfile(
            image,
            source,
            ImageCms.createProfile("sRGB"),
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        )
    except Exception:  # noqa: BLE001 - 壞掉的描述檔不應該令整張圖載入失敗
        return image


def _capture_info(opened: Image.Image, path: Path) -> SourceInfo:
    """由已開啟的影像捕獲完整描述。載入與匯出都要用同一份。"""
    raw_icc = opened.info.get("icc_profile")
    return SourceInfo(
        path=path,
        sha256=_sha256(path),
        format=opened.format or "UNKNOWN",
        width=opened.width,
        height=opened.height,
        bit_depth=_bit_depth(opened),
        icc_profile=raw_icc,
        icc_description=(
            _describe_profile(ImageCms.ImageCmsProfile(io.BytesIO(raw_icc))) if raw_icc else None
        ),
        exif=opened.info.get("exif"),
        had_orientation_tag=_EXIF_ORIENTATION in opened.getexif(),
    )


def load(path: str | Path) -> LoadedImage:
    """載入原圖，轉成內部工作空間（線性光 sRGB、float32）。

    回傳的 ``linear`` 是 ``float32``、範圍大致 ``[0,1]``，但**不裁剪**——
    色彩轉換可以在極端顏色上產生稍微超出範圍的值，過早裁剪會失去資訊。

    ⚠️ **這是分析用的表示，不是儲存格式**（§6.1c）。42 MP 佔 506 MB。
    編輯 pipeline 要用 :func:`load_srgb`。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with Image.open(path) as opened:
        info = _capture_info(opened, path)
        # 先套用 EXIF 方向再轉色彩——轉色彩是有方向的運算，
        # 在躺平的圖上做要多一次轉置，而且容易寫錯。
        upright = ImageOps.exif_transpose(opened)
        converted = _convert_to_srgb(upright)
        array = _to_float01(np.asarray(converted.convert("RGB")))

    return LoadedImage(info=info, linear=srgb_to_linear(array))


def load_srgb(path: str | Path) -> LoadedSrgb:
    """載入原圖為 **uint8 sRGB**——編輯 pipeline 的實際底圖（§6.1c）。

    與 :func:`load` 的分工：

    - :func:`load` 回傳整張圖的線性光 float32。那是**分析用的表示**
      （42 MP 要 506 MB），適合用來做預覽或統計，不適合當底圖。
    - 本函數回傳原檔編碼的位元組（42 MP 只要 127 MB）。
      合成時遮罩外是**字面上的位元組複製**，那些像素根本沒有
      經過任何浮點運算。

    兩者的像素內容一致（同一個 ICC 轉換、同一個 EXIF 方向套用），
    差別只在表示方式。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with Image.open(path) as opened:
        info = _capture_info(opened, path)
        upright = ImageOps.exif_transpose(opened)
        converted = _convert_to_srgb(upright)
        array = np.asarray(converted.convert("RGB"))

    if array.dtype != np.uint8:
        raise ValueError(f"底圖必須是 uint8，得到 {array.dtype}")
    return LoadedSrgb(info=info, srgb=array)


def save(
    linear: np.ndarray,
    path: str | Path,
    *,
    info: SourceInfo | None = None,
    bit_depth: int = 8,
) -> None:
    """把內部工作空間的影像匯出。

    ``info`` 有給的話，會把 ICC 描述檔與 EXIF 寫回去（方向標籤除外——
    像素已經轉正了，留著原本的方向標籤會令看圖軟體再轉一次）。
    """
    path = Path(path)
    srgb = np.clip(linear_to_srgb(linear), 0.0, 1.0)

    if bit_depth == 8:
        array = (srgb * 255.0 + 0.5).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
    else:
        raise NotImplementedError(f"{bit_depth} 位元匯出尚未實作——見 docs/PROGRESS.md 的已知缺口")

    params: dict[str, object] = {}
    if info is not None:
        if info.icc_profile:
            params["icc_profile"] = info.icc_profile
        if info.exif:
            params["exif"] = _exif_without_orientation(info.exif)

    image.save(path, **params)


def _bit_depth(image: Image.Image) -> int:
    """由 Pillow 的模式推斷原始位元深度。

    ⚠️ **陷阱：16 位元的 RGB PNG 會被 Pillow 悄悄轉成 8 位元 ``RGB``。**
    所以這個函數對那類檔案會回報 8，而真實來源是 16。
    要準確知道的話需要自己解析檔案標頭——列在 docs/PROGRESS.md 的已知缺口。
    """
    return 16 if image.mode.startswith("I;16") else 8


def _to_float01(array: np.ndarray) -> np.ndarray:
    """把整數像素轉成 ``[0,1]`` 的 float32。"""
    if array.dtype == np.uint8:
        return (array.astype(np.float32) / 255.0).astype(np.float32)
    if array.dtype == np.uint16:
        return (array.astype(np.float32) / 65535.0).astype(np.float32)
    raise ValueError(f"不支援的像素型別：{array.dtype}")


def _exif_without_orientation(raw: bytes) -> bytes:
    """移除 EXIF 的方向標籤。

    像素在載入時已經轉正了。留著原本的方向標籤，會令看圖軟體
    再轉一次——照片就變成躺平的，而使用者不會知道為甚麼。
    """
    try:
        exif = Image.Exif()
        exif.load(raw)
        if _EXIF_ORIENTATION in exif:
            del exif[_EXIF_ORIENTATION]
        return exif.tobytes()
    except Exception:  # noqa: BLE001 - 壞掉的 EXIF 不應該令匯出失敗
        return raw
