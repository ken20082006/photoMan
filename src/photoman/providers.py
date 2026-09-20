"""雲端模型的客戶端（見 docs/design.md §8）。

**BYO API key**：使用者自己付錢，所以要有快取、要能選 model、要跑前估價。

⚠️ **圖像模型在獨立的端點 `/api/v1/images/models`，不在 `/models`。**
後者只列文字模型（四百多個），一個圖像模型都沒有。
查錯端點會得出「OpenRouter 沒有圖像模型」這個完全錯誤的結論——
我第一次就查錯了，而 Seedream 一直都在。

⚠️ **圖像 API 沒有 mask 參數。** 所以做不到真正的遮罩式 inpainting：
只能送裁切圖加文字指令，收回整張，再靠本地合成器只取遮罩內的像素。
這不是缺陷而是現實——而且因為保證在合成器，這樣做是安全的：
就算模型把整張裁切圖都改爛，遮罩外的像素仍然逐位元組不變。
"""

from __future__ import annotations

import base64
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# 送出之前把裁切圖縮到這個尺寸。
# 模型的原生輸出多數是 1–4 MP，送更大的圖只會被內部縮掉，
# 白白增加上傳時間與 token 成本。
MAX_SEND_PX = 1536


class ProviderError(RuntimeError):
    """呼叫雲端模型失敗。訊息要能直接顯示給使用者看。"""


@dataclass(frozen=True)
class ImageModel:
    """一個可用的圖像模型。"""

    id: str
    name: str
    price_per_image: float | None
    max_resolution: str
    max_references: int


@dataclass(frozen=True)
class EditResponse:
    image: np.ndarray
    cost_usd: float | None
    seconds: float
    model: str


def _data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _decode(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


class OpenRouterClient:
    """OpenRouter 的圖像編輯客戶端。"""

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL) -> None:
        if not api_key:
            raise ProviderError(
                "尚未設定 API key。在介面的「設定」填入，或改用本機的移除功能（那個不需要 key）。"
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=300.0)
        self._models: list[ImageModel] | None = None

    # ── 模型清單 ────────────────────────────────────────────────

    def models(self) -> list[ImageModel]:
        """列出可用的圖像模型，按價格排序。

        清單會快取——它很少變動，而每次開介面都抓一次是浪費。
        """
        if self._models is not None:
            return self._models

        try:
            listing = self._client.get(f"{self.base_url}/images/models").json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderError(f"無法取得模型清單：{exc}") from exc

        found: list[ImageModel] = []
        for entry in listing.get("data", []):
            model_id = entry.get("id", "")
            if not model_id or "vector" in model_id or model_id.endswith("-preview"):
                continue
            try:
                detail = self._client.get(f"https://openrouter.ai{entry['endpoints']}").json()
            except (httpx.HTTPError, KeyError, json.JSONDecodeError):
                continue
            for endpoint in detail.get("endpoints", []):
                price = next(
                    (
                        item["cost_usd"]
                        for item in endpoint.get("pricing", [])
                        if item.get("billable") == "output_image"
                    ),
                    None,
                )
                params = endpoint.get("supported_parameters", {})
                resolution = params.get("resolution")
                references = params.get("input_references")
                found.append(
                    ImageModel(
                        id=model_id,
                        name=entry.get("name", model_id),
                        price_per_image=price,
                        max_resolution="/".join(resolution.get("values", []))
                        if isinstance(resolution, dict)
                        else "—",
                        max_references=references.get("max", 0)
                        if isinstance(references, dict)
                        else 0,
                    )
                )

        # 同一個模型可能有多個供應商——只留最便宜的那一個。
        cheapest: dict[str, ImageModel] = {}
        for model in found:
            current = cheapest.get(model.id)
            if current is None or (
                model.price_per_image is not None
                and (
                    current.price_per_image is None
                    or model.price_per_image < current.price_per_image
                )
            ):
                cheapest[model.id] = model

        self._models = sorted(
            cheapest.values(),
            key=lambda m: (m.price_per_image is None, m.price_per_image or 0.0),
        )
        return self._models

    # ── 編輯 ────────────────────────────────────────────────────

    def edit(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        model: str,
        resolution: str | None = None,
        references: Sequence[np.ndarray] = (),
    ) -> EditResponse:
        """送一張圖加指令，回傳模型產生的圖。

        ``references`` 是使用者另外給的參考圖（例如「把這隻狗加進去」的那隻狗）。
        它們會接在主要那張之後，順序即指令裡說的「第幾張」。

        **不需要遮罩。** OpenRouter 的圖像 API 沒有這個參數。
        我們只取回傳圖在遮罩內的部分，其餘由本地合成器保留原樣。
        """
        if not prompt.strip():
            raise ProviderError("請描述要做甚麼，例如「移除中間的人」或「加一隻小狗」。")

        sent = [_fit(image, MAX_SEND_PX)] + [_fit(ref, MAX_SEND_PX) for ref in references]
        body: dict = {
            "model": model,
            "prompt": prompt.strip(),
            "input_references": [
                {"type": "image_url", "image_url": {"url": _data_url(prepared)}}
                for prepared in sent
            ],
            "n": 1,
        }
        if resolution:
            body["resolution"] = resolution

        started = time.time()
        try:
            response = self._client.post(f"{self.base_url}/images/generations", json=body)
        except httpx.HTTPError as exc:
            raise ProviderError(f"連線失敗：{exc}") from exc
        elapsed = time.time() - started

        if response.status_code == 401:
            raise ProviderError("API key 無效或已撤銷。請在「設定」重新填入。")
        if response.status_code == 402:
            raise ProviderError("帳戶餘額不足。")
        if response.status_code != 200:
            raise ProviderError(
                f"模型回報錯誤（HTTP {response.status_code}）：{response.text[:300]}"
            )

        payload = response.json()
        items = payload.get("data") or []
        if not items:
            raise ProviderError("模型沒有回傳圖像。")

        image_bytes = base64.b64decode(items[0]["b64_json"])
        produced = _decode(image_bytes)

        # 模型的輸出尺寸未必等於送出的尺寸，要縮放回去才能合成。
        if produced.shape[:2] != image.shape[:2]:
            produced = np.asarray(
                Image.fromarray(produced).resize((image.shape[1], image.shape[0]), Image.LANCZOS)
            )

        return EditResponse(
            image=produced,
            cost_usd=(payload.get("usage") or {}).get("cost"),
            seconds=elapsed,
            model=model,
        )


def _fit(image: np.ndarray, limit: int) -> np.ndarray:
    """縮到長邊不超過 ``limit``。"""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= limit:
        return image
    scale = limit / longest
    return np.asarray(
        Image.fromarray(image).resize(
            (max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS
        )
    )


def client_from_config() -> OpenRouterClient:
    """由使用者設定建立客戶端。"""
    from photoman import config

    settings = config.load().get("providers", {}).get("openrouter", {})
    return OpenRouterClient(
        settings.get("api_key", ""),
        base_url=settings.get("base_url", DEFAULT_BASE_URL),
    )


def models_cache_path() -> Path:
    from photoman import config

    return config.config_dir() / "image_models.json"
