"""API 裁切圖與參考圖按原尺寸送出，回傳圖亦保持供應商尺寸。"""

import base64
import io

import httpx
import numpy as np
from PIL import Image

from photoman.providers import OpenRouterClient


def test_no_input_downscale_or_output_upscale(monkeypatch):
    client = OpenRouterClient("test-only")
    output = io.BytesIO()
    Image.new("RGB", (32, 32)).save(output, format="PNG")

    def post(url, *, json):
        for item in json["input_references"]:
            raw = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
            assert Image.open(io.BytesIO(raw)).size == (2048, 32)
        return httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(output.getvalue()).decode()}]}
        )

    monkeypatch.setattr(client._client, "post", post)
    image = np.zeros((32, 2048, 3), dtype=np.uint8)
    response = client.edit(image, "修改", model="test-model", references=[image])
    assert response.image.shape == (32, 32, 3)
