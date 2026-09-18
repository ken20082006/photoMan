"""模型與專案的路徑。

模型權重刻意**不進版本庫**（`.gitignore` 排除 `models/`）：
它們有數十至數百 MB，而且部分權重的授權與本 repo 不同。
"""

from __future__ import annotations

from pathlib import Path

# src/photoman/paths.py → 上三層就是專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# LaMa，Apache-2.0。來源：huggingface.co/opencv/inpainting_lama
LAMA_MODEL_FILE = "inpainting_lama.onnx"

# MobileSAM，Apache-2.0。編碼器與解碼器分開是刻意的——
# 圖像嵌入只算一次，之後每次點擊只跑解碼器（見 segment.py）。
# 來源：huggingface.co/Acly/MobileSAM
SAM_ENCODER_FILE = "mobile_sam_encoder.onnx"
SAM_DECODER_FILE = "mobile_sam_decoder.onnx"


def lama_model_path() -> Path:
    return MODELS_DIR / LAMA_MODEL_FILE
