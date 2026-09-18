"""使用者設定（見 docs/design.md §8）。

**設定檔刻意放在 `%APPDATA%\\photoMan\\config.json`，不在 repo 之內。**
理由與 mangaMan 相同：`git add -A` 不會把 API key 一併提交。
那個意外一旦發生就無法收回——key 已經在 git 歷史裡了。

**API key 回傳給瀏覽器時一律遮蔽。** 介面只需要知道「有沒有填」，
不需要知道內容。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 遮蔽後的字串。介面用它判斷「有沒有填」。
MASK = "••••••••"

DEFAULT_CONFIG: dict[str, Any] = {
    "providers": {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "",
        }
    },
    "tasks": {
        # 局部生成的預設引擎。本地 LaMa 是預設——免費、離線、
        # 而且它是真正的遮罩式 inpainting（見 §7.4.3）。
        "inpaint": "lama",
    },
}


def config_dir() -> Path:
    return Path(os.environ.get("APPDATA", Path.home())) / "photoMan"


def config_path() -> Path:
    return config_dir() / "config.json"


def work_dir() -> Path:
    """上傳的圖片放這裡。"""
    return config_dir() / "work"


def load() -> dict[str, Any]:
    """讀出設定。缺少的欄位以預設值補齊（深層合併）。"""
    path = config_path()
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # 壞掉的設定檔不應該令工具無法啟動——用預設值繼續，
        # 使用者至少還能改回來。
        return json.loads(json.dumps(DEFAULT_CONFIG))
    return _merge(DEFAULT_CONFIG, stored)


def save(config: dict[str, Any]) -> None:
    config_dir().mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def get_api_key(provider: str = "openrouter") -> str:
    """取得 API key。**這個函數的結果永遠不可以送到瀏覽器。**"""
    return load().get("providers", {}).get(provider, {}).get("api_key", "")


def masked() -> dict[str, Any]:
    """給瀏覽器看的版本——API key 換成遮蔽字串。"""
    config = load()
    view = json.loads(json.dumps(config))
    for provider in view.get("providers", {}).values():
        if provider.get("api_key"):
            provider["api_key"] = MASK
    return view


def update_api_key(value: str, provider: str = "openrouter") -> None:
    """更新 API key。

    送來遮蔽字串時**不覆蓋**——那代表使用者沒有改動它，
    只是介面把讀到的值原樣送回來。
    """
    if value == MASK:
        return
    config = load()
    config.setdefault("providers", {}).setdefault(provider, {})["api_key"] = value
    save(config)


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result
