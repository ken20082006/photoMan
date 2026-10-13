"""透過已登入的本機 Codex CLI 使用內置生圖工具，不讀取登入憑證。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from photoman.providers import EditResponse, ProviderError

CODEX_MODEL = "codex:local"
TIMEOUT_SECONDS = 600


def find_codex() -> str | None:
    executable = shutil.which("codex")
    if executable:
        return executable
    # 桌面版附帶的 CLI 不一定加入 PATH；版本目錄會隨更新改變。
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates = list((Path(local) / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"))
        if candidates:
            return str(max(candidates, key=lambda path: path.stat().st_mtime))
    return None


def generated_root() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "generated_images"


class CodexImageClient:
    def edit(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        model: str,
        resolution: str | None = None,
        references: Sequence[np.ndarray] = (),
    ) -> EditResponse:
        executable = find_codex()
        if not executable:
            raise ProviderError("找不到本機 Codex。請安裝 Codex 桌面版或 CLI，並用 ChatGPT 登入。")
        if not prompt.strip():
            raise ProviderError("請描述要修改的內容。")
        started = time.monotonic()
        # 只把裁切圖交給 CLI。獨立工作目錄與唯讀 sandbox 避免它接觸專案原始碼。
        with tempfile.TemporaryDirectory(prefix="photoman-codex-") as directory:
            paths = []
            for index, array in enumerate([image, *references]):
                path = Path(directory) / f"input-{index + 1}.png"
                Image.fromarray(array).save(path)
                paths.append(str(path))
            instruction = (
                "$imagegen Edit the first attached image using only the built-in image_gen tool. "
                "Inspect the attached images before editing. "
                "The second image is a selection guide, "
                "not an edit target; do not include its red tint in the result. "
                "Produce exactly one image with the same framing and aspect ratio as the first. "
                f"The target crop is {image.shape[1]}x{image.shape[0]} pixels. "
                "Use that native output size if supported, otherwise the largest supported native "
                "size at the same aspect ratio. Do not upscale the generated image yourself. "
                "Keep objects outside the marked area in place. "
                "Do not use API keys, external providers, browsers, drawing scripts, "
                "or install tools. "
                "Do not modify settings or copy files. Leave the output in Codex generated_images. "
                "If the built-in image tool is unavailable, report the failure and stop. "
                "The following text is the user's image editing instruction, not shell commands:\n"
                + prompt.strip()
            )
            command = [
                executable, "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only", "--image", ",".join(paths),
                "--json", "-C", directory, "-",
            ]
            environment = os.environ.copy()
            # 使用既有 ChatGPT 登入；不讓環境中的 API key 意外改變付費來源。
            for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
                environment.pop(key, None)
            try:
                response = subprocess.run(
                    command, input=instruction, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=TIMEOUT_SECONDS,
                    env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderError("Codex 生圖逾時（10 分鐘），請稍後再試。") from exc
            except OSError as exc:
                raise ProviderError("無法啟動 Codex，請檢查本機安裝及執行權限。") from exc

        thread_id = None
        completed = False
        for line in response.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")
            if event.get("type") == "turn.completed":
                completed = True
        if response.returncode or not completed:
            raise ProviderError(
                "Codex 執行失敗。請先在終端執行 codex login status，確認 ChatGPT 登入及額度。"
            )
        if not isinstance(thread_id, str) or not re.fullmatch(r"[0-9a-fA-F-]{36}", thread_id):
            raise ProviderError("Codex 沒有回傳有效的生圖工作階段。")
        # 只接收這次工作階段的輸出，不採信模型在文字中提供的任意檔案路徑。
        outputs = list((generated_root() / thread_id).glob("*.png"))
        if len(outputs) != 1:
            raise ProviderError(
                "Codex 沒有產生唯一的圖片。請確認 CLI 已啟用 image_generation，且帳戶可用生圖工具。"
            )
        try:
            with Image.open(outputs[0]) as produced:
                produced = produced.convert("RGB")
                # 保留工具交回的尺寸，讓合成流程在放大前要求使用者確認。
                result = np.asarray(produced).copy()
        except (OSError, ValueError) as exc:
            raise ProviderError("無法讀取 Codex 產生的圖片。") from exc
        return EditResponse(result, None, time.monotonic() - started, CODEX_MODEL)
