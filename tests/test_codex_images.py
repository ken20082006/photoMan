"""Codex 接駁必須使用既有登入，且只接收本次生圖的輸出。"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photoman import codex_images
from photoman.providers import ProviderError

THREAD = "01a13427-7701-7ba3-ba21-8c7c4419ebf6"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_images, "find_codex", lambda: "codex.exe")
    monkeypatch.setattr(codex_images, "generated_root", lambda: tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    monkeypatch.setenv("CODEX_API_KEY", "should-not-be-used")
    return tmp_path


def events(*, completed=True, thread=THREAD):
    items = [{"type": "thread.started", "thread_id": thread}]
    if completed:
        items.append({"type": "turn.completed"})
    return "\n".join(json.dumps(item) for item in items)


def test_uses_builtin_tool_and_only_this_threads_image(configured, monkeypatch):
    def run(command, **options):
        assert options["env"].get("OPENAI_API_KEY") is None
        assert options["env"].get("CODEX_API_KEY") is None
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert "--ignore-user-config" in command
        assert "$imagegen" in options["input"]
        assert options["input"].endswith("改成藍色")
        inputs = command[command.index("--image") + 1].split(",")
        assert len(inputs) == 2
        assert all(Path(path).is_file() for path in inputs)
        assert all(Image.open(path).size == (2048, 32) for path in inputs)
        directory = configured / THREAD
        directory.mkdir()
        Image.new("RGB", (20, 20), "blue").save(directory / "generated.png")
        Image.new("RGB", (20, 20), "red").save(configured / "unrelated.png")
        return subprocess.CompletedProcess(command, 0, events(), "")

    monkeypatch.setattr(subprocess, "run", run)
    array = np.zeros((32, 2048, 3), dtype=np.uint8)
    response = codex_images.CodexImageClient().edit(
        array, "改成藍色", model="codex:local", references=[array],
    )
    assert response.image.shape == (20, 20, 3)
    assert np.all(response.image[..., 2] == 255)
    assert response.cost_usd is None


@pytest.mark.parametrize("output,code", [(events(completed=False), 1), (events(), 0),
                                         (events(thread="../../private"), 0)])
def test_failure_never_uses_unrelated_output(configured, monkeypatch, output, code):
    Image.new("RGB", (20, 20)).save(configured / "other.png")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, code, output, "private diagnostics"))
    with pytest.raises(ProviderError) as error:
        codex_images.CodexImageClient().edit(
            np.zeros((20, 20, 3), dtype=np.uint8), "修改", model="codex:local",
        )
    assert "private diagnostics" not in str(error.value)


def test_timeout_is_actionable(configured, monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 600)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ProviderError, match="逾時"):
        codex_images.CodexImageClient().edit(
            np.zeros((20, 20, 3), dtype=np.uint8), "修改", model="codex:local",
        )


def test_missing_codex_does_not_fall_back_to_api(monkeypatch):
    monkeypatch.setattr(codex_images, "find_codex", lambda: None)
    with pytest.raises(ProviderError, match="找不到本機 Codex"):
        codex_images.CodexImageClient().edit(
            np.zeros((20, 20, 3), dtype=np.uint8), "修改", model="codex:local",
        )
