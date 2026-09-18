"""命令列入口。"""

from __future__ import annotations

import socket
import subprocess
import sys

import typer
from rich.console import Console

from photoman.paths import MODELS_DIR

app = typer.Typer(add_completion=False, help="photoMan — 本地優先的修圖工具")
console = Console()

DEFAULT_PORT = 8765


@app.command()
def ui(
    port: int = typer.Option(DEFAULT_PORT, help="介面的連接埠"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不要自動開啟瀏覽器"),
) -> None:
    """開啟介面。"""
    _warn_about_missing_models()

    if _port_in_use(port):
        console.print(
            f"[yellow]連接埠 {port} 已經被佔用。[/yellow]\n"
            "可能已經有一個 photoMan 在執行——先開瀏覽器看看：\n"
            f"  http://127.0.0.1:{port}/\n"
            f"或用 [bold]photoman ui --port {port + 1}[/bold] 換一個埠。"
        )
        raise typer.Exit(code=1)

    console.print(f"[green]photoMan[/green] 介面在 http://127.0.0.1:{port}/")
    console.print("按 Ctrl+C 停止。")

    from photoman.web.server import run

    run(port=port, open_browser=not no_browser)


@app.command()
def shortcut() -> None:
    """在桌面建立捷徑。

    **捷徑不進版本庫**：它內含絕對路徑，資料夾一移動就失效。
    由這個指令重建。

    ⚠️ 這裡刻意用 PowerShell 的**行內** ``-Command``，而不是執行一個
    ``.ps1`` 檔案。行內指令不受執行原則管轄（那一項管的是指令碼檔案），
    所以不需要 ``-ExecutionPolicy Bypass``——那等於為了建立一個捷徑
    而關掉一個安全控制，代價與目的不相稱。
    """
    if sys.platform != "win32":
        console.print("[red]建立捷徑只支援 Windows。[/red]")
        raise typer.Exit(code=1)

    from photoman.paths import PROJECT_ROOT

    launcher = PROJECT_ROOT / "photoMan.bat"
    if not launcher.exists():
        console.print(f"[red]找不到 {launcher}[/red]")
        raise typer.Exit(code=1)

    script = (
        "$d = [Environment]::GetFolderPath('Desktop'); "
        '$s = (New-Object -ComObject WScript.Shell).CreateShortcut("$d\\photoMan.lnk"); '
        f"$s.TargetPath = '{launcher}'; "
        f"$s.WorkingDirectory = '{PROJECT_ROOT}'; "
        "$s.Description = 'photoMan - local photo editing'; "
        '$s.IconLocation = "$env:SystemRoot\\System32\\shell32.dll,325"; '
        "$s.Save()"
    )

    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        console.print(f"[red]建立捷徑失敗：[/red]\n{completed.stderr.strip()}")
        raise typer.Exit(code=1)

    console.print("[green]已在桌面建立 photoMan 捷徑。[/green]")


@app.command()
def models() -> None:
    """檢查模型檔案的狀態。"""
    from photoman.paths import LAMA_MODEL_FILE, SAM_DECODER_FILE, SAM_ENCODER_FILE

    entries = [
        ("物件選取（MobileSAM 編碼器）", SAM_ENCODER_FILE),
        ("物件選取（MobileSAM 解碼器）", SAM_DECODER_FILE),
        ("移除（LaMa）", LAMA_MODEL_FILE),
    ]
    missing = False
    for label, filename in entries:
        path = MODELS_DIR / filename
        if path.exists():
            size = path.stat().st_size / 1e6
            console.print(f"  [green]✓[/green] {label}　{filename}　{size:.1f} MB")
        else:
            console.print(f"  [red]✗[/red] {label}　{filename}　[red]未下載[/red]")
            missing = True

    if missing:
        console.print("\n下載指令見 docs/PROGRESS.md。")


def _warn_about_missing_models() -> None:
    from photoman.paths import LAMA_MODEL_FILE, SAM_ENCODER_FILE

    missing = [
        name
        for name, filename in (("物件選取", SAM_ENCODER_FILE), ("移除", LAMA_MODEL_FILE))
        if not (MODELS_DIR / filename).exists()
    ]
    if missing:
        console.print(
            f"[yellow]未下載：{'、'.join(missing)}[/yellow]　"
            "介面仍然開得起來，但那兩個功能不能用。\n"
            "用 [bold]photoman models[/bold] 查看狀態。"
        )


def _port_in_use(port: int) -> bool:
    """檢查連接埠是否已被佔用。

    mangaMan 吃過這個苦：無視窗啟動失敗時錯誤只寫進 log，
    使用者只看到「按了沒反應」。所以這裡要在啟動之前先檢查，
    並把可能的原因直接講出來。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
