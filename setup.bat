@echo off
REM ============================================================
REM  photoMan - one-time setup
REM
REM  IMPORTANT: this file must stay PURE ASCII.
REM  Windows console uses OEM 437 when launched from Explorer,
REM  so non-ASCII characters become garbage. Adding "chcp 65001"
REM  to work around that corrupts the file offsets instead.
REM  Keep every message in this file in plain English.
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo   photoMan - setup
echo ============================================================
echo.

REM ---- 1. Python 3.12 -----------------------------------------
REM The "python" on PATH may be an older version. Always use the
REM "py" launcher with an explicit version, otherwise the venv is
REM built with the wrong interpreter and torch/onnxruntime fail
REM later with a confusing error.
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.12 was not found.
  echo.
  echo   Install it from https://www.python.org/downloads/
  echo   and keep the "py launcher" option ticked during install.
  echo.
  goto :failed
)
for /f "delims=" %%v in ('py -3.12 --version 2^>^&1') do echo [1/6] Found %%v

REM ---- 2. Virtual environment ---------------------------------
if exist ".venv\Scripts\python.exe" (
  echo [2/6] Virtual environment already exists - keeping it.
) else (
  echo [2/6] Creating virtual environment .venv ...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    goto :failed
  )
)

REM ---- 3. Dependencies ----------------------------------------
echo [3/6] Installing dependencies ^(this takes a few minutes^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -e ".[dev,models]" --quiet
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  goto :failed
)

REM ---- 4. Models ----------------------------------------------
REM Kept out of the repository: they are ~140 MB and their
REM licences differ from this project's.
mkdir models 2>nul

if exist "models\inpainting_lama.onnx" (
  echo [4/6] LaMa model already present.
) else (
  echo [4/6] Downloading LaMa ^(93 MB^) ...
  curl -L --fail --progress-bar -o "models\inpainting_lama.onnx" ^
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx"
  if errorlevel 1 (
    echo [ERROR] Could not download the LaMa model.
    goto :failed
  )
)

if exist "models\mobile_sam_encoder.onnx" (
  echo       MobileSAM already present.
) else (
  echo       Downloading MobileSAM ^(45 MB^) ...
  curl -L --fail --progress-bar -o "models\mobile_sam_encoder.onnx" ^
    "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx"
  if errorlevel 1 (
    echo [ERROR] Could not download the MobileSAM encoder.
    goto :failed
  )
  curl -L --fail --progress-bar -o "models\mobile_sam_decoder.onnx" ^
    "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx"
  if errorlevel 1 (
    echo [ERROR] Could not download the MobileSAM decoder.
    goto :failed
  )
)

REM ---- 5. Desktop shortcut ------------------------------------
REM Not committed to the repository: a shortcut embeds an absolute
REM path, so it breaks as soon as the folder moves. Rebuild it here.
".venv\Scripts\python.exe" -m photoman.cli shortcut >nul 2>&1
if errorlevel 1 (
  echo [5/6] Could not create a desktop shortcut - not fatal.
) else (
  echo [5/6] Desktop shortcut created.
)

REM ---- 6. Verify ----------------------------------------------
echo [6/6] Running the test suite to confirm everything works ...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 (
  echo.
  echo [WARNING] Some tests failed. The tool may still work -
  echo           run the command above to see what failed.
  echo.
)

echo.
echo ============================================================
echo   Setup finished.
echo.
echo   Double-click photoMan.bat to start, or use the desktop
echo   shortcut that was just created.
echo ============================================================
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo   Setup did NOT finish. Nothing was left half-installed -
echo   running this file again will pick up where it stopped.
echo ============================================================
echo.
pause
exit /b 1