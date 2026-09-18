@echo off
REM ============================================================
REM  photoMan - launcher
REM
REM  Keeps a console window on purpose. Without one, a startup
REM  failure is invisible: the window just does not appear and
REM  the only record is a log file nobody reads. A visible
REM  console also gives Ctrl+C to stop the server.
REM
REM  Keep this file PURE ASCII - see the note in setup.bat.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ============================================================
  echo   photoMan has not been set up yet.
  echo.
  echo   Double-click setup.bat first - it creates the Python
  echo   environment and downloads the models.
  echo ============================================================
  echo.
  pause
  exit /b 1
)

echo Starting photoMan...
echo.
echo   The browser will open in a moment.
echo   Close this window to stop, or press Ctrl+C.
echo.

".venv\Scripts\python.exe" -m photoman.cli ui

if errorlevel 1 (
  echo.
  echo ============================================================
  echo   photoMan stopped with an error. The message is above.
  echo ============================================================
  echo.
  pause
)