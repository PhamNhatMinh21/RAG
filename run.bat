@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    py -3.12 run.py %*
) else (
    python run.py %*
)
if errorlevel 1 (
    echo.
    echo Startup failed. Install Python 3.12, enable "Add python.exe to PATH", and try again.
    pause
    exit /b 1
)
