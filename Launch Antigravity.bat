@echo off
title A+Antigravity Launcher
setlocal

set BACKEND=%~dp0
set FRONTEND=%~dp0..\pixel-perfect-replica

echo.
echo  ============================================
echo   A+Antigravity
echo  ============================================
echo.

:: ── First-time setup: Python venv ─────────────────────────────────────────
if not exist "%BACKEND%.venv\Scripts\python.exe" (
    echo [SETUP] Installing Python environment (first time only, ~2 min)...
    python -m venv "%BACKEND%.venv"
    "%BACKEND%.venv\Scripts\pip.exe" install -r "%BACKEND%requirements.txt" --quiet
    echo [SETUP] Python environment ready.
    echo.
)

:: ── First-time setup: Node modules ────────────────────────────────────────
if not exist "%FRONTEND%\node_modules" (
    echo [SETUP] Installing app dependencies (first time only, ~2 min)...
    cd /d "%FRONTEND%"
    npm install --silent
    echo [SETUP] App dependencies ready.
    echo.
)

:: ── Launch ─────────────────────────────────────────────────────────────────
echo [1/2] Starting analytics engine...
start "Antigravity - Engine" cmd /k "cd /d "%BACKEND%" && .venv\Scripts\python.exe -m uvicorn api_server:app --port 8502"

timeout /t 3 /nobreak >nul

echo [2/2] Starting app interface...
start "Antigravity - App" cmd /k "cd /d "%FRONTEND%" && npm run dev"

timeout /t 5 /nobreak >nul

echo Opening browser...
start http://localhost:8080

echo.
echo  App is running at http://localhost:8080
echo  Close the two black windows to stop the app.
echo.
