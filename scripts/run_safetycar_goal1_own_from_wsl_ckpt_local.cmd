@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_safetycar_goal1_own_from_wsl_ckpt_local.ps1"
endlocal
