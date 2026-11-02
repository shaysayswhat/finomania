@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m finomania %*
) else (
  start "" pythonw -m finomania %*
)
