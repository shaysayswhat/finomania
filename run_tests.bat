@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
%PY% -m pip install pytest
%PY% -m pytest -q tests
pause
