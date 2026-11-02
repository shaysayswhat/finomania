@echo off
rem Finomania with a console window (shows error tracebacks) or the text command prompt with --cli
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
%PY% -m finomania %*
if errorlevel 1 pause
