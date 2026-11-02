@echo off
rem Run a Finomania program and write an HTML report next to it:   run_program.bat examples\01_volatility.fprg
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")
%PY% -m finomania "%~1" --html "%~dpn1.html"
pause
