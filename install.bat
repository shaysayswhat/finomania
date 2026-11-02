@echo off
setlocal
cd /d "%~dp0"
echo ==============================================
echo   Finomania installer
echo ==============================================
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
%PY% --version
if errorlevel 1 (
  echo Python 3.10 or newer is required. Download it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed - see the messages above.
  pause
  exit /b 1
)
echo.
echo Installed. Start Finomania with Finomania.bat
echo Optional extras:  .venv\Scripts\pip install torch      (GRU + faster LSTM)
echo                   .venv\Scripts\pip install yfinance
pause
