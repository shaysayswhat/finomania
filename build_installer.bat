@echo off
rem ==========================================================================
rem  Build the stand-alone Finomania application and Windows installer on this PC.
rem  Needs Python 3.10+ only for building; the result runs without Python.
rem    dist\Finomania\Finomania.exe          ready-to-run application folder
rem    installer\Finomania-x.y.z-Setup-win64.exe   installer (needs Inno Setup 6)
rem ==========================================================================
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")
if not exist ".venv\Scripts\python.exe" (
  echo Creating build environment...
  %PY% -m venv .venv || goto :nopython
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r requirements.txt "pyinstaller>=6.6" pillow || goto :fail
echo.
echo === Building executables (this takes a few minutes) ===
"%VPY%" -m PyInstaller --noconfirm --clean packaging\finomania.spec || goto :fail
echo.
echo === Self-test of the built application ===
"dist\Finomania\finomania-cli.exe" --selftest || goto :fail
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo Inno Setup 6 not found - installing it with winget ...
  winget install --id JRSoftware.InnoSetup -e --silent --accept-package-agreements --accept-source-agreements
  set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
  if not exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)
"%VPY%" -c "import finomania; print(finomania.__version__)" > "%TEMP%\fino_ver.txt"
set /p VER=<"%TEMP%\fino_ver.txt"
if exist "%ISCC%" (
  "%ISCC%" /DMyAppVersion=%VER% packaging\installer.iss || goto :fail
  echo.
  echo Installer created:  installer\Finomania-%VER%-Setup-win64.exe
) else (
  echo Inno Setup is not available, so no installer was made.
  echo The finished application is in dist\Finomania  - start Finomania.exe
)
pause
exit /b 0
:nopython
echo Python 3.10+ is needed to build. Get it from https://www.python.org/downloads/
:fail
echo Build failed - see the messages above.
pause
exit /b 1
