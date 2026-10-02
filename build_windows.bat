@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1
python -m PyInstaller --noconfirm --windowed --onedir --name BrickTally ^
  --icon "installer\bricktally.ico" ^
  --collect-all PySide6 ^
  --collect-all cv2 ^
  --collect-all numpy ^
  --collect-all platformdirs ^
  --collect-submodules bricktally ^
  --collect-submodules bricktally_ui ^
  --hidden-import bricktally.selftest ^
  --add-data "bricktally\data\colors.json;bricktally\data" ^
  bricktally_ui\main.py
if errorlevel 1 exit /b 1
python -m PyInstaller --noconfirm --console --onefile --name BrickTallyUpdater ^
  --icon "installer\bricktally.ico" ^
  updater\updater.py
if errorlevel 1 exit /b 1
copy /Y dist\BrickTallyUpdater.exe dist\BrickTally\ >nul
if errorlevel 1 exit /b 1
echo Built dist\BrickTally\BrickTally.exe
