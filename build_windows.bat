@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1
python -m PyInstaller --noconfirm --windowed --onedir --name BrickTally ^
  --add-data "bricktally\data\colors.json;bricktally\data" ^
  bricktally_ui\main.py
if errorlevel 1 exit /b 1
python -m PyInstaller --noconfirm --console --onefile --name BrickTallyUpdater updater\updater.py
if errorlevel 1 exit /b 1
copy /Y dist\BrickTallyUpdater.exe dist\BrickTally\ >nul
echo Built dist\BrickTally\BrickTally.exe
