# BrickTally

Count a bin of the same LEGO part, mixed colors, and save a BrickStore file.

Download the installer (this link stays the same):

https://github.com/NEXUS-Systems-Core/bricktally/releases/latest/download/BrickTallySetup.exe

The installer is unsigned. Windows may show a blue SmartScreen box the first time. Click More info, then Run anyway. Short steps are in docs/OPERATOR.md.

## Using it

1. Double-click BrickTally.
2. Pick the camera if the picture is wrong. The camera list is Camera 1 through Camera 6.
3. Put one piece on the table and click **Identify part**. Pick the matching part, or type the part number and click **Use this part**. If the guess is wrong, click that piece in the photo to try again.
4. Dump the bin on a plain mat. Click **Count this pile**.
5. Boxes that say "check" need a look. Click the box, pick the color, click **Save color sample**. The app remembers that under your lights.
6. Click **Add this pile to the list**. Repeat for the next bin. A different part is fine.
7. Click **Export BSX**. Open that file in BrickStore.

Set **New** or **Used** before you add a pile. That is the condition written into the file.

If the table is not a plain color, clear it and click **Empty table** once. If the photo looks too blue or too yellow, hold a grey card in the middle and click **Grey card**.

If a button says **Update available - Install now**, click it. The app closes, installs, and opens again. The old copy is kept next to the new one, in a folder ending in `.prev`. If you are offline, ignore it and keep counting.

## First-time setup on the laptop

The operator does not install Python. Download BrickTallySetup.exe from the link above and double-click it. It installs for that Windows user only, asks for no admin password, and puts a BrickTally icon on the desktop. Steps are in docs/OPERATOR.md.

## For the person who ships updates

Tag `vX.Y.Z` and push the tag. The release build writes that version into the app and the installer, then publishes the Windows zip, its SHA256, and BrickTallySetup.exe. The update button reads the zip and the SHA256. It will not install a zip whose checksum does not match.

The installer is not code-signed. Operators see SmartScreen until a certificate is added.

To build the app folder on a Windows machine without GitHub, run `build_windows.bat`. Compiling `installer/bricktally.iss` with Inno Setup produces BrickTallySetup.exe.
