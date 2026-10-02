# BrickTally

Count a bin of the same LEGO part, mixed colors, and save a BrickStore file.

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

Someone who can install Python does this once, on the Windows laptop:

1. Install Python 3.12 from python.org. Tick "Add python to PATH".
2. Copy this folder to the laptop.
3. Double-click `build_windows.bat`. It builds `dist\BrickTally\BrickTally.exe`.
4. Copy that whole `dist\BrickTally` folder wherever the operator will run it. They double-click `BrickTally.exe`.

After that, the operator does not need Python. Updates come from the button.

## For the person who ships updates

The update button does nothing until `UPDATE_REPO` in `bricktally/config.py` is a real `owner/repo`. Create that GitHub repo, push this folder, then:

1. Bump `__version__` in `bricktally/config.py` (for example `0.1.1`).
2. Commit, tag `v0.1.1`, and push the tag.
3. GitHub Actions builds the Windows zip, writes a SHA256 file, and publishes the release.

The app compares that tag to its own version. It will not install a zip whose SHA256 does not match.

To build on the laptop without GitHub, run `build_windows.bat`.
