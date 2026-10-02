"""Launch the installed updater and quit. Missing helper is a message, not a crash."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def install_dir() -> Path | None:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def spawn_updater(zip_path: Path) -> str:
    folder = install_dir()
    if folder is None:
        return "Updates install only in the packaged app, not when running from source."
    helper = folder / "BrickTallyUpdater.exe"
    if not helper.is_file():
        return "The updater program is missing next to BrickTally. Download the zip by hand."
    flags = 0
    if os.name == "nt":
        flags = 0x00000008 | 0x00000200
    subprocess.Popen(
        [
            str(helper),
            "--pid",
            str(os.getpid()),
            "--zip",
            str(zip_path),
            "--install",
            str(folder),
            "--exe",
            Path(sys.executable).name,
        ],
        creationflags=flags,
        close_fds=True,
    )
    return ""
