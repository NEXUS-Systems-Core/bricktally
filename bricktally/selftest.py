"""Bundle check: imports, colors.json, one synthetic count. No window, no camera."""

from __future__ import annotations

import os
import sys


def _emit(text: str) -> None:
    data = (text + "\n").encode("utf-8", errors="replace")
    if sys.platform == "win32":
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            invalid = {0, -1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
            if handle not in invalid:
                written = ctypes.c_ulong(0)
                kernel.WriteFile(handle, data, len(data), ctypes.byref(written), None)
        except Exception:
            pass
    for fd in (1, 2):
        try:
            os.write(fd, data)
        except OSError:
            pass
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        if stream is None:
            continue
        try:
            stream.write(text + "\n")
            stream.flush()
        except Exception:
            pass


def run_selftest() -> int:
    try:
        return _run()
    except Exception as exc:
        _emit(f"FAIL {exc}")
        return 1


def _run() -> int:
    import cv2
    import numpy as np
    import platformdirs
    from PySide6 import QtCore, QtGui, QtWidgets

    import bricktally
    import bricktally.apply_update
    import bricktally.bsx
    import bricktally.camera
    import bricktally.colors
    import bricktally.count
    import bricktally.segment
    import bricktally.settings
    import bricktally.updates
    import bricktally_ui.window

    _ = (
        bricktally.__version__,
        bricktally.apply_update.install_dir,
        bricktally.bsx.to_bsx,
        bricktally.camera.open_camera,
        bricktally.segment.segment_pieces,
        bricktally.settings.default_data_dir,
        bricktally.updates.check_for_update,
        bricktally_ui.window.MainWindow,
        QtCore.qVersion(),
        QtGui.QImage,
        platformdirs.user_data_dir("BrickTally", "EricManno"),
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    palette = bricktally.colors.Palette.load()
    color = palette.by_id[5]
    text = color.hex.lstrip("#")
    bgr = (int(text[4:6], 16), int(text[2:4], 16), int(text[0:2], 16))
    image = np.full((400, 520, 3), 176, dtype=np.uint8)
    cv2.ellipse(image, (180, 170), (42, 30), 0, 0, 360, bgr, thickness=-1)
    result = bricktally.count.count_image(image, palette, min_area=400)
    total = sum(piece.count for piece in result.pieces)
    if total < 1:
        _emit(f"FAIL count={total}")
        return 1
    _emit("OK")
    try:
        from pathlib import Path

        Path.cwd().joinpath("bricktally-selftest.txt").write_text("OK\n", encoding="utf-8")
    except OSError:
        pass
    return 0
