"""Camera open helper. DirectShow on Windows, default backend elsewhere."""

from __future__ import annotations

import os

import cv2


def capture_backend() -> int:
    if os.name == "nt":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camera(index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(index), capture_backend())
