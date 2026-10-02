"""Synthetic pile photos: colored ellipses with a lighter stud on a grey mat."""

from __future__ import annotations

import cv2
import numpy as np

from bricktally.colors import Palette


def bgr_of(palette: Palette, color_id: int) -> tuple[int, int, int]:
    text = palette.by_id[color_id].hex.lstrip("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return blue, green, red


def paint_piece(
    image: np.ndarray,
    center: tuple[int, int],
    bgr: tuple[int, int, int],
    rx: int = 42,
    ry: int = 30,
) -> None:
    cv2.ellipse(image, center, (rx, ry), 0, 0, 360, bgr, thickness=-1)
    stud = tuple(min(255, channel + 36) for channel in bgr)
    cv2.circle(image, (center[0], center[1] - 6), 9, stud, thickness=-1)


def grey_mat(width: int = 900, height: int = 700, shade: int = 176) -> np.ndarray:
    return np.full((height, width, 3), shade, dtype=np.uint8)


def pile_image(palette: Palette) -> tuple[np.ndarray, dict[int, int]]:
    """Four separate pieces plus two overlapping reds. Totals are piece counts."""
    image = grey_mat()
    red = bgr_of(palette, 5)
    blue = bgr_of(palette, 7)
    yellow = bgr_of(palette, 3)
    black = bgr_of(palette, 11)
    paint_piece(image, (160, 180), red)
    paint_piece(image, (198, 180), red)  # overlaps the first red
    paint_piece(image, (420, 200), blue)
    paint_piece(image, (640, 220), blue)
    paint_piece(image, (240, 460), yellow)
    paint_piece(image, (560, 480), black)
    return image, {5: 2, 7: 2, 3: 1, 11: 1}
