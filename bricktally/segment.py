"""Split a pile photo into piece blobs against a plain backdrop."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bricktally.lab import bgr_uint8_to_lab, delta_e


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    mean_lab: tuple[float, float, float]
    count: int
    touching: bool


def _border_median_bgr(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    band = max(4, int(min(height, width) * 0.06))
    strips = [
        image_bgr[:band, :],
        image_bgr[-band:, :],
        image_bgr[:, :band],
        image_bgr[:, -band:],
    ]
    pixels = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    return np.median(pixels, axis=0)


def foreground_mask(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None = None,
    threshold: float = 14.0,
) -> np.ndarray:
    lab = bgr_uint8_to_lab(image_bgr)
    if background_bgr is not None:
        if background_bgr.shape[:2] != image_bgr.shape[:2]:
            background_bgr = cv2.resize(
                background_bgr,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        bg_lab = bgr_uint8_to_lab(background_bgr)
        distance = delta_e(lab, bg_lab)
    else:
        bg_bgr = _border_median_bgr(image_bgr).reshape(1, 1, 3).astype(np.uint8)
        bg_lab = bgr_uint8_to_lab(bg_bgr)[0, 0]
        distance = delta_e(lab, bg_lab)
    mask = (distance > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _mean_lab(lab: np.ndarray, contour: np.ndarray) -> tuple[float, float, float]:
    mask = np.zeros(lab.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    pixels = lab[eroded > 0] if int((eroded > 0).sum()) > 20 else lab[mask > 0]
    if len(pixels) == 0:
        return 0.0, 0.0, 0.0
    mean = pixels.mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def _assign_counts(blobs: list[Blob], touch_ratio: float) -> None:
    if not blobs:
        return
    areas = sorted(blob.area for blob in blobs)
    seed = float(np.median(areas))
    singles = [area for area in areas if area <= seed * touch_ratio]
    unit = float(np.median(singles or areas))
    if unit <= 0:
        return
    for blob in blobs:
        if blob.area > unit * touch_ratio:
            blob.count = max(2, int(round(blob.area / unit)))
            blob.touching = True
        else:
            blob.count = 1
            blob.touching = False


def segment_pieces(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None = None,
    min_area: int = 400,
    touch_ratio: float = 1.55,
    threshold: float = 14.0,
) -> list[Blob]:
    mask = foreground_mask(image_bgr, background_bgr, threshold=threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lab = bgr_uint8_to_lab(image_bgr)
    blobs: list[Blob] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        blobs.append(
            Blob(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=area,
                mean_lab=_mean_lab(lab, contour),
                count=1,
                touching=False,
            )
        )
    _assign_counts(blobs, touch_ratio)
    blobs.sort(key=lambda blob: (blob.y, blob.x))
    return blobs
