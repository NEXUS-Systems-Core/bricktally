"""Split a pile photo into pieces, then colour each piece.

The mask finds pieces against the mat. Colour is taken from the interior of
each piece, not from every pixel. A region is split only when it clearly
holds two colours, or a real joint line crosses it. Area division is a
fallback for a low-solidity clump, and it is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from bricktally.colors import Palette
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
    contour: np.ndarray | None = None
    low_confidence: bool = False
    notes: str = field(default="", repr=False)


def foreground_mask(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None = None,
    threshold: float = 14.0,
) -> np.ndarray:
    if background_bgr is not None:
        return _difference_mask(image_bgr, background_bgr, threshold)
    mask, _mat = _photo_mask(image_bgr)
    return mask


def segment_pieces(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None = None,
    min_area: int = 400,
    touch_ratio: float = 1.55,
    threshold: float = 14.0,
    palette: Palette | None = None,
    piece_area: int | None = None,
) -> list[Blob]:
    del palette  # matching happens after the pieces exist
    if image_bgr is None or image_bgr.size == 0:
        return []
    smooth = cv2.bilateralFilter(image_bgr, 7, 35, 35)
    lab = bgr_uint8_to_lab(smooth)
    if background_bgr is not None:
        if background_bgr.shape[:2] != smooth.shape[:2]:
            background_bgr = cv2.resize(
                background_bgr,
                (smooth.shape[1], smooth.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        mask = _difference_mask(smooth, background_bgr, threshold)
        mat = np.median(bgr_uint8_to_lab(background_bgr).reshape(-1, 3), axis=0)
        scale = 1.0
    else:
        mask, mat = _photo_mask(smooth)
        scale = _exposure_scale(smooth)
    brick_width, unit_area = _estimate_size(mask, min_area)
    if piece_area is not None and piece_area > 0:
        unit_area = float(piece_area)
        brick_width = max(brick_width, (unit_area / 1.8) ** 0.5)
    match_lab = lab
    if scale > 1.05:
        scaled = np.clip(smooth.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        match_lab = bgr_uint8_to_lab(scaled)
    return _blobs_from_mask(
        smooth, lab, match_lab, mask, mat, min_area, touch_ratio, brick_width, unit_area
    )


def _difference_mask(image_bgr: np.ndarray, background_bgr: np.ndarray, threshold: float) -> np.ndarray:
    distance = delta_e(bgr_uint8_to_lab(image_bgr), bgr_uint8_to_lab(background_bgr))
    mask = (distance > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return _fill_small_holes(mask, 600)


def _photo_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pieces against an uneven mat.

    A global threshold swallows the brighter middle of the mat. The local
    median is taken over a window larger than one brick, so a piece is a
    deviation from the mat next to it, not from the dark border.
    """

    lab = bgr_uint8_to_lab(image_bgr)
    lightness = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    height, width = lightness.shape
    ksize = _odd(int(min(height, width) * 0.11), 31, 121)
    local_l = cv2.medianBlur(np.clip(lightness * 2.55, 0, 255).astype(np.uint8), ksize)
    local_l = local_l.astype(np.float64) / 2.55
    local_c = cv2.medianBlur(np.clip(chroma * 3.0, 0, 255).astype(np.uint8), ksize)
    local_c = local_c.astype(np.float64) / 3.0
    delta = np.abs(lightness - local_l)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    mat = _mat_from_border(lab, chroma)
    # Flat black or white is a piece even when the local window is also that
    # piece. Without this, the interior is thrown out as mat and only the
    # edge remains.
    dark_piece = lightness < float(mat[0]) - 8.0
    light_piece = lightness > float(mat[0]) + 8.0
    matish = (gradient < 12) & (chroma < 8.0) & (delta < 14.0) & ~dark_piece & ~light_piece
    raw = ((delta > 8.0) | ((chroma - local_c) > 6.0) | dark_piece) & ~matish
    mask = raw.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = _fill_small_holes(mask, 700)
    return mask, _mat_from_border(lab, chroma)


def _odd(value: int, low: int, high: int) -> int:
    value = int(np.clip(value, low, high))
    if value % 2 == 0:
        value += 1
    return int(np.clip(value, low, high))


def _cut_internal_edges(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Open the dark joint between touching bricks. Leave the outer silhouette."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 25, 25)
    edges = cv2.Canny(blur, 35, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    barriers = cv2.bitwise_and(edges, inner)
    cut = mask.copy()
    cut[barriers > 0] = 0
    return cv2.morphologyEx(cut, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _mat_from_border(lab: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    height, width = chroma.shape
    band = max(8, int(min(height, width) * 0.04))
    border = np.zeros(chroma.shape, dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    sample = lab[border & (chroma < 9.0)]
    if len(sample) < 50:
        sample = lab[border]
    return np.median(sample, axis=0)


def _exposure_scale(image_bgr: np.ndarray) -> float:
    """Lift a dark webcam frame. A bright synthetic mat is left alone."""

    height, width = image_bgr.shape[:2]
    band = max(8, int(min(height, width) * 0.04))
    border = np.concatenate(
        [
            image_bgr[:band].reshape(-1, 3),
            image_bgr[-band:].reshape(-1, 3),
            image_bgr[:, :band].reshape(-1, 3),
            image_bgr[:, -band:].reshape(-1, 3),
        ]
    )
    median = float(np.median(border))
    if median < 8.0:
        return 1.0
    scale = 176.0 / median
    if scale < 1.05:
        return 1.0
    return float(min(scale, 1.65))


def _enclosed_pieces(image_bgr: np.ndarray, lab: np.ndarray, mat: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    flood = cv2.bitwise_not(edges)
    height, width = flood.shape
    ff_mask = np.zeros((height + 2, width + 2), np.uint8)
    for x in range(0, width, 10):
        if flood[1, x] == 255:
            cv2.floodFill(flood, ff_mask, (int(x), 1), 128)
        if flood[height - 2, x] == 255:
            cv2.floodFill(flood, ff_mask, (int(x), height - 2), 128)
    for y in range(0, height, 10):
        if flood[y, 1] == 255:
            cv2.floodFill(flood, ff_mask, (1, int(y)), 128)
        if flood[y, width - 2] == 255:
            cv2.floodFill(flood, ff_mask, (width - 2, int(y)), 128)
    enclosed = ((flood != 128) & (edges == 0)).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, 8)
    keep = np.zeros_like(enclosed)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 700 or area > 16000:
            continue
        region = (labels == index).astype(np.uint8) * 255
        if _is_mat(lab, region, mat) or _solidity(region) < 0.72:
            continue
        keep[region > 0] = 255
    if not np.any(keep):
        return None
    return keep


def _estimate_size(mask: np.ndarray, min_area: int) -> tuple[float, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    widths: list[float] = []
    areas: list[int] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < max(min_area, 700) or area > 16000:
            continue
        x, y, w, h = (int(stats[index, key]) for key in range(4))
        region = np.zeros((h, w), np.uint8)
        region[labels[y : y + h, x : x + w] == index] = 255
        if _solidity(region) < 0.78:
            continue
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        widths.append(2.0 * float(distance.max()))
        areas.append(area)
    if len(widths) < 3:
        return 48.0, 4000.0
    return float(np.median(widths)), float(np.median(areas))


def _blobs_from_mask(
    image_bgr: np.ndarray,
    lab: np.ndarray,
    match_lab: np.ndarray,
    mask: np.ndarray,
    mat: np.ndarray,
    min_area: int,
    touch_ratio: float,
    brick_width: float,
    unit_area: float,
) -> list[Blob]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions: list[np.ndarray] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > 0.25 * mask.size:
            continue
        region = np.zeros(mask.shape, np.uint8)
        region[labels == index] = 255
        if _is_mat(lab, region, mat):
            continue
        regions.extend(_separate(region, image_bgr, lab, brick_width, unit_area))
    blobs: list[Blob] = []
    for region in regions:
        if _is_mat(lab, region, mat):
            continue
        blob = _blob_from_region(
            region, match_lab, lab, brick_width, unit_area, min_area, touch_ratio
        )
        if blob is not None:
            blobs.append(blob)
    blobs.sort(key=lambda blob: (blob.y, blob.x))
    return blobs


def _separate(
    region: np.ndarray,
    image_bgr: np.ndarray,
    lab: np.ndarray,
    brick_width: float,
    unit_area: float,
) -> list[np.ndarray]:
    area = int(np.count_nonzero(region))
    if area < unit_area * 1.25:
        return [region]
    return _split_by_colour(lab, region, unit_area)


def _split_by_colour(lab: np.ndarray, region: np.ndarray, unit_area: float, depth: int = 0) -> list[np.ndarray]:
    """Split a region only when it holds two colours, each big enough to be a piece."""

    area = int(np.count_nonzero(region))
    if area < unit_area * 0.90 or depth > 6:
        return [region]
    ys, xs = np.where(region > 0)
    pixels = lab[ys, xs].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 16, 0.8)
    try:
        _compact, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return [region]
    labels = labels.reshape(-1)
    counts = [int(np.count_nonzero(labels == cluster)) for cluster in (0, 1)]
    if min(counts) < 0.28 * unit_area:
        return [region]
    if not _distinct_colours(centers[0], centers[1]):
        return [region]
    parts: list[np.ndarray] = []
    kernel = np.ones((3, 3), np.uint8)
    for cluster in (0, 1):
        part = np.zeros(region.shape, np.uint8)
        part[ys[labels == cluster], xs[labels == cluster]] = 255
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, kernel)
        part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, kernel)
        count, comp, stats, _ = cv2.connectedComponentsWithStats(part, 8)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) < 0.30 * unit_area:
                continue
            one = np.zeros(region.shape, np.uint8)
            one[comp == index] = 255
            parts.extend(_split_by_colour(lab, one, unit_area, depth + 1))
    return parts if len(parts) >= 2 else [region]


def _distinct_colours(left: np.ndarray, right: np.ndarray) -> bool:
    """Shade on one brick is not a second colour. A real hue change is."""

    distance = float(np.linalg.norm(left - right))
    chroma_left = float(np.hypot(left[1], left[2]))
    chroma_right = float(np.hypot(right[1], right[2]))
    if chroma_left < 12.0 and chroma_right < 12.0:
        return abs(float(left[0] - right[0])) >= 16.0
    hue_left = np.arctan2(left[2], left[1])
    hue_right = np.arctan2(right[2], right[1])
    hue = abs(float(hue_left - hue_right))
    if hue > np.pi:
        hue = 2 * np.pi - hue
    if np.degrees(hue) >= 18.0 and distance >= 14.0:
        return True
    return distance >= 22.0


def _split_on_edges(
    image_bgr: np.ndarray,
    region: np.ndarray,
    brick_width: float,
    unit_area: float,
) -> list[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 5, 25, 25)
    edges = cv2.Canny(blur, 45, 130)
    inner = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edges[inner == 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
    min_span = max(12, int(round(0.62 * brick_width)))
    barriers = np.zeros(region.shape, np.uint8)
    for index in range(1, count):
        span = max(int(stats[index, cv2.CC_STAT_WIDTH]), int(stats[index, cv2.CC_STAT_HEIGHT]))
        if span >= min_span and int(stats[index, cv2.CC_STAT_AREA]) >= min_span // 2:
            barriers[labels == index] = 255
    if not np.any(barriers):
        return [region]
    barriers = cv2.dilate(barriers, np.ones((3, 3), np.uint8))
    cut = region.copy()
    cut[barriers > 0] = 0
    cut = cv2.morphologyEx(cut, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cut, 8)
    floor = 0.32 * unit_area
    parts: list[np.ndarray] = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < floor:
            continue
        part = np.zeros(region.shape, np.uint8)
        part[labels == index] = 255
        parts.append(part)
    if len(parts) < 2:
        return [region]
    if any(_minor_width(part) < 0.55 * brick_width for part in parts):
        return [region]
    return parts


def _single_brick(region: np.ndarray, brick_width: float) -> bool:
    """A long rectangle is one brick, not a row of pieces to divide by area."""

    if _solidity(region) < 0.86:
        return False
    minor = _minor_width(region)
    return 0.65 * brick_width <= minor <= 2.4 * brick_width


def _is_mat(lab: np.ndarray, region: np.ndarray, mat: np.ndarray) -> bool:
    inner = cv2.erode(region, np.ones((3, 3), np.uint8))
    if int(np.count_nonzero(inner)) < 20:
        inner = region
    pixels = lab[inner > 0]
    if len(pixels) < 8:
        return False
    mean = np.median(pixels, axis=0)
    chroma = float(np.hypot(mean[1], mean[2]))
    return float(delta_e(mean, mat)) < 9.0 and chroma < 10.0


def _blob_from_region(
    region: np.ndarray,
    match_lab: np.ndarray,
    lab: np.ndarray,
    brick_width: float,
    unit_area: float,
    min_area: int,
    touch_ratio: float,
) -> Blob | None:
    area = int(np.count_nonzero(region))
    if area < min_area:
        return None
    mean = _interior_lab(match_lab, region, brick_width)
    if mean is None:
        mean = _interior_lab(lab, region, brick_width)
    if mean is None:
        return None
    ys, xs = np.where(region > 0)
    x, y = int(xs.min()), int(ys.min())
    w = int(xs.max() - x + 1)
    h = int(ys.max() - y + 1)
    count = 1
    touching = False
    note = ""
    if area > unit_area * touch_ratio and not _single_brick(region, brick_width):
        count = max(2, int(round(area / unit_area)))
        touching = True
        note = "area"
    return Blob(
        x=x,
        y=y,
        w=w,
        h=h,
        area=area,
        mean_lab=(float(mean[0]), float(mean[1]), float(mean[2])),
        count=count,
        touching=touching,
        contour=_contour_of(region, (0, 0)),
        low_confidence=touching,
        notes=note,
    )


def _interior_lab(lab: np.ndarray, region: np.ndarray, brick_width: float) -> np.ndarray | None:
    radius = max(2, int(round(0.16 * brick_width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    inner = cv2.erode(region, kernel)
    if int(np.count_nonzero(inner)) < 20:
        inner = region
    pixels = lab[inner > 0]
    if len(pixels) < 8:
        return None
    lightness = pixels[:, 0]
    low, high = np.percentile(lightness, [16, 93])
    kept = pixels[(lightness >= low) & (lightness <= high)]
    if len(kept) >= 8:
        pixels = kept
    return np.median(pixels, axis=0)


def _minor_width(region: np.ndarray) -> float:
    distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    return 2.0 * float(distance.max())


def _solidity(region: np.ndarray) -> float:
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(contour))
    area = float(np.count_nonzero(region))
    if hull <= 1.0:
        return 1.0
    return area / hull


def _fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    inverse = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    filled = mask.copy()
    height, width = mask.shape
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x, y, w, h = (int(stats[index, key]) for key in range(4))
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        if area <= max_hole:
            filled[labels == index] = 255
    return filled


def _contour_of(region: np.ndarray, origin: tuple[int, int]) -> np.ndarray | None:
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).copy()
    contour[:, 0, 0] += origin[0]
    contour[:, 0, 1] += origin[1]
    return contour
