"""Split a pile photo into piece blobs against a plain backdrop.

Colour is classified per pixel in CIELAB, so a mixed-colour clump is not one
blob. Touching pieces of the same colour are split with a distance-transform
watershed when the piece size can be estimated. Area division is only a
fallback, and it is flagged.
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
    # Internal, not part of the public count. Kept so a caller can see why.
    notes: str = field(default="", repr=False)


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


def _global_mask(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None,
    threshold: float,
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


def _odd_kernel(value: int, low: int, high: int) -> int:
    value = int(np.clip(value, low, high))
    if value % 2 == 0:
        value += 1
    return int(np.clip(value, low, high))


def _local_contrast_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Foreground when uneven light makes a global threshold swallow the mat.

    The kernel is a fraction of the frame and larger than a piece, so the
    local median is the mat and piece interiors stay foreground. Smooth grey
    pixels are the vignette, not a piece, and are removed. Later thresholds
    come from the estimated piece size, not from this kernel.
    """
    lab = bgr_uint8_to_lab(image_bgr)
    lightness = lab[:, :, 0]
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    height, width = lightness.shape
    ksize = _odd_kernel(int(min(height, width) * 0.11), 31, 121)
    local_l = cv2.medianBlur(np.clip(lightness * 2.55, 0, 255).astype(np.uint8), ksize)
    local_l = local_l.astype(np.float64) / 2.55
    local_c = cv2.medianBlur(np.clip(chroma * 3.0, 0, 255).astype(np.uint8), ksize)
    local_c = local_c.astype(np.float64) / 3.0
    delta_l = np.abs(lightness - local_l)
    raw = (delta_l > 7.0) | ((chroma - local_c) > 6.0)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    matish = (gradient < 10) & (chroma < 8.0) & (delta_l < 16.0)
    mask = (raw & ~matish).astype(np.uint8) * 255
    # Stud highlights, not the gaps between pieces.
    mask = _fill_small_holes(mask, max_hole=max(400, int(0.0015 * mask.size)))
    opened = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opened)
    return mask


def _largest_component_area(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def foreground_mask(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None = None,
    threshold: float = 14.0,
) -> np.ndarray:
    mask = _global_mask(image_bgr, background_bgr, threshold)
    if background_bgr is not None:
        return mask
    image_area = int(mask.shape[0] * mask.shape[1])
    # A blob covering a large share of the frame is the mat leaking through
    # uneven light, not a pile. Fall back to local contrast.
    if _largest_component_area(mask) > 0.12 * image_area:
        return _local_contrast_mask(image_bgr)
    return mask


def _mat_lab(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    background = mask == 0
    if int(background.sum()) < 50:
        return np.median(lab.reshape(-1, 3), axis=0)
    return np.median(lab[background], axis=0)


def _estimate_unit_area(mask: np.ndarray, min_area: int, piece_area: int | None) -> float:
    if piece_area is not None and piece_area > 0:
        return float(piece_area)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    convex: list[int] = []
    areas: list[int] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        areas.append(area)
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        crop = (labels[y : y + h, x : x + w] == index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0:
            continue
        if area / hull_area >= 0.84:
            convex.append(area)
    if len(convex) >= 3:
        convex.sort()
        # The upper tail is clumps that happened to look convex. Keep the body.
        cut = max(3, int(round(len(convex) * 0.75)))
        return float(np.median(convex[:cut]))
    if not areas:
        return float(max(min_area * 4, 600))
    areas.sort()
    body = areas[: max(1, int(round(len(areas) * 0.6)))]
    return float(np.median(body))


def _smooth_lab(lab: np.ndarray) -> np.ndarray:
    """Bilateral smooth so a stud highlight does not open its own cluster."""
    encoded = np.clip(
        np.stack((lab[:, :, 0] * 2.55, lab[:, :, 1] + 128.0, lab[:, :, 2] + 128.0), axis=-1),
        0,
        255,
    ).astype(np.uint8)
    blurred = cv2.bilateralFilter(encoded, 9, 28, 28)
    out = np.empty_like(lab)
    out[:, :, 0] = blurred[:, :, 0].astype(np.float64) / 2.55
    out[:, :, 1] = blurred[:, :, 1].astype(np.float64) - 128.0
    out[:, :, 2] = blurred[:, :, 2].astype(np.float64) - 128.0
    return out


def _cluster_count(pixels: np.ndarray) -> int:
    """How many colours are actually in the pile, not a fixed image scale.

    Stops when a new cluster barely reduces the error, and stays in a range
    that can separate a mixed pile without splitting one brick into specks.
    """
    if len(pixels) < 80:
        return 2
    rng = np.random.default_rng(7)
    take = min(len(pixels), 2500)
    sample = pixels[rng.choice(len(pixels), size=take, replace=False)]
    chosen = 4
    previous = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.8)
    for k in range(4, 13):
        cv2.setRNGSeed(7)
        compact, _, _ = cv2.kmeans(sample, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        if previous is not None and compact > previous * 0.90:
            return max(4, k - 1)
        previous = compact
        chosen = k
    return chosen


def _color_labels(
    lab: np.ndarray,
    mask: np.ndarray,
    unit_area: float,
    palette: Palette,
    mat: np.ndarray,
) -> np.ndarray:
    """Cluster interior pixels in CIELAB, then name each cluster from the palette.

    Edges, glare and mat-like pixels stay unlabeled so they cannot open a
    phantom colour or bridge two pieces. A small close fills stud holes
    inside a colour without crossing into a different one.
    """
    del unit_area
    smooth = _smooth_lab(lab)
    foreground = mask > 0
    lightness = smooth[:, :, 0]
    chroma = np.hypot(smooth[:, :, 1], smooth[:, :, 2])
    highlight = (lightness > 90.0) & (chroma < 16.0)
    mat_like = (delta_e(smooth, mat) < 10.0) & (chroma < 9.0)
    usable = foreground & ~highlight & ~mat_like
    labels = np.zeros(mask.shape, dtype=np.uint16)
    points = np.argwhere(usable)
    if len(points) < 40:
        return labels
    pixels = smooth[points[:, 0], points[:, 1]].astype(np.float32)
    k = _cluster_count(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.4)
    cv2.setRNGSeed(7)
    _, assigned, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    refs = np.stack([palette.reference_lab(color) for color in palette.colors]).astype(np.float64)
    ids = np.asarray([color.id for color in palette.colors], dtype=np.uint16)
    # Lightness down-weighted so a shaded cluster keeps the hue family's id.
    scale = np.array([0.55, 1.0, 1.0], dtype=np.float64)
    named = np.empty(k, dtype=np.uint16)
    for index, center in enumerate(centers):
        diff = (refs - center.astype(np.float64)) * scale
        named[index] = ids[int(np.linalg.norm(diff, axis=1).argmin())]
    labels[points[:, 0], points[:, 1]] = named[assigned.ravel()]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grow_into = foreground & ~mat_like & (labels == 0)
    for color_id in np.unique(labels):
        if color_id == 0:
            continue
        region = (labels == color_id).astype(np.uint8) * 255
        closed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)
        fill = grow_into & (closed > 0)
        labels[fill] = color_id
        grow_into[fill] = False
    return labels


def _touching_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    left = labels[:, :-1]
    right = labels[:, 1:]
    horizontal = (left > 0) & (right > 0) & (left != right)
    if np.any(horizontal):
        a = left[horizontal].astype(np.int32)
        b = right[horizontal].astype(np.int32)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pairs.update(zip(lo.tolist(), hi.tolist()))
    up = labels[:-1, :]
    down = labels[1:, :]
    vertical = (up > 0) & (down > 0) & (up != down)
    if np.any(vertical):
        a = up[vertical].astype(np.int32)
        b = down[vertical].astype(np.int32)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pairs.update(zip(lo.tolist(), hi.tolist()))
    return pairs


def _hue_delta(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    hue_a = np.arctan2(lab_a[2], lab_a[1])
    hue_b = np.arctan2(lab_b[2], lab_b[1])
    delta = abs(float(hue_a - hue_b))
    if delta > np.pi:
        delta = 2 * np.pi - delta
    return float(np.degrees(delta))


def _similar(lab_a: np.ndarray, lab_b: np.ndarray) -> bool:
    chroma_a = float(np.hypot(lab_a[1], lab_a[2]))
    chroma_b = float(np.hypot(lab_b[1], lab_b[2]))
    distance = float(delta_e(lab_a, lab_b))
    if chroma_a < 12.0 and chroma_b < 12.0:
        return abs(float(lab_a[0] - lab_b[0])) < 16.0 and distance < 22.0
    # A bright low-chroma speck against a coloured piece is glare, not a colour.
    if (chroma_a < 10.0 and lab_a[0] > 82.0) or (chroma_b < 10.0 and lab_b[0] > 82.0):
        return True
    if distance > 26.0:
        return False
    return _hue_delta(lab_a, lab_b) < 18.0 and abs(float(lab_a[0] - lab_b[0])) < 24.0


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def _region_mean(lab: np.ndarray, region: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8))
    use = eroded if int((eroded > 0).sum()) >= 25 else region
    pixels = lab[use > 0]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float64)
    chroma = np.hypot(pixels[:, 1], pixels[:, 2])
    keep = ~((pixels[:, 0] > 90.0) & (chroma < 16.0))
    if int(keep.sum()) >= 15:
        pixels = pixels[keep]
    return np.median(pixels, axis=0)


def _fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill enclosed holes up to a stud, not the gaps between pieces."""
    inverse = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    filled = mask.copy()
    height, width = mask.shape[:2]
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        if area <= max_hole:
            filled[labels == index] = 255
    return filled


def _fill_holes(region: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def _watershed_masks(region: np.ndarray, min_distance: int, unit_area: float) -> list[np.ndarray]:
    filled = _fill_small_holes(region, max_hole=int(max(300, 0.45 * unit_area)))
    if int((filled > 0).sum()) < 30:
        return [region]
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    radius = float(dist.max())
    if radius < 3.5:
        return [filled]
    kernel = _odd_kernel(min_distance, 3, 61)
    dilated = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    expected = 0.45 * np.sqrt(max(unit_area, 1.0) / np.pi)
    peak_floor = max(2.5, expected * 0.55)
    peaks = (dist >= dilated - 0.05) & (dist >= peak_floor) & (filled > 0)
    peak_count, peak_labels = cv2.connectedComponents(peaks.astype(np.uint8))
    if peak_count <= 2:
        return [filled]
    markers = np.zeros(filled.shape, dtype=np.int32)
    markers[peaks] = peak_labels[peaks]
    height = np.zeros(filled.shape, dtype=np.uint8)
    inside = filled > 0
    height[inside] = np.clip(255.0 - dist[inside] * (220.0 / (radius + 1e-6)), 0, 255).astype(np.uint8)
    markers_in = markers.copy()
    cv2.watershed(cv2.cvtColor(height, cv2.COLOR_GRAY2BGR), markers_in)
    masks: list[np.ndarray] = []
    for label in range(1, int(markers_in.max()) + 1):
        piece = np.zeros(filled.shape, dtype=np.uint8)
        piece[markers_in == label] = 255
        piece[filled == 0] = 0
        if int(piece.sum() // 255) < 20:
            continue
        masks.append(piece)
    if len(masks) <= 1:
        return [filled]
    # Reject a split that chews one piece into slivers. Keep only parts that
    # are at least a large fraction of one piece; fold the rest back.
    floor = 0.42 * unit_area
    solid = [item for item in masks if int((item > 0).sum()) >= floor]
    scraps = [item for item in masks if int((item > 0).sum()) < floor]
    if len(solid) <= 1:
        return [filled]
    if scraps and solid:
        # Scraps stay with the split rather than being dropped. They are
        # assigned to the nearest solid mask by dilation below, in the caller
        # we just return the solid pieces plus scraps that are still big
        # enough to be a partial piece.
        partial_floor = 0.22 * unit_area
        solid.extend(item for item in scraps if int((item > 0).sum()) >= partial_floor)
    return solid


def _contour_of(region: np.ndarray, origin: tuple[int, int]) -> np.ndarray | None:
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = contour.copy()
    contour[:, 0, 0] += origin[0]
    contour[:, 0, 1] += origin[1]
    return contour


def _refine_unit(regions: list[np.ndarray], min_area: int, fallback: float) -> float:
    """Piece size from isolated convex regions, not from a mixed clump."""
    convex: list[int] = []
    areas: list[int] = []
    for region in regions:
        area = int(np.count_nonzero(region))
        if area < max(min_area // 2, 80):
            continue
        areas.append(area)
        contours, _ = cv2.findContours(
            (region > 0).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull <= 0:
            continue
        if area / hull >= 0.82:
            convex.append(area)
    pool = convex if len(convex) >= 3 else areas
    if not pool:
        return float(fallback)
    pool.sort()
    # Drop the upper tail: those are clumps that still look fairly solid.
    cut = max(1, int(round(len(pool) * 0.7)))
    return float(np.median(pool[:cut]))


def _components_by_class(color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Connected components inside each colour class.

    cv2.connectedComponents treats every non-zero pixel as one foreground, so
    it cannot be called on the class image directly.
    """
    labels = np.zeros(color.shape, dtype=np.int32)
    areas: list[int] = [0]
    next_id = 1
    classes = np.unique(color)
    for class_id in classes:
        if int(class_id) == 0:
            continue
        binary = (color == class_id).astype(np.uint8)
        count, local, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for index in range(1, count):
            labels[local == index] = next_id
            areas.append(int(stats[index, cv2.CC_STAT_AREA]))
            next_id += 1
    return labels, np.asarray(areas, dtype=np.int32)


def _split_mask_to_blobs(
    image_lab: np.ndarray,
    mask: np.ndarray,
    unit_area: float,
    min_area: int,
    touch_ratio: float,
    mat: np.ndarray,
    palette: Palette,
    unit_locked: bool = False,
) -> list[Blob]:
    color = _color_labels(image_lab, mask, unit_area, palette, mat)
    labels, areas = _components_by_class(color)
    count = int(len(areas))
    if count <= 1:
        return []

    means: dict[int, np.ndarray] = {}
    keep: list[int] = []
    noise_floor = max(80, int(0.00012 * mask.size))
    for index in range(1, count):
        area = int(areas[index])
        if area < noise_floor:
            continue
        region = (labels == index).astype(np.uint8) * 255
        means[index] = _region_mean(image_lab, region)
        keep.append(index)
    if not keep:
        return []

    # Merge adjacent regions that are the same colour under shading, studs,
    # or a highlight. Do not merge across a real hue change.
    parent = _UnionFind(count)
    for left, right in _touching_pairs(labels):
        if left not in means or right not in means:
            continue
        if _similar(means[left], means[right]):
            parent.union(left, right)

    merged: dict[int, np.ndarray] = {}
    for index in keep:
        root = parent.find(index)
        region = labels == index
        if root not in merged:
            merged[root] = region.copy()
        else:
            merged[root] |= region

    if not unit_locked:
        unit_area = _refine_unit(list(merged.values()), min_area, unit_area)
    expected_radius = 0.5 * np.sqrt(max(unit_area, 1.0) / np.pi)
    min_distance = max(8, int(round(0.85 * np.sqrt(unit_area))))
    # Thin side-walls and shadow fringes get absorbed into the neighbour they
    # share the longest border with, instead of becoming a phantom piece.
    thin_ids = []
    for root, region_bool in list(merged.items()):
        region = region_bool.astype(np.uint8) * 255
        dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        if float(dist.max()) < max(2.5, 0.42 * expected_radius):
            thin_ids.append(root)
    if thin_ids and len(merged) > len(thin_ids):
        label_img = np.zeros(mask.shape, dtype=np.int32)
        for root, region_bool in merged.items():
            label_img[region_bool] = root
        for root in thin_ids:
            region_bool = merged[root]
            dilated = cv2.dilate(region_bool.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            neighbours = label_img[dilated & ~region_bool]
            neighbours = neighbours[neighbours > 0]
            neighbours = neighbours[neighbours != root]
            if len(neighbours) == 0:
                continue
            values, counts = np.unique(neighbours, return_counts=True)
            host = int(values[int(np.argmax(counts))])
            if host not in merged or host in thin_ids:
                continue
            merged[host] = merged[host] | region_bool
            del merged[root]

    blobs: list[Blob] = []
    drop_floor = max(min_area * 0.45, 0.20 * unit_area)
    for region_bool in merged.values():
        region = region_bool.astype(np.uint8) * 255
        area = int((region > 0).sum())
        if area < drop_floor or area > 0.12 * mask.size:
            continue
        mean = _region_mean(image_lab, region)
        # Mat or a soft shadow that survived the mask. Black pieces are far
        # from the mat in lightness, so they stay.
        mat_distance = float(delta_e(mean, mat))
        mean_chroma = float(np.hypot(mean[1], mean[2]))
        if mat_distance < 9.0 and mean_chroma < 10.0:
            continue
        pieces = [region]
        low = False
        note = ""
        if area > unit_area * touch_ratio:
            split = _watershed_masks(region, min_distance, unit_area)
            if len(split) >= 2:
                pieces = split
                note = "watershed"
            else:
                low = True
                note = "area"
        for piece_mask in pieces:
            piece_area_px = int((piece_mask > 0).sum())
            if piece_area_px < drop_floor * 0.7:
                continue
            ys, xs = np.where(piece_mask > 0)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            crop = piece_mask[y0 : y1 + 1, x0 : x1 + 1]
            contour = _contour_of(crop, (x0, y0))
            piece_mean = _region_mean(image_lab, piece_mask)
            count_estimate = 1
            touching = False
            piece_low = low
            if note == "area":
                count_estimate = max(2, int(round(piece_area_px / unit_area)))
                touching = True
                piece_low = True
            elif piece_area_px < 0.55 * unit_area:
                piece_low = True
                note = (note + "; partial").strip("; ")
            elif piece_area_px > unit_area * touch_ratio:
                count_estimate = max(2, int(round(piece_area_px / unit_area)))
                touching = True
                piece_low = True
            blobs.append(
                Blob(
                    x=x0,
                    y=y0,
                    w=x1 - x0 + 1,
                    h=y1 - y0 + 1,
                    area=piece_area_px,
                    mean_lab=(float(piece_mean[0]), float(piece_mean[1]), float(piece_mean[2])),
                    count=count_estimate,
                    touching=touching,
                    contour=contour,
                    low_confidence=piece_low,
                    notes=note,
                )
            )
    return blobs


def segment_pieces(
    image_bgr: np.ndarray,
    background_bgr: np.ndarray | None = None,
    min_area: int = 400,
    touch_ratio: float = 1.55,
    threshold: float = 14.0,
    palette: Palette | None = None,
    piece_area: int | None = None,
) -> list[Blob]:
    if palette is None:
        palette = Palette.load()
    mask = foreground_mask(image_bgr, background_bgr, threshold=threshold)
    lab = bgr_uint8_to_lab(image_bgr)
    unit = _estimate_unit_area(mask, min_area, piece_area)
    mat = _mat_lab(lab, mask)
    blobs = _split_mask_to_blobs(
        lab,
        mask,
        unit,
        min_area,
        touch_ratio,
        mat,
        palette,
        unit_locked=piece_area is not None and piece_area > 0,
    )
    blobs.sort(key=lambda blob: (blob.y, blob.x))
    return blobs
