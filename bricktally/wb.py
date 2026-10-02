"""Grey-card white balance. Gains are BGR, matching OpenCV frames."""

from __future__ import annotations

import numpy as np


def grey_card_gains(image_bgr: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    """Gains that pull the center of a grey card back to neutral."""
    height, width = image_bgr.shape[:2]
    side_y = max(1, int(height * fraction))
    side_x = max(1, int(width * fraction))
    y0 = (height - side_y) // 2
    x0 = (width - side_x) // 2
    patch = image_bgr[y0:y0 + side_y, x0:x0 + side_x]
    mean = patch.astype(np.float64).mean(axis=(0, 1))
    target = float(mean.mean())
    return target / np.maximum(mean, 1.0)


def apply_gains(image_bgr: np.ndarray, gains: np.ndarray | None) -> np.ndarray:
    if gains is None:
        return image_bgr
    gains = np.asarray(gains, dtype=np.float64).reshape(3)
    if np.allclose(gains, 1.0):
        return image_bgr
    scaled = image_bgr.astype(np.float64) * gains.reshape(1, 1, 3)
    return np.clip(scaled, 0, 255).astype(np.uint8)
