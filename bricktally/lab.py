"""sRGB to CIELAB (D65). Images elsewhere in the app are OpenCV BGR."""

from __future__ import annotations

import numpy as np

_D65 = (0.95047, 1.0, 1.08883)
_DELTA = 6.0 / 29.0


def _linearize(channel: np.ndarray) -> np.ndarray:
    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3.0 * _DELTA * _DELTA) + 4.0 / 29.0)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb is float 0-1, shape (..., 3). Returns L*a*b* with the same shape."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = _linearize(rgb[..., 0]), _linearize(rgb[..., 1]), _linearize(rgb[..., 2])
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / _D65[0]
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / _D65[2]
    fx, fy, fz = _f(x), _f(y), _f(z)
    lab = np.empty(rgb.shape, dtype=np.float64)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


def bgr_uint8_to_lab(image_bgr: np.ndarray) -> np.ndarray:
    rgb = image_bgr[..., ::-1].astype(np.float64) / 255.0
    return srgb_to_lab(rgb)


def hex_to_lab(hex_color: str) -> tuple[float, float, float]:
    text = hex_color.strip().lstrip("#")
    rgb = np.array(
        [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)],
        dtype=np.float64,
    ) / 255.0
    lab = srgb_to_lab(rgb)
    return float(lab[0]), float(lab[1]), float(lab[2])


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt(np.sum(diff * diff, axis=-1))
