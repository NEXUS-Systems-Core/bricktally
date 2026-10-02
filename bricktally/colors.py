"""BrickLink color palette and CIELAB matching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from bricktally.lab import delta_e, hex_to_lab

# Pairs the camera regularly swaps when the margin is thin.
CONFUSABLE_PAIRS = {
    frozenset({9, 86}),    # Light Gray / Light Bluish Gray
    frozenset({11, 85}),   # Black / Dark Bluish Gray
    frozenset({88, 68}),   # Reddish Brown / Dark Orange
}

FAR_DISTANCE = 18.0
THIN_MARGIN = 5.0
CONFUSABLE_MARGIN = 12.0


@dataclass(frozen=True)
class BrickColor:
    id: int
    name: str
    hex: str
    type: str
    lab: tuple[float, float, float]


@dataclass(frozen=True)
class ColorMatch:
    color: BrickColor
    distance: float
    second: BrickColor | None
    margin: float
    confidence: float
    review: bool
    reason: str


def colors_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "colors.json"


class Palette:
    def __init__(self, colors: list[BrickColor]):
        self.colors = colors
        self.by_id = {c.id: c for c in colors}
        # color id -> measured LAB samples (calibration and corrections)
        self.samples: dict[int, list[tuple[float, float, float]]] = {}

    @classmethod
    def load(cls, path: Path | None = None) -> Palette:
        payload = json.loads((path or colors_path()).read_text(encoding="utf-8"))
        colors = []
        for row in payload["colors"]:
            colors.append(
                BrickColor(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    hex=str(row["hex"]).upper(),
                    type=str(row["type"]),
                    lab=hex_to_lab(row["hex"]),
                )
            )
        return cls(colors)

    def set_samples(self, samples: dict[int, list[tuple[float, float, float]]]) -> None:
        self.samples = {int(k): list(v) for k, v in samples.items() if v}

    def add_sample(self, color_id: int, lab: tuple[float, float, float]) -> None:
        self.samples.setdefault(int(color_id), []).append(tuple(float(x) for x in lab))

    def reference_lab(self, color: BrickColor) -> np.ndarray:
        samples = self.samples.get(color.id)
        if samples:
            return np.mean(np.asarray(samples, dtype=np.float64), axis=0)
        return np.asarray(color.lab, dtype=np.float64)

    def match(self, lab: tuple[float, float, float] | np.ndarray) -> ColorMatch:
        target = np.asarray(lab, dtype=np.float64)
        ranked: list[tuple[float, BrickColor]] = []
        for color in self.colors:
            dist = float(delta_e(target, self.reference_lab(color)))
            ranked.append((dist, color))
        ranked.sort(key=lambda item: item[0])
        best_dist, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else None
        second_dist = ranked[1][0] if second is not None else best_dist
        margin = second_dist - best_dist
        by_id = {color.id: dist for dist, color in ranked}
        reasons: list[str] = []
        if best_dist > FAR_DISTANCE:
            reasons.append("far from palette")
        if second is not None and margin < THIN_MARGIN:
            reasons.append("close call")
        for pair in CONFUSABLE_PAIRS:
            left, right = tuple(pair)
            if left not in by_id or right not in by_id:
                continue
            if by_id[left] - best_dist < CONFUSABLE_MARGIN and by_id[right] - best_dist < CONFUSABLE_MARGIN:
                reasons.append("confusable pair")
                break
        review = bool(reasons)
        confidence = max(0.0, min(1.0, 1.0 - best_dist / 40.0))
        if "confusable pair" in reasons:
            confidence = min(confidence, 0.45)
        elif review:
            confidence = min(confidence, 0.55)
        return ColorMatch(
            color=best,
            distance=best_dist,
            second=second,
            margin=margin,
            confidence=confidence,
            review=review,
            reason="; ".join(reasons),
        )


@lru_cache(maxsize=1)
def load_palette() -> Palette:
    return Palette.load()
