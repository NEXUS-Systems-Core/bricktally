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

# Effect finishes photograph as grey, gold or brown. They are not candidates
# unless the operator has saved a sample of that exact finish.
EFFECT_TYPES = frozenset({"pearl", "metallic", "glitter", "chrome", "milky", "speckle", "satin", "modulex"})

# Solid and transparent colours that show up in a normal bin. Rare solids
# (Rust, Umber, Fabuland, Modulex) stay in the correction list, not the guess.
COMMON_PHOTO_IDS = frozenset({
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    23, 24, 28, 34, 36, 39, 42, 47, 49,
    55, 59, 63, 68, 69, 71, 76, 80,
    85, 86, 88, 89, 90, 94,
    103, 104, 105, 110, 150, 153, 154, 156, 220,
    12, 14, 15, 17, 18, 19, 20, 50, 98, 107, 108,
})

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

    def photo_colors(self) -> list[BrickColor]:
        """Colours the camera is allowed to guess.

        Effect finishes are included only when a sample was saved for them.
        The correction dropdown still lists the full palette.
        """

        chosen: list[BrickColor] = []
        for color in self.colors:
            if color.id in self.samples:
                chosen.append(color)
                continue
            if color.type in EFFECT_TYPES:
                continue
            if color.id in COMMON_PHOTO_IDS:
                chosen.append(color)
        return chosen

    def match(self, lab: tuple[float, float, float] | np.ndarray, photo: bool = False) -> ColorMatch:
        target = np.asarray(lab, dtype=np.float64)
        if not photo:
            return self._rank(target, self.colors)
        pool = self.photo_colors() or self.colors
        solids = [color for color in pool if color.type != "trans" or color.id in self.samples]
        trans = [color for color in pool if color.type == "trans" and color.id not in self.samples]
        solid_match = self._rank(target, solids or pool)
        # A solid brick photographed dark is closer to a transparent hex than to
        # its own catalog colour. Prefer the solid unless the solid fit is poor
        # and a transparent colour is clearly nearer.
        if trans and solid_match.distance > 16.0:
            trans_match = self._rank(target, trans)
            if trans_match.distance + 4.0 < solid_match.distance:
                return self._photo_review(trans_match)
        return self._photo_review(solid_match)

    def _rank(self, target: np.ndarray, pool: list[BrickColor]) -> ColorMatch:
        ranked: list[tuple[float, BrickColor]] = []
        for color in pool:
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

    def _photo_review(self, match: ColorMatch) -> ColorMatch:
        """Flag a photo guess only when the colour really is in doubt.

        A thin catalog margin is normal under a webcam. Review is for a poor
        fit, a known confusable pair, or a coin-flip between two colours.
        """

        reasons: list[str] = []
        if match.distance > 20.0:
            reasons.append("far from palette")
        if "confusable pair" in match.reason and match.margin < 8.0:
            reasons.append("confusable pair")
        if match.margin < 2.5 and match.distance > 8.0:
            reasons.append("close call")
        review = bool(reasons)
        confidence = match.confidence
        if review and "confusable pair" in reasons:
            confidence = min(confidence, 0.45)
        elif review:
            confidence = min(confidence, 0.55)
        return ColorMatch(
            color=match.color,
            distance=match.distance,
            second=match.second,
            margin=match.margin,
            confidence=confidence,
            review=review,
            reason="; ".join(reasons),
        )


@lru_cache(maxsize=1)
def load_palette() -> Palette:
    return Palette.load()
