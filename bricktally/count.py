"""Classify segmented pieces and total them by BrickLink color."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from bricktally.colors import Palette
from bricktally.segment import Blob, segment_pieces
from bricktally.wb import apply_gains


@dataclass
class Piece:
    index: int
    box: tuple[int, int, int, int]
    area: int
    count: int
    touching: bool
    lab: tuple[float, float, float]
    color_id: int
    color_name: str
    distance: float
    confidence: float
    review: bool
    reason: str
    locked: bool = False


@dataclass
class CountResult:
    pieces: list[Piece] = field(default_factory=list)

    @property
    def totals(self) -> dict[int, int]:
        totals: dict[int, int] = {}
        for piece in self.pieces:
            totals[piece.color_id] = totals.get(piece.color_id, 0) + piece.count
        return totals

    @property
    def review_indexes(self) -> list[int]:
        return [piece.index for piece in self.pieces if piece.review or piece.touching]


def _piece_from_blob(index: int, blob: Blob, palette: Palette) -> Piece:
    match = palette.match(blob.mean_lab)
    review = match.review or blob.touching
    reason = match.reason
    if blob.touching:
        extra = f"touching, about {blob.count}"
        reason = f"{reason}; {extra}" if reason else extra
    return Piece(
        index=index,
        box=(blob.x, blob.y, blob.w, blob.h),
        area=blob.area,
        count=blob.count,
        touching=blob.touching,
        lab=blob.mean_lab,
        color_id=match.color.id,
        color_name=match.color.name,
        distance=match.distance,
        confidence=match.confidence,
        review=review,
        reason=reason,
    )


def count_image(
    image_bgr: np.ndarray,
    palette: Palette,
    background_bgr: np.ndarray | None = None,
    wb_gains: np.ndarray | None = None,
    min_area: int = 400,
) -> CountResult:
    frame = apply_gains(image_bgr, wb_gains)
    background = None
    if background_bgr is not None:
        background = apply_gains(background_bgr, wb_gains)
    blobs = segment_pieces(frame, background_bgr=background, min_area=min_area)
    pieces = [_piece_from_blob(i, blob, palette) for i, blob in enumerate(blobs)]
    return CountResult(pieces=pieces)


def correct_piece(result: CountResult, index: int, color_id: int, palette: Palette) -> Piece:
    piece = result.pieces[index]
    color = palette.by_id[int(color_id)]
    piece.color_id = color.id
    piece.color_name = color.name
    piece.locked = True
    piece.review = piece.touching
    piece.reason = "set by operator" if not piece.touching else piece.reason
    piece.confidence = 1.0
    return piece


def rematch_unlocked(result: CountResult, palette: Palette) -> None:
    for piece in result.pieces:
        if piece.locked:
            continue
        match = palette.match(piece.lab)
        piece.color_id = match.color.id
        piece.color_name = match.color.name
        piece.distance = match.distance
        piece.confidence = match.confidence
        piece.review = match.review or piece.touching
        reason = match.reason
        if piece.touching:
            extra = f"touching, about {piece.count}"
            reason = f"{reason}; {extra}" if reason else extra
        piece.reason = reason


def draw_overlay(image_bgr: np.ndarray, result: CountResult) -> np.ndarray:
    canvas = image_bgr.copy()
    for piece in result.pieces:
        x, y, w, h = piece.box
        color = (40, 180, 255) if piece.review or piece.touching else (40, 180, 60)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = piece.color_name
        if piece.count > 1:
            label = f"{label} x{piece.count}"
        cv2.putText(
            canvas,
            label[:28],
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
