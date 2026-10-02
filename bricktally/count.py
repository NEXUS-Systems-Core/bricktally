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
    contour: np.ndarray | None = None
    low_confidence: bool = False
    color_bgr: tuple[int, int, int] = (40, 180, 60)


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


def _bgr_of_hex(text: str) -> tuple[int, int, int]:
    text = text.lstrip("#")
    red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    return blue, green, red


def _piece_from_blob(index: int, blob: Blob, palette: Palette) -> Piece:
    match = palette.match(blob.mean_lab)
    review = match.review or blob.touching or blob.low_confidence
    reason = match.reason
    if blob.touching:
        extra = f"touching, about {blob.count}"
        reason = f"{reason}; {extra}" if reason else extra
    if blob.low_confidence and "low confidence" not in reason:
        extra = "low confidence"
        reason = f"{reason}; {extra}" if reason else extra
    # Colour-match confidence stays as matched. A split or area estimate does
    # not get a made-up higher score; the review flag carries the doubt.
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
        contour=blob.contour,
        low_confidence=blob.low_confidence,
        color_bgr=_bgr_of_hex(match.color.hex),
    )


def count_image(
    image_bgr: np.ndarray,
    palette: Palette,
    background_bgr: np.ndarray | None = None,
    wb_gains: np.ndarray | None = None,
    min_area: int = 400,
    piece_area: int | None = None,
) -> CountResult:
    frame = apply_gains(image_bgr, wb_gains)
    background = None
    if background_bgr is not None:
        background = apply_gains(background_bgr, wb_gains)
    blobs = segment_pieces(
        frame,
        background_bgr=background,
        min_area=min_area,
        piece_area=piece_area,
        palette=palette,
    )
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


def _dashed_rect(canvas: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    step = 8
    x2, y2 = x + w, y + h
    for start in range(x, x2, step * 2):
        cv2.line(canvas, (start, y), (min(start + step, x2), y), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (start, y2), (min(start + step, x2), y2), color, 2, cv2.LINE_AA)
    for start in range(y, y2, step * 2):
        cv2.line(canvas, (x, start), (x, min(start + step, y2)), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (x2, start), (x2, min(start + step, y2)), color, 2, cv2.LINE_AA)


def draw_overlay(image_bgr: np.ndarray, result: CountResult) -> np.ndarray:
    canvas = image_bgr.copy()
    for piece in result.pieces:
        color = piece.color_bgr
        if piece.contour is not None and len(piece.contour) >= 3:
            cv2.drawContours(canvas, [piece.contour], -1, color, 2, cv2.LINE_AA)
        else:
            x, y, w, h = piece.box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        flagged = piece.review or piece.touching or piece.low_confidence
        if flagged:
            x, y, w, h = piece.box
            _dashed_rect(canvas, x, y, w, h, (0, 220, 255))
        if piece.count > 1 or flagged:
            x, y, _, _ = piece.box
            label = piece.color_name
            if piece.count > 1:
                label = f"{label} x{piece.count}"
            cv2.putText(
                canvas,
                label[:28],
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 255) if flagged else color,
                1,
                cv2.LINE_AA,
            )
    return canvas
