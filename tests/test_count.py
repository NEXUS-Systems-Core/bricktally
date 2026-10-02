import cv2
import numpy as np

from bricktally.colors import Palette
from bricktally.count import correct_piece, count_image, rematch_unlocked
from bricktally.segment import segment_pieces
from bricktally.wb import apply_gains, grey_card_gains
from tests.synthetic import grey_mat, paint_piece, pile_image, bgr_of


def test_count_synthetic_pile():
    palette = Palette.load()
    image, expected = pile_image(palette)
    result = count_image(image, palette, min_area=500)
    assert result.totals == expected
    assert sum(piece.count for piece in result.pieces) == sum(expected.values())
    touching = [piece for piece in result.pieces if piece.touching]
    assert len(touching) == 1
    assert touching[0].count == 2
    assert touching[0].color_id == 5
    assert 0 in result.review_indexes or touching[0].index in result.review_indexes


def test_empty_table_background_finds_pieces():
    palette = Palette.load()
    empty = grey_mat(shade=200)
    cv2.rectangle(empty, (0, 0), (40, 40), (190, 190, 190), thickness=-1)
    pile = empty.copy()
    paint_piece(pile, (300, 250), bgr_of(palette, 5))
    paint_piece(pile, (520, 360), bgr_of(palette, 11))
    blobs = segment_pieces(pile, background_bgr=empty, min_area=500)
    assert len(blobs) == 2
    result = count_image(pile, palette, background_bgr=empty, min_area=500)
    assert result.totals == {5: 1, 11: 1}


def test_correction_is_remembered_and_refines_match():
    palette = Palette.load()
    image, _expected = pile_image(palette)
    result = count_image(image, palette, min_area=500)
    black = next(piece for piece in result.pieces if piece.color_id == 11)
    correct_piece(result, black.index, 5, palette)
    assert result.pieces[black.index].locked
    assert result.pieces[black.index].color_id == 5
    palette.add_sample(5, black.lab)
    rematch_unlocked(result, palette)
    assert result.pieces[black.index].color_id == 5


def test_grey_card_neutralizes_center():
    image = np.full((120, 160, 3), (70, 110, 150), dtype=np.uint8)
    gains = grey_card_gains(image)
    balanced = apply_gains(image, gains)
    mean = balanced.astype(np.float64).mean(axis=(0, 1))
    assert abs(mean[0] - mean[1]) < 1.5
    assert abs(mean[1] - mean[2]) < 1.5
