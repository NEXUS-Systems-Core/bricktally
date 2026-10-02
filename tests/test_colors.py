from bricktally.colors import Palette
from bricktally.lab import hex_to_lab, srgb_to_lab
import numpy as np


EXPECTED = {
    1: ("White", "#F3F4EF", "solid"),
    5: ("Red", "#D82424", "solid"),
    11: ("Black", "#2E2E2E", "solid"),
    9: ("Light Gray", "#B8BAAE", "solid"),
    86: ("Light Bluish Gray", "#B9B9B9", "solid"),
    85: ("Dark Bluish Gray", "#7D7C78", "solid"),
    88: ("Reddish Brown", "#7C442C", "solid"),
    68: ("Dark Orange", "#AC5112", "solid"),
    12: ("Trans-Clear", "#EEEEEE", "trans"),
    21: ("Chrome Gold", "#F1F2E1", "chrome"),
    115: ("Pearl Gold", "#E79E1D", "pearl"),
}


def test_palette_matches_bricklink_guide():
    palette = Palette.load()
    assert len(palette.colors) == 214
    assert len(palette.by_id) == 214
    for color_id, (name, hex_color, kind) in EXPECTED.items():
        color = palette.by_id[color_id]
        assert color.name == name
        assert color.hex == hex_color
        assert color.type == kind


def test_white_is_neutral_lab():
    lab = srgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert abs(lab[0] - 100) < 0.2
    assert abs(lab[1]) < 0.2
    assert abs(lab[2]) < 0.2


def test_catalog_red_matches_itself():
    palette = Palette.load()
    match = palette.match(palette.by_id[5].lab)
    assert match.color.id == 5
    assert match.distance < 0.1
    assert match.review is False


def test_calibrated_sample_beats_catalog():
    palette = Palette.load()
    weird = (18.0, 72.0, -68.0)
    before = palette.match(weird)
    assert before.distance > 5
    palette.add_sample(5, weird)
    after = palette.match(weird)
    assert after.color.id == 5
    assert after.distance < 0.1


def test_confusable_greys_go_to_review():
    palette = Palette.load()
    light = np.array(palette.by_id[9].lab)
    bluish = np.array(palette.by_id[86].lab)
    mid = tuple((light + bluish) / 2.0)
    match = palette.match(mid)
    assert match.review
    assert "confusable pair" in match.reason
    assert match.confidence <= 0.45
    # Both system greys sit inside the thin-margin band, even if a Modulex
    # grey is the nearest catalog hex.
    light = palette.by_id[9].lab
    bluish = palette.by_id[86].lab
    assert palette.match(light).review
    assert palette.match(bluish).review


def test_hex_roundtrip_used_by_palette():
    lab = hex_to_lab("#D82424")
    assert lab[0] > 30
