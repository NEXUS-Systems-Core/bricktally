import pytest

from scripts.stamp_version import normalize, stamp_text


def test_normalize_strips_v():
    assert normalize("v0.1.0") == "0.1.0"
    assert normalize("1.2.3") == "1.2.3"


def test_normalize_rejects_junk():
    with pytest.raises(ValueError):
        normalize("v0.1")
    with pytest.raises(ValueError):
        normalize("1.2.3-rc1")


def test_stamp_text_replaces_once():
    text = '__version__ = "0.1.0"\nUSER_AGENT = "BrickTally/0.1.0"\n'
    updated = stamp_text(text, "0.2.0", r'__version__ = "[^"]*"', '__version__ = "0.2.0"')
    assert '__version__ = "0.2.0"' in updated
    assert updated.count("__version__") == 1
