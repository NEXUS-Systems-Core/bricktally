import pytest

from bricktally.bsx import to_bsx, validate_bsx
from bricktally.inventory import Inventory


def test_merge_batches_and_export():
    inv = Inventory()
    inv.add_batch("3001", "Brick 2 x 4", {5: 3, 11: 1}, {5: "Red", 11: "Black"}, "U")
    inv.add_batch("3001", "Brick 2 x 4", {5: 2}, {5: "Red"}, "U")
    inv.add_batch("3001", "Brick 2 x 4", {5: 4}, {5: "Red"}, "N")
    assert len(inv.lines) == 3
    red_used = next(line for line in inv.lines if line.color_id == 5 and line.condition == "U")
    assert red_used.qty == 5
    text = to_bsx(inv)
    parsed = validate_bsx(text)
    assert text.startswith("<?xml")
    assert "<BrickStoreXML>" in text
    assert parsed[0]["ItemTypeID"] == "P"
    assert {row["Condition"] for row in parsed} == {"N", "U"}
    assert sum(row["Qty"] for row in parsed if row["ColorID"] == 5 and row["Condition"] == "U") == 5


def test_rejects_bad_condition():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add_batch("3001", "", {5: 1}, {5: "Red"}, "X")


def test_roundtrip_json():
    inv = Inventory()
    inv.add_batch("3024", "Plate 1 x 1", {86: 8}, {86: "Light Bluish Gray"}, "N")
    again = Inventory.from_json(inv.to_json())
    assert again.lines[0].qty == 8
    assert again.lines[0].condition == "N"
