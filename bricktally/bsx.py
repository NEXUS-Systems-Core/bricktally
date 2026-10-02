"""BrickStore BSX writer.

Shape checked against BrickStoreXML.rnc in rgriebl/brickstore: root
BrickStoreXML, Inventory, Item with ItemID, ItemTypeID, ColorID, Qty,
Condition (N or U).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from bricktally.inventory import Inventory, Line

_REQUIRED = ("ItemID", "ItemTypeID", "ColorID", "Qty")


def to_bsx(inventory: Inventory) -> str:
    root = ET.Element("BrickStoreXML")
    inv = ET.SubElement(root, "Inventory")
    for line in inventory.lines:
        _append_item(inv, line)
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def _append_item(parent: ET.Element, line: Line) -> None:
    if line.condition not in ("N", "U"):
        raise ValueError(f"bad condition {line.condition!r}")
    if not line.part_id.strip():
        raise ValueError("empty ItemID")
    item = ET.SubElement(parent, "Item")
    ET.SubElement(item, "ItemID").text = line.part_id.strip()
    ET.SubElement(item, "ItemTypeID").text = "P"
    ET.SubElement(item, "ColorID").text = str(int(line.color_id))
    if line.part_name:
        ET.SubElement(item, "ItemName").text = line.part_name
    ET.SubElement(item, "ItemTypeName").text = "Part"
    if line.color_name:
        ET.SubElement(item, "ColorName").text = line.color_name
    ET.SubElement(item, "Status").text = "I"
    ET.SubElement(item, "Qty").text = str(int(line.qty))
    ET.SubElement(item, "Price").text = "0.000"
    ET.SubElement(item, "Condition").text = line.condition


def validate_bsx(text: str) -> list[dict]:
    """Raise ValueError if the document is not a usable BrickStore inventory."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed XML: {exc}") from exc
    if root.tag not in ("BrickStoreXML", "BrickStockXML"):
        raise ValueError(f"unexpected root {root.tag}")
    inv = root.find("Inventory")
    if inv is None:
        raise ValueError("missing Inventory")
    parsed: list[dict] = []
    for item in inv.findall("Item"):
        fields = {child.tag: (child.text or "").strip() for child in item}
        for tag in _REQUIRED:
            if not fields.get(tag):
                raise ValueError(f"Item missing {tag}")
        if fields["ItemTypeID"] != "P":
            raise ValueError(f"ItemTypeID must be P, got {fields['ItemTypeID']}")
        if not fields["ColorID"].isdigit():
            raise ValueError(f"ColorID is not an integer: {fields['ColorID']}")
        qty = int(fields["Qty"])
        if qty < 0:
            raise ValueError("Qty is negative")
        condition = fields.get("Condition") or "N"
        if condition not in ("N", "U"):
            raise ValueError(f"Condition must be N or U, got {condition}")
        parsed.append(
            {
                "ItemID": fields["ItemID"],
                "ItemTypeID": "P",
                "ColorID": int(fields["ColorID"]),
                "Qty": qty,
                "Condition": condition,
                "ColorName": fields.get("ColorName", ""),
            }
        )
    return parsed
