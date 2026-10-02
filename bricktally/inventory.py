"""Running inventory across batches, merged by part, color, and condition."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Line:
    part_id: str
    part_name: str
    color_id: int
    color_name: str
    qty: int
    condition: str


@dataclass
class Inventory:
    lines: list[Line] = field(default_factory=list)

    def add_batch(
        self,
        part_id: str,
        part_name: str,
        totals: dict[int, int],
        color_names: dict[int, str],
        condition: str,
    ) -> None:
        part_id = part_id.strip()
        if not part_id:
            raise ValueError("part number is empty")
        if condition not in ("N", "U"):
            raise ValueError("condition must be N or U")
        for color_id, qty in totals.items():
            qty = int(qty)
            if qty <= 0:
                continue
            color_id = int(color_id)
            self._add(part_id, part_name, color_id, color_names.get(color_id, ""), qty, condition)

    def _add(
        self,
        part_id: str,
        part_name: str,
        color_id: int,
        color_name: str,
        qty: int,
        condition: str,
    ) -> None:
        for line in self.lines:
            if line.part_id == part_id and line.color_id == color_id and line.condition == condition:
                line.qty += qty
                if part_name and not line.part_name:
                    line.part_name = part_name
                if color_name and not line.color_name:
                    line.color_name = color_name
                return
        self.lines.append(
            Line(part_id, part_name, color_id, color_name, qty, condition)
        )

    def clear(self) -> None:
        self.lines.clear()

    def to_json(self) -> list[dict]:
        return [
            {
                "part_id": line.part_id,
                "part_name": line.part_name,
                "color_id": line.color_id,
                "color_name": line.color_name,
                "qty": line.qty,
                "condition": line.condition,
            }
            for line in self.lines
        ]

    @classmethod
    def from_json(cls, rows: list[dict]) -> Inventory:
        inv = cls()
        for row in rows:
            inv._add(
                str(row["part_id"]),
                str(row.get("part_name") or ""),
                int(row["color_id"]),
                str(row.get("color_name") or ""),
                int(row["qty"]),
                str(row.get("condition") or "U"),
            )
        return inv
