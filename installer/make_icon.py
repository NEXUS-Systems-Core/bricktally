"""Write a simple brick icon. No external art."""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def _pixel(x: int, y: int, size: int) -> tuple[int, int, int, int]:
    gx = x * 32 // size
    gy = y * 32 // size
    brick = 4 <= gx <= 27 and 10 <= gy <= 26
    stud = False
    for cx in (11, 21):
        if (gx - cx) ** 2 + (gy - 8) ** 2 <= 7:
            stud = True
    if stud:
        return (232, 96, 72, 255)
    if brick:
        edge = gx in (4, 27) or gy in (10, 26)
        return (140, 24, 18, 255) if edge else (196, 40, 28, 255)
    return (36, 40, 46, 255)


def _dib(size: int) -> bytes:
    header = struct.pack(
        "<IiiHHIIiiII",
        40,
        size,
        size * 2,
        1,
        32,
        0,
        size * size * 4,
        0,
        0,
        0,
        0,
    )
    rows = []
    for y in range(size - 1, -1, -1):
        row = bytearray()
        for x in range(size):
            red, green, blue, alpha = _pixel(x, y, size)
            row += bytes((blue, green, red, alpha))
        rows.append(bytes(row))
    mask_stride = ((size + 31) // 32) * 4
    mask = b"\x00" * (mask_stride * size)
    return header + b"".join(rows) + mask


def build_ico() -> bytes:
    sizes = (16, 32, 48, 256)
    blobs = [_dib(size) for size in sizes]
    offset = 6 + 16 * len(blobs)
    entries = []
    for size, blob in zip(sizes, blobs):
        shown = 0 if size >= 256 else size
        entries.append(struct.pack("<BBBBHHII", shown, shown, 0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    return struct.pack("<HHH", 0, 1, len(blobs)) + b"".join(entries) + b"".join(blobs)


def main() -> int:
    dest = Path(__file__).resolve().parent / "bricktally.ico"
    dest.write_bytes(build_ico())
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
