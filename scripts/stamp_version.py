"""Write a vX.Y.Z tag into the app version before the release build."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def normalize(raw: str) -> str:
    text = (raw or "").strip()
    if text[:1].lower() == "v":
        text = text[1:]
    if not re.fullmatch(r"\d+\.\d+\.\d+", text):
        raise ValueError(f"version must be X.Y.Z, got {raw!r}")
    return text


def stamp_text(text: str, version: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"pattern did not match once: {pattern}")
    return updated


def stamp_tree(root: Path, version: str) -> None:
    config = root / "bricktally" / "config.py"
    config_text = config.read_text(encoding="utf-8")
    config_text = stamp_text(
        config_text,
        version,
        r'__version__ = "[^"]*"',
        f'__version__ = "{version}"',
    )
    config_text = stamp_text(
        config_text,
        version,
        r'USER_AGENT = "BrickTally/[^"]*"',
        f'USER_AGENT = "BrickTally/{version}"',
    )
    config.write_text(config_text, encoding="utf-8")

    pyproject = root / "pyproject.toml"
    pyproject_text = stamp_text(
        pyproject.read_text(encoding="utf-8"),
        version,
        r'(?m)^version = "[^"]*"',
        f'version = "{version}"',
    )
    pyproject.write_text(pyproject_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: stamp_version.py X.Y.Z", file=sys.stderr)
        return 2
    try:
        version = normalize(args[0])
        root = Path(__file__).resolve().parents[1]
        stamp_tree(root, version)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
