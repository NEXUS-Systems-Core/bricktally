"""Settings in AppData (Windows) or the platform user-data dir."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from bricktally.config import APP_NAME


def choose_data_dir(platform: str, appdata: str | None, platform_dir: str, home: str) -> str:
    if platform == "nt" and appdata:
        return os.path.join(appdata, APP_NAME)
    if platform_dir:
        return platform_dir
    if appdata:
        return os.path.join(appdata, APP_NAME)
    return os.path.join(home, ".bricktally")


def default_data_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    platform_dir = ""
    try:
        from platformdirs import user_data_dir

        platform_dir = user_data_dir(APP_NAME, "EricManno")
    except Exception:
        platform_dir = ""
    return Path(choose_data_dir(os.name, appdata, platform_dir, str(Path.home())))


class Settings:
    def __init__(self, path: Path | None = None):
        self.path = path or (default_data_dir() / "settings.json")
        self.camera_index = 0
        self.condition = "U"
        self.wb_gains = [1.0, 1.0, 1.0]
        self.calibrated: dict[str, list[list[float]]] = {}
        self.learned: dict[str, list[list[float]]] = {}
        self.inventory: list[dict] = []
        # 0 means estimate piece area from the photo. A positive value is the
        # operator override, in pixels, for one piece.
        self.piece_area = 0

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def background_path(self) -> Path:
        return self.directory / "background.png"

    def user_samples(self) -> dict[int, list[tuple[float, float, float]]]:
        merged: dict[int, list[tuple[float, float, float]]] = {}
        for bucket in (self.calibrated, self.learned):
            for key, rows in bucket.items():
                color_id = int(key)
                merged.setdefault(color_id, [])
                for row in rows:
                    merged[color_id].append((float(row[0]), float(row[1]), float(row[2])))
        return merged

    def add_calibrated(self, color_id: int, lab: tuple[float, float, float]) -> None:
        self.calibrated.setdefault(str(int(color_id)), []).append([float(x) for x in lab])

    def add_learned(self, color_id: int, lab: tuple[float, float, float]) -> None:
        self.learned.setdefault(str(int(color_id)), []).append([float(x) for x in lab])

    def load(self) -> Settings:
        if not self.path.is_file():
            return self
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self
        self.camera_index = int(data.get("camera_index") or 0)
        condition = str(data.get("condition") or "U")
        self.condition = condition if condition in ("N", "U") else "U"
        gains = data.get("wb_gains") or [1.0, 1.0, 1.0]
        if len(gains) == 3:
            self.wb_gains = [float(g) for g in gains]
        self.calibrated = dict(data.get("calibrated") or {})
        self.learned = dict(data.get("learned") or {})
        self.inventory = list(data.get("inventory") or [])
        try:
            self.piece_area = max(0, int(data.get("piece_area") or 0))
        except (TypeError, ValueError):
            self.piece_area = 0
        return self

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "camera_index": self.camera_index,
            "condition": self.condition,
            "wb_gains": self.wb_gains,
            "calibrated": self.calibrated,
            "learned": self.learned,
            "inventory": self.inventory,
            "piece_area": self.piece_area,
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def gains_array(self) -> np.ndarray:
        return np.asarray(self.wb_gains, dtype=np.float64)
