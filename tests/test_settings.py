import json
from pathlib import Path

from bricktally.settings import Settings, choose_data_dir


def test_roundtrip(tmp_path: Path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    settings.camera_index = 2
    settings.condition = "N"
    settings.wb_gains = [1.1, 0.9, 1.0]
    settings.add_calibrated(5, (40.0, 50.0, 30.0))
    settings.add_learned(86, (70.0, 0.0, -2.0))
    settings.inventory = [{"part_id": "3001", "color_id": 5, "qty": 2, "condition": "N"}]
    settings.save()
    loaded = Settings(path).load()
    assert loaded.camera_index == 2
    assert loaded.condition == "N"
    assert loaded.user_samples()[5][0] == (40.0, 50.0, 30.0)
    assert loaded.user_samples()[86][0][0] == 70.0
    assert loaded.inventory[0]["part_id"] == "3001"
    assert json.loads(path.read_text())["condition"] == "N"


def test_appdata_on_windows():
    chosen = choose_data_dir("nt", r"C:\Users\op\AppData\Roaming", "/ignored", "/home/op")
    assert chosen.replace("\\", "/").endswith("AppData/Roaming/BrickTally")
