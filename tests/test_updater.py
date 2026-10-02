import shutil
import zipfile
from pathlib import Path

from updater import apply_zip, find_payload, swap_install


def _zip_with_exe(path: Path, exe_name: str = "BrickTally.exe", nested: bool = True) -> None:
    folder = path.parent / "payload"
    target = folder / "BrickTally" if nested else folder
    target.mkdir(parents=True, exist_ok=True)
    (target / exe_name).write_bytes(b"new-exe")
    (target / "colors.json").write_text("{}", encoding="utf-8")
    with zipfile.ZipFile(path, "w") as handle:
        for item in target.rglob("*"):
            if item.is_file():
                handle.write(item, item.relative_to(folder).as_posix())
    shutil.rmtree(folder)


def test_swap_keeps_previous(tmp_path: Path):
    install = tmp_path / "BrickTally"
    install.mkdir()
    (install / "BrickTally.exe").write_bytes(b"old-exe")
    (install / "note.txt").write_text("keep-me-in-prev", encoding="utf-8")
    payload = tmp_path / "new"
    payload.mkdir()
    (payload / "BrickTally.exe").write_bytes(b"new-exe")
    prev = swap_install(payload, install)
    assert (install / "BrickTally.exe").read_bytes() == b"new-exe"
    assert (prev / "note.txt").read_text(encoding="utf-8") == "keep-me-in-prev"
    assert not (install / "note.txt").exists()


def test_apply_nested_zip(tmp_path: Path):
    install = tmp_path / "app"
    install.mkdir()
    (install / "BrickTally.exe").write_bytes(b"old")
    archive = tmp_path / "BrickTally-windows.zip"
    _zip_with_exe(archive, nested=True)
    apply_zip(archive, install, "BrickTally.exe")
    assert (install / "BrickTally.exe").read_bytes() == b"new-exe"
    assert (install / "colors.json").is_file()
    assert (Path(str(install) + ".prev") / "BrickTally.exe").read_bytes() == b"old"


def test_find_payload_at_root(tmp_path: Path):
    root = tmp_path / "extract"
    root.mkdir()
    (root / "BrickTally.exe").write_bytes(b"x")
    assert find_payload(root, "BrickTally.exe") == root
