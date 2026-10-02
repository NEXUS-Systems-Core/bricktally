"""Wait for BrickTally to exit, swap the install folder, relaunch.

Keeps the previous folder as <install>.prev for rollback.
If this exe is inside the install folder, it copies itself to temp and
restarts from there first, so Windows will allow the rename.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os_kill = __import__("os").kill
        os_kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_exit(pid: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.3)
    raise TimeoutError(f"app pid {pid} did not exit")


def find_payload(extracted: Path, exe_name: str) -> Path:
    if (extracted / exe_name).is_file():
        return extracted
    matches = [path.parent for path in extracted.rglob(exe_name) if path.is_file()]
    if not matches:
        raise RuntimeError(f"{exe_name} was not in the update zip")
    return matches[0]


def swap_install(payload: Path, install_dir: Path) -> Path:
    install_dir = install_dir.resolve()
    prev = Path(str(install_dir) + ".prev")
    incoming = Path(str(install_dir) + ".incoming")
    if incoming.exists():
        shutil.rmtree(incoming)
    shutil.copytree(payload, incoming)
    if prev.exists():
        shutil.rmtree(prev)
    renamed = False
    if install_dir.exists():
        install_dir.rename(prev)
        renamed = True
    try:
        incoming.rename(install_dir)
    except Exception:
        if renamed and prev.exists() and not install_dir.exists():
            prev.rename(install_dir)
        raise
    return prev


def apply_zip(zip_path: Path, install_dir: Path, exe_name: str) -> Path:
    with tempfile.TemporaryDirectory(prefix="bricktally-upd-") as tmp:
        extracted = Path(tmp) / "extract"
        shutil.unpack_archive(str(zip_path), str(extracted))
        payload = find_payload(extracted, exe_name)
        return swap_install(payload, install_dir)


def relaunch(install_dir: Path, exe_name: str) -> None:
    exe = install_dir / exe_name
    if not exe.is_file():
        raise RuntimeError(f"relaunched exe missing: {exe}")
    if sys.platform == "win32":
        subprocess.Popen([str(exe)], cwd=str(install_dir), close_fds=True)
    else:
        subprocess.Popen([str(exe)], cwd=str(install_dir), start_new_session=True)


def _restart_outside_install(install_dir: Path) -> None:
    if not getattr(sys, "frozen", False):
        return
    exe = Path(sys.executable).resolve()
    install_dir = install_dir.resolve()
    if install_dir != exe.parent and install_dir not in exe.parents:
        return
    temp_exe = Path(tempfile.gettempdir()) / "BrickTallyUpdater.exe"
    if exe.resolve() == temp_exe.resolve():
        return
    shutil.copy2(exe, temp_exe)
    subprocess.Popen([str(temp_exe), *sys.argv[1:]], close_fds=True)
    raise SystemExit(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BrickTally folder swapper")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--zip", dest="zip_path", required=True)
    parser.add_argument("--install", required=True)
    parser.add_argument("--exe", default="BrickTally.exe")
    args = parser.parse_args(argv)
    install = Path(args.install)
    log_path = install.parent / "bricktally-update.log"
    try:
        _restart_outside_install(install)
        wait_for_exit(args.pid)
        apply_zip(Path(args.zip_path), install, args.exe)
        relaunch(install, args.exe)
    except Exception as exc:
        try:
            log_path.write_text(f"update failed: {exc}\n", encoding="utf-8")
        except OSError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
