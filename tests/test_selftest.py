def test_selftest_counts_without_a_window(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("cv2")
    pytest.importorskip("numpy")
    pytest.importorskip("PySide6")
    monkeypatch.chdir(tmp_path)
    from bricktally.selftest import run_selftest

    assert run_selftest() == 0
    assert (tmp_path / "bricktally-selftest.txt").read_text(encoding="utf-8").strip() == "OK"
