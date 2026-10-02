import hashlib
from pathlib import Path

import pytest
import requests

from bricktally.updates import (
    check_for_update,
    download_verified,
    is_newer,
    parse_sha256_file,
    parse_version,
    repo_configured,
)


def test_versions():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("1.2.3-beta") == (1, 2, 3)
    assert is_newer("v0.2.0", "0.1.0")
    assert not is_newer("v0.1.0", "0.1.0")
    assert not is_newer("v0.1.0", "0.2.0")


def test_placeholder_does_not_call_network():
    class Boom:
        def get(self, *args, **kwargs):
            raise AssertionError("should not call GitHub")

    assert repo_configured("OWNER/REPO") is False
    assert check_for_update("OWNER/REPO", session=Boom()) is None


def test_offline_returns_none():
    class Down:
        def get(self, *args, **kwargs):
            raise requests.ConnectionError("offline")

    assert check_for_update("eric/bricktally", current="0.1.0", session=Down()) is None


def test_newer_release_uses_digest():
    digest = "ab" * 32

    class Session:
        def get(self, url, headers=None, timeout=None):
            return _Resp(
                {
                    "tag_name": "v0.9.0",
                    "body": "notes",
                    "assets": [
                        {
                            "name": "BrickTally-windows.zip",
                            "browser_download_url": "https://example.test/BrickTally-windows.zip",
                            "digest": f"sha256:{digest}",
                        }
                    ],
                }
            )

    update = check_for_update("eric/bricktally", current="0.1.0", session=Session())
    assert update is not None
    assert update.tag == "v0.9.0"
    assert update.sha256 == digest
    assert update.download_url.endswith(".zip")


def test_sha_mismatch_refuses(tmp_path: Path):
    payload = b"not-a-real-zip"
    good = hashlib.sha256(b"other").hexdigest()

    class Session:
        def get(self, url, headers=None, timeout=None):
            resp = _Resp({})
            resp.content = payload
            return resp

    from bricktally.updates import Update

    update = Update(
        tag="v0.2.0",
        version=(0, 2, 0),
        notes="",
        download_url="https://example.test/x.zip",
        asset_name="BrickTally-windows.zip",
        sha256=good,
        sha_url="",
    )
    with pytest.raises(RuntimeError, match="SHA256"):
        download_verified(update, tmp_path / "x.zip", session=Session())
    assert not (tmp_path / "x.zip").exists()


def test_parse_sha_file():
    assert parse_sha256_file("abcd" * 16 + "  BrickTally-windows.zip\n") == "abcd" * 16


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.content = b""
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))
