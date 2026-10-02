"""GitHub Releases check. Offline or a placeholder repo never blocks the app."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from bricktally.config import UPDATE_REPO, USER_AGENT, WINDOWS_ASSET, __version__

_PLACEHOLDERS = {"", "OWNER/REPO", "owner/repo"}


@dataclass
class Update:
    tag: str
    version: tuple[int, ...]
    notes: str
    download_url: str
    asset_name: str
    sha256: str
    sha_url: str


def parse_version(text: str) -> tuple[int, ...]:
    raw = (text or "").strip()
    if raw[:1].lower() == "v":
        raw = raw[1:]
    raw = raw.split("-", 1)[0].split("+", 1)[0]
    parts: list[int] = []
    for piece in raw.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts) or (0,)


def is_newer(remote: str, local: str | None = None) -> bool:
    return parse_version(remote) > parse_version(local or __version__)


def repo_configured(repo: str | None = None) -> bool:
    text = (repo if repo is not None else UPDATE_REPO).strip()
    if text in _PLACEHOLDERS or "/" not in text:
        return False
    owner, name = text.split("/", 1)
    return bool(owner) and bool(name) and " " not in text


def _sha_from_digest(digest: str) -> str:
    digest = (digest or "").strip().lower()
    if digest.startswith("sha256:"):
        return digest.split(":", 1)[1]
    return ""


def _pick_assets(assets: list[dict]) -> tuple[dict | None, str]:
    by_name = {str(item.get("name") or ""): item for item in assets}
    zip_asset = by_name.get(WINDOWS_ASSET)
    if zip_asset is None:
        for item in assets:
            name = str(item.get("name") or "")
            if name.endswith(".zip") and "windows" in name.lower():
                zip_asset = item
                break
    sha = ""
    if zip_asset is not None:
        sha = _sha_from_digest(str(zip_asset.get("digest") or ""))
    sha_asset = by_name.get(f"{WINDOWS_ASSET}.sha256")
    sha_url = str(sha_asset.get("browser_download_url") or "") if sha_asset else ""
    return zip_asset, sha if sha else sha_url


def check_for_update(
    repo: str | None = None,
    current: str | None = None,
    timeout: float = 8.0,
    session: requests.Session | None = None,
) -> Update | None:
    repo = (repo if repo is not None else UPDATE_REPO).strip()
    if not repo_configured(repo):
        return None
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    get = session.get if session is not None else requests.get
    try:
        response = get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    tag = str(data.get("tag_name") or "")
    if not tag or not is_newer(tag, current or __version__):
        return None
    asset, sha_or_url = _pick_assets(list(data.get("assets") or []))
    if asset is None:
        return None
    sha256 = sha_or_url if sha_or_url and not sha_or_url.startswith("http") else ""
    sha_url = sha_or_url if sha_or_url.startswith("http") else ""
    return Update(
        tag=tag,
        version=parse_version(tag),
        notes=str(data.get("body") or ""),
        download_url=str(asset.get("browser_download_url") or ""),
        asset_name=str(asset.get("name") or WINDOWS_ASSET),
        sha256=sha256,
        sha_url=sha_url,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_file(text: str) -> str:
    match = re.search(r"\b([0-9a-fA-F]{64})\b", text or "")
    if not match:
        raise ValueError("checksum file has no SHA256")
    return match.group(1).lower()


def download_verified(
    update: Update,
    dest: Path,
    timeout: float = 120.0,
    session: requests.Session | None = None,
) -> Path:
    if not update.download_url:
        raise RuntimeError("update has no download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    get = session.get if session is not None else requests.get
    response = get(update.download_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    dest.write_bytes(response.content)
    expected = update.sha256
    if not expected and update.sha_url:
        sha_resp = get(update.sha_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        sha_resp.raise_for_status()
        expected = parse_sha256_file(sha_resp.text)
    if not expected:
        dest.unlink(missing_ok=True)
        raise RuntimeError("release has no SHA256; not installing")
    actual = file_sha256(dest)
    if actual.lower() != expected.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError("downloaded file did not match the published SHA256")
    return dest
