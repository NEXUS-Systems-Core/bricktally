"""Client for https://api.brickognize.com/predict/.

POST multipart field name is query_image. Response is listing_id,
bounding_box, and items[] with id, name, img_url, score, type,
external_sites. Confirmed against the public OpenAPI and the sorter-v2 client.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import requests

from bricktally.config import BRICKOGNIZE_URL, USER_AGENT


@dataclass
class Candidate:
    part_id: str
    name: str
    score: float
    image_url: str
    item_type: str
    external_sites: list[dict] = field(default_factory=list)


@dataclass
class IdentifyResult:
    listing_id: str
    candidates: list[Candidate]
    bounding_box: dict


def crop_around(image_bgr: np.ndarray, x: int, y: int, side: int = 220) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    side = max(40, min(side, height, width))
    half = side // 2
    x0 = int(x) - half
    y0 = int(y) - half
    x0 = min(max(0, x0), max(0, width - side))
    y0 = min(max(0, y0), max(0, height - side))
    return image_bgr[y0:y0 + side, x0:x0 + side].copy()


def parse_response(data: dict) -> IdentifyResult:
    candidates: list[Candidate] = []
    for raw in data.get("items") or []:
        part_id = str(raw.get("id") or "").strip()
        if not part_id:
            continue
        candidates.append(
            Candidate(
                part_id=part_id,
                name=str(raw.get("name") or ""),
                score=float(raw.get("score") or 0.0),
                image_url=str(raw.get("img_url") or ""),
                item_type=str(raw.get("type") or ""),
                external_sites=list(raw.get("external_sites") or []),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return IdentifyResult(
        listing_id=str(data.get("listing_id") or ""),
        candidates=candidates,
        bounding_box=dict(data.get("bounding_box") or {}),
    )


def identify_image(
    image_bgr: np.ndarray,
    timeout: float = 25.0,
    top_k: int = 5,
    session: requests.Session | None = None,
) -> IdentifyResult:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("could not encode the photo")
    files = {"query_image": ("query.jpg", buf.tobytes(), "image/jpeg")}
    params = {
        "top_k_items": int(top_k),
        "predict_color": "false",
        "min_similarity_items": 0.2,
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    post = session.post if session is not None else requests.post
    response = post(BRICKOGNIZE_URL, files=files, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return parse_response(response.json())
