import requests

from bricktally.brickognize import crop_around, identify_image, parse_response
from bricktally.config import BRICKOGNIZE_URL
import numpy as np


FIXTURE = {
    "listing_id": "res-d492bca0",
    "bounding_box": {
        "left": 1,
        "upper": 2,
        "right": 80,
        "lower": 90,
        "image_width": 200,
        "image_height": 200,
        "score": 0.5,
    },
    "items": [
        {
            "id": "3001",
            "name": "Brick 2 x 4",
            "img_url": "https://example.test/3001.webp",
            "external_sites": [
                {"name": "bricklink", "url": "https://www.bricklink.com/v2/catalog/catalogitem.page?P=3001"}
            ],
            "category": "Brick",
            "type": "part",
            "score": 0.91,
        },
        {
            "id": "3002",
            "name": "Brick 2 x 3",
            "img_url": "",
            "external_sites": [],
            "category": "Brick",
            "type": "part",
            "score": 0.4,
        },
    ],
}


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, files=None, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "files": files, "params": params, "timeout": timeout})
        return _Response(self.payload)


def test_parse_openapi_shape():
    result = parse_response(FIXTURE)
    assert result.listing_id == "res-d492bca0"
    assert result.candidates[0].part_id == "3001"
    assert result.candidates[0].name == "Brick 2 x 4"
    assert result.candidates[0].score == 0.91
    assert result.bounding_box["left"] == 1
    assert [item.part_id for item in result.candidates] == ["3001", "3002"]


def test_identify_posts_query_image(monkeypatch):
    session = _Session(FIXTURE)
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 180)
    result = identify_image(image, session=session, top_k=5)
    assert result.candidates[0].part_id == "3001"
    call = session.calls[0]
    assert call["url"] == BRICKOGNIZE_URL
    assert "query_image" in call["files"]
    filename, payload, mime = call["files"]["query_image"]
    assert filename.endswith(".jpg")
    assert payload[:2] == b"\xff\xd8"
    assert mime == "image/jpeg"
    assert call["params"]["top_k_items"] == 5


def test_crop_stays_inside():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    crop = crop_around(image, 0, 0, side=40)
    assert crop.shape == (40, 40, 3)


def test_http_error_surfaces():
    class Bad(_Session):
        def post(self, url, files=None, params=None, headers=None, timeout=None):
            response = _Response({})
            response.status_code = 500

            def raise_for_status():
                raise requests.HTTPError("500")

            response.raise_for_status = raise_for_status
            return response

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    try:
        identify_image(image, session=Bad({}))
    except requests.HTTPError:
        return
    raise AssertionError("expected HTTPError")
