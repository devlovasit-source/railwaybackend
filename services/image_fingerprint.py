import base64
from typing import Any

import cv2
import numpy as np
import requests


_URL_HASH_CACHE: dict[str, str] = {}
_URL_HASH_CACHE_MAX = 2048


def _normalize_base64(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "," in text:
        text = text.split(",", 1)[1]
    return text.strip()


def _cache_get(url: str) -> str:
    return _URL_HASH_CACHE.get(url, "")


def _cache_set(url: str, pixel_hash: str) -> None:
    if not url or not pixel_hash:
        return
    _URL_HASH_CACHE[url] = pixel_hash
    if len(_URL_HASH_CACHE) > _URL_HASH_CACHE_MAX:
        oldest = next(iter(_URL_HASH_CACHE.keys()), None)
        if oldest:
            _URL_HASH_CACHE.pop(oldest, None)


def _decode_image_bytes(image_bytes: bytes):
    if not image_bytes:
        return None
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)


def _foreground_crop(decoded):
    if decoded is None:
        return None
    if decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)

    if decoded.ndim != 3:
        return None

    if decoded.shape[2] == 4:
        alpha = decoded[:, :, 3]
        ys, xs = np.where(alpha > 12)
        if len(xs) and len(ys):
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            return decoded[y1:y2, x1:x2, :3]
        return decoded[:, :, :3]

    return decoded[:, :, :3]


def compute_pixel_hash_from_bytes(image_bytes: bytes, hash_size: int = 8) -> str:
    try:
        decoded = _decode_image_bytes(image_bytes)
        crop = _foreground_crop(decoded)
        if crop is None or crop.size == 0:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        bits = "".join("1" if v else "0" for v in diff.flatten())
        if not bits:
            return ""

        width = (hash_size * hash_size + 3) // 4
        return f"{int(bits, 2):0{width}x}"
    except Exception:
        return ""


def compute_pixel_hash_from_base64(value: Any, hash_size: int = 8) -> str:
    text = _normalize_base64(value)
    if not text:
        return ""
    try:
        image_bytes = base64.b64decode(text, validate=True)
    except Exception:
        return ""
    return compute_pixel_hash_from_bytes(image_bytes, hash_size=hash_size)


def compute_pixel_hash_from_url(url: Any, timeout_seconds: float = 8.0, hash_size: int = 8) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        return ""

    cached = _cache_get(normalized)
    if cached:
        return cached

    try:
        response = requests.get(normalized, timeout=timeout_seconds)
        response.raise_for_status()
        pixel_hash = compute_pixel_hash_from_bytes(response.content, hash_size=hash_size)
        if pixel_hash:
            _cache_set(normalized, pixel_hash)
        return pixel_hash
    except Exception:
        return ""


def hamming_distance_hex(left: Any, right: Any):
    l = str(left or "").strip().lower()
    r = str(right or "").strip().lower()
    if not l or not r:
        return None
    if len(l) != len(r):
        return None
    try:
        return (int(l, 16) ^ int(r, 16)).bit_count()
    except Exception:
        return None
