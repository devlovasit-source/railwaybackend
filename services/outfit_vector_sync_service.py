import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

from services import data_access_service
from services.appwrite_proxy import AppwriteProxyError
from services.embedding_service import encode_metadata
from services.image_embedding_service import (
    encode_image_base64,
    encode_image_bytes,
    encode_image_url,
)
from services.image_fingerprint import (
    compute_pixel_hash_from_base64,
    compute_pixel_hash_from_bytes,
    compute_pixel_hash_from_url,
)
from services.qdrant_service import qdrant_service
from services.r2_storage import R2Storage


def _to_uuid_point_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _coerce_vector(value: Any) -> list:
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                return []
        return out
    return []


def _normalize_object_name(value: Any) -> str:
    text = str(value or "").strip().strip("/")
    if not text:
        return ""
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "?" in text:
        text = text.split("?", 1)[0]
    if "#" in text:
        text = text.split("#", 1)[0]
    return text.strip()


def _basename_from_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        path = parsed.path or text
    except Exception:
        path = text
    return path.rsplit("/", 1)[-1].strip()


def _derive_prefixed_png_name(value: Any, prefix: str) -> str:
    base = _normalize_object_name(value)
    if not base:
        return ""
    if "." in base:
        return base
    if not base.startswith(prefix):
        base = f"{prefix}{base}"
    return f"{base}.png"


def _collect_outfit_r2_candidates(payload: Dict[str, Any]) -> Dict[str, List[str]]:
    raw_candidates: List[str] = []
    masked_candidates: List[str] = []

    def add_unique(target: List[str], value: Any):
        name = _normalize_object_name(value)
        if name and name not in target:
            target.append(name)

    image_id = payload.get("image_id")
    masked_id = payload.get("masked_id")
    image_url = payload.get("image_url")
    masked_url = payload.get("masked_url")
    seed = str(payload.get("qdrant_point_id") or "").strip() or str(payload.get("$id") or "").strip()

    add_unique(raw_candidates, image_id)
    add_unique(masked_candidates, masked_id)
    add_unique(raw_candidates, _basename_from_url(image_url))
    add_unique(masked_candidates, _basename_from_url(masked_url))
    add_unique(raw_candidates, _derive_prefixed_png_name(image_id, "raw_"))
    add_unique(masked_candidates, _derive_prefixed_png_name(masked_id, "wardrobe_"))

    if seed:
        add_unique(raw_candidates, _derive_prefixed_png_name(seed, "raw_"))
        add_unique(masked_candidates, _derive_prefixed_png_name(seed, "wardrobe_"))

    return {"raw": raw_candidates, "masked": masked_candidates}


def _compute_payload_image_vector(payload: Dict[str, Any]) -> list:
    vector = _coerce_vector(payload.get("image_vector"))
    if vector:
        return vector
    vector = _coerce_vector(payload.get("imageVector"))
    if vector:
        return vector

    for key in ("masked_image_base64", "maskedImageBase64", "processed_image_base64", "image_base64", "imageBase64"):
        value = payload.get(key)
        if value:
            vector = encode_image_base64(value)
            if vector:
                return vector

    try:
        storage = R2Storage()
        candidates = _collect_outfit_r2_candidates(payload)
        for name in candidates.get("masked", []):
            image_bytes = storage.read_object_bytes(bucket=storage.wardrobe_bucket, object_name=name)
            if not image_bytes:
                continue
            vector = encode_image_bytes(image_bytes)
            if vector:
                return vector
        for name in candidates.get("raw", []):
            image_bytes = storage.read_object_bytes(bucket=storage.raw_bucket, object_name=name)
            if not image_bytes:
                continue
            vector = encode_image_bytes(image_bytes)
            if vector:
                return vector
    except Exception:
        pass

    for key in ("masked_url", "maskedUrl", "image_url", "imageUrl"):
        value = payload.get(key)
        if value:
            vector = encode_image_url(value)
            if vector:
                return vector

    return []


def _compute_payload_pixel_hash(payload: Dict[str, Any]) -> str:
    for key in ("pixel_hash", "pixelHash", "masked_pixel_hash", "maskedPixelHash"):
        value = str(payload.get(key) or "").strip().lower()
        if value:
            return value

    for key in ("masked_image_base64", "maskedImageBase64", "processed_image_base64", "image_base64", "imageBase64"):
        value = payload.get(key)
        if value:
            pixel_hash = compute_pixel_hash_from_base64(value)
            if pixel_hash:
                return pixel_hash

    try:
        storage = R2Storage()
        candidates = _collect_outfit_r2_candidates(payload)
        for name in candidates.get("masked", []):
            image_bytes = storage.read_object_bytes(bucket=storage.wardrobe_bucket, object_name=name)
            if not image_bytes:
                continue
            pixel_hash = compute_pixel_hash_from_bytes(image_bytes)
            if pixel_hash:
                return pixel_hash
        for name in candidates.get("raw", []):
            image_bytes = storage.read_object_bytes(bucket=storage.raw_bucket, object_name=name)
            if not image_bytes:
                continue
            pixel_hash = compute_pixel_hash_from_bytes(image_bytes)
            if pixel_hash:
                return pixel_hash
    except Exception:
        pass

    for key in ("masked_url", "maskedUrl", "image_url", "imageUrl"):
        value = payload.get(key)
        if value:
            pixel_hash = compute_pixel_hash_from_url(value)
            if pixel_hash:
                return pixel_hash

    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mark_outfit_vector_status(
    *,
    document_id: str,
    status: str,
    job_id: str = "",
    error: str = "",
    point_id: str = "",
) -> None:
    doc_id = str(document_id or "").strip()
    if not doc_id:
        return
    update_payload = {
        "qdrant_sync_status": str(status or "").strip().lower(),
        "qdrant_sync_updated_at": _now_iso(),
    }
    if job_id:
        update_payload["qdrant_sync_job_id"] = str(job_id)
    if error:
        update_payload["qdrant_sync_error"] = str(error)[:500]
    if point_id:
        update_payload["qdrant_point_id"] = str(point_id)
    try:
        data_access_service.update_document("outfits", doc_id, update_payload)
    except AppwriteProxyError:
        pass
    except Exception:
        pass


def sync_outfit_vectors(
    *,
    document: Dict[str, Any],
    payload: Dict[str, Any],
    user_id: str = "",
) -> Dict[str, Any]:
    if not qdrant_service.enabled():
        return {
            "qdrant_saved": False,
            "qdrant_error": "qdrant_disabled",
            "image_qdrant_saved": False,
            "image_qdrant_error": "qdrant_disabled",
            "qdrant_point_id": None,
        }

    doc = dict(document or {})
    src = dict(payload or {})
    merged = dict(src)
    merged.update(doc)

    payload_image_vector = (
        _coerce_vector(src.get("image_vector"))
        or _coerce_vector(src.get("imageVector"))
        or _compute_payload_image_vector(merged)
    )
    payload_pixel_hash = (
        str(src.get("pixel_hash") or "").strip().lower()
        or str(src.get("pixelHash") or "").strip().lower()
        or _compute_payload_pixel_hash(merged)
    )

    point_id = (
        str(doc.get("qdrant_point_id") or "").strip()
        or str(src.get("qdrant_point_id") or "").strip()
        or _to_uuid_point_id(doc.get("$id") or src.get("$id") or src.get("document_id"))
    )

    qdrant_saved = False
    qdrant_error = None
    image_qdrant_saved = False
    image_qdrant_error = None

    try:
        vector_input = {
            "category": doc.get("category", "") or src.get("category", ""),
            "sub_category": doc.get("sub_category", "") or src.get("sub_category", ""),
            "color_code": doc.get("color_code", "") or src.get("color_code", ""),
            "pattern": doc.get("pattern", "") or src.get("pattern", ""),
            "occasions": doc.get("occasions", []) if isinstance(doc.get("occasions", []), list) else [],
        }
        vector = encode_metadata(vector_input)
        qdrant_payload = dict(doc)
        qdrant_payload["userId"] = str(
            doc.get("userId")
            or src.get("userId")
            or src.get("user_id")
            or user_id
            or ""
        )
        if payload_pixel_hash:
            qdrant_payload["pixel_hash"] = payload_pixel_hash
        qdrant_service.upsert_item(point_id, vector, qdrant_payload)
        qdrant_saved = True
    except Exception as exc:
        qdrant_error = str(exc)

    try:
        if payload_image_vector:
            image_payload = {
                "userId": str(doc.get("userId") or src.get("userId") or src.get("user_id") or user_id or ""),
                "category": doc.get("category", "") or src.get("category", ""),
                "sub_category": doc.get("sub_category", "") or src.get("sub_category", ""),
                "color_code": doc.get("color_code", "") or src.get("color_code", ""),
                "image_url": doc.get("masked_url")
                or doc.get("image_url")
                or src.get("masked_url")
                or src.get("image_url")
                or "",
                "pixel_hash": payload_pixel_hash or "",
            }
            qdrant_service.upsert_image_vector(point_id, payload_image_vector, image_payload)
            image_qdrant_saved = True
    except Exception as exc:
        image_qdrant_error = str(exc)

    if qdrant_saved and str(doc.get("$id") or "").strip():
        mark_outfit_vector_status(
            document_id=str(doc.get("$id")),
            status="completed" if image_qdrant_saved else "partial",
            error=image_qdrant_error or "",
            point_id=point_id,
        )

    return {
        "qdrant_saved": qdrant_saved,
        "qdrant_error": qdrant_error,
        "image_qdrant_saved": image_qdrant_saved,
        "image_qdrant_error": image_qdrant_error,
        "qdrant_point_id": point_id,
        "pixel_hash": payload_pixel_hash,
    }
