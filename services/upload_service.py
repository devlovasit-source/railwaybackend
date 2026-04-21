import base64
from urllib.parse import urlparse

from fastapi import HTTPException

from services.r2_storage import R2Storage


def _decode_base64_image(value: str, *, max_bytes: int, field_name: str) -> bytes:
    text = value or ""
    if "," in text:
        text = text.split(",", 1)[1]
    text = text.strip()
    text += "=" * ((4 - len(text) % 4) % 4)
    try:
        data = base64.b64decode(text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid base64: {exc}")
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{field_name} too large (max {max_bytes // (1024 * 1024)}MB)")
    return data


def upload_avatar(*, user_id: str, image_base64: str) -> str:
    image_bytes = _decode_base64_image(
        image_base64,
        max_bytes=8 * 1024 * 1024,
        field_name="image_base64",
    )
    return R2Storage().upload_avatar(user_id=user_id, image_bytes=image_bytes)


def _object_name_from_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    path = (parsed.path or "").strip("/")
    if not path:
        return ""
    return path.rsplit("/", 1)[-1].strip()


def upload_wardrobe_images(
    *,
    file_id: str,
    raw_image_base64: str | None = None,
    masked_image_base64: str | None = None,
    raw_image_url: str | None = None,
    masked_image_url: str | None = None,
):
    raw_url = str(raw_image_url or "").strip()
    masked_url = str(masked_image_url or "").strip()
    if raw_url and masked_url:
        # Client uploaded directly to storage and only needs canonical metadata
        # persisted; skip memory-heavy decode/relay in backend.
        return {
            "raw_file_name": _object_name_from_url(raw_url) or str(file_id or "").strip(),
            "masked_file_name": _object_name_from_url(masked_url) or str(file_id or "").strip(),
            "raw_image_url": raw_url,
            "masked_image_url": masked_url,
        }

    if not raw_image_base64 or not masked_image_base64:
        raise HTTPException(
            status_code=400,
            detail="Provide both raw/masked base64 images or both raw/masked image URLs.",
        )

    raw_bytes = _decode_base64_image(
        raw_image_base64,
        max_bytes=12 * 1024 * 1024,
        field_name="raw_image_base64",
    )
    masked_bytes = _decode_base64_image(
        masked_image_base64,
        max_bytes=12 * 1024 * 1024,
        field_name="masked_image_base64",
    )
    return R2Storage().upload_wardrobe_images(
        file_id=file_id,
        raw_image_bytes=raw_bytes,
        masked_image_bytes=masked_bytes,
    )
