import base64

from fastapi import HTTPException

from services.r2_storage import R2Storage


def _decode_base64_image(value: str, *, max_bytes: int, field_name: str) -> bytes:
    text = value or ""
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=True)
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


def upload_wardrobe_images(*, file_id: str, raw_image_base64: str, masked_image_base64: str):
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
