import re
import uuid
from typing import Any, Dict, List

from services import data_access_service
from services.appwrite_proxy import AppwriteProxyError
from services.upload_service import upload_wardrobe_images


_UNKNOWN_ATTR_RE = re.compile(r'Unknown attribute:\s*"([^"]+)"')


def _unknown_attribute_from_error(exc: Exception) -> str:
    msg = str(exc or "")
    matched = _UNKNOWN_ATTR_RE.search(msg)
    return (matched.group(1) if matched else "").strip()


def _create_outfit_with_schema_retries(payload: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
    safe_payload = dict(payload or {})
    for _ in range(max_retries):
        try:
            return data_access_service.create_document(
                resource="outfits",
                payload=safe_payload,
                document_id="unique()",
            )
        except AppwriteProxyError as exc:
            unknown_attr = _unknown_attribute_from_error(exc)
            if unknown_attr and unknown_attr in safe_payload:
                safe_payload.pop(unknown_attr, None)
                continue
            raise
    return data_access_service.create_document(
        resource="outfits",
        payload=safe_payload,
        document_id="unique()",
    )


def _normalize_hex_color(value: Any) -> str:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"#[0-9A-F]{6}", text):
        return text
    return "#000000"


def _normalize_pattern(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "plain"


def _normalize_occasions(value: Any) -> List[str]:
    if isinstance(value, list):
        out = [str(v).strip().lower() for v in value if str(v).strip()]
        return out if out else ["casual"]
    if isinstance(value, str):
        out = [part.strip().lower() for part in value.split(",") if part.strip()]
        return out if out else ["casual"]
    return ["casual"]


def persist_selected_items(
    *,
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Dict[str, Any]],
    duplicate_threshold: float = 0.97,
    pixel_max_distance: int = 6,
    image_duplicate_threshold: float = 0.985,
) -> Dict[str, Any]:
    del duplicate_threshold
    del pixel_max_distance
    del image_duplicate_threshold

    user_id = str(user_id or "").strip()
    if not user_id:
        return {
            "success": False,
            "message": "user_id is required",
            "saved_count": 0,
            "saved_items": [],
            "failed_items": [{"reason": "missing_user_id"}],
        }

    selected = {str(item_id).strip() for item_id in (selected_item_ids or []) if str(item_id).strip()}
    if not selected:
        return {
            "success": False,
            "message": "No selected items to save.",
            "saved_count": 0,
            "saved_items": [],
            "failed_items": [{"reason": "no_selected_items"}],
        }

    items_by_id: Dict[str, Dict[str, Any]] = {}
    for raw in detected_items or []:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("item_id") or "").strip()
        if item_id:
            items_by_id[item_id] = raw

    saved_items: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []

    for item_id in selected:
        row = items_by_id.get(item_id)
        if not row:
            failed_items.append({"item_id": item_id, "reason": "item_not_found"})
            continue

        raw_crop_base64 = str(row.get("raw_crop_base64") or "").strip()
        segmented_png_base64 = str(row.get("segmented_png_base64") or "").strip()
        if not raw_crop_base64 or not segmented_png_base64:
            failed_items.append({"item_id": item_id, "reason": "missing_image_payload"})
            continue

        file_id = uuid.uuid4().hex

        try:
            upload = upload_wardrobe_images(
                file_id=file_id,
                raw_image_base64=raw_crop_base64,
                masked_image_base64=segmented_png_base64,
            )

            payload = {
                "userId": user_id,
                "name": str(row.get("name") or row.get("sub_category") or "Outfit Item").strip(),
                "category": str(row.get("category") or "Tops").strip(),
                "sub_category": str(row.get("sub_category") or "Item").strip(),
                "color_code": _normalize_hex_color(row.get("color_code")),
                "pattern": _normalize_pattern(row.get("pattern")),
                "occasions": _normalize_occasions(row.get("occasions")),
                "image_url": str(upload.get("raw_image_url") or "").strip(),
                "masked_url": str(upload.get("masked_image_url") or "").strip(),
                "image_id": str(upload.get("raw_file_name") or file_id).strip(),
                "masked_id": str(upload.get("masked_file_name") or file_id).strip(),
                "status": "active",
                "worn": 0,
                "liked": False,
            }

            created = _create_outfit_with_schema_retries(payload)
            saved_items.append(
                {
                    "item_id": item_id,
                    "document_id": str(created.get("$id") or created.get("id") or "").strip(),
                    "name": payload["name"],
                    "image_url": payload["image_url"],
                }
            )
        except Exception as exc:
            failed_items.append({"item_id": item_id, "reason": str(exc)})

    return {
        "success": bool(saved_items),
        "message": f"Saved {len(saved_items)} items.",
        "saved_count": len(saved_items),
        "saved_items": saved_items,
        "failed_items": failed_items,
    }

