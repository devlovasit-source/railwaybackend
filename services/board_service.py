import base64
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException

from services.appwrite_proxy import AppwriteProxy, AppwriteProxyError
from services.r2_storage import R2Storage, R2StorageError


def clean_occasion(raw: str) -> str:
    v = (raw or "").strip().lower()
    mapping = {
        "party looks": "Party",
        "party": "Party",
        "office fit": "Office",
        "office": "Office",
        "vacation": "Vacation",
        "occasion": "Occasion",
    }
    return mapping.get(v, (raw or "Occasion").strip().title())


def decode_image_base64(value: str) -> tuple[bytes, str]:
    text = (value or "").strip()
    if not text:
        return b"", "png"

    extension = "png"
    if text.startswith("data:image/"):
        match = re.match(r"^data:image/([a-zA-Z0-9]+);base64,", text)
        if match:
            extension = match.group(1).lower()
        text = text.split(",", 1)[1] if "," in text else text

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    if not data:
        raise HTTPException(status_code=400, detail="image_base64 is empty")
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image_base64 too large (max 12MB)")
    return data, extension


def list_saved_boards(*, user_id: str, occasion: Optional[str] = None, limit: int = 100):
    proxy = AppwriteProxy()
    return proxy.list_documents(
        "saved_boards",
        user_id=user_id,
        occasion=clean_occasion(occasion) if occasion else None,
        limit=limit,
    )


def save_board(
    *,
    user_id: str,
    occasion: str,
    image_url: str = "",
    image_base64: str = "",
    board_ids: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
):
    proxy = AppwriteProxy()
    final_image_url = (image_url or "").strip()
    if str(image_base64 or "").strip():
        image_bytes, extension = decode_image_base64(image_base64)
        storage = R2Storage()
        uploaded = storage.upload_style_board_image(
            user_id=user_id,
            image_bytes=image_bytes,
            extension=extension,
        )
        final_image_url = uploaded.get("image_url", final_image_url)

    item_ids: list[str] = []
    if board_ids:
        item_ids = [x.strip() for x in board_ids.split(",") if x.strip()]
    elif isinstance(payload, dict):
        raw_item_ids = payload.get("itemIds") or payload.get("boardIds") or []
        if isinstance(raw_item_ids, list):
            item_ids = [str(x).strip() for x in raw_item_ids if str(x).strip()]

    doc = {
        "userId": user_id,
        "occasion": clean_occasion(occasion),
        "imageUrl": final_image_url,
        "itemIds": item_ids,
    }
    return proxy.create_document("saved_boards", doc)


def list_life_boards(*, user_id: str, limit: int = 100):
    proxy = AppwriteProxy()

    # First try explicit life_boards collection if it exists and has data.
    try:
        docs = proxy.list_documents("life_boards", user_id=user_id, limit=limit)
        if docs:
            return docs
    except AppwriteProxyError:
        # Fallback to synthesized boards from existing collections.
        pass

    def _safe_docs(resource: str) -> list[dict]:
        try:
            data = proxy.list_documents(resource, user_id=user_id, limit=200)
            if isinstance(data, list):
                return data
        except AppwriteProxyError:
            return []
        return []

    outfits = _safe_docs("outfits")
    bills = _safe_docs("bills")
    coupons = _safe_docs("coupons")
    skincare_profiles = _safe_docs("skincare_profiles")
    meal_plans = _safe_docs("meal_plans")
    workout_outfits = _safe_docs("workout_outfits")
    life_goals = _safe_docs("life_goals")

    return [
        {
            "title": "Daily Wear",
            "boardType": "daily_wear",
            "description": f"{len(outfits)} wardrobe item(s) ready",
        },
        {
            "title": "Home Utilities",
            "boardType": "home_utilities",
            "description": f"{len(bills)} bill(s) and {len(coupons)} coupon(s)",
        },
        {
            "title": "Skincare",
            "boardType": "skincare",
            "description": f"{len(skincare_profiles)} profile(s) saved",
        },
        {
            "title": "Diet & Fitness",
            "boardType": "diet_fitness",
            "description": (
                f"{len(meal_plans)} meal plan(s), "
                f"{len(workout_outfits)} workout set(s), "
                f"{len(life_goals)} goal(s)"
            ),
        },
    ]


def save_life_board(*, user_id: str, title: str, board_type: str, description: str, payload: Dict[str, Any]):
    now_iso = datetime.now(timezone.utc).isoformat()
    doc = {
        "userId": user_id,
        "title": (title or "").strip() or "Life Board",
        "boardType": (board_type or "").strip() or "daily_wear",
        "description": (description or "").strip(),
        "payload": payload or {},
        "createdAt": now_iso,
        "updatedAt": now_iso,
    }
    return AppwriteProxy().create_document("life_boards", doc)


def delete_saved_board(*, document_id: str):
    AppwriteProxy().delete_document("saved_boards", document_id)


__all__ = [
    "AppwriteProxyError",
    "R2StorageError",
    "clean_occasion",
    "list_saved_boards",
    "save_board",
    "list_life_boards",
    "save_life_board",
    "delete_saved_board",
]
