import base64
import concurrent.futures
import os
import time
from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from services import ai_gateway
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_base64
from services.image_fingerprint import compute_pixel_hash_from_base64
from services.qdrant_service import qdrant_service
from services.task_queue import enqueue_task

try:
    from worker import vision_analyze_task
except Exception:
    vision_analyze_task = None

try:
    from routers.bg_remover import BGRemoveRequest, remove_background_sync
except Exception:
    BGRemoveRequest = None
    remove_background_sync = None

router = APIRouter()


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _vision_enable_similarity() -> bool:
    return _env_bool("VISION_ANALYZE_ENABLE_SIMILARITY", False)


def _duplicate_threshold() -> float:
    try:
        val = float(os.getenv("WARDROBE_DUPLICATE_THRESHOLD", "0.97"))
        return val if 0.0 < val <= 1.0 else 0.97
    except Exception:
        return 0.97


def _pixel_duplicate_distance() -> int:
    try:
        val = int(os.getenv("WARDROBE_PIXEL_DUPLICATE_DISTANCE", "6"))
        return max(0, min(val, 64))
    except Exception:
        return 6


def _image_duplicate_threshold() -> float:
    try:
        val = float(os.getenv("WARDROBE_IMAGE_DUPLICATE_THRESHOLD", "0.985"))
        return val if 0.0 < val <= 1.0 else 0.985
    except Exception:
        return 0.985


def _vision_ai_timeout_seconds() -> int:
    try:
        val = int(os.getenv("VISION_ANALYZE_AI_TIMEOUT_SECONDS", "18"))
        return max(3, min(val, 60))
    except Exception:
        return 18


class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    userId: str = "demo_user"


def _normalize_base64_for_model(value: str) -> str:
    text = (value or "").strip()
    return text.split(",", 1)[1] if "," in text else text


def _to_png_data_uri(base64_text: str) -> str:
    text = _normalize_base64_for_model(base64_text)
    return f"data:image/png;base64,{text}"


def _input_has_alpha(image_base64: str) -> bool:
    try:
        b64 = _normalize_base64_for_model(image_base64)
        img_data = base64.b64decode(b64, validate=True)
        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return bool(decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 4)
    except Exception:
        return False


def _remove_bg_first(image_base64: str):
    if _input_has_alpha(image_base64):
        return image_base64, True, "input_already_has_alpha"
    if BGRemoveRequest is None or remove_background_sync is None:
        return image_base64, False, "bg_remover_unavailable"
    try:
        req = BGRemoveRequest(image_base64=image_base64)
        result = remove_background_sync(req.image_base64)
        fallback_reason = ""
        if isinstance(result, dict):
            fallback_reason = str(result.get("fallback_reason") or "").strip().lower()
        should_retry = (
            isinstance(result, dict)
            and result.get("success")
            and not bool(result.get("bg_removed", False))
            and any(token in fallback_reason for token in ("warm", "load", "download", "init"))
        )
        if should_retry:
            time.sleep(2)
            result = remove_background_sync(req.image_base64)
        if isinstance(result, dict) and result.get("success") and result.get("image_base64"):
            processed = _to_png_data_uri(result.get("image_base64"))
            return processed, bool(result.get("bg_removed", True)), result.get("fallback_reason")
        return image_base64, False, "bg_remove_no_image"
    except Exception as exc:
        return image_base64, False, f"bg_remove_failed: {exc}"


def get_dominant_color(cv_image, k=3):
    try:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        crop_h, crop_w = int(h * 0.25), int(w * 0.25)
        center_image = image[crop_h:h - crop_h, crop_w:w - crop_w]
        center_image = cv2.resize(center_image, (100, 100), interpolation=cv2.INTER_AREA)

        hsv_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV)
        pixels_rgb = center_image.reshape((-1, 3))
        pixels_hsv = hsv_image.reshape((-1, 3))

        mask = (pixels_hsv[:, 1] > 20) & (pixels_hsv[:, 2] > 70) & (pixels_hsv[:, 2] < 245)
        filtered_rgb = pixels_rgb[mask]
        if len(filtered_rgb) < 100:
            filtered_rgb = pixels_rgb[(pixels_hsv[:, 2] > 30) & (pixels_hsv[:, 2] < 250)]
            if len(filtered_rgb) == 0:
                filtered_rgb = pixels_rgb

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(filtered_rgb)
        dominant_rgb = [int(x) for x in kmeans.cluster_centers_[Counter(kmeans.labels_).most_common(1)[0][0]]]
        return "#{:02x}{:02x}{:02x}".format(*dominant_rgb).upper()
    except Exception:
        return "#000000"


def _hex_to_color_name(hex_color: str) -> str:
    try:
        color = hex_color.lstrip("#")
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    except Exception:
        return "Multicolor"

    if max(r, g, b) < 40:
        return "Black"
    if min(r, g, b) > 220:
        return "White"
    if abs(r - g) < 14 and abs(g - b) < 14:
        return "Gray"
    if r > 180 and g < 110 and b < 110:
        return "Red"
    if r > 170 and g > 120 and b < 90:
        return "Orange"
    if r > 170 and g > 170 and b < 90:
        return "Yellow"
    if g > 150 and r < 130 and b < 130:
        return "Green"
    if b > 150 and r < 130 and g < 150:
        return "Blue"
    if r > 150 and b > 150 and g < 130:
        return "Purple"
    if r > 140 and g > 100 and b > 70:
        return "Brown"
    return "Multicolor"


# Emergency fallbacks used only when model output is missing fields.
def _extract_foreground_mask(decoded_img) -> np.ndarray | None:
    try:
        if decoded_img is None:
            return None
        if decoded_img.ndim == 3 and decoded_img.shape[2] == 4:
            return decoded_img[:, :, 3] > 18

        bgr = decoded_img if decoded_img.ndim == 3 else None
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        maxc, minc = rgb.max(axis=2), rgb.min(axis=2)
        near_white = (rgb[:, :, 0] >= 236) & (rgb[:, :, 1] >= 236) & (rgb[:, :, 2] >= 236) & ((maxc - minc) <= 20)
        saturated = hsv[:, :, 1] > 18
        mask = (~near_white) | saturated

        mask_u8 = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = mask_u8 > 0
        return final_mask if final_mask.sum() >= 200 else None
    except Exception:
        return None


def _infer_garment_hint(decoded_img) -> tuple[str, str]:
    mask = _extract_foreground_mask(decoded_img)
    if mask is None:
        return ("Tops", "Shirt")
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return ("Tops", "Shirt")
    box_w, box_h = max(1, xs.max() - xs.min() + 1), max(1, ys.max() - ys.min() + 1)
    if float(box_h) / float(box_w) > 1.15:
        return ("Bottoms", "Trousers")
    return ("Tops", "Shirt")


def _infer_pattern(cv_image) -> str:
    try:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        return "checked" if (float(np.count_nonzero(edges)) / float(edges.size)) > 0.09 else "plain"
    except Exception:
        return "plain"


MASTER_VISION_PROMPT = """
You are a high-end fashion stylist vision classifier.
Analyze the garment image and return STRICT JSON with this exact shape:
{
  "name": "Short 2-to-3 word name combining the color and specific item type (e.g., 'Black T-Shirt', 'Blue Jeans'). DO NOT include gender (men's, women's) or unnecessary adjectives.",
  "category": "Main category (Choose ONE: Tops, Bottoms, Dresses, Outerwear, Footwear, Bags, Accessories, Jewelry, Indian Wear)",
  "sub_category": "Specific type (e.g., T-Shirt, Chinos, Sneakers, Watch, Kurta)",
  "pattern": "one short value like plain/striped/checked/floral",
  "occasions": ["list 5 to 8 specific occasions where this item can be worn"]
}

CRITICAL RULES:
- Pants/Jeans/Shorts MUST be 'Bottoms'. 
- Shoes/Sneakers/Boots MUST be 'Footwear'. 
- DO NOT categorize clothing or shoes as 'Accessories'.
- Output ONLY raw JSON, no markdown tags.
"""


def _clean_text(val):
    return str(val).strip() if val else ""


def _normalize_occasions(raw_occ) -> list[str]:
    if isinstance(raw_occ, str):
        raw_occ = [x.strip() for x in raw_occ.split(",")]
    if not isinstance(raw_occ, list):
        return []
    out = []
    seen = set()
    for item in raw_occ:
        text = _clean_text(item).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_VALID_CATEGORIES = {
    "Tops",
    "Bottoms",
    "Footwear",
    "Outerwear",
    "Dresses",
    "Bags",
    "Accessories",
    "Jewelry",
    "Indian Wear",
}


def _normalize_category_from_subcategory(category: str, sub_category: str) -> str:
    cat = _clean_text(category).title()
    sub = _clean_text(sub_category).lower()
    if not sub:
        return cat

    footwear_tokens = (
        "shoe", "shoes", "sneaker", "loafer", "sandal", "heel", "boot",
        "slipper", "flip flop", "flipflop", "crocs", "croc",
    )
    bottoms_tokens = (
        "pant", "pants", "trouser", "trousers", "jean", "jeans", "jogger",
        "legging", "leggings", "short", "shorts", "skirt", "cargo", "chino",
    )
    bags_tokens = ("bag", "backpack", "tote", "clutch", "purse", "sling")

    if any(tok in sub for tok in footwear_tokens):
        return "Footwear"
    if any(tok in sub for tok in bottoms_tokens):
        return "Bottoms"
    if any(tok in sub for tok in bags_tokens):
        return "Bags"
    return cat


def _shape_vision_output(raw_data, color_hex: str, decoded_img, cv_image) -> dict:
    data = dict(raw_data) if isinstance(raw_data, dict) else {}

    name = _clean_text(data.get("name") or data.get("title"))
    category = _clean_text(data.get("category") or data.get("main_category")).title()
    sub_category = _clean_text(data.get("sub_category") or data.get("subcategory") or data.get("subType")).title()
    pattern = _clean_text(data.get("pattern") or data.get("texture")).lower()
    occasions = _normalize_occasions(data.get("occasions") or data.get("occasion"))

    if not category or not sub_category:
        print("[vision] AI missing category/sub_category -> using emergency geometry fallback")
        fallback_cat, fallback_sub = _infer_garment_hint(decoded_img)
        category = category or fallback_cat
        sub_category = sub_category or fallback_sub

    if not name:
        name = f"{_hex_to_color_name(color_hex)} {sub_category}"

    if not pattern:
        print("[vision] AI missing pattern -> using emergency edge fallback")
        pattern = _infer_pattern(cv_image)

    if len(occasions) < 3:
        print("[vision] AI missing occasions -> using emergency generic fallback")
        occasions = ["daily wear", "casual outing", "weekend", "travel", "office", "hangout"]

    # Guardrail: if LLM gives a mismatched category (e.g., shoes as accessories),
    # infer the main category from sub-category keywords.
    category = _normalize_category_from_subcategory(category, sub_category)

    if category not in _VALID_CATEGORIES:
        category = "Tops"

    return {
        "name": name,
        "category": category,
        "sub_category": sub_category,
        "pattern": pattern,
        "occasions": occasions[:8],
        "color_code": color_hex,
    }


def vision_analyze_core(image_base64: str, user_id: str = "demo_user"):
    vision_input_base64, bg_removed, bg_fallback_reason = _remove_bg_first(image_base64)
    if not bg_removed:
        raise HTTPException(
            status_code=422,
            detail=(
                "Background removal failed. Please take a photo of the garment "
                "against a plain, contrasting background."
            ),
        )

    base64_data = _normalize_base64_for_model(vision_input_base64)
    try:
        img_data = base64.b64decode(base64_data, validate=True)
        if len(img_data) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="image payload too large (max 12MB)")
        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        cv_image = (
            cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
            if (decoded.ndim == 3 and decoded.shape[2] == 4)
            else (decoded if decoded.ndim == 3 else cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
        )
        if cv_image is None:
            raise HTTPException(status_code=400, detail="invalid image payload")
        extracted_color_hex = get_dominant_color(cv_image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {str(e)}")

    model_used = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                ai_gateway.ollama_vision_json,
                prompt=MASTER_VISION_PROMPT,
                image_base64=base64_data,
                usecase="vision",
            )
            final_data, model_used = future.result(
                timeout=_vision_ai_timeout_seconds()
            )
    except concurrent.futures.TimeoutError:
        print(f"[vision] AI Vision Timeout after {_vision_ai_timeout_seconds()}s.")
        raise Exception("The Vision AI timed out.")
    except Exception as e:
        print(f"[vision] AI Vision Error: {e}")
        raise Exception(f"The Vision AI failed: {e}")

    final_data = _shape_vision_output(final_data, extracted_color_hex, decoded, cv_image)
    final_data["userId"] = user_id

    image_duplicate = {"checked": False, "is_duplicate": False, "id": None, "score": 0.0}
    pixel_duplicate = {"checked": False, "is_duplicate": False, "id": None, "distance": None}
    vector = None
    similar_items = []
    image_vector = []
    pixel_hash = ""
    image_duplicate_threshold = _image_duplicate_threshold()
    pixel_max_distance = _pixel_duplicate_distance()

    if _vision_enable_similarity():
        try:
            vector = encode_metadata(final_data)
            similar_items = qdrant_service.search_similar(vector, user_id, limit=5)
        except Exception as e:
            print(f"[vision] Similarity metadata search error: {e}")

        image_vector = encode_image_base64(vision_input_base64)
        if image_vector:
            try:
                image_duplicate = qdrant_service.find_image_duplicate(
                    image_vector, user_id, threshold=image_duplicate_threshold
                )
            except Exception as e:
                print(f"[vision] Image duplicate check error: {e}")

        pixel_hash = compute_pixel_hash_from_base64(vision_input_base64)
        if pixel_hash:
            try:
                pixel_duplicate = qdrant_service.find_pixel_duplicate(
                    user_id, pixel_hash, max_distance=pixel_max_distance
                )
            except Exception as e:
                print(f"[vision] Pixel duplicate check error: {e}")

    top_similarity_score = float(similar_items[0].get("score") or 0.0) if similar_items else 0.0
    probable_duplicate = bool(
        image_duplicate.get("is_duplicate")
        or pixel_duplicate.get("is_duplicate")
        or top_similarity_score >= _duplicate_threshold()
    )

    return {
        "success": True,
        "data": final_data,
        "processed_image_base64": vision_input_base64,
        "similar_items": similar_items,
        "meta": {
            "bg_removed": bg_removed,
            "bg_fallback_reason": bg_fallback_reason,
            "llm_fallback": False,
            "vision_model_used": model_used,
            "similarity_enabled": _vision_enable_similarity(),
            "embedding_created": vector is not None,
            "similar_items_found": len(similar_items),
            "image_duplicate_checked": bool(image_duplicate.get("checked")),
            "image_duplicate_threshold": image_duplicate_threshold,
            "image_duplicate_score": float(image_duplicate.get("score") or 0.0),
            "pixel_duplicate_checked": bool(pixel_duplicate.get("checked")),
            "pixel_duplicate_distance": pixel_duplicate.get("distance"),
            "pixel_duplicate_max_distance": pixel_max_distance,
            "pixel_hash": pixel_hash or None,
            "probable_duplicate": probable_duplicate,
        },
    }


@router.post("/analyze-image/async", status_code=status.HTTP_202_ACCEPTED)
def analyze_image_async(http_request: Request, request: ImageAnalyzeRequest):
    if vision_analyze_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")
    task_id = enqueue_task(
        task_func=vision_analyze_task,
        args=[request.image_base64, request.userId],
        kwargs={"request_id": str(getattr(http_request.state, "request_id", "") or "")},
        kind="vision_analyze",
        user_id=request.userId,
    )
    return {"success": True, "status": "queued", "task_id": task_id}
