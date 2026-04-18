import base64
import io
import uuid
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

# Import the new prompt from the prompts folder
from prompts.core_prompts import WARDROBE_CAPTURE_PROMPT

from services.wardrobe_persistence_service import persist_selected_items
from services import ai_gateway
try:
    from worker import capture_analyze_task, capture_save_selected_task, process_upload_task
except Exception:
    capture_analyze_task = None
    capture_save_selected_task = None
    process_upload_task = None
try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task


router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])


class CaptureAnalyzeRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


class DetectedItem(BaseModel):
    item_id: str
    name: str
    category: str
    sub_category: str
    color_code: str
    pattern: str = "plain"
    occasions: List[str] = Field(default_factory=lambda: ["casual"])
    confidence: float = 0.0
    reasoning: str
    bbox: Dict[str, int]
    raw_crop_base64: str
    segmented_png_base64: str


class SaveSelectedRequest(BaseModel):
    user_id: str
    selected_item_ids: List[str]
    detected_items: List[DetectedItem]


class ProcessUploadRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


def _duplicate_threshold() -> float:
    raw = str(os.getenv("WARDROBE_DUPLICATE_THRESHOLD", "0.97") or "").strip()
    try:
        value = float(raw)
        if value <= 0.0:
            return 0.97
        if value > 1.0:
            return 1.0
        return value
    except Exception:
        return 0.97


def _pixel_duplicate_distance() -> int:
    raw = str(os.getenv("WARDROBE_PIXEL_DUPLICATE_DISTANCE", "6") or "").strip()
    try:
        value = int(raw)
        if value < 0:
            return 0
        if value > 64:
            return 64
        return value
    except Exception:
        return 6


def _image_duplicate_threshold() -> float:
    raw = str(os.getenv("WARDROBE_IMAGE_DUPLICATE_THRESHOLD", "0.985") or "").strip()
    try:
        value = float(raw)
        if value <= 0.0:
            return 0.985
        if value > 1.0:
            return 1.0
        return value
    except Exception:
        return 0.985


def _decode_image_base64(value: str) -> Image.Image:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB)")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}")
    return image


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _dominant_hex(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = arr.reshape((-1, 3))
    if len(pixels) == 0:
        return "#000000"
    mean = np.mean(pixels, axis=0).astype(int)
    rgb = (int(mean[2]), int(mean[1]), int(mean[0]))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _segment_png_base64(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        return base64.b64encode(cv2.imencode(".png", rgba)[1].tobytes()).decode("utf-8")

    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (2, 2, max(1, w - 4), max(1, h - 4))
    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    except Exception:
        alpha = np.full((h, w), 255, dtype=np.uint8)

    rgba = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode segmented PNG")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _extract_hex_from_text(value: str) -> str:
    m = re.search(r"#(?:[0-9a-fA-F]{6})\b", str(value or ""))
    if not m:
        return ""
    return m.group(0).upper()


def _safe_bbox(raw_bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw_bbox, dict):
        return None
    try:
        x1 = max(0, min(width - 1, int(float(raw_bbox.get("x1", 0)))))
        y1 = max(0, min(height - 1, int(float(raw_bbox.get("y1", 0)))))
        x2 = max(1, min(width, int(float(raw_bbox.get("x2", width)))))
        y2 = max(1, min(height, int(float(raw_bbox.get("y2", height)))))
    except Exception:
        return None
    if x2 <= x1:
        x2 = min(width, x1 + 2)
    if y2 <= y1:
        y2 = min(height, y1 + 2)
    if (x2 - x1) < 24 or (y2 - y1) < 24:
        return None
    return (x1, y1, x2, y2)


def _llama_detect_items(image_base64: str, *, request_id: str = "") -> List[Dict[str, Any]]:
    result, _model = ai_gateway.ollama_vision_json(
        prompt=WARDROBE_CAPTURE_PROMPT,
        image_base64=(image_base64 or "").split(",", 1)[1] if "," in str(image_base64 or "") else str(image_base64 or ""),
        request_id=request_id,
        usecase="vision",
    )
    items = result.get("items", []) if isinstance(result, dict) else []
    return [row for row in items if isinstance(row, dict)]


def _fallback_single_item(image: Image.Image) -> List[Dict[str, Any]]:
    width, height = image.size
    crop = image.crop((0, 0, width, height))
    return [
        {
            "item_id": str(uuid.uuid4()),
            "name": "Detected Outfit Item",
            "category": "Tops",
            "sub_category": "item",
            "color_code": _dominant_hex(crop),
            "pattern": "plain",
            "occasions": ["casual"],
            "confidence": 0.35,
            "reasoning": "Fallback item generated because vision JSON was empty or invalid.",
            "bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height},
            "raw_crop_base64": _image_to_base64(crop, fmt="JPEG"),
            "segmented_png_base64": _segment_png_base64(crop),
        }
    ]


def _draw_overlay(image: Image.Image, items: List[Dict[str, Any]]) -> str:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(items):
        bbox = item.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=3)
        label = f"{idx + 1}. {item.get('name', item.get('sub_category', 'item'))}"
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill="cyan")
    return _image_to_base64(canvas, fmt="JPEG")


@router.post("/analyze")
def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):
    request_id = str(getattr(http_request.state, "request_id", "") or "")
    return analyze_capture_core(request.user_id, request.image_base64, request_id=request_id)


def analyze_capture_core(user_id: str, image_base64: str, request_id: str = ""):
    image = _decode_image_base64(image_base64)
    width, height = image.size

    llm_items: List[Dict[str, Any]] = []
    llm_error = ""
    try:
        llm_items = _llama_detect_items(image_base64, request_id=request_id)
    except Exception as exc:
        llm_error = str(exc)
        llm_items = []

    items: List[Dict[str, Any]] = []
    if llm_items:
        for row in llm_items:
            bbox_tuple = _safe_bbox(row.get("bbox"), width, height)
            if bbox_tuple is None:
                continue
            x1, y1, x2, y2 = bbox_tuple
            crop = image.crop((x1, y1, x2, y2))
            
            # Prioritize LLM color code if provided, otherwise fallback to math
            color_code = _extract_hex_from_text(row.get("color_code")) or _dominant_hex(crop)
            segmented_png_base64 = _segment_png_base64(crop)
            raw_crop_base64 = _image_to_base64(crop, fmt="JPEG")

            # RAW PASS-THROUGH (No normalization filters)
            raw_name = str(row.get("name", "Item")).strip()
            raw_category = str(row.get("category", "")).strip()
            raw_sub_category = str(row.get("sub_category", "")).strip()
            raw_pattern = str(row.get("pattern", "")).strip()
            raw_occasions = row.get("occasions", [])

            try:
                confidence = float(row.get("confidence", 0.5))
            except Exception:
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            
            reasoning = str(row.get("reasoning", "")).strip()

            items.append(
                {
                    "item_id": str(uuid.uuid4()),
                    "name": raw_name,
                    "category": raw_category,
                    "sub_category": raw_sub_category,
                    "color_code": color_code,
                    "pattern": raw_pattern,
                    "occasions": raw_occasions,
                    "confidence": round(confidence, 4),
                    "reasoning": reasoning,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "raw_crop_base64": raw_crop_base64,
                    "segmented_png_base64": segmented_png_base64,
                }
            )

    if not items:
        items = _fallback_single_item(image)

    overlay_base64 = _draw_overlay(image, items)

    return {
        "success": True,
        "board": "wardrobe_capture",
        "type": "multi_detect",
        "message": f"Detected {len(items)} items. Select and save the ones you want.",
        "overlay_preview_base64": overlay_base64,
        "items": items,
        "meta": {
            "pipeline": "single_shot_llama" if llm_items else "single_shot_fallback",
            "llm_error": llm_error or None,
        },
    }


@router.post("/analyze/async", status_code=status.HTTP_202_ACCEPTED)
def analyze_capture_async(http_request: Request, request: CaptureAnalyzeRequest):
    if capture_analyze_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        request_id = str(getattr(http_request.state, "request_id", "") or "")
        task_id = enqueue_task(
            task_func=capture_analyze_task,
            args=[request.user_id, request.image_base64],
            kwargs={"request_id": request_id},
            kind="capture_analyze",
            user_id=request.user_id,
            request_id=request_id,
            source="api:/api/wardrobe/capture/analyze/async",
            meta={"task_type": "capture_analyze_task"},
        )
        return {
            "success": True,
            "status": "queued",
            "task_id": task_id,
            "task_type": "capture_analyze_task",
            "request_id": request_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue capture analyze: {exc}")


@router.post("/save-selected")
def save_selected(request: SaveSelectedRequest):
    return save_selected_core(
        user_id=request.user_id,
        selected_item_ids=request.selected_item_ids,
        detected_items=request.detected_items,
    )


def _normalize_detected_items(items: List[Any]) -> List[DetectedItem]:
    normalized: List[DetectedItem] = []
    for item in items or []:
        if isinstance(item, DetectedItem):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(DetectedItem(**item))
        else:
            if hasattr(item, "model_dump"):
                normalized.append(DetectedItem(**item.model_dump()))
            else:
                normalized.append(DetectedItem(**item.dict()))
    return normalized


def save_selected_core(
    *,
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Any],
):
    selected = set(selected_item_ids or [])
    if not selected:
        raise HTTPException(status_code=400, detail="selected_item_ids cannot be empty")

    normalized_items = _normalize_detected_items(detected_items)
    duplicate_threshold = _duplicate_threshold()
    pixel_max_distance = _pixel_duplicate_distance()
    image_duplicate_threshold = _image_duplicate_threshold()
    normalized_payload = [
        item.model_dump() if hasattr(item, "model_dump") else item.dict()
        for item in normalized_items
    ]
    return persist_selected_items(
        user_id=user_id,
        selected_item_ids=list(selected),
        detected_items=normalized_payload,
        duplicate_threshold=duplicate_threshold,
        pixel_max_distance=pixel_max_distance,
        image_duplicate_threshold=image_duplicate_threshold,
    )


@router.post("/save-selected/async", status_code=status.HTTP_202_ACCEPTED)
def save_selected_async(http_request: Request, request: SaveSelectedRequest):
    if capture_save_selected_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        detected = []
        for item in request.detected_items:
            if hasattr(item, "model_dump"):
                detected.append(item.model_dump())
            else:
                detected.append(item.dict())
        payload = {
            "user_id": request.user_id,
            "selected_item_ids": request.selected_item_ids,
            "detected_items": detected,
        }
        request_id = str(getattr(http_request.state, "request_id", "") or "")
        task_id = enqueue_task(
            task_func=capture_save_selected_task,
            args=[payload],
            kwargs={"request_id": request_id},
            kind="capture_save_selected",
            user_id=request.user_id,
            request_id=request_id,
            source="api:/api/wardrobe/capture/save-selected/async",
            meta={"task_type": "capture_save_selected_task"},
        )
        return {
            "success": True,
            "status": "queued",
            "task_id": task_id,
            "task_type": "capture_save_selected_task",
            "request_id": request_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue save-selected: {exc}")


@router.post("/process-upload/async", status_code=status.HTTP_202_ACCEPTED)
def process_upload_async(http_request: Request, request: ProcessUploadRequest):
    if process_upload_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        request_id = str(getattr(http_request.state, "request_id", "") or "")
        task_id = enqueue_task(
            task_func=process_upload_task,
            args=[request.user_id, request.image_base64],
            kwargs={"request_id": request_id},
            kind="process_upload",
            user_id=request.user_id,
            request_id=request_id,
            source="api:/api/wardrobe/capture/process-upload/async",
            meta={"task_type": "process_upload"},
        )
        return {
            "success": True,
            "status": "queued",
            "task_id": task_id,
            "task_type": "process_upload",
            "request_id": request_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue process-upload: {exc}")