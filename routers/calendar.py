from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import traceback

from brain.archive.calendar_engine import calendar_engine
from brain.utils.calendar_formatter import build_calendar_checklist_bundle
from middleware.auth_middleware import get_current_user  # 🔥 NEW

router = APIRouter()


# =========================
# REQUEST MODELS
# =========================
class CalendarEventRequest(BaseModel):
    text: str = Field(..., min_length=2, description="User event text")


class DailyBriefRequest(BaseModel):
    date: str


# =========================
# EVENT PROCESSING
# =========================
@router.post("/calendar/process")
def process_event(
    req: CalendarEventRequest,
    user=Depends(get_current_user)  # 🔥 USER CONTEXT
):
    try:
        user_id = user["user_id"]

        # 🧠 Step 1: engine with user context (important upgrade)
        result = calendar_engine.process_event(req.text, user_id=user["user_id"])

        # 🧩 Step 2: UI formatter
        checklist_bundle = build_calendar_checklist_bundle(
            result.get("classification", {}),
            {
                "packing": result.get("packing", []),
                "prep_tasks": result.get("prep_tasks", []),
                "outfit": result.get("outfit", {})
            }
        )

        return {
            "success": True,
            "meta": {
                "user_id": user_id
            },
            "data": {
                **result,
                "checklist_bundle": checklist_bundle
            }
        }

    except Exception:
        print("❌ /calendar/process error:\n", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Calendar processing failed"
        )


# =========================
# QUICK CLASSIFICATION ONLY
# =========================
@router.post("/calendar/classify")
def classify_event(
    req: CalendarEventRequest,
    user=Depends(get_current_user)  # 🔥 protect endpoint
):
    try:
        result = calendar_engine.classify_event(req.text)

        return {
            "success": True,
            "data": result
        }

    except Exception:
        print("❌ /calendar/classify error:\n", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Classification failed"
        )


# =========================
# HEALTH CHECK
# =========================
@router.get("/calendar/health")
def calendar_health():
    return {
        "status": "ok",
        "engine": "calendar_engine_v3",
        "auth": "enabled",  # 🔥 added clarity
        "ready": True
    }