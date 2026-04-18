from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import traceback

from middleware.auth_middleware import get_current_user

# 🔥 NEW: use orchestrator (NOT engine)
from brain.engines.calendar.calendar_utils import (
    classify_intent,
    format_event,
    build_reminder
)
router = APIRouter()


# =========================
# REQUEST MODELS
# =========================
class CalendarEventRequest(BaseModel):
    text: str = Field(..., min_length=2)


class DailyEventsRequest(BaseModel):
    events: list  # list of event dicts


# =========================
# SINGLE EVENT PIPELINE
# =========================
@router.post("/calendar/process")
def process_event(
    req: CalendarEventRequest,
    user=Depends(get_current_user)
):
    try:
        user_id = user["user_id"]

        event = {
            "title": req.text,
            "user_id": user_id
        }

        # 🔥 MASTER PIPELINE
        result = run_calendar_runtime(event)

        return {
            "success": True,
            "meta": {"user_id": user_id},
            "data": result
        }

    except Exception:
        print("❌ /calendar/process error:\n", traceback.format_exc())
        raise HTTPException(500, "Calendar processing failed")


# =========================
# DAILY BRIEFING PIPELINE
# =========================
@router.post("/calendar/daily")
def daily_briefing(
    req: DailyEventsRequest,
    user=Depends(get_current_user)
):
    try:
        user_id = user["user_id"]

        # attach user_id to all events
        events = [
            {**event, "user_id": user_id}
            for event in req.events
        ]

        result = run_daily_calendar_runtime(events)

        return {
            "success": True,
            "data": result
        }

    except Exception:
        print("❌ /calendar/daily error:\n", traceback.format_exc())
        raise HTTPException(500, "Daily briefing failed")


# =========================
# HEALTH CHECK
# =========================
@router.get("/calendar/health")
def calendar_health():
    return {
        "status": "ok",
        "engine": "calendar_orchestrator_v1",
        "auth": "enabled",
        "mode": "pipeline",
        "ready": True
    }
    from datetime import datetime, timedelta


# =========================
# PREP TASKS
# =========================
def build_prep_tasks(event):
    tasks = set()
    group = event.get("group")
    subtype = event.get("subtype")

    if group == "travel":
        tasks.update(["Check documents", "Pack essentials", "Set alarm", "Leave with buffer"])

    elif group == "social":
        tasks.add("Decide outfit")
        if subtype in ["wedding", "birthday_party", "cocktail"]:
            tasks.add("Check shoes and bag")

    elif group in ["kids", "school"]:
        tasks.update(["Pack child items", "Confirm pickup/reporting time"])

    elif group == "health":
        tasks.update(["Keep reports ready", "Leave with buffer"])
        if subtype == "lab_test":
            tasks.add("Check fasting instructions")

    elif group == "finance":
        tasks.update(["Keep payment method ready", "Clear before due window"])

    elif group == "work":
        if subtype in ["presentation", "interview"]:
            tasks.update(["Review deck/CV", "Charge laptop", "Set outfit aside"])
        else:
            tasks.add("Review agenda")

    else:
        tasks.add("Quick prep check")

    if event.get("dressCode"):
        tasks.add(f"Dress code: {event['dressCode']}")

    return list(tasks)


# =========================
# PACKING
# =========================
def build_packing_list(event):
    mapping = {
        "domestic_flight": ["ID", "Phone", "Wallet", "Charger", "Tickets"],
        "international_flight": ["Passport", "Visa", "Wallet", "Tickets"],
        "doctor_appointment": ["Reports", "ID"],
        "presentation": ["Laptop", "Charger", "Deck"],
    }
    return mapping.get(event.get("subtype"), [])


# =========================
# OUTFIT
# =========================
def build_outfit(event):
    rules = {
        "presentation": ["structured", "clean", "confident"],
        "wedding": ["event-ready", "occasionwear"],
        "gym_class": ["activewear", "breathable"],
    }

    subtype = event.get("subtype")
    if subtype in rules:
        return {
            "outfitKeywords": rules[subtype]
        }

    return None


# =========================
# BUFFER PLAN
# =========================
def build_buffer(event):
    try:
        start = datetime.fromisoformat(event.get("startAtISO"))
    except:
        return None

    leave_minutes = 30

    if event.get("group") == "travel":
        leave_minutes = 120

    leave_by = start - timedelta(minutes=leave_minutes)

    return {
        "leaveByISO": leave_by.isoformat()
    }


# =========================
# STRESS SCORE
# =========================
def compute_stress(event):
    score = 20

    if event.get("priority") == "critical":
        score += 25

    if event.get("group") == "travel":
        score += 20

    if event.get("subtype") in ["wedding", "presentation"]:
        score += 15

    return min(score, 100)


# =========================
# FOLLOWUPS
# =========================
def build_followups(event):
    subtype = event.get("subtype")
    followups = []

    if "flight" in subtype:
        followups.append("Check hotel")

    if subtype == "interview":
        followups.append("Send follow-up email")

    return followups


# =========================
# MAIN ENGINE
# =========================
def run_calendar_predictive_engine(event, preferences=None):

    return {
        "prepTasks": build_prep_tasks(event),
        "packingList": build_packing_list(event),
        "outfitPrompt": build_outfit(event),
        "bufferPlan": build_buffer(event),
        "stressLoadScore": compute_stress(event),
        "followupCandidates": build_followups(event),
        "linkedErrands": []
    }