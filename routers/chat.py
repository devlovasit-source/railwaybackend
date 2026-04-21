from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import json
import re
import os
import logging

from deep_translator import GoogleTranslator

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

from brain.orchestrator import ahvi_orchestrator
from brain.outfit_pipeline import save_feedback
try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task

# 🔥 NEW
from services.weather_service import get_hourly_weather
from services.thread_memory_service import thread_memory_service

router = APIRouter()
logger = logging.getLogger("ahvi.routers.chat")

def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


MAX_CHAT_MESSAGES = _env_int("CHAT_MESSAGES_MAX", 300, 30, 1000)
MAX_HISTORY_CONTEXT = _env_int("CHAT_HISTORY_CONTEXT_MAX", 200, 20, 800)
MAX_HISTORY_CONTEXT_CHARS = _env_int("CHAT_HISTORY_CONTEXT_MAX_CHARS", 14000, 2000, 120000)


def _build_history(messages: List["Message"], limit: int = MAX_HISTORY_CONTEXT) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages[-limit:]:
        role = str(getattr(msg, "role", "user")).lower()
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        history.append({"role": role, "text": content})
    return history


def _parse_memory_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return decoded
            return {"summary": text}
        except Exception:
            return {"summary": text}
    return {}


def _merge_history(memory_history: List[Dict[str, Any]], request_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base: List[Dict[str, str]] = []
    incoming: List[Dict[str, str]] = []

    for row in memory_history:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        text = str(row.get("text") or row.get("content") or "").strip()
        if role and text:
            base.append({"role": role, "text": text})

    for row in request_history:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        text = str(row.get("text") or row.get("content") or "").strip()
        if role and text:
            incoming.append({"role": role, "text": text})

    overlap = 0
    max_overlap = min(len(base), len(incoming))
    for size in range(max_overlap, 0, -1):
        if base[-size:] == incoming[:size]:
            overlap = size
            break

    merged = [*base, *incoming[overlap:]]
    cleaned: List[Dict[str, Any]] = []
    last_role = ""
    last_text = ""
    for row in merged:
        role = str(row.get("role") or "").strip().lower()
        text = str(row.get("text") or "").strip()
        if not role or not text:
            continue
        if role == last_role and text == last_text:
            continue
        cleaned.append({"role": role, "text": text})
        last_role = role
        last_text = text
    return cleaned[-MAX_HISTORY_CONTEXT:]


def _prune_history_by_chars(history: List[Dict[str, Any]], max_chars: int = MAX_HISTORY_CONTEXT_CHARS) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    budget = max(500, int(max_chars))
    for row in reversed(history or []):
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower()
        text = str(row.get("text") or row.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not text:
            continue
        cost = len(text)
        if rows and budget - cost < 0:
            break
        if cost > budget and not rows:
            rows.append({"role": role, "text": text[-budget:]})
            budget = 0
            break
        rows.append({"role": role, "text": text})
        budget -= cost
        if budget <= 0:
            break
    rows.reverse()
    return rows


def _sanitize_thread_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    safe: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", ":"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:160]


# -------------------------
# MODELS
# -------------------------
class Message(BaseModel):
    role: str = Field(..., min_length=1, max_length=24)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        role = str(value or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of user/assistant/system")
        return role


class TextChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=MAX_CHAT_MESSAGES)
    language: str = Field(default="en", min_length=2, max_length=8)
    current_memory: Any = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    wardrobe_items: List[Dict[str, Any]] = Field(default_factory=list)
    wardrobe_attached: bool = False
    user_id: str | None = None
    userID: str | None = None
    module_context: str | None = None
    thread_id: str | None = None
    threadId: str | None = None


class OutfitFeedbackRequest(BaseModel):
    user_id: str
    feedback: str
    outfit: Dict[str, Any]


class OrganizeHubRequest(BaseModel):
    user_id: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)
    include_counts: bool = False


class PlanPackRequest(BaseModel):
    user_id: str
    prompt: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)
    wardrobe_items: List[Dict[str, Any]] = Field(default_factory=list)


class DailyCardsRequest(BaseModel):
    user_id: str
    time_slot: str | None = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)


# -------------------------
# CHAT ENDPOINT
# -------------------------
@router.post("/text")
def text_chat(request: TextChatRequest, http_request: Request):

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_input = request.messages[-1].content.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    # -------------------------
    # LANGUAGE DETECTION
    # -------------------------
    try:
        preferred_lang = str(request.language or "en").lower()
        has_telugu = bool(re.search(r"[\u0C00-\u0C7F]", user_input))
        has_hindi = bool(re.search(r"[\u0900-\u097F]", user_input))

        if preferred_lang == "te" or has_telugu:
            english_input = GoogleTranslator(source="te", target="en").translate(user_input)
            target_lang = "te"
        elif preferred_lang == "hi" or has_hindi:
            english_input = GoogleTranslator(source="hi", target="en").translate(user_input)
            target_lang = "hi"
        else:
            english_input = user_input
            target_lang = "en"

    except Exception:
        english_input = user_input
        target_lang = "en"

    # -------------------------
    # 🔥 WEATHER INJECTION (NEW)
    # -------------------------
    weather_data = {}

    try:
        location = request.user_profile.get("location", {})

        if location.get("lat") and location.get("lon"):
            weather_data = get_hourly_weather(
                lat=location.get("lat"),
                lon=location.get("lon")
            )
    except Exception as e:
        logger.warning("weather lookup failed user_id=%s error=%s", request.user_id or request.userID or "user_1", e)

    user_id = request.user_id or request.userID or "user_1"
    memory_payload = _parse_memory_payload(request.current_memory)
    requested_thread_id = request.thread_id or request.threadId or memory_payload.get("thread_id")
    fallback_thread = f"{str(request.module_context or 'chat').strip().lower() or 'chat'}:default"
    thread_id = _sanitize_thread_id(requested_thread_id) or _sanitize_thread_id(fallback_thread) or "chat_default"

    # -------------------------
    # ORCHESTRATOR CALL
    # -------------------------
    history = _build_history(request.messages[:-1], limit=MAX_HISTORY_CONTEXT) if len(request.messages) > 1 else []
    memory_history = memory_payload.get("history", []) if isinstance(memory_payload, dict) else []
    stored_memory = thread_memory_service.get(user_id=user_id, thread_id=thread_id)
    stored_history = stored_memory.get("history", []) if isinstance(stored_memory, dict) else []
    merged_summary = str(stored_memory.get("summary") or memory_payload.get("summary") or "").strip()
    stored_plus_memory = _merge_history(
        [h for h in stored_history if isinstance(h, dict)],
        [h for h in memory_history if isinstance(h, dict)],
    )
    merged_history = _merge_history(stored_plus_memory, history)
    merged_history = _prune_history_by_chars(merged_history, MAX_HISTORY_CONTEXT_CHARS)

    slot_hints: Dict[str, Any] = {}
    if request.module_context:
        module = str(request.module_context).lower()
        if "occasion" in module:
            slot_hints["occasion"] = request.user_profile.get("occasion")
        if "work" in module or "office" in module:
            slot_hints["occasion"] = slot_hints.get("occasion") or "office"

    try:
        result = ahvi_orchestrator.run(
            text=english_input,
            user_id=user_id,
            context={
                "memory": {
                    **memory_payload,
                    "summary": merged_summary,
                    "thread_id": thread_id,
                },
                "user_profile": request.user_profile,
                "wardrobe_items": request.wardrobe_items,
                "wardrobe_attached": bool(request.wardrobe_attached),
                "module_context": request.module_context,
                "history": merged_history,
                "slots": slot_hints,
                "request_id": str(getattr(http_request.state, "request_id", "") or ""),

                # 🔥 NEW CONTEXT
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestrator failed: {exc}")

    # -------------------------
    # TRANSLATE RESPONSE BACK
    # -------------------------
    message = result.get("message", "")

    try:
        if target_lang != "en" and message:
            message = GoogleTranslator(source="en", target=target_lang).translate(message)
    except Exception:
        pass

    # -------------------------
    # AUDIO (OPTIONAL)
    # -------------------------
    try:
        if (
            run_heavy_audio_task is not None
            and os.getenv("ENABLE_AUDIO_TASKS", "false").lower() in ("1", "true", "yes")
        ):
            request_id = str(getattr(http_request.state, "request_id", "") or "")
            audio_job_id = enqueue_task(
                task_func=run_heavy_audio_task,
                args=[message, target_lang],
                kwargs={"request_id": request_id},
                kind="audio_generate",
                user_id=user_id,
                request_id=request_id,
                source="api:/api/text",
                meta={"task_type": "generate_audio"},
            )
        else:
            audio_job_id = "offline"
    except Exception:
        audio_job_id = "offline"

    persisted = thread_memory_service.append_turns(
        user_id=user_id,
        thread_id=thread_id,
        turns=[
            {"role": "user", "text": english_input},
            {"role": "assistant", "text": message},
        ],
        incoming_summary=str(memory_payload.get("summary") or ""),
    )

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    return {
        "success": True,
        "message": message,
        "board": result.get("board"),
        "type": result.get("type"),
        "cards": result.get("cards", []),
        "chips": result.get("chips", []),
        "board_ids": result.get("board_ids"),
        "pack_ids": result.get("pack_ids"),
        "data": result.get("data", {}),
        "meta": {
            **result.get("meta", {}),
            "weather": weather_data,
            "history_used": len(merged_history),
            "thread_id": thread_id,
        },
        "thread_id": thread_id,
        "updated_memory": persisted.get("summary", ""),
        "audio_job_id": audio_job_id,
    }


# -------------------------
# FEEDBACK
# -------------------------
@router.post("/feedback/outfit")
def outfit_feedback(request: OutfitFeedbackRequest):

    fb = str(request.feedback).strip().lower()

    mapped = "up" if fb in ("up", "like", "liked", "thumbs_up", "👍") else "down"

    if fb not in (
        "up", "down", "like", "liked", "dislike", "disliked",
        "thumbs_up", "thumbs_down", "👍", "👎"
    ):
        raise HTTPException(status_code=400, detail="feedback must be up/down")

    try:
        return save_feedback(
            user_id=request.user_id,
            outfit=request.outfit,
            feedback=mapped
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {exc}")


@router.post("/organize/chips")
def organize_chips(request: OrganizeHubRequest, http_request: Request):
    try:
        result = ahvi_orchestrator.run(
            text="open organize",
            user_id=request.user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": "organize",
                "history": [],
                "include_counts": request.include_counts,
                "request_id": str(getattr(http_request.state, "request_id", "") or ""),
            },
        )
        return {
            "success": True,
            "message": result.get("message", "Choose what you want to organize."),
            "board": result.get("board", "organize"),
            "type": result.get("type", "chips"),
            "chips": result.get("cards", []),
            "data": result.get("data", {}),
            "meta": result.get("meta", {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load organize chips: {exc}")


@router.post("/plan-pack")
def plan_pack(request: PlanPackRequest, http_request: Request):
    try:
        weather_data = {}
        try:
            location = request.user_profile.get("location", {})
            if location.get("lat") and location.get("lon"):
                weather_data = get_hourly_weather(
                    lat=location.get("lat"),
                    lon=location.get("lon")
                )
        except Exception:
            weather_data = {}

        result = ahvi_orchestrator.run(
            text=request.prompt,
            user_id=request.user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "wardrobe_items": request.wardrobe_items,
                "module_context": "plan_pack",
                "history": [],
                "request_id": str(getattr(http_request.state, "request_id", "") or ""),
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
                "weather_data": weather_data,
            },
        )
        return {
            "success": True,
            "message": result.get("message", ""),
            "board": result.get("board", "plan_pack"),
            "type": result.get("type", "checklists"),
            "cards": result.get("cards", []),
            "chips": result.get("chips", []),
            "pack_ids": result.get("pack_ids"),
            "data": result.get("data", {}),
            "meta": {
                **result.get("meta", {}),
                "weather": weather_data,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build plan & pack flow: {exc}")


@router.post("/daily/cards")
@router.post("/text/daily/cards")
def daily_cards(request: DailyCardsRequest, http_request: Request):
    try:
        weather_data = {}
        try:
            location = request.user_profile.get("location", {})
            if location.get("lat") and location.get("lon"):
                weather_data = get_hourly_weather(
                    lat=location.get("lat"),
                    lon=location.get("lon"),
                )
        except Exception:
            weather_data = {}

        slot_hint = (request.time_slot or "").strip().lower() if request.time_slot else ""
        prompt = f"{slot_hint} daily cards".strip() if slot_hint else "daily cards"

        result = ahvi_orchestrator.run(
            text=prompt,
            user_id=request.user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": "daily_dependency",
                "history": [],
                "request_id": str(getattr(http_request.state, "request_id", "") or ""),
                "time_slot": slot_hint or None,
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
                "weather_data": weather_data,
            },
        )

        return {
            "success": True,
            "message": result.get("message", ""),
            "board": result.get("board", "daily_dependency"),
            "type": result.get("type", "cards"),
            "cards": result.get("cards", [])[:3],
            "data": result.get("data", {}),
            "meta": {
                **result.get("meta", {}),
                "weather": weather_data,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build daily cards: {exc}")
