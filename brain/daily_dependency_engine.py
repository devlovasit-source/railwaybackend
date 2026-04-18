from datetime import datetime
from typing import Any, Dict, List, Optional

from services.appwrite_proxy import AppwriteProxy
from brain.decision_engine import decision_engine


TIME_SLOTS = ("morning", "midday", "afternoon", "evening", "night")


def _normalize_documents(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [doc for doc in payload if isinstance(doc, dict)]
    if isinstance(payload, dict):
        docs = payload.get("documents", [])
        if isinstance(docs, list):
            return [doc for doc in docs if isinstance(doc, dict)]
    return []


def _time_slot_from_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "evening"
    return "night"


def _resolve_time_slot(context: Dict[str, Any]) -> str:
    requested = str(context.get("time_slot") or "").lower().strip()
    if requested in TIME_SLOTS:
        return requested

    weather_data = context.get("weather_data", {}) if isinstance(context.get("weather_data"), dict) else {}
    weather_slot = str(weather_data.get("time_of_day") or context.get("time_of_day") or "").lower().strip()
    if weather_slot == "day":
        weather_slot = "afternoon"
    if weather_slot in TIME_SLOTS:
        return weather_slot

    return _time_slot_from_hour(datetime.now().hour)


def _resolve_persona(user_profile: Dict[str, Any]) -> str:
    profile = user_profile if isinstance(user_profile, dict) else {}
    p = str(profile.get("persona") or profile.get("life_stage") or profile.get("role") or "").lower()
    if profile.get("has_kids") is True or "parent" in p:
        return "busy_parent"
    if "student" in p:
        return "student"
    if "single" in p:
        return "single"
    return "working_individual"


def _count_resource(appwrite: AppwriteProxy, resource: str, user_id: str) -> int:
    try:
        docs = _normalize_documents(appwrite.list_documents(resource, user_id=user_id, limit=50))
        return len(docs)
    except Exception:
        return 0


def _first_title(appwrite: AppwriteProxy, resource: str, user_id: str, field: str = "title") -> Optional[str]:
    try:
        docs = _normalize_documents(appwrite.list_documents(resource, user_id=user_id, limit=1))
        if docs:
            value = docs[0].get(field) or docs[0].get("name")
            return str(value) if value else None
    except Exception:
        return None
    return None


def _card(
    card_type: str,
    title: str,
    reason: str,
    *,
    priority: int,
    notification_needed: bool = False,
    action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    card_id = f"{card_type}:{title}".lower().replace(" ", "_")
    return {
        "id": card_id,
        "type": card_type,
        "title": title,
        "reason": reason,
        "priority": int(priority),
        "notification_needed": bool(notification_needed),
        "action": action or {"type": "open_dashboard"},
    }


def _candidate_cards(
    *,
    time_slot: str,
    persona: str,
    counts: Dict[str, int],
    weather: str,
    next_event: Optional[str],
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    has_outfits = counts.get("outfits", 0) > 0
    has_meds = counts.get("meds", 0) > 0

    if time_slot == "morning":
        cards.append(
            _card(
                "outfit_suggestion",
                "Morning Outfit",
                "Styled for your morning context and weather.",
                priority=100 if has_outfits else 70,
                action={"type": "open_module", "module": "outfit", "route": "/outfit"},
            )
        )
        cards.append(
            _card(
                "skincare_routine",
                "AM Skincare",
                "Quick morning routine to start fresh.",
                priority=90,
                action={"type": "open_module", "module": "skincare", "route": "/organize/skincare"},
            )
        )
        cards.append(
            _card(
                "key_reminder",
                "Key Reminder",
                "Important medicine or event cue for this morning.",
                priority=95 if has_meds or next_event else 65,
                notification_needed=bool(has_meds or next_event),
                action={"type": "open_module", "module": "calendar", "route": "/organize/calendar"},
            )
        )

    elif time_slot == "midday":
        cards.append(
            _card(
                "meal_suggestion",
                "Midday Meal",
                "Suggested based on your daytime energy needs.",
                priority=95,
                action={"type": "open_module", "module": "meal_planner", "route": "/organize/meal-planner"},
            )
        )
        cards.append(
            _card(
                "grocery_alert",
                "Grocery Low Alert",
                "Check essentials before evening rush.",
                priority=85,
                notification_needed=True,
                action={"type": "open_module", "module": "meal_planner", "route": "/organize/meal-planner"},
            )
        )

    elif time_slot == "afternoon":
        cards.append(
            _card(
                "event_reminder",
                "Upcoming Event",
                f"Next event: {next_event or 'Check your schedule'}",
                priority=100 if next_event else 75,
                notification_needed=bool(next_event),
                action={"type": "open_module", "module": "calendar", "route": "/organize/calendar"},
            )
        )
        cards.append(
            _card(
                "task_focus",
                "Top Tasks",
                "Focus on the two highest-impact tasks now.",
                priority=90,
                action={"type": "open_module", "module": "life_boards", "route": "/organize/life-boards"},
            )
        )

    elif time_slot == "evening":
        cards.append(
            _card(
                "dinner_suggestion",
                "Dinner Suggestion",
                "Easy dinner card for your evening window.",
                priority=95,
                action={"type": "open_module", "module": "meal_planner", "route": "/organize/meal-planner"},
            )
        )
        cards.append(
            _card(
                "medication_reminder",
                "Medication Reminder",
                "Evening meds check-in.",
                priority=98 if has_meds else 60,
                notification_needed=bool(has_meds),
                action={"type": "open_module", "module": "medicines", "route": "/organize/medicines"},
            )
        )
        cards.append(
            _card(
                "outfit_switch",
                "Outfit Switch",
                "Optional switch for evening plans.",
                priority=75 if has_outfits else 55,
                action={"type": "open_module", "module": "outfit", "route": "/outfit"},
            )
        )

    else:  # night
        cards.append(
            _card(
                "tomorrow_preview",
                "Tomorrow Preview",
                "Quick look at next-day priorities.",
                priority=100,
                action={"type": "open_module", "module": "calendar", "route": "/organize/calendar"},
            )
        )
        cards.append(
            _card(
                "outfit_prep",
                "Outfit Prep",
                "Prepare tomorrow's outfit tonight.",
                priority=90 if has_outfits else 65,
                action={"type": "open_module", "module": "outfit", "route": "/outfit"},
            )
        )
        cards.append(
            _card(
                "pm_skincare",
                "PM Skincare",
                "Wind-down skincare sequence.",
                priority=85,
                action={"type": "open_module", "module": "skincare", "route": "/organize/skincare"},
            )
        )

    if persona == "busy_parent":
        cards.append(
            _card(
                "family_prep",
                "Family Prep",
                "Kid schedule and essentials check.",
                priority=99 if time_slot in ("morning", "evening") else 80,
                notification_needed=time_slot in ("morning", "evening"),
                action={"type": "open_module", "module": "calendar", "route": "/organize/calendar"},
            )
        )
    elif persona == "student":
        cards.append(
            _card(
                "study_priority",
                "Study Priority",
                "Top class/deadline item for this slot.",
                priority=92 if time_slot in ("afternoon", "night") else 75,
                action={"type": "open_module", "module": "life_goals", "route": "/organize/life-goals"},
            )
        )

    # Fallback generic suggestion (always available).
    cards.append(
        _card(
            "generic_fallback",
            "Quick Suggestion",
            f"Context-aware default for {time_slot} in {weather or 'mild'} conditions.",
            priority=50,
            action={"type": "open_dashboard"},
        )
    )
    return cards


def build_daily_dependency_response(
    *,
    user_id: str,
    context: Dict[str, Any],
    appwrite: Optional[AppwriteProxy] = None,
) -> Dict[str, Any]:
    app = appwrite or AppwriteProxy()
    user_profile = context.get("user_profile", {}) if isinstance(context.get("user_profile"), dict) else {}
    time_slot = _resolve_time_slot(context)
    persona = _resolve_persona(user_profile)
    weather = str(context.get("weather") or (context.get("weather_data", {}) or {}).get("condition") or "mild")

    counts = {
        "outfits": _count_resource(app, "outfits", user_id),
        "meal_plans": _count_resource(app, "meal_plans", user_id),
        "meds": _count_resource(app, "meds", user_id),
        "plans": _count_resource(app, "plans", user_id),
        "skincare": _count_resource(app, "skincare", user_id),
    }
    next_event = _first_title(app, "plans", user_id, field="title")

    candidates = _candidate_cards(
        time_slot=time_slot,
        persona=persona,
        counts=counts,
        weather=weather,
        next_event=next_event,
    )

    # Product rule: never show more than 3 cards.
    cards, decision_meta = decision_engine.rank_actions(
        candidates=candidates,
        context={
            "time_slot": time_slot,
            "persona": persona,
            "weather": weather,
            "counts": counts,
        },
        top_n=3,
    )

    return {
        "success": True,
        "message": f"{time_slot.title()} plan ready. I picked your top {len(cards)} actions.",
        "board": "daily_dependency",
        "type": "cards",
        "cards": cards,
        "data": {
            "time_slot": time_slot,
            "persona": persona,
            "counts": counts,
            "max_cards": 3,
            "weather": weather,
            "decision": decision_meta,
        },
        "meta": {
            "intent": "daily_dependency",
            "domain": "daily_planner",
            "rules_applied": [
                "max_3_cards",
                "actionable_notifications_only",
                "context_aware_output",
                "fallback_available",
                "decision_engine_v1",
            ],
        },
    }
