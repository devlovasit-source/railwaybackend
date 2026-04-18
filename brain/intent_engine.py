import json
from typing import Dict, Any

from services.ai_gateway import generate_text


INTENT_PROMPT = """
You are an intent classification engine for an AI stylist and organizer app.

Return ONLY JSON.

Schema:
{
  "intent": "daily_dependency | daily_outfit | occasion_outfit | explore_styles | wardrobe_query | try_on | organize_hub | plan_pack | general",
  "slots": {
    "occasion": "string or null",
    "style": "string or null",
    "vibe": "string or null",
    "time": "morning | midday | afternoon | evening | night | null",
    "module": "life_boards | meal_planner | medicines | bills | calendar | workout | skincare | contacts | life_goals | null"
  },
  "confidence": 0.0-1.0
}

Rules:
- "morning plan / daily cards / today plan / tomorrow preview" -> daily_dependency
- "what should I wear today" -> daily_outfit
- wedding/party/event -> occasion_outfit
- "show styles / casual / trending" -> explore_styles
- "how many tops do I have / count my wardrobe items" -> wardrobe_query
- "try this / try on" -> try_on
- "organize / life planner / bills / medicines / calendar / workout / skincare / contacts / goals" -> organize_hub
- "plan trip / pack for travel / wedding checklist / business travel packing" -> plan_pack
- Fill slots if clearly mentioned
- If unsure -> general

User:
"""


def _safe_parse(text: str) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"intent": "general", "slots": {}, "confidence": 0.3}


def _fallback_intent(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    slots: Dict[str, Any] = {}

    if "office" in t or "work" in t:
        slots["occasion"] = "office"
    elif "party" in t:
        slots["occasion"] = "party"
    elif "wedding" in t:
        slots["occasion"] = "wedding"
    elif "date" in t:
        slots["occasion"] = "date_night"

    if "morning" in t:
        slots["time"] = "morning"
    elif "midday" in t or "noon" in t:
        slots["time"] = "midday"
    elif "afternoon" in t:
        slots["time"] = "afternoon"
    elif "evening" in t:
        slots["time"] = "evening"
    elif "night" in t:
        slots["time"] = "night"

    if any(x in t for x in ["wear", "outfit", "style me", "recommend look", "what should i wear", "dress me"]):
        return {"intent": "daily_outfit", "slots": slots, "confidence": 0.68}

    daily_words = [
        "daily plan", "daily cards", "morning flow", "midday flow", "afternoon flow",
        "evening flow", "night flow", "tomorrow preview", "day planner", "daily dependency",
    ]
    if any(x in t for x in daily_words):
        return {"intent": "daily_dependency", "slots": slots, "confidence": 0.8}

    if any(x in t for x in ["wedding", "party", "event"]):
        return {"intent": "occasion_outfit", "slots": slots or {"occasion": "event"}, "confidence": 0.7}

    if any(x in t for x in ["try", "try on", "virtual try", "preview this"]):
        return {"intent": "try_on", "slots": slots, "confidence": 0.72}

    if any(x in t for x in ["trend", "style ideas", "inspiration", "new styles"]):
        return {"intent": "explore_styles", "slots": slots, "confidence": 0.62}

    count_words = ["how many", "count", "number of", "total", "do i have"]
    wardrobe_words = [
        "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
        "tshirt", "t-shirt", "pants", "trousers", "jeans", "bottoms", "shoes",
        "footwear", "dress", "dresses", "accessories", "jewelry", "bags", "bag",
    ]
    if any(x in t for x in count_words) and any(x in t for x in wardrobe_words):
        return {"intent": "wardrobe_query", "slots": slots, "confidence": 0.8}

    organize_words = [
        "organize", "life board", "meal planner", "medicine", "meds", "bills",
        "calendar", "workout", "skincare", "contacts", "life goals", "goals"
    ]
    if any(x in t for x in organize_words):
        if "life board" in t:
            slots["module"] = "life_boards"
        elif "meal" in t:
            slots["module"] = "meal_planner"
        elif "med" in t:
            slots["module"] = "medicines"
        elif "bill" in t:
            slots["module"] = "bills"
        elif "calendar" in t:
            slots["module"] = "calendar"
        elif "workout" in t:
            slots["module"] = "workout"
        elif "skin" in t:
            slots["module"] = "skincare"
        elif "contact" in t:
            slots["module"] = "contacts"
        elif "goal" in t:
            slots["module"] = "life_goals"
        return {"intent": "organize_hub", "slots": slots, "confidence": 0.75}

    plan_pack_words = [
        "plan trip", "trip plan", "travel plan", "packing list", "pack for",
        "pack my", "business travel", "wedding checklist", "checklist for trip",
        "goa trip", "vacation packing"
    ]
    if any(x in t for x in plan_pack_words):
        return {"intent": "plan_pack", "slots": slots, "confidence": 0.78}

    return {"intent": "general", "slots": slots, "confidence": 0.4}


def detect_intent(user_text: str, history=None, model: str | None = None) -> Dict[str, Any]:
    if not user_text:
        return {"intent": "general", "slots": {}, "confidence": 0.0}

    # Fast deterministic path first; avoids unnecessary model latency for obvious intents.
    fallback = _fallback_intent(user_text)
    if float(fallback.get("confidence", 0.0)) >= 0.75:
        return fallback

    prompt = INTENT_PROMPT + user_text
    response = generate_text(
        prompt,
        options={"temperature": 0.2, "num_predict": 200},
        usecase="intent",
        model=model,
    )

    parsed = _safe_parse(response)

    if not parsed.get("intent"):
        parsed = fallback
    elif str(response).strip().lower() in ("none", ""):
        parsed = fallback
    elif parsed.get("intent") == "general" and float(parsed.get("confidence", 0.0)) < 0.55:
        parsed = fallback

    if history:
        last = history[-1] if history else {}
        if parsed["intent"] == "general" and last.get("intent"):
            parsed["intent"] = last.get("intent")
            parsed["confidence"] = max(parsed.get("confidence", 0.0), 0.6)

    return parsed
