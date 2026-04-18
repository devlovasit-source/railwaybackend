import json
import uuid
from typing import Any, Dict, List

from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy


def _dress_options_from_wardrobe(raw_items: List[Dict[str, Any]]) -> List[str]:
    picks: List[str] = []
    seen = set()
    for row in raw_items:
        if not isinstance(row, dict):
            continue
        category_blob = " ".join(
            [
                str(row.get("category") or ""),
                str(row.get("sub_category") or ""),
                str(row.get("type") or ""),
                str(row.get("style") or ""),
                str(row.get("name") or ""),
                str(row.get("title") or ""),
            ]
        ).lower()
        if "dress" not in category_blob:
            continue
        label = str(row.get("name") or row.get("title") or row.get("sub_category") or "Dress").strip()
        if not label:
            label = "Dress"
        if label.lower() in seen:
            continue
        seen.add(label.lower())
        picks.append(f"Wear: {label}")
        if len(picks) >= 5:
            break
    return picks


def _common_dress_options() -> List[str]:
    return [
        "Wear: Light cotton day dress",
        "Wear: Easy midi dress",
        "Wear: Flowy maxi dress",
        "Wear: Simple black dress",
    ]


def _history_snippet(context: Dict[str, Any]) -> str:
    history = context.get("history", []) or []
    lines: List[str] = []
    for row in history[-12:]:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().lower() or "user"
        text = str(row.get("text") or row.get("content") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _load_wardrobe_items(user_id: str) -> List[Dict[str, Any]]:
    if not str(user_id or "").strip():
        return []
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=200)
        return [dict(doc) for doc in docs if isinstance(doc, dict)]
    except Exception:
        return []


def _wardrobe_summary(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in items[:60]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "name": str(row.get("name") or row.get("title") or row.get("sub_category") or "item").strip(),
                "category": str(row.get("category") or "").strip(),
                "sub_category": str(row.get("sub_category") or "").strip(),
                "color": str(row.get("color_code") or "").strip(),
                "pattern": str(row.get("pattern") or "").strip(),
            }
        )
    return out


def _planning_prompt(
    *,
    user_text: str,
    history_text: str,
    weather: str,
    time_of_day: str,
    wardrobe: List[Dict[str, str]],
) -> str:
    return f"""
You are AHVI, an expert trip planner and packing assistant.
Reason deeply from the conversation and produce practical decisions.
Do not use generic filler text.
Use simple, easy English.

User latest message:
{user_text}

Recent chat history:
{history_text or "none"}

Context:
- weather_hint: {weather or "unknown"}
- time_of_day_hint: {time_of_day or "unknown"}
- wardrobe_items: {json.dumps(wardrobe, ensure_ascii=False)}

Return ONLY valid JSON object with this schema:
{{
  "message": "short friendly summary in easy English",
  "cards": [
    {{
      "id": "string",
      "title": "string",
      "kind": "checklist",
      "items": ["string", "string"]
    }}
  ],
  "chips": ["string", "string", "string"],
  "data": {{
    "trip_days": 0,
    "destination": "string",
    "trip_style": "string",
    "bag_mode": "string",
    "notes": "string"
  }}
}}

Rules:
- 3 to 5 cards only.
- each card must have 4 to 8 concise checklist items.
- reflect carry-on / domestic / international / weather constraints if implied.
- use wardrobe context to suggest mix-and-match outfits when possible.
- avoid repeating the same sentence patterns.
""".strip()


def build_plan_pack_response(text: str, context: Dict[str, Any] | None = None, user_id: str = "") -> Dict[str, Any]:
    context = context or {}
    resolved_user_id = str(user_id or context.get("user_id") or "").strip()

    wardrobe_items = context.get("wardrobe_items")
    if not isinstance(wardrobe_items, list) or not wardrobe_items:
        wardrobe_items = _load_wardrobe_items(resolved_user_id)
    raw_wardrobe = [dict(item) for item in wardrobe_items if isinstance(item, dict)]
    wardrobe = _wardrobe_summary(raw_wardrobe)

    history_text = _history_snippet(context)
    weather = str(context.get("weather") or (context.get("weather_data") or {}).get("condition") or "").strip()
    time_of_day = str(context.get("time_of_day") or "").strip()

    prompt = _planning_prompt(
        user_text=str(text or "").strip(),
        history_text=history_text,
        weather=weather,
        time_of_day=time_of_day,
        wardrobe=wardrobe,
    )

    pack_id = f"pack-{uuid.uuid4().hex[:10]}"

    try:
        raw = ai_gateway.generate_text(
            prompt=prompt,
            usecase="general",
            signals={"context_mode": "planning", "reasoning_mode": "plan_pack"},
            request_id=str(context.get("request_id") or ""),
        )
        parsed = ai_gateway.parse_json_object(raw)
    except Exception:
        # LLM fallback path (still model-based, no static templates).
        fallback_raw = ai_gateway.generate_text(
            prompt=(
                "Create a simple trip packing checklist response in easy English for this user message: "
                f"{str(text or '').strip()}"
            ),
            usecase="general",
            signals={"context_mode": "planning", "reasoning_mode": "plan_pack_fallback"},
            request_id=str(context.get("request_id") or ""),
        )
        parsed = {
            "message": str(fallback_raw or "I can help with your trip packing plan.").strip(),
            "cards": [],
            "chips": [],
            "data": {},
        }

    cards = parsed.get("cards")
    if not isinstance(cards, list):
        cards = []
    normalized_cards: List[Dict[str, Any]] = []
    for idx, row in enumerate(cards[:5]):
        if not isinstance(row, dict):
            continue
        items = row.get("items")
        if not isinstance(items, list):
            items = []
        normalized_cards.append(
            {
                "id": str(row.get("id") or f"pack_{idx+1}").strip(),
                "title": str(row.get("title") or f"Checklist {idx+1}").strip(),
                "kind": "checklist",
                "items": [str(item).strip() for item in items if str(item).strip()][:8],
            }
        )

    dress_items = _dress_options_from_wardrobe(raw_wardrobe)
    dress_source = "wardrobe"
    if not dress_items:
        dress_items = _common_dress_options()
        dress_source = "common_fallback"
    if dress_items:
        normalized_cards.insert(
            0,
            {
                "id": "dress_options",
                "title": "Dress Options",
                "kind": "checklist",
                "items": dress_items,
            },
        )

    chips = parsed.get("chips")
    if not isinstance(chips, list):
        chips = []
    chips = [str(chip).strip() for chip in chips if str(chip).strip()][:3]

    data = parsed.get("data")
    if not isinstance(data, dict):
        data = {}
    data = dict(data)
    data["pack_id"] = pack_id
    data["wardrobe_count_used"] = len(wardrobe)
    data["dress_source"] = dress_source
    data["dress_options"] = dress_items
    data["source_text"] = text
    data["can_save_to_life_board"] = True

    base_message = str(parsed.get("message") or "Here is your trip packing plan.").strip()
    if dress_source == "common_fallback":
        base_message = (
            f"{base_message} Upload more items to your wardrobe so I can give you more specific picks."
        )

    return {
        "intent": "plan_pack",
        "message": base_message,
        "board": "plan_pack",
        "type": "checklists",
        "pack_ids": pack_id,
        "cards": normalized_cards,
        "chips": chips,
        "data": data,
    }
