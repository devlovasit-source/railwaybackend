import re
from typing import Any, Dict


_CODE_FENCE_RE = re.compile(r"```(?:json|python|text)?|```", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def to_plain_text(value: Any, *, fallback: str = "I can help with that.") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback

    text = _CODE_FENCE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r", "\n")
    text = _MULTISPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    text = text.strip()
    if not text:
        return fallback
    if len(text) > 2000:
        text = text[:2000].rstrip() + "..."
    return text


def _to_two_sentences(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "I can help with that. What vibe are we going for?"

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(raw) if p.strip()]
    if len(parts) >= 2:
        combined = f"{parts[0]} {parts[1]}".strip()
    else:
        first = parts[0] if parts else raw
        first = first.rstrip(".!?")
        combined = f"{first}. What vibe are we going for?"

    if "?" not in combined:
        combined = combined.rstrip(".!?") + ". What vibe are we going for?"
    return combined.strip()


def clean_llm_json_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _CODE_FENCE_RE.sub("", text).strip()
    # Remove obvious preambles before the first JSON token.
    first_obj = text.find("{")
    first_arr = text.find("[")
    starts = [x for x in (first_obj, first_arr) if x >= 0]
    if starts:
        text = text[min(starts):].strip()
    # Remove trailing commas before object/array close tokens.
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def validate_orchestrator_response(
    payload: Dict[str, Any] | Any,
    *,
    request_id: str = "",
) -> Dict[str, Any]:
    row = dict(payload) if isinstance(payload, dict) else {}

    # Required top-level safety defaults.
    row["success"] = bool(row.get("success", True))
    row["request_id"] = str(row.get("request_id") or request_id or "")
    row["message"] = to_plain_text(
        row.get("message"),
        fallback="I can help with that.",
    )
    row["message"] = _to_two_sentences(row["message"])

    cards = row.get("cards", [])
    row["cards"] = cards if isinstance(cards, list) else []
    data = row.get("data", {})
    row["data"] = data if isinstance(data, dict) else {}
    meta = row.get("meta", {})
    row["meta"] = meta if isinstance(meta, dict) else {}
    row["board"] = str(row.get("board") or "general")
    row["type"] = str(row.get("type") or "text")

    return row
