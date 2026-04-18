import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from brain.tone.tone_engine import tone_engine

# =========================
# CONFIG
# =========================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_MODEL_FALLBACKS",
        "llama3.1:latest,llama3.1",
    ).split(",")
    if m.strip()
]
ALLOW_HEAVY_MODELS = os.getenv("OLLAMA_ALLOW_HEAVY_MODELS", "false").lower() in {"1", "true", "yes"}
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))


def _is_heavy_model(model_name: str) -> bool:
    m = str(model_name or "").lower()
    heavy_markers = [":7b", ":8b", ":9b", ":13b", ":14b", ":32b", ":70b", "latest", "vl"]
    return any(marker in m for marker in heavy_markers)

# =========================
# SESSION WITH RETRIES
# =========================
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


def _model_candidates(requested_model: str | None) -> list[str]:
    ordered: list[str] = []
    requested = str(requested_model or "").strip()
    if requested and (ALLOW_HEAVY_MODELS or not _is_heavy_model(requested)):
        first_model = requested
    else:
        first_model = DEFAULT_MODEL

    for index, model in enumerate([first_model, *MODEL_FALLBACKS]):
        m = str(model or "").strip()
        if not m:
            continue
        # Always honor the configured first model (DEFAULT_MODEL/requested),
        # then filter heavy fallback models when disabled.
        if index > 0 and not ALLOW_HEAVY_MODELS and _is_heavy_model(m):
            continue
        if m not in ordered:
            ordered.append(m)
    return ordered


def _merged_options(incoming: dict | None) -> dict:
    merged = {
        "num_ctx": DEFAULT_NUM_CTX,
        "num_predict": DEFAULT_NUM_PREDICT,
        "temperature": 0.7,
    }
    if incoming:
        merged.update(incoming)
    return merged


def _stylist_guidance(user_profile=None, signals=None) -> str:
    user_profile = user_profile or {}
    signals = signals or {}
    context_mode = str(signals.get("context_mode", "general")).lower()
    if context_mode != "styling":
        return ""

    preferred_colors = user_profile.get("preferred_colors", user_profile.get("colors", []))
    style = user_profile.get("style", "")
    body_type = user_profile.get("body_type", "")
    budget = user_profile.get("budget", "")

    return f"""
Advanced Stylist Rules:
- Prioritize occasion, weather, and comfort first.
- Use wardrobe-aware recommendations and avoid generic trends.
- Give one best choice first, then one alternative.
- Add one practical upgrade (accessory, layer, or color swap).
- Mention confidence rationale in plain language.
- Keep output actionable and premium.

User style profile:
- style: {style}
- preferred colors: {preferred_colors}
- body type: {body_type}
- budget: {budget}
"""


# =========================
# SAFE REQUEST HANDLER
# =========================
def safe_request(endpoint: str, payload: dict, timeout: int = 30):
    candidates = _model_candidates(payload.get("model"))
    last_error = ""

    for model in candidates:
        try:
            local_payload = dict(payload)
            local_payload["model"] = model
            local_payload["options"] = _merged_options(local_payload.get("options"))
            response = session.post(f"{OLLAMA_URL}/{endpoint}", json=local_payload, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    data["_model_used"] = model
                return data

            last_error = f"{response.status_code}: {response.text}"
            # Try next model when model is missing.
            if response.status_code == 404 and "not found" in response.text.lower():
                continue

        except Exception as e:
            last_error = str(e)
            continue

    if last_error:
        print(f"OLLAMA ERROR ({endpoint}): {last_error}")
    return None


# =========================
# TONE-AWARE TEXT GENERATION
# =========================
def generate_text(
    prompt: str,
    options: dict = None,
    user_profile=None,
    signals=None,
    model: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    if not prompt:
        return "none"

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
You are AHVI, a premium AI fashion stylist.

Tone Instructions:
{tone.get("tone_instruction", "")}

Guidelines:
- Be natural and human
- Keep responses concise
- Sound confident but not arrogant
- Avoid robotic phrasing

{_stylist_guidance(user_profile=user_profile, signals=signals)}

{prompt}
"""

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }

    if options:
        payload["options"] = options

    data = safe_request("generate", payload, timeout=int(timeout_seconds or 30))

    if not data:
        return "none"

    response = data.get("response", "").strip() or "none"
    response = tone_engine.apply(response, user_profile=user_profile, signals=signals)
    return response


# =========================
# CHAT COMPLETION
# =========================
def chat_completion(
    messages: list,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
    user_profile=None,
    signals=None,
    timeout_seconds: int | None = None,
) -> str:

    if not messages:
        return "I didn't catch that!"

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    formatted_messages = []

    system_msg = f"""
You are AHVI, an AI fashion stylist.

Tone:
{tone.get("tone_instruction", "")}

Rules:
- Speak naturally
- Keep it concise
- Be stylish and modern

{_stylist_guidance(user_profile=user_profile, signals=signals)}
"""

    if system_instruction:
        system_msg += "\n" + system_instruction[:2000]

    formatted_messages.append({"role": "system", "content": system_msg})

    safe_messages = messages[-10:]

    for msg in safe_messages:
        role = str(msg.get("role", "user")).lower()
        if role not in ["user", "assistant", "system"]:
            role = "assistant"

        content = str(msg.get("content", ""))[:4000]

        if content:
            formatted_messages.append({"role": role, "content": content})

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": formatted_messages,
        "stream": False,
    }

    data = safe_request("chat", payload, timeout=int(timeout_seconds or 45))

    if not data:
        return "I'm having trouble thinking right now. Try again in a moment."

    try:
        response = data.get("message", {}).get("content", "").strip()
        response = tone_engine.apply(response, user_profile=user_profile, signals=signals)
        return response or "Something went wrong."
    except Exception:
        return "AI response parsing failed."


# =========================
# WARDROBE FORMATTER
# =========================
def format_wardrobe_for_llm(items):
    if not items:
        return "The user's wardrobe is empty."

    msg = "User wardrobe:\n"

    for item in items[:50]:
        category = item.get("category_group", "")
        sub = item.get("subcategory", "")
        color = item.get("colors", {}).get("primary", "") if isinstance(item.get("colors"), dict) else item.get("color", "")
        msg += f"- {color} {sub} ({category})\n"

    return msg


# =========================
# OUTFIT EXPLANATION
# =========================
def generate_outfit_explanation(outfits: list, context: str = "", user_profile=None, signals=None):
    prompt = f"""
User wardrobe:
{context}

Outfits:
{outfits}

Explain:
- why these outfits work
- when to wear them

Keep it short (2-3 lines).
"""
    return generate_text(prompt, user_profile=user_profile, signals=signals)


# =========================
# STYLE ADVICE
# =========================
def generate_style_advice(user_input: str, wardrobe_summary: str, user_profile=None, signals=None):
    prompt = f"""
User request:
{user_input}

Wardrobe:
{wardrobe_summary}

Give practical styling advice using available wardrobe.
Keep it concise and helpful.
"""
    return generate_text(prompt, user_profile=user_profile, signals=signals)


# =========================
# SMART RESPONSE GENERATOR
# =========================
def generate_ai_response(user_input: str, outfits: list, wardrobe_items: list, user_profile=None, signals=None):
    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(outfits, wardrobe_summary, user_profile=user_profile, signals=signals)

    return generate_style_advice(user_input, wardrobe_summary, user_profile=user_profile, signals=signals)
