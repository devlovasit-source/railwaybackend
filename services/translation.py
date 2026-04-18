# backend/services/translation.py

import re
from deep_translator import GoogleTranslator
from functools import lru_cache

# =========================
# FAST LANGUAGE DETECTION (NO LLM)
# =========================
def dynamic_nlp_language_detector(text: str) -> str:
    if not text or len(text.strip()) < 2:
        return "english"

    text = text.strip()

    # Native script detection (FAST)
    if re.search(r'[\u0C00-\u0C7F]', text):
        return "telugu_script"

    if re.search(r'[\u0900-\u097F]', text):
        return "hindi_script"

    # Romanized detection heuristics
    lower = text.lower()

    hinglish_keywords = ["hai", "kaise", "kya", "nahi", "haan", "acha"]
    tanglish_keywords = ["enna", "epadi", "irukka", "seri", "illa"]

    if any(word in lower for word in hinglish_keywords):
        return "hinglish"

    if any(word in lower for word in tanglish_keywords):
        return "tanglish"

    return "english"


# =========================
# TRANSLATOR CACHE
# =========================
@lru_cache(maxsize=10)
def get_translator(source: str, target: str):
    return GoogleTranslator(source=source, target=target)


# =========================
# TRANSLITERATE + TRANSLATE
# =========================
def transliterate_and_translate(text: str, target_lang_code: str) -> str:
    try:
        if not text or len(text) > 2000:
            return text

        translator = get_translator(target_lang_code, 'en')
        return translator.translate(text)

    except Exception as e:
        print(f"⚠️ Transliteration Error: {e}")
        return text


# =========================
# EN → SCRIPT
# =========================
def translate_to_script_and_romanized(english_text: str, target_lang_code: str) -> dict:
    try:
        if not english_text:
            return {"native_script": "", "romanized": ""}

        translator = get_translator('en', target_lang_code)
        native_script = translator.translate(english_text[:2000])

        return {
            "native_script": native_script,
            "romanized": native_script  # fallback
        }

    except Exception as e:
        print(f"⚠️ Script Translation Error: {e}")
        return {
            "native_script": english_text,
            "romanized": english_text
        }


# =========================
# ROMANIZED STYLE GENERATION (LIGHTWEIGHT)
# =========================
def generate_natural_romanized(english_text: str, style: str) -> str:

    if style == "english" or not english_text:
        return english_text

    # 🔥 Avoid LLM for simple cases (speed boost)
    basic_map = {
        "hinglish": lambda t: t.replace("you", "tum").replace("are", "ho"),
        "tanglish": lambda t: t.replace("you", "nee").replace("are", "irukka")
    }

    try:
        if style in basic_map:
            return basic_map[style](english_text)

        return english_text

    except Exception as e:
        print(f"⚠️ Romanized Error: {e}")
        return english_text