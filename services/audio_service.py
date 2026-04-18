import base64
import asyncio
from typing import Optional

import edge_tts

from brain.tone.tone_engine import tone_engine


# -------------------------
# VOICE MAP (MULTI-LANG)
# -------------------------
VOICE_MAP = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
}


# -------------------------
# TONE → VOICE MODULATION
# -------------------------
def _get_voice_modulation(user_profile=None, signals=None):
    tone = tone_engine.build_prompt_tone(user_profile, signals)
    instruction = tone.get("tone_instruction", "").lower()

    rate = "+0%"
    pitch = "+0Hz"

    # calm / soft
    if "soft" in instruction or "calm" in instruction:
        rate = "-10%"
        pitch = "-2Hz"

    # energetic / high energy
    elif "energetic" in instruction or "high" in instruction:
        rate = "+12%"
        pitch = "+3Hz"

    # premium / luxury
    if "premium" in instruction or "luxury" in instruction:
        rate = "-5%"
        pitch = "+1Hz"

    # casual / relaxed
    if "casual" in instruction or "relaxed" in instruction:
        rate = "+5%"
        pitch = "+2Hz"

    return rate, pitch


# -------------------------
# SSML BUILDER (PAUSES)
# -------------------------
def _build_ssml(text: str, voice: str) -> str:
    text = text.replace(".", ". <break time='400ms'/>")

    return f"""
<speak version="1.0">
  <voice name="{voice}">
    <prosody>
      {text}
    </prosody>
  </voice>
</speak>
"""


# -------------------------
# ASYNC AUDIO GENERATOR
# -------------------------
async def _generate_audio_async(text: str, voice: str, rate: str, pitch: str) -> bytes:
    ssml = _build_ssml(text, voice)

    communicator = edge_tts.Communicate(
        ssml,
        voice,
        rate=rate,
        pitch=pitch,
        method="ssml"
    )

    audio_bytes = b""

    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    return audio_bytes


# -------------------------
# MAIN FUNCTION
# -------------------------
def generate_cloned_audio(
    text_to_clone: str,
    lang: Optional[str] = "en",
    user_profile=None,
    signals=None
) -> str:
    """
    🔥 AHVI Voice Engine (Edge TTS)

    Features:
    - multi-language
    - tone-aware voice modulation
    - SSML pauses
    - safe fallback
    """

    if not text_to_clone:
        return ""

    try:
        voice = VOICE_MAP.get(lang, VOICE_MAP["en"])

        rate, pitch = _get_voice_modulation(user_profile, signals)

        audio_bytes = asyncio.run(
            _generate_audio_async(text_to_clone, voice, rate, pitch)
        )

        return base64.b64encode(audio_bytes).decode("utf-8")

    except Exception as e:
        print("❌ Edge TTS error:", str(e))

        # SAFE FALLBACK (never crash system)
        return base64.b64encode(b"").decode("utf-8")