from typing import Dict, Any, List, Optional
from datetime import datetime

from services.weather_service import get_hourly_weather


class ContextEngine:

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def build_context(
        self,
        user_id: str,
        intent_data: Dict[str, Any],
        wardrobe: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        vision: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        wardrobe = wardrobe or []
        user_profile = user_profile or {}
        history = history or []
        vision = vision or {}

        slots = intent_data.get("slots", {})

        # -------------------------
        # 🔥 ENRICH CORE SIGNALS
        # -------------------------
        weather_data = self._get_weather(user_profile)
        time_data = self._get_time_context()

        enriched_slots = self._enrich_slots(
            slots=slots,
            history=history,
            vision=vision,
            weather=weather_data,
            time_data=time_data
        )

        wardrobe_meta = self._analyze_wardrobe(wardrobe)
        user_meta = self._analyze_user_profile(user_profile)

        return {
            "user_id": user_id,

            # 🔥 intent layer
            "intent": intent_data.get("intent"),
            "confidence": intent_data.get("confidence", 0.0),

            # 🔥 enriched slots
            "slots": enriched_slots,

            # 🔥 raw data
            "user_profile": user_profile,
            "wardrobe": wardrobe,
            "history": history,
            "vision": vision,

            # 🔥 intelligence layers
            "weather": weather_data,
            "time": time_data,
            "wardrobe_meta": wardrobe_meta,
            "user_meta": user_meta,

            # 🔥 system meta
            "meta": {
                "has_wardrobe": len(wardrobe) > 0,
                "has_profile": bool(user_profile),
                "has_history": len(history) > 0,
                "has_vision": bool(vision),
                "wardrobe_size": len(wardrobe)
            }
        }

    # -------------------------
    # 🔥 SLOT ENRICHMENT
    # -------------------------
    def _enrich_slots(
        self,
        slots: Dict[str, Any],
        history: List[Dict[str, Any]],
        vision: Dict[str, Any],
        weather: Dict[str, Any],
        time_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        enriched = slots.copy()

        # -------------------------
        # FROM HISTORY
        # -------------------------
        if history:
            prev_slots = history[-1].get("slots", {})
            for key in ["occasion", "style", "vibe"]:
                if not enriched.get(key) and prev_slots.get(key):
                    enriched[key] = prev_slots.get(key)

        # -------------------------
        # FROM VISION
        # -------------------------
        if vision:
            if not enriched.get("vibe"):
                enriched["vibe"] = vision.get("detected_style")

        # -------------------------
        # FROM WEATHER
        # -------------------------
        if weather:
            if not enriched.get("weather"):
                enriched["weather"] = weather.get("condition")

        # -------------------------
        # FROM TIME
        # -------------------------
        if not enriched.get("time"):
            enriched["time"] = time_data.get("time_of_day")

        return enriched

    # -------------------------
    # 🌦️ WEATHER
    # -------------------------
    def _get_weather(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:

        try:
            location = user_profile.get("location", {})

            if location.get("lat") and location.get("lon"):
                return get_hourly_weather(
                    lat=location.get("lat"),
                    lon=location.get("lon")
                )

        except Exception as e:
            print("Weather error:", str(e))

        return {"condition": "mild"}

    # -------------------------
    # ⏰ TIME CONTEXT
    # -------------------------
    def _get_time_context(self) -> Dict[str, Any]:

        now = datetime.now().hour

        if 5 <= now < 12:
            tod = "morning"
        elif 12 <= now < 17:
            tod = "afternoon"
        elif 17 <= now < 21:
            tod = "evening"
        else:
            tod = "night"

        return {
            "hour": now,
            "time_of_day": tod
        }

    # -------------------------
    # 👕 WARDROBE INTELLIGENCE
    # -------------------------
    def _analyze_wardrobe(self, wardrobe: List[Dict[str, Any]]) -> Dict[str, Any]:

        categories = {}
        colors = {}

        for item in wardrobe:
            cat = item.get("category", "unknown")
            color = item.get("color")

            categories[cat] = categories.get(cat, 0) + 1

            if color:
                colors[color] = colors.get(color, 0) + 1

        dominant_color = max(colors, key=colors.get) if colors else None

        return {
            "category_distribution": categories,
            "dominant_color": dominant_color,
            "has_formal": any("formal" in str(i.get("type", "")).lower() for i in wardrobe),
            "has_casual": any("casual" in str(i.get("type", "")).lower() for i in wardrobe)
        }

    # -------------------------
    # 👤 USER PROFILE INTELLIGENCE
    # -------------------------
    def _analyze_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "preferred_style": user_profile.get("style"),
            "preferred_colors": user_profile.get("colors", []),
            "gender": user_profile.get("gender")
        }


# singleton
context_engine = ContextEngine()