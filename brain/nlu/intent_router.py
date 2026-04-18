# backend/brain/nlu/intent_router.py

import re
from typing import Dict

class IntentRouter:
    def __init__(self):

        # Precompile patterns for speed
        self.styling_patterns = self._compile_patterns([
            "wear", "outfit", "dress", "clothes", "style", "look", "matching", "fit"
        ])

        self.occasions = self._compile_dict_patterns({
            "party looks": ["party", "club", "birthday", "pub"],
            "office fits": ["office", "work", "interview", "meeting", "corporate"],
            "vacation": ["vacation", "trip", "holiday", "beach", "travel", "goa"],
            "occasion": ["wedding", "reception", "festival", "event", "pooja"],
            "casual": ["casual", "daily", "everyday", "grocery"]
        })

        self.weather_conditions = self._compile_dict_patterns({
            "rainy": ["rain", "rainy", "monsoon"],
            "summer": ["hot", "summer", "sunny"],
            "winter": ["cold", "winter", "freezing"]
        })

        self.life_keywords = self._compile_dict_patterns({
            "meal_planning": ["meal", "diet", "food", "protein", "recipe"],
            "life_goals": ["goal", "habit", "progress"],
            "health_wellness": ["workout", "gym", "skincare", "fitness"],
            "finance_home": ["bill", "budget", "expense", "savings"]
        })

    # =========================
    # HELPERS
    # =========================
    def _compile_patterns(self, keywords):
        return [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in keywords]

    def _compile_dict_patterns(self, data):
        return {
            key: [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in values]
            for key, values in data.items()
        }

    def normalize_text(self, text: str) -> str:
        return text.lower().strip()

    # =========================
    # SLOT EXTRACTION
    # =========================
    def extract_slots(self, text: str) -> Dict:
        text = self.normalize_text(text)

        slots = {
            "occasion": None,
            "weather": None,
            "life_category": None
        }

        # Occasion
        for occasion, patterns in self.occasions.items():
            if any(p.search(text) for p in patterns):
                slots["occasion"] = occasion
                break

        # Weather
        for weather, patterns in self.weather_conditions.items():
            if any(p.search(text) for p in patterns):
                slots["weather"] = weather
                break

        # Life category
        for category, patterns in self.life_keywords.items():
            if any(p.search(text) for p in patterns):
                slots["life_category"] = category
                break

        return slots

    # =========================
    # INTENT CLASSIFICATION
    # =========================
    def classify_intent(self, text: str) -> Dict:
        text = self.normalize_text(text)
        slots = self.extract_slots(text)

        score = 0

        # Life intent (highest priority)
        if slots["life_category"]:
            return {
                "status": "success",
                "intent": slots["life_category"],
                "slots": slots,
                "confidence": 0.95
            }

        # Styling detection
        styling_hits = sum(1 for p in self.styling_patterns if p.search(text))
        if styling_hits:
            score += styling_hits

        if slots["occasion"]:
            score += 2

        if slots["weather"]:
            score += 1

        if score > 0:
            return {
                "status": "success",
                "intent": "styling",
                "slots": slots,
                "confidence": min(0.5 + (score * 0.1), 0.95)
            }

        # Fallback
        return {
            "status": "unrecognized",
            "intent": "unknown",
            "slots": slots,
            "confidence": 0.0
        }


# Singleton instance
nlu_router = IntentRouter()