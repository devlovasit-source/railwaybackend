import json
import os
from typing import Any, Dict


class StyleRulesEngine:
    """
    Rules provider only.
    No outfit generation logic should live here.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.banks_dir = os.path.join(base_dir, "banks")
        self.data_dir = os.path.join(base_dir, "data")

        self.events_bank = self._load_json(os.path.join(self.banks_dir, "events", "events_bank_v1.json"))
        self.weather_bank = self._load_json(
            os.path.join(self.banks_dir, "contextual", "season_weather_overlays_bank_v1.json")
        )
        self.style_knowledge = self._load_json(os.path.join(self.data_dir, "style_knowledge_v1.json"))

    def _load_json(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def get_scoring_rules(self, style_dna: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        style_dna = style_dna or {}
        context = context or {}

        gender = str(style_dna.get("gender", "women")).lower()
        body_type = str(style_dna.get("body_type", "")).lower()
        body_rules = (
            self.style_knowledge.get(gender, {})
            .get("body_types", {})
            .get(body_type, {})
        )

        preferred_keywords = []
        best = body_rules.get("best", {}) if isinstance(body_rules, dict) else {}
        for key in ("tops", "pants", "skirts", "dresses"):
            values = best.get(key, [])
            if isinstance(values, list):
                preferred_keywords.extend([str(v).lower() for v in values])

        weather = str(context.get("weather", "")).strip()
        weather_rule = self.weather_bank.get(weather, {}) if weather else {}
        occasion = str(context.get("occasion", "")).strip()
        event_rule = self.events_bank.get(occasion, {}) if occasion else {}

        return {
            "preferred_colors": [str(c).lower() for c in style_dna.get("preferred_colors", [])],
            "preferred_fabrics": [str(f).lower() for f in style_dna.get("preferred_fabrics", [])],
            "avoided_items": [str(i).lower() for i in style_dna.get("disliked_items", [])],
            "preferred_keywords": preferred_keywords,
            "event_tags": [str(t).lower() for t in event_rule.get("tags", [])] if isinstance(event_rule, dict) else [],
            "weather_tags": [str(t).lower() for t in weather_rule.get("tags", [])] if isinstance(weather_rule, dict) else [],
            "compatibility": {
                "shirt": ["trousers"],
                "tshirt": ["jeans"],
            },
            "invalid_combo_penalty": -8.0,
        }


style_engine = StyleRulesEngine()
