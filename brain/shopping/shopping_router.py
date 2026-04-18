import os
import json


class ShoppingRouter:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "config", "shopping_router_map.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.map = json.load(f).get("shopping_router_map", {})
        except Exception as e:
            print("⚠️ Shopping router load failed:", e)
            self.map = {}

    # =========================
    # KEYWORD MATCH
    # =========================
    def _match_keywords(self, text, keywords):
        return any(k in text for k in keywords)

    # =========================
    # MAIN ROUTER
    # =========================
    def route(self, text: str, signals: dict):
        text = (text or "").lower()

        intents = self.map.get("keyword_intents", {})
        routes = self.map.get("routes", [])

        # 🔥 Detect keyword matches
        keyword_match = {}

        for key, data in intents.items():
            keywords = data.get("keywords_any", []) + data.get("keywords_soft", [])
            keyword_match[key] = self._match_keywords(text, keywords)

        # 🔥 Check routes in priority order
        for route in routes:
            mode = route.get("mode")

            conditions = route.get("when_any_of", [])

            for cond in conditions:
                if "keyword_match" in cond:
                    key = cond.split(".")[-1].replace(" == true", "")
                    if keyword_match.get(key):
                        return route

                if "signals" in cond:
                    key = cond.split(".")[-1].replace(" == true", "")
                    if signals.get(key):
                        return route

        # 🔥 fallback
        return self.map.get("fallback")


# Singleton
shopping_router = ShoppingRouter()