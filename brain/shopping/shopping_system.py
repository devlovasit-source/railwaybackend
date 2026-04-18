import os
import json

from brain.shopping.shopping_router import shopping_router
from brain.shopping.shopping_engine import shopping_engine
from brain.wardrobe.wardrobe_normalizer import wardrobe_normalizer


class ShoppingSystem:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "shopping_engine_pack_v1.json")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("shopping_engine_pack_v1", {})
        except Exception as e:
            print("⚠️ Shopping system config load failed:", e)
            self.config = {}

    # =========================
    # MAIN ENTRYPOINT
    # =========================
    def run(self, text: str, context: dict):
        """
        text = user input
        context = {
            product_candidate,
            wardrobe,
            signals
        }
        """

        signals = context.get("signals", {})

        # 🔥 STEP 1: NORMALIZE WARDROBE
        wardrobe_raw = context.get("wardrobe", [])

        wardrobe_index = [
            wardrobe_normalizer.normalize_item(item)
            for item in wardrobe_raw
        ]

        context["wardrobe_index"] = {
            "items": wardrobe_index,
            "coverage_score": self._calculate_coverage(wardrobe_index)
        }

        # 🔥 STEP 2: ROUTE
        route = shopping_router.route(text, signals)

        # 🔥 STEP 3: ENGINE EXECUTION
        output = shopping_engine.run(route, context)

        # 🔥 STEP 4: CONTRACT ENFORCEMENT (basic)
        output = self._enforce_contract(route, output)

        return output

    # =========================
    # COVERAGE SCORE (simple)
    # =========================
    def _calculate_coverage(self, wardrobe_items):
        if not wardrobe_items:
            return 0

        return min(len(wardrobe_items) / 10, 1)  # simple heuristic

    # =========================
    # CONTRACT ENFORCEMENT
    # =========================
    def _enforce_contract(self, route, output):
        mode = route.get("mode")

        contracts = self.config.get("mode_contracts", {})
        contract = contracts.get(mode, {})

        if not contract:
            return output

        # Example check
        if mode == "purchase_conviction_mode":
            if "combos" not in output:
                output["combos"] = ["Fallback combo 1", "Fallback combo 2", "Fallback combo 3"]

        return output


# Singleton
shopping_system = ShoppingSystem()