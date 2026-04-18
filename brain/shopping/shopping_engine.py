import os
import json


class ShoppingEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.rules_dir = os.path.join(base_dir, "shopping", "rules")

    # =========================
    # LOAD RULES
    # =========================
    def _load_rules(self, filename):
        try:
            path = os.path.join(self.rules_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("⚠️ Failed loading shopping rules:", e)
            return {}

    # =========================
    # DECISION ENGINE (BUY / SKIP)
    # =========================
    def _run_decision_engine(self, context):
        product = context.get("product_candidate", {})
        wardrobe = context.get("wardrobe_index", {})

        coverage = wardrobe.get("coverage_score", 0)

        reasons = []

        if coverage > 0.5:
            reasons.append("You already have pieces that pair well with this")

        if product.get("formality") in ["casual", "smart_casual"]:
            reasons.append("It’s versatile across multiple occasions")

        if product.get("color_family") in ["black", "white", "neutral"]:
            reasons.append("Neutral tones increase rewear value")

        # 🔥 Verdict logic
        if len(reasons) >= 2:
            verdict = "buy"
        else:
            verdict = "skip"

        # 🔥 Message
        if verdict == "buy":
            message = "I’d buy this. It fits well into your wardrobe and won’t be a one-time wear."
        else:
            message = "I’d skip this. It might not integrate easily with your current wardrobe."

        return {
            "type": "shopping_decision",
            "verdict": verdict,
            "reasons": reasons[:2],
            "message": message,
            "next_step": "Want me to style it or find better alternatives?"
        }

    # =========================
    # CONVICTION ENGINE (3 COMBOS)
    # =========================
    def _run_conviction_engine(self, context):
        product = context.get("product_candidate", {})

        item = product.get("category", "item")

        combos = [
            f"• Casual: white tee + blue jeans + {item} + sneakers",
            f"• Smart: shirt + tailored trousers + {item} + loafers",
            f"• Elevated: blazer + structured pants + {item} + heels"
        ]

        return {
            "type": "purchase_conviction",
            "combos": combos,
            "why": "This works because it integrates easily with staple pieces you already wear.",
            "confidence": "This is a smart buy if you want maximum wear.",
            "question": "Want me to tailor this for work or weekend?"
        }

    # =========================
    # MAIN ROUTER EXECUTION
    # =========================
    def run(self, route, context):
        engine_key = route.get("engine", {}).get("key")
        file_path = route.get("engine", {}).get("file")

        if not file_path:
            return {"error": "No engine file"}

        filename = os.path.basename(file_path)
        _ = self._load_rules(filename)  # optional use later

        # 🔥 Route to correct logic
        if engine_key == "shopping_decision_engine":
            return self._run_decision_engine(context)

        if engine_key == "purchase_conviction_engine":
            return self._run_conviction_engine(context)

        # fallback
        return {
            "type": "shopping_fallback",
            "message": "Tell me what you're looking for — I’ll help you decide."
        }


# Singleton
shopping_engine = ShoppingEngine()