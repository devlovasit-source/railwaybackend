# backend/brain/engines/meal_planner_engine.py

import time
from collections import defaultdict


class MealPlannerEngine:

    # =========================
    # HELPERS
    # =========================
    def norm(self, s):
        return (s or "").lower().strip()

    def includes_any(self, text, words):
        t = self.norm(text)
        return any(w.lower() in t for w in words)

    def uniq(self, arr):
        return list(set(arr))

    # =========================
    # SCORING
    # =========================
    def score_recipe(self, r, input_data):
        score = 0

        focus = input_data.get("goals", {}).get("focus")
        if focus and focus in r.get("goal_tags", []):
            score += 4

        diet = input_data.get("user", {}).get("diet_type")
        if diet and diet in r.get("diet_type", []):
            score += 4

        cap = input_data.get("constraints", {}).get("cooking_time_cap_min")
        if cap is not None:
            score += 2 if r.get("time_min", 999) <= cap else -3

        # allergies
        allergies = [self.norm(x) for x in input_data.get("user", {}).get("allergies", [])]
        for a in allergies:
            if any(a in self.norm(i) for i in r.get("ingredients", [])):
                score -= 10

        return score

    # =========================
    # PICK TOP
    # =========================
    def pick_top(self, recipes, input_data, n=50):
        scored = [(r, self.score_recipe(r, input_data)) for r in recipes]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:n]]

    # =========================
    # MEAL TYPE
    # =========================
    def meal_type(self, r):
        if r.get("meal_type"):
            return r["meal_type"]

        t = self.norm(r.get("title", ""))

        if self.includes_any(t, ["idli", "dosa", "upma", "poha", "oats"]):
            return "breakfast"
        if self.includes_any(t, ["soup", "salad", "chaat"]):
            return "snack"
        if self.includes_any(t, ["rice", "biryani", "pulao"]):
            return "lunch"

        return "dinner"

    def note(self, r):
        tags = r.get("goal_tags", [])

        if "high_protein" in tags:
            return "Protein rich"
        if "fat_loss" in tags:
            return "Low calorie"
        if "gut_friendly" in tags:
            return "Easy digestion"

        return "Balanced"

    # =========================
    # MAIN
    # =========================
    def build_weekly_plan(self, input_data):

        recipes = input_data.get("recipes", [])
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        top = self.pick_top(recipes, input_data, 50)

        buckets = {
            "breakfast": [],
            "lunch": [],
            "snack": [],
            "dinner": []
        }

        for r in top:
            buckets[self.meal_type(r)].append(r)

        used = set()
        plan = []

        def pick(bucket):
            for r in bucket:
                if r["id"] not in used:
                    return r
            return bucket[0] if bucket else None

        for d in days:
            b = pick(buckets["breakfast"])
            l = pick(buckets["lunch"])
            s = pick(buckets["snack"])
            dn = pick(buckets["dinner"])

            for x in [b, l, s, dn]:
                if x:
                    used.add(x["id"])

            plan.append({
                "day": d,
                "breakfast": {"id": b["id"], "title": b["title"], "note": self.note(b)} if b else {},
                "lunch": {"id": l["id"], "title": l["title"], "note": self.note(l)} if l else {},
                "snack": {"id": s["id"], "title": s["title"], "note": self.note(s)} if s else {},
                "dinner": {"id": dn["id"], "title": dn["title"], "note": self.note(dn)} if dn else {}
            })

        # =========================
        # GROCERY
        # =========================
        ingredient_map = defaultdict(list)

        for r in recipes:
            for i in r.get("ingredients", []):
                ingredient_map[self.norm(i)].append(r["id"])

        grocery = []
        for k, v in ingredient_map.items():
            grocery.append({
                "item": k,
                "used_in": len(set(v))
            })

        return {
            "week_id": f"wk_{int(time.time())}",
            "plan": plan,
            "grocery_list": grocery[:40]
        }


# Singleton
meal_planner_engine = MealPlannerEngine()