import random
import copy


class RecipeEngine:
    def __init__(self):
        pass

    # =========================
    # HELPERS
    # =========================
    def pick(self, arr):
        return random.choice(arr) if arr else None

    def difficulty(self, time_min):
        return "easy" if time_min <= 25 else "medium"

    def unique_title(self, title, used):
        count = used.get(title, 0)
        used[title] = count + 1
        return title if count == 0 else f"{title} {count+1}"

    # =========================
    # RULE ENGINE
    # =========================
    def apply_regional(self, card, rules):
        r = self.pick(rules)
        if not r:
            return card

        card["ingredients"] = list(set(card["ingredients"] + r.get("adds", [])))
        card["goal_tags"] = list(set(card.get("goal_tags", []) + [r.get("tag")]))

        if r.get("note"):
            card.setdefault("notes", []).append(r["note"])

        return card

    def apply_constraint(self, card, constraints):
        if random.random() > 0.3:
            return card

        c = self.pick(constraints)
        if not c:
            return card

        if c.get("avoid"):
            card["ingredients"] = [
                i for i in card["ingredients"]
                if i not in c["avoid"]
            ]

        if c.get("note"):
            card.setdefault("notes", []).append(c["note"])

        return card

    def apply_grain_swap(self, card, swaps):
        if random.random() > 0.4:
            return card

        s = self.pick(swaps)
        if not s:
            return card

        f, t = s.get("from"), s.get("to")

        card["ingredients"] = [t if x == f else x for x in card["ingredients"]]
        card["steps"] = [step.replace(f, t) for step in card["steps"]]

        card.setdefault("notes", []).append(s.get("note", ""))

        return card

    def apply_protein_swap(self, card, protein_swaps):
        diet = (card.get("diet_type") or ["veg"])[0]
        pool = protein_swaps.get(diet, [])

        if random.random() > 0.5:
            return card

        s = self.pick(pool)
        if not s:
            return card

        f, t = s.get("from"), s.get("to")

        card["ingredients"] = [t if x == f else x for x in card["ingredients"]]
        card["steps"] = [step.replace(f, t) for step in card["steps"]]
        card["title"] = card["title"].replace(f, t)

        return card

    # =========================
    # MAIN ENGINE
    # =========================
    def generate(self, config: dict):
        random.seed(config.get("seed", 42))

        base_cards = config.get("base_cards", [])
        rules = config.get("variant_rules", {})

        target = config.get("count", 100)
        time_opts = config.get("time_options_min", [10, 20, 30])

        regional = rules.get("regional_variants", [])
        grain_swaps = rules.get("grain_swaps", [])
        protein_swaps = rules.get("protein_swaps", {})
        constraints = rules.get("style_constraints", [])
        max_var = rules.get("max_variants_per_base", 5)

        used_titles = {}
        recipes = []

        # =========================
        # 1. BASE RECIPES
        # =========================
        for base in base_cards:
            if len(recipes) >= target:
                break

            time_min = base.get("time_min") or self.pick(time_opts)

            card = {
                "id": base.get("id"),
                "title": self.unique_title(base.get("title"), used_titles),
                "diet_type": base.get("diet_type", ["veg"]),
                "goal_tags": base.get("goal_tags", []),
                "time_min": time_min,
                "difficulty": self.difficulty(time_min),
                "ingredients": base.get("ingredients", []),
                "steps": base.get("steps", []),
                "notes": base.get("notes", [])[:3]
            }

            recipes.append(card)

        # =========================
        # 2. VARIANTS
        # =========================
        i = 0
        while len(recipes) < target and base_cards:
            base = base_cards[i % len(base_cards)]
            i += 1

            for _ in range(max_var):
                if len(recipes) >= target:
                    break

                card = copy.deepcopy(base)

                card["id"] = f"r_{len(recipes)+1}"
                card["title"] = self.unique_title(
                    f"{card['title']} Variant", used_titles
                )

                time_min = card.get("time_min") or self.pick(time_opts)
                card["time_min"] = time_min
                card["difficulty"] = self.difficulty(time_min)

                # APPLY RULES
                card = self.apply_regional(card, regional)
                card = self.apply_constraint(card, constraints)
                card = self.apply_grain_swap(card, grain_swaps)
                card = self.apply_protein_swap(card, protein_swaps)

                # CLEAN NOTES
                card["notes"] = list(set(card.get("notes", [])))[:3]

                recipes.append(card)

        return {
            "version": config.get("version", "v1"),
            "count": len(recipes[:target]),
            "recipes": recipes[:target]
        }


# Singleton
recipe_engine = RecipeEngine()