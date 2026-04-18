# backend/brain/engines/recipe_rewriter.py

import copy


class RecipeRewriter:

    # =========================
    # HELPERS
    # =========================
    def replace_pairs(self, text, pairs):
        for f, t in pairs:
            text = text.replace(f, t)
        return text

    def remove_if_contains(self, arr, terms):
        terms = [t.lower() for t in terms]
        return [
            x for x in arr
            if not any(term in x.lower() for term in terms)
        ]

    # =========================
    # MAIN FUNCTION
    # =========================
    def rewrite(self, recipe: dict, opts: dict):

        r = copy.deepcopy(recipe)

        appliance = opts.get("appliance", "tawa")
        spice = opts.get("spice_tolerance", "medium")
        toggles = opts.get("toggles", {})

        # =========================
        # JAIN
        # =========================
        if toggles.get("jain"):
            r["ingredients"] = self.remove_if_contains(r["ingredients"], ["onion", "garlic"])
            r["steps"] = [s.replace("onion", "").replace("garlic", "") for s in r["steps"]]

            r.setdefault("goal_tags", []).append("jain")
            r.setdefault("notes", []).append("Jain edit: no onion/garlic.")

        # =========================
        # NO DAIRY
        # =========================
        if toggles.get("no_dairy"):
            pairs = [
                ("curd", "coconut yogurt"),
                ("yogurt", "coconut yogurt"),
                ("milk", "plant milk"),
                ("paneer", "tofu"),
                ("ghee", "oil")
            ]

            r["ingredients"] = [self.replace_pairs(x, pairs) for x in r["ingredients"]]
            r["steps"] = [self.replace_pairs(s, pairs) for s in r["steps"]]

            r.setdefault("goal_tags", []).append("no_dairy")
            r.setdefault("notes", []).append("No-dairy edit applied.")

        # =========================
        # NO EGG
        # =========================
        if toggles.get("no_egg"):
            r["ingredients"] = self.remove_if_contains(r["ingredients"], ["egg"])

            r["steps"] = [
                s.replace("egg", "tofu scramble")
                for s in r["steps"]
            ]

            r.setdefault("goal_tags", []).append("no_egg")
            r.setdefault("notes", []).append("No-egg edit applied.")

        # =========================
        # NO PEANUTS
        # =========================
        if toggles.get("no_peanuts"):
            r["ingredients"] = self.remove_if_contains(r["ingredients"], ["peanut"])

            r["steps"] = [
                s.replace("peanuts", "roasted seeds (optional)")
                for s in r["steps"]
            ]

            r.setdefault("goal_tags", []).append("no_peanuts")
            r.setdefault("notes", []).append("Peanut-free edit applied.")

        # =========================
        # SPICE
        # =========================
        if spice == "low":
            r["ingredients"] = [
                x.replace("chilli", "chilli (optional)").replace("pepper", "pepper (light)")
                for x in r["ingredients"]
            ]

            r["steps"] = [
                s.replace("chilli", "chilli (optional)")
                for s in r["steps"]
            ]

            r.setdefault("notes", []).append("Low-spice edit applied.")

        elif spice == "high":
            r.setdefault("notes", []).append("High-spice ok.")

        # =========================
        # APPLIANCE
        # =========================
        if appliance == "pressure_cooker":
            r["steps"] = [
                "Pressure cooker method: cook everything together efficiently."
            ] + r["steps"]

        elif appliance == "airfryer":
            r["steps"] = [
                f"{s} (Airfryer option available)"
                for s in r["steps"]
            ]

        elif appliance == "microwave":
            r["steps"] = [
                "Microwave shortcut: pre-cook components then mix."
            ] + r["steps"]

        elif appliance == "no_cook":
            r["steps"] = [
                "No-cook: assemble ready ingredients into a bowl."
            ]
            r.setdefault("goal_tags", []).append("no_cook")

        # =========================
        # CLEANUP
        # =========================
        r["notes"] = list(set(r.get("notes", [])))[:3]
        r["goal_tags"] = list(set(r.get("goal_tags", [])))

        return r


# Singleton
recipe_rewriter = RecipeRewriter()