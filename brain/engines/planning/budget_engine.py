import json
import os


class BudgetEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "event_budget.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except:
            self.data = {}

    # =========================
    # GET TIER
    # =========================
    def get_tier(self, tier_key):
        return next((t for t in self.data.get("tiers", []) if t["key"] == tier_key), None)

    # =========================
    # SIMPLE ESTIMATE
    # =========================
    def estimate_simple(self, guest_count, tier_key, venue_type):
        tier = self.get_tier(tier_key)
        if not tier:
            return 0

        venue_range = tier["per_guest_ranges"].get(venue_type, {})
        avg_cost = (venue_range.get("min", 0) + venue_range.get("max", 0)) / 2

        return guest_count * avg_cost

    # =========================
    # ADVANCED ESTIMATE
    # =========================
    def estimate_advanced(
        self,
        guest_count,
        tier_key,
        venue_type,
        city_tier="tier_2_city",
        season="regular",
        day_part="evening",
        guest_band="81_200",
        event_type="wedding",
        function_type="wedding_day"
    ):
        base = self.estimate_simple(guest_count, tier_key, venue_type)

        multipliers = self.data.get("multipliers", {})

        city_m = multipliers.get("city_multiplier", {}).get(city_tier, 1)
        season_m = multipliers.get("season_multiplier", {}).get(season, 1)
        day_m = multipliers.get("day_part_multiplier", {}).get(day_part, 1)
        guest_m = multipliers.get("guest_count_band_multiplier", {}).get(guest_band, 1)

        event_m = self.data.get("event_type_multiplier", {}).get(event_type, 1)
        func_m = self.data.get("rules", {}).get("wedding_functions_multiplier", {}).get(function_type, 1)

        total = base * city_m * season_m * day_m * guest_m * event_m * func_m

        return round(total, 2)

    # =========================
    # COST BREAKDOWN
    # =========================
    def get_breakdown(self, total_cost, tier_key):
        buckets = self.data.get("cost_buckets", {}).get("default_percent_split", [])

        result = []
        for b in buckets:
            avg_pct = (b["min_pct"] + b["max_pct"]) / 2
            amount = total_cost * (avg_pct / 100)

            result.append({
                "category": b["label"],
                "percentage": avg_pct,
                "amount": round(amount, 2)
            })

        return result

    # =========================
    # FULL RESPONSE
    # =========================
    def build_budget_plan(self, inputs):
        total = self.estimate_advanced(**inputs)
        breakdown = self.get_breakdown(total, inputs["tier_key"])

        return {
            "total_estimate": total,
            "currency": self.data.get("currency", "INR"),
            "breakdown": breakdown
        }


# Singleton
budget_engine = BudgetEngine()