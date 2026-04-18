# backend/brain/engines/life_engine.py

import os
import json


class LifeEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "life_data.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print("⚠️ LifeEngine load failed:", e)
            self.data = {}

        self.bills = self.data.get("bills_utilities", [])
        self.expenses = self.data.get("expenses", {})
        self.meds = self.data.get("medicine_schedules", [])
        self.routines = self.data.get("weekly_routines", [])

    # =========================
    # 💰 BUDGET
    # =========================
    def get_budget_plan(self, income: float, template_key="50_30_20"):
        template = next(
            (t for t in self.expenses.get("budget_templates", []) if t["key"] == template_key),
            None
        )

        if not template:
            return {}

        result = []
        for part in template["split"]:
            amount = income * (part["pct"] / 100)
            result.append({
                "category": part["name"],
                "percentage": part["pct"],
                "amount": round(amount, 2)
            })

        return result

    # =========================
    # 🧾 EXPENSE CATEGORIES
    # =========================
    def get_expense_categories(self):
        return self.expenses.get("categories", [])

    # =========================
    # 💊 MEDICINE SCHEDULE
    # =========================
    def get_medicine_schedule(self, key="daily_meds_basic"):
        return next(
            (m for m in self.meds if m["key"] == key),
            None
        )

    # =========================
    # 🏠 ROUTINES
    # =========================
    def get_weekly_routine(self, key="home_reset_sunday_90min"):
        return next(
            (r for r in self.routines if r["key"] == key),
            None
        )

    # =========================
    # 🔥 MASTER RESPONSE
    # =========================
    def build_life_plan(self, input_data: dict):
        """
        input_data = {
            "income": 50000,
            "needs": ["budget", "routine", "meds"]
        }
        """

        response = {}

        if "budget" in input_data.get("needs", []):
            response["budget"] = self.get_budget_plan(input_data.get("income", 0))

        if "routine" in input_data.get("needs", []):
            response["routine"] = self.get_weekly_routine()

        if "meds" in input_data.get("needs", []):
            response["medicine"] = self.get_medicine_schedule()

        return response


# Singleton
life_engine = LifeEngine()