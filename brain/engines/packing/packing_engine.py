# backend/brain/engines/packing_engine.py

import os
import json
import math


class PackingEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Base packing
        base_file = os.path.join(base_dir, "data", "packing_data.json")

        # Smart layers
        smart_file = os.path.join(base_dir, "data", "packing_intelligence.json")

        try:
            with open(base_file, "r", encoding="utf-8") as f:
                self.base_data = json.load(f)
        except:
            self.base_data = {}

        try:
            with open(smart_file, "r", encoding="utf-8") as f:
                self.smart_data = json.load(f)
        except:
            self.smart_data = {}

        # Base
        self.categories = self.base_data.get("categories", [])
        self.addons = self.base_data.get("purpose_addons", {})
        self.slots = self.base_data.get("outfit_slots", {})

        # Smart
        self.destinations = self.smart_data.get("destination_bases", [])
        self.weather_layers = self.smart_data.get("weather_layers", [])
        self.activity_layers = self.smart_data.get("activity_layers", [])

    # =========================
    # BASE ITEMS
    # =========================
    def get_base_items(self):
        return [
            {"category": c["label"], "items": c["items"]}
            for c in self.categories
        ]

    # =========================
    # PURPOSE ADDONS
    # =========================
    def get_addons(self, purpose):
        return self.addons.get(purpose, [])

    # =========================
    # DESTINATION
    # =========================
    def get_destination(self, key):
        return next((d for d in self.destinations if d["key"] == key), None)

    def get_weather(self, key):
        return next((w for w in self.weather_layers if w["key"] == key), None)

    def get_activity(self, key):
        return next((a for a in self.activity_layers if a["key"] == key), None)

    # =========================
    # OUTFIT CALCULATION
    # =========================
    def calculate_outfits(self, days, gender="women"):

        base = self.slots.get("per_day_default", {}).get(gender, [])
        multipliers = self.slots.get("multipliers", {})

        if days <= 3:
            m = multipliers.get("short_trip_1_3_days", {})
        elif days <= 7:
            m = multipliers.get("mid_trip_4_7_days", {})
        else:
            m = multipliers.get("long_trip_8_plus_days", {})

        result = []

        for item in base:
            if "Top" in item:
                count = math.ceil(days * m.get("tops", 1))
            elif "Bottom" in item:
                count = math.ceil(days * m.get("bottoms", 1))
            elif "Innerwear" in item:
                count = math.ceil(days * m.get("innerwear", 1))
            else:
                count = days

            result.append(f"{item.replace('x1','x'+str(count))}")

        return result

    # =========================
    # MERGE HELPERS
    # =========================
    def merge_cards(self, base_cards, new_items, title):
        if not new_items:
            return base_cards

        base_cards.append({
            "category": title,
            "items": new_items
        })
        return base_cards

    # =========================
    # 🔥 MAIN ENGINE
    # =========================
    def build_packing(self, input_data: dict):
        """
        input_data = {
            "days": 5,
            "purpose": "beach",
            "gender": "women",
            "destination": "beach_tropical",
            "weather": "rainy",
            "activity": "hiking"
        }
        """

        days = input_data.get("days", 3)
        purpose = input_data.get("purpose")
        gender = input_data.get("gender", "women")

        # BASE
        base_items = self.get_base_items()
        addons = self.get_addons(purpose)
        outfits = self.calculate_outfits(days, gender)

        # SMART LAYERS
        dest = self.get_destination(input_data.get("destination"))
        weather = self.get_weather(input_data.get("weather"))
        activity = self.get_activity(input_data.get("activity"))

        cards = []

        # Start with base categories
        for b in base_items:
            cards.append({
                "title": b["category"],
                "items": b["items"]
            })

        # Destination override
        if dest:
            cards.extend(dest.get("cards", []))

        # Add layers
        if weather:
            cards = self.merge_cards(cards, weather.get("add_items"), "Weather")

        if activity:
            cards = self.merge_cards(cards, activity.get("add_items"), "Activity")

        # Purpose addons
        if addons:
            cards = self.merge_cards(cards, addons, "Purpose")

        return {
            "type": "packing_board",
            "trip_summary": {
                "days": days,
                "purpose": purpose,
                "destination": input_data.get("destination")
            },
            "cards": cards,
            "outfits": outfits
        }


# Singleton
packing_engine = PackingEngine()
