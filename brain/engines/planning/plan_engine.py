# backend/brain/engines/plan_engine.py

import os
import json


class PlanEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "data", "plan_data.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"WARN: PlanEngine load failed: {e}")
            self.data = {}

        self.trip_templates = self.data.get("trip_templates", [])
        self.event_templates = self.data.get("event_templates", [])

    # =========================
    # GET TEMPLATE
    # =========================
    def get_trip_template(self, key):
        return next((t for t in self.trip_templates if t["key"] == key), None)

    def get_event_template(self, key):
        return next((t for t in self.event_templates if t["key"] == key), None)

    # =========================
    # BUILD TRIP PLAN
    # =========================
    def build_trip_plan(self, input_data: dict):
        """
        input_data = {
            "template": "weekend_city",
            "destination": "Hyderabad"
        }
        """

        template_key = input_data.get("template", "weekend_city")
        destination = input_data.get("destination", "your trip")

        template = self.get_trip_template(template_key)

        if not template:
            return {"error": "Trip template not found"}

        return {
            "type": "trip_board",
            "title": f"{template['label']} - {destination}",
            "sections": template.get("sections", [])
        }

    # =========================
    # BUILD EVENT PLAN
    # =========================
    def build_event_plan(self, input_data: dict):
        """
        input_data = {
            "template": "house_party",
            "event_name": "Birthday"
        }
        """

        template_key = input_data.get("template", "house_party")
        event_name = input_data.get("event_name", "Your Event")

        template = self.get_event_template(template_key)

        if not template:
            return {"error": "Event template not found"}

        return {
            "type": "event_board",
            "title": f"{template['label']} - {event_name}",
            "sections": template.get("sections", [])
        }

    # =========================
    # SMART ROUTER
    # =========================
    def build_plan(self, input_data: dict):
        """
        input_data = {
            "mode": "trip" | "event"
        }
        """

        mode = input_data.get("mode", "trip")

        if mode == "trip":
            return self.build_trip_plan(input_data)

        elif mode == "event":
            return self.build_event_plan(input_data)

        return {"error": "Invalid mode"}


# Singleton
plan_engine = PlanEngine()
