# backend/brain/engines/template_engine.py

import os
import json


class TemplateEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "event_templates.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print("❌ Template load failed:", e)
            self.data = {}

        self.templates = self.data.get("templates", [])
        self.timelines = self.data.get("timelines", [])
        self.packing = self.data.get("packing_for_events", [])

    # =========================
    # FIND TEMPLATE
    # =========================
    def get_template(self, event_type: str):
        return [
            t for t in self.templates
            if t.get("event_type") == event_type
        ]

    # =========================
    # SMART MATCH (BEST TEMPLATE)
    # =========================
    def select_template(self, context: dict):
        """
        context = {
            "event": "mehendi",
            "people_count": 120,
            "style": "modern"
        }
        """

        event = context.get("event")
        people = context.get("people_count", 0)

        candidates = self.get_template(event)

        if not candidates:
            return None

        # Pick closest by people_count
        best = sorted(
            candidates,
            key=lambda x: abs(x.get("people_count", 0) - people)
        )[0]

        return best

    # =========================
    # GET TIMELINE
    # =========================
    def get_timeline(self, event_type: str):
        return next(
            (t for t in self.timelines if t["event_type"] == event_type),
            None
        )

    # =========================
    # GET PACKING
    # =========================
    def get_packing(self, key: str):
        return next(
            (p for p in self.packing if p["key"] == key),
            None
        )

    # =========================
    # BUILD RESPONSE
    # =========================
    def build_event_plan(self, context: dict):
        template = self.select_template(context)

        if not template:
            return {"error": "No template found"}

        timeline = self.get_timeline(template.get("event_type"))
        packing = self.get_packing(template.get("linked_packing_key", ""))

        return {
            "event": template.get("event_type"),
            "style": template.get("style"),
            "people_count": template.get("people_count"),
            "cards": template.get("cards"),
            "timeline": timeline,
            "packing": packing
        }


# Singleton