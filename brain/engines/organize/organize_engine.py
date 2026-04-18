# backend/brain/engines/organize_engine.py

import os
import json
import datetime


class OrganizeEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "data", "organize_data.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"WARN: OrganizeEngine load failed: {e}")
            self.data = {}

        self.areas = self.data.get("areas", [])
        self.schema = self.data.get("task_card_schema_hint", {})

    # =========================
    # GET AREA
    # =========================
    def get_area(self, key):
        return next((a for a in self.areas if a["key"] == key), None)

    # =========================
    # BUILD TASK BOARD
    # =========================
    def build_task_board(self, area_key: str):

        area = self.get_area(area_key)

        if not area:
            return {"error": "Area not found"}

        tasks = area.get("default_tasks", [])

        # Simple split logic
        today_tasks = tasks[:2]
        week_tasks = tasks[2:]

        return {
            "type": "task_board",
            "area": area["label"],
            "cards": [
                {
                    "kind": "today",
                    "title": "Today",
                    "items": today_tasks
                },
                {
                    "kind": "this_week",
                    "title": "This week",
                    "items": week_tasks
                }
            ],
            "cta": {
                "primary": "Mark done",
                "secondary": "Add task"
            }
        }

    # =========================
    # BUILD MULTI-AREA DASHBOARD
    # =========================
    def build_dashboard(self, selected_areas=None):
        """
        selected_areas = ["groceries", "bills"]
        """

        boards = []

        for area in self.areas:
            if selected_areas and area["key"] not in selected_areas:
                continue

            board = self.build_task_board(area["key"])
            boards.append(board)

        return {
            "type": "organize_dashboard",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "boards": boards
        }


# Singleton
organize_engine = OrganizeEngine()
