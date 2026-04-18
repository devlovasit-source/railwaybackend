import os
import json


class FitnessEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "data", "fitness_data.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"WARN: FitnessEngine load failed: {e}")
            self.data = {}

        # Core
        self.sessions = {s["key"]: s for s in self.data.get("session_templates", [])}

        # Routes
        self.goal_routes = self.data.get("goal_routes", {})
        self.gender_routes = self.data.get("gender_routes", {})
        self.duration_routes = self.data.get("duration_routes", {})
        self.location_routes = self.data.get("location_routes", {})
        self.equipment_routes = self.data.get("equipment_routes", {})
        self.constraint_routes = self.data.get("constraint_routes", {})

    # =========================
    # FILTER ENGINE
    # =========================
    def filter_sessions(self, input_data):
        """
        input_data = {
            "goal": "fat_loss",
            "gender": "women",
            "duration": 20,
            "location": "home",
            "equipment": "none"
        }
        """

        candidates = set(self.sessions.keys())

        # Goal filter
        if input_data.get("goal") in self.goal_routes:
            candidates &= set(self.goal_routes[input_data["goal"]])

        # Gender filter
        if input_data.get("gender") in self.gender_routes:
            candidates &= set(self.gender_routes[input_data["gender"]])

        # Duration filter
        if str(input_data.get("duration")) in self.duration_routes:
            candidates &= set(self.duration_routes[str(input_data["duration"])])

        # Location
        if input_data.get("location") in self.location_routes:
            candidates &= set(self.location_routes[input_data["location"]])

        # Equipment
        if input_data.get("equipment") in self.equipment_routes:
            candidates &= set(self.equipment_routes[input_data["equipment"]])

        # Constraint
        if input_data.get("constraint") in self.constraint_routes:
            candidates &= set(self.constraint_routes[input_data["constraint"]])

        return [self.sessions[k] for k in candidates]

    # =========================
    # MAIN RECOMMENDER
    # =========================
    def recommend_workout(self, input_data):
        results = self.filter_sessions(input_data)

        if not results:
            return {
                "message": "No exact match found, try relaxing filters",
                "fallback": list(self.sessions.values())[:3]
            }

        # Pick top 3
        return {
            "type": "fitness_recommendation",
            "count": len(results),
            "recommendations": results[:3]
        }

    # =========================
    # WEEKLY PROGRAM
    # =========================
    def get_weekly_program(self, key):
        programs = self.data.get("weekly_programs_extended", [])
        return next((p for p in programs if p["key"] == key), None)


# Singleton
fitness_engine = FitnessEngine()
