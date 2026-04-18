import os
import json


class PromptEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "data", "prompt_bank_v1.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print("⚠️ PromptEngine load failed:", e)
            self.data = {}

    def get_prompt(self, category: str):
        return self.data.get(category, [])


# Singleton
prompt_engine = PromptEngine()