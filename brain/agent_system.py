from typing import Any, Dict, List
import logging

# FIX: Import extract_json instead of strict parse_json_array
from services.ai_gateway import generate_text, extract_json


class AgentSystem:
    """
    Hybrid planner:
    - Uses LLM (Ollama) for dynamic planning
    - Falls back to deterministic rules
    """

    def plan(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            ai_plan = self._llm_plan(intent, context)
            if ai_plan:
                return ai_plan
        except Exception:
            logging.exception("LLM planning failed")

        return self._rule_based_plan(intent)

    def _llm_plan(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(intent, context)
        response = generate_text(prompt, usecase="intent")

        if not response or response == "none":
            return []

        try:
            # FIX: Use extract_json to handle both arrays and wrapped objects safely
            parsed = extract_json(str(response))
            
            # Scenario A: The AI correctly returned a list
            if isinstance(parsed, list):
                return parsed
                
            # Scenario B: The AI wrapped the list in a dict (e.g. {"steps": [...]})
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if isinstance(value, list):
                        return value
                        
        except Exception:
            logging.exception("Invalid LLM plan JSON")

        return []

    def _build_prompt(self, intent: str, context: Dict[str, Any]) -> str:
        slots = context.get("slots", {})

        return f"""
You are an AI planning engine for a fashion assistant.

Your job:
Return a JSON list of steps to execute.

Rules:
- Output ONLY a valid JSON array. DO NOT wrap it in a JSON object.
- No explanation or markdown
- Each step must have:
  - "step"
  - "agent"

Available steps:
- normalize_context
- build_style_graph
- generate_score_rank
- persist_and_feedback_hooks
- generate_boards
- prepare_tryon
- run_tryon_model

User intent: {intent}

Context:
- occasion: {slots.get("occasion")}
- weather: {slots.get("weather")}
- time: {slots.get("time_of_day")}
- location: {slots.get("location")}

Examples:

daily_outfit ->
[
  {{"step": "normalize_context", "agent": "context_agent"}},
  {{"step": "build_style_graph", "agent": "style_graph_agent"}},
  {{"step": "generate_score_rank", "agent": "outfit_agent"}},
  {{"step": "persist_and_feedback_hooks", "agent": "memory_agent"}}
]

tryon ->
[
  {{"step": "prepare_tryon", "agent": "tryon_agent"}},
  {{"step": "run_tryon_model", "agent": "vision_agent"}}
]

Now generate plan:
"""

    def _rule_based_plan(self, intent: str) -> List[Dict[str, Any]]:
        if intent == "daily_outfit":
            return [
                {"step": "normalize_context", "agent": "context_agent"},
                {"step": "build_style_graph", "agent": "style_graph_agent"},
                {"step": "generate_score_rank", "agent": "outfit_agent"},
                {"step": "persist_and_feedback_hooks", "agent": "memory_agent"},
            ]

        if intent == "tryon":
            return [
                {"step": "prepare_tryon", "agent": "tryon_agent"},
                {"step": "run_tryon_model", "agent": "vision_agent"},
            ]

        return [{"step": "no_op", "agent": "fallback_agent"}]


agent_system = AgentSystem()