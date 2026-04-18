from typing import Dict, Any


def run_calendar_runtime(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal, stable calendar runtime stub.
    Returns a predictable structure for the orchestrator.
    """
    title = event.get("title", "")
    return {
        "event": {
            "title": title,
        },
        "classification": {
            "group": "general",
            "subtype": None,
            "priority": "normal",
        },
        "packing": [],
        "prep_tasks": [],
        "outfit": {},
    }

