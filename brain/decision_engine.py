from __future__ import annotations

from typing import Any, Dict, List, Tuple


class DecisionEngine:
    """
    Lightweight control-layer ranker.
    Takes candidate actions/cards and returns top-N with deterministic scoring.
    """

    def __init__(self) -> None:
        self._base_weight = 1.0

    def rank_actions(
        self,
        *,
        candidates: List[Dict[str, Any]],
        context: Dict[str, Any],
        top_n: int = 3,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        slot = str(context.get("time_slot") or "").lower()
        persona = str(context.get("persona") or "").lower()

        for idx, card in enumerate(candidates or []):
            priority = int(card.get("priority", 0))
            score = float(priority) * self._base_weight

            # Notification cards get a small boost during urgent slots.
            if bool(card.get("notification_needed")) and slot in {"morning", "evening"}:
                score += 8.0

            # Persona-aware nudge.
            if persona == "busy_parent" and card.get("type") in {"family_prep", "key_reminder"}:
                score += 6.0
            if persona == "student" and card.get("type") in {"study_priority", "task_focus"}:
                score += 5.0

            # Stable tie-break by original order.
            score -= idx * 0.001

            normalized = dict(card)
            normalized["decision_score"] = round(score, 3)
            scored.append((score, normalized))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [item for _, item in scored[: max(1, int(top_n))]]
        meta = {
            "strategy": "weighted_priority_v1",
            "top_n": max(1, int(top_n)),
            "slot": slot,
            "persona": persona,
            "candidate_count": len(candidates or []),
            "selected_count": len(selected),
        }
        return selected, meta


decision_engine = DecisionEngine()

