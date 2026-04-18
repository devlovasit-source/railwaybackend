import json
import math
import os
from threading import Lock
from typing import Any, Dict, List


class OutfitRanker:
    """
    Lightweight online ranker:
    - Uses weighted feature scoring for ranking.
    - Learns per-user preferences via feedback updates.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._state_path = os.path.join(base_dir, "data", "ranker_state.json")
        self._lock = Lock()

    def rank(self, user_id: str, outfits: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        if not outfits:
            return []
        state = self._load_state()
        profile = self._ensure_user_profile(state, user_id)
        weights = profile.get("weights", {})
        bias = float(profile.get("bias", 0.0))

        scored: List[Dict[str, Any]] = []
        for outfit in outfits:
            feature_map = outfit.get("ml_features", {}) if isinstance(outfit, dict) else {}
            linear = bias + sum(float(feature_map.get(k, 0.0)) * float(v) for k, v in weights.items())
            ml_score = self._sigmoid(linear)

            item = dict(outfit)
            item["ml_score"] = round(ml_score, 4)
            item["rank_score"] = round((ml_score * 100.0) + float(item.get("score", 0.0)), 3)
            scored.append(item)

        scored.sort(key=lambda x: float(x.get("rank_score", 0.0)), reverse=True)
        return scored[: max(1, int(top_n))]

    def learn_from_feedback(self, user_id: str, features: Dict[str, Any], feedback: str) -> None:
        label = 1.0 if str(feedback).lower() in ("up", "like", "liked") else 0.0
        with self._lock:
            state = self._load_state()
            profile = self._ensure_user_profile(state, user_id)
            weights = profile.get("weights", {})
            bias = float(profile.get("bias", 0.0))

            x = {k: float(v) for k, v in (features or {}).items()}
            pred = self._sigmoid(bias + sum(float(weights.get(k, 0.0)) * v for k, v in x.items()))
            err = label - pred

            lr = 0.08
            for key, value in x.items():
                current = float(weights.get(key, 0.0))
                weights[key] = round(current + (lr * err * value), 6)
            profile["bias"] = round(bias + (lr * err), 6)

            self._trim_weights(weights)
            self._save_state(state)

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self._state_path):
            return {"users": {}}
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("users", {})
                    return data
        except Exception:
            pass
        return {"users": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, indent=2)

    def _ensure_user_profile(self, state: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        users = state.setdefault("users", {})
        profile = users.setdefault(
            str(user_id),
            {
                "bias": 0.0,
                "weights": {
                    "occasion_rules": 0.9,
                    "color_intelligence": 0.8,
                    "layering": 0.7,
                    "style_graph": 0.7,
                    "memory": 0.8,
                    "feedback": 1.0,
                    "semantic_relevance": 0.9,
                },
            },
        )
        profile.setdefault("weights", {})
        profile.setdefault("bias", 0.0)
        return profile

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(-20.0, min(20.0, float(value)))
        return 1.0 / (1.0 + math.exp(-clipped))

    @staticmethod
    def _trim_weights(weights: Dict[str, Any], limit: int = 24) -> None:
        if len(weights) <= limit:
            return
        by_importance = sorted(weights.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        keep = dict(by_importance[:limit])
        weights.clear()
        weights.update(keep)


outfit_ranker = OutfitRanker()

