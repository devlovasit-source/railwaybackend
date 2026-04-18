from itertools import combinations
from typing import Any, Dict, List


class StyleGraphEngine:
    """
    Lightweight style graph:
    - Nodes: wardrobe items
    - Edges: compatibility signals (color/fabric/type)
    """

    def build_graph(self, wardrobe: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for key in ("tops", "bottoms", "shoes"):
            for item in wardrobe.get(key, []):
                data = item if isinstance(item, dict) else {}
                item_id = str(data.get("id") or data.get("name") or "")
                if not item_id:
                    continue
                items.append(
                    {
                        "id": item_id,
                        "type": str(data.get("type", "")).lower(),
                        "color": str(data.get("color", "")).lower(),
                        "fabric": str(data.get("fabric", "")).lower(),
                    }
                )

        edges: List[Dict[str, Any]] = []
        edge_map: Dict[str, float] = {}
        for left, right in combinations(items, 2):
            weight = self._edge_weight(left, right)
            if weight <= 0:
                continue
            key = self._pair_key(left["id"], right["id"])
            edge_map[key] = weight
            edges.append({"from": left["id"], "to": right["id"], "weight": weight})

        return {"nodes": items, "edges": edges, "edge_map": edge_map}

    def pair_weight(self, graph: Dict[str, Any], item_a_id: str, item_b_id: str) -> float:
        edge_map = graph.get("edge_map", {}) if isinstance(graph, dict) else {}
        return float(edge_map.get(self._pair_key(item_a_id, item_b_id), 0.0))

    def _pair_key(self, item_a_id: str, item_b_id: str) -> str:
        left, right = sorted([str(item_a_id), str(item_b_id)])
        return f"{left}|{right}"

    def _edge_weight(self, left: Dict[str, str], right: Dict[str, str]) -> float:
        score = 0.0
        if left.get("color") and left.get("color") == right.get("color"):
            score += 0.5
        if left.get("fabric") and left.get("fabric") == right.get("fabric"):
            score += 0.5

        lt = left.get("type", "")
        rt = right.get("type", "")
        complementary_pairs = [
            ("shirt", "trousers"),
            ("tshirt", "jeans"),
            ("top", "bottom"),
        ]
        for a, b in complementary_pairs:
            if (a in lt and b in rt) or (a in rt and b in lt):
                score += 1.0
                break
        return score


style_graph_engine = StyleGraphEngine()
