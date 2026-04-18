import json
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from threading import Lock
from typing import Any, Dict, List, Tuple

from brain.engines.styling.style_builder import style_engine
from brain.ml.outfit_ranker import outfit_ranker
from brain.style_graph_engine import style_graph_engine
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy
from services.embedding_service import get_model
from services.qdrant_service import qdrant_service

_MEMORY_LOCK = Lock()
_MEMORY_FILE = os.path.join(os.path.dirname(__file__), "data", "outfit_memory.json")


def _contains_word(text: str, words: List[str]) -> bool:
    # FIX: Removed the strict space padding so "Tops" will successfully match "top" and "tops"
    text = str(text or "").lower()
    return any(w in text for w in words)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_memory() -> Dict[str, Any]:
    if not os.path.exists(_MEMORY_FILE):
        return {"users": {}}
    try:
        with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("users", {})
                return data
    except Exception:
        pass
    return {"users": {}}


def _save_memory(memory: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_MEMORY_FILE), exist_ok=True)
    with open(_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=True, indent=2)


def _memory_doc_id(user_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(user_id or "anonymous"))
    return f"outfit_memory_{safe}"[:64]


def _default_user_memory() -> Dict[str, Any]:
    return {
        "recent_outfits": [],
        "liked_outfits": [],
        "disliked_outfits": [],
    }


def _load_user_memory(user_id: str) -> Dict[str, Any]:
    proxy = AppwriteProxy()
    try:
        doc = proxy.get_document("memories", _memory_doc_id(user_id))
        payload = doc.get("payload") if isinstance(doc, dict) else None
        if isinstance(payload, str) and payload.strip():
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                user_memory = _default_user_memory()
                user_memory.update(parsed)
                return user_memory
    except Exception:
        pass

    memory = _load_memory()
    user = _ensure_user_memory(memory, user_id)
    return dict(user)


def _save_user_memory(user_id: str, user_memory: Dict[str, Any]) -> None:
    proxy = AppwriteProxy()
    try:
        payload = {
            "userId": str(user_id),
            "name": "outfit_memory",
            "payload": json.dumps(user_memory, ensure_ascii=True),
        }
        doc_id = _memory_doc_id(user_id)
        try:
            proxy.update_document("memories", doc_id, payload)
        except Exception:
            proxy.create_document("memories", payload, document_id=doc_id)
        return
    except Exception:
        pass

    memory = _load_memory()
    user = _ensure_user_memory(memory, user_id)
    user.clear()
    user.update(user_memory or {})
    _save_memory(memory)


def _ensure_user_memory(memory: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    users = memory.setdefault("users", {})
    user = users.setdefault(
        user_id,
        {
            "recent_outfits": [],
            "liked_outfits": [],
            "disliked_outfits": [],
        },
    )
    user.setdefault("recent_outfits", [])
    user.setdefault("liked_outfits", [])
    user.setdefault("disliked_outfits", [])
    return user


def _normalize_item(item: Dict[str, Any], fallback_type: str) -> Dict[str, Any]:
    item = item or {}
    item_type = str(item.get("type") or item.get("category") or fallback_type).lower()
    return {
        "id": str(item.get("id") or item.get("$id") or item.get("item_id") or item.get("name") or ""),
        "name": str(item.get("name") or item_type.title()),
        "type": item_type,
        "color": str(item.get("color") or item.get("color_name") or "").lower(),
        "fabric": str(item.get("fabric") or "").lower(),
        "style": str(item.get("style") or item.get("vibe") or "").lower(),
        "occasion_tags": [str(v).lower() for v in item.get("occasion_tags", item.get("occasions", [])) if v],
        "weather_tags": [str(v).lower() for v in item.get("weather_tags", item.get("weather", [])) if v],
        "layerable": bool(item.get("layerable", False)),
    }


def _normalize_wardrobe(raw_wardrobe: Any) -> Dict[str, List[Dict[str, Any]]]:
    parts = {
        "tops": [],
        "bottoms": [],
        "shoes": [],
        "outerwear": [],
        "dresses": [],
        "accessories": [],
    }

    def _add(raw: Dict[str, Any]) -> None:
        category = str(
            raw.get("type")
            or raw.get("category")
            or raw.get("main_category")
            or ""
        ).lower()

        name = str(raw.get("name", "")).lower()

        # HARD OVERRIDE (MOST IMPORTANT)
        if any(x in name for x in ["shoe", "sneaker", "boot", "heel", "sandal"]):
            parts["shoes"].append(_normalize_item(raw, "shoes"))
            return
        if "dress" in name or "gown" in name:
            parts["dresses"].append(_normalize_item(raw, "dress"))
            return
        if any(x in name for x in ["bag", "watch", "necklace", "earring", "bracelet", "belt", "scarf", "cap", "hat"]):
            parts["accessories"].append(_normalize_item(raw, "accessory"))
            return

        # FIX: Added plurals (shoes, dresses, accessories, bottoms, outerwear, tops) to ensure matching
        if _contains_word(category, ["shoe", "shoes", "footwear", "sneaker", "heel", "boot", "sandal"]):
            parts["shoes"].append(_normalize_item(raw, "shoes"))
        elif _contains_word(category, ["dress", "dresses", "gown", "onepiece", "one-piece", "jumpsuit"]):
            parts["dresses"].append(_normalize_item(raw, "dress"))
        elif _contains_word(category, ["accessory", "accessories", "jewelry", "jewellery", "bag", "watch", "belt", "scarf", "hat"]):
            parts["accessories"].append(_normalize_item(raw, "accessory"))
        elif _contains_word(category, ["bottom", "bottoms", "jean", "pant", "pants", "trouser", "skirt", "short"]):
            parts["bottoms"].append(_normalize_item(raw, "bottom"))
        elif _contains_word(category, ["outer", "outerwear", "jacket", "blazer", "coat", "hoodie"]):
            parts["outerwear"].append(_normalize_item(raw, "outerwear"))
        elif _contains_word(category, ["top", "tops", "shirt", "tee", "blouse", "sweater"]):
            parts["tops"].append(_normalize_item(raw, "top"))

    if isinstance(raw_wardrobe, dict):
        for item in raw_wardrobe.get("tops", []) or []:
            if isinstance(item, dict):
                _add(item)
        for item in raw_wardrobe.get("bottoms", []) or []:
            if isinstance(item, dict):
                _add(item)
        for item in raw_wardrobe.get("shoes", raw_wardrobe.get("footwear", [])) or []:
            if isinstance(item, dict):
                _add(item)
        for item in raw_wardrobe.get("outerwear", []) or []:
            if isinstance(item, dict):
                _add(item)
        for item in raw_wardrobe.get("dresses", []) or []:
            if isinstance(item, dict):
                _add(item)
        for item in raw_wardrobe.get("accessories", raw_wardrobe.get("jewelry", [])) or []:
            if isinstance(item, dict):
                _add(item)
    elif isinstance(raw_wardrobe, list):
        for item in raw_wardrobe:
            if isinstance(item, dict):
                _add(item)

    return parts


def _outfit_vector(outfit: Dict[str, Any]) -> List[float]:
    def _hash_fraction(value: str) -> float:
        raw = str(value or "")
        return (sum(ord(ch) for ch in raw) % 100) / 100.0

    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}
    shoes = outfit.get("shoes", {}) or {}
    score = float(outfit.get("score", 0.0))
    return [
        _hash_fraction(top.get("type")),
        _hash_fraction(top.get("color")),
        _hash_fraction(bottom.get("type")),
        _hash_fraction(bottom.get("color")),
        _hash_fraction(shoes.get("type")),
        _hash_fraction(shoes.get("color")),
        max(-1.0, min(1.0, score / 10.0)),
        1.0,
    ]


def _index_outfit_vector(user_id: str, outfit: Dict[str, Any], label: str) -> None:
    if not qdrant_service.enabled():
        return
    try:
        vector = _outfit_vector(outfit)
        combo_id = str(outfit.get("combo_id", ""))
        seed = f"{user_id}:{label}:{combo_id or _utcnow_iso()}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))
        payload = {"user_id": user_id, "label": label, "combo_id": combo_id}
        qdrant_service.upsert_memory_vector(point_id=point_id, vector=vector, payload=payload)
    except Exception:
        return


def _semantic_retrieval(
    user_id: str,
    context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not qdrant_service.enabled():
        return [], {}

    try:
        query_text = " ".join(
            [
                str(context.get("query", "")),
                str(context.get("occasion", "")),
                str(context.get("weather", "")),
                str(context.get("time_of_day", "")),
                str((context.get("style_dna", {}) or {}).get("style", "")),
                " ".join((context.get("style_dna", {}) or {}).get("preferred_colors", [])),
            ]
        ).strip()
        if not query_text:
            query_text = "daily outfit"

        model = get_model()
        query_vector = model.encode(query_text).tolist()
        results = qdrant_service.semantic_retrieve(query_vector, user_id=user_id, limit=40)

        wardrobe_items: List[Dict[str, Any]] = []
        semantic_map: Dict[str, float] = {}
        for row in results:
            payload = row.get("payload", {}) if isinstance(row, dict) else {}
            normalized = _normalize_item(payload, str(payload.get("category", "item")))
            item_id = normalized.get("id")
            if not item_id:
                continue
            semantic_map[item_id] = max(float(semantic_map.get(item_id, 0.0)), float(row.get("score", 0.0)))
            wardrobe_items.append(normalized)

        return wardrobe_items, semantic_map
    except Exception:
        return [], {}


def _merge_wardrobe(
    base: Dict[str, List[Dict[str, Any]]],
    semantic_items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    merged = {
        "tops": list(base.get("tops", [])),
        "bottoms": list(base.get("bottoms", [])),
        "shoes": list(base.get("shoes", [])),
        "outerwear": list(base.get("outerwear", [])),
        "dresses": list(base.get("dresses", [])),
        "accessories": list(base.get("accessories", [])),
    }

    seen = set()
    for key in merged:
        for item in merged[key]:
            seen.add(str(item.get("id", "")))

    for item in semantic_items:
        item_type = str(item.get("type", "")).lower()
        item_id = str(item.get("id", ""))
        if not item_id or item_id in seen:
            continue

        # FIX: Added plurals here as well
        if _contains_word(item_type, ["shoe", "shoes", "footwear", "sneaker", "boot", "heel", "sandal"]):
            merged["shoes"].append(item)
        elif _contains_word(item_type, ["bottom", "bottoms", "pant", "pants", "trouser", "jean", "skirt", "short"]):
            merged["bottoms"].append(item)
        elif _contains_word(item_type, ["outer", "outerwear", "jacket", "coat", "blazer", "hoodie"]):
            merged["outerwear"].append(item)
        elif _contains_word(item_type, ["dress", "dresses", "gown", "onepiece", "one-piece", "jumpsuit"]):
            merged["dresses"].append(item)
        elif _contains_word(item_type, ["accessory", "accessories", "jewel", "jewelry", "bag", "watch", "belt", "scarf", "hat"]):
            merged["accessories"].append(item)
        elif _contains_word(item_type, ["top", "tops", "shirt", "tee", "blouse", "sweater"]):
            merged["tops"].append(item)

        seen.add(item_id)

    return merged


def _occasion_filter(wardrobe: Dict[str, List[Dict[str, Any]]], occasion: str) -> Dict[str, List[Dict[str, Any]]]:
    occ = str(occasion or "").strip().lower()
    if not occ:
        return wardrobe

    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for key, items in wardrobe.items():
        matched = []
        for item in items or []:
            tags = [str(v).strip().lower() for v in (item.get("occasion_tags") or []) if str(v).strip()]
            if occ in tags or any(occ in t for t in tags):
                matched.append(item)
        filtered[key] = matched if matched else list(items or [])
    return filtered


def _master_piece_score(item: Dict[str, Any], occasion: str, semantic_map: Dict[str, float]) -> float:
    tags = [str(v).strip().lower() for v in (item.get("occasion_tags") or []) if str(v).strip()]
    score = 0.0
    if occasion and (occasion in tags or any(occasion in t for t in tags)):
        score += 2.0
    item_id = str(item.get("id", "")).strip()
    if item_id:
        score += float(semantic_map.get(item_id, 0.0))
    return score


def _pick_master_piece(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    occasion: str,
    semantic_map: Dict[str, float],
) -> Tuple[str, Dict[str, Any]]:
    candidates: List[Tuple[str, Dict[str, Any], float]] = []
    for row in wardrobe.get("tops", []) or []:
        candidates.append(("top", row, _master_piece_score(row, occasion, semantic_map)))
    for row in wardrobe.get("dresses", []) or []:
        candidates.append(("dress", row, _master_piece_score(row, occasion, semantic_map) + 0.1))

    if not candidates:
        return "", {}
    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


def _build_master_combos(
    wardrobe: Dict[str, List[Dict[str, Any]]],
    master_type: str,
    master_item: Dict[str, Any],
    *,
    max_combos: int = 36,
) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    if master_type == "top":
        for bottom in (wardrobe.get("bottoms", []) or []):
            for shoe in (wardrobe.get("shoes", []) or []):
                combos.append(
                    {
                        "combo_id": f"{master_item.get('id')}|{bottom.get('id')}|{shoe.get('id')}",
                        "master_type": "top",
                        "master_piece": master_item,
                        "top": master_item,
                        "bottom": bottom,
                        "shoes": shoe,
                        "dress": {},
                        "outerwear": {},
                        "accessories": [],
                    }
                )
                if len(combos) >= max_combos:
                    return combos
    elif master_type == "dress":
        for shoe in (wardrobe.get("shoes", []) or []):
            combos.append(
                {
                    "combo_id": f"{master_item.get('id')}|{shoe.get('id')}",
                    "master_type": "dress",
                    "master_piece": master_item,
                    "top": {},
                    "bottom": {},
                    "shoes": shoe,
                    "dress": master_item,
                    "outerwear": {},
                    "accessories": [],
                }
            )
            if len(combos) >= max_combos:
                return combos
    return combos


def _combo_palette(combo: Dict[str, Any]) -> List[str]:
    colors: List[str] = []
    for part in ("master_piece", "top", "bottom", "shoes", "dress"):
        item = combo.get(part, {}) or {}
        color = str(item.get("color", "")).strip().lower()
        if color:
            colors.append(color)
    return colors


def _combo_patterns(combo: Dict[str, Any]) -> List[str]:
    patterns: List[str] = []
    for part in ("master_piece", "top", "bottom", "shoes", "dress"):
        item = combo.get(part, {}) or {}
        p = str(item.get("fabric", "")).strip().lower()
        if p:
            patterns.append(p)
    return patterns


def _llm_filter_combo_ids(
    *,
    occasion: str,
    stage: str,
    master_type: str,
    master_piece: Dict[str, Any],
    combos: List[Dict[str, Any]],
) -> List[str]:
    if not combos:
        return []

    compact = []
    for row in combos:
        compact.append(
            {
                "combo_id": row.get("combo_id"),
                "palette": _combo_palette(row),
                "patterns": _combo_patterns(row),
                "bottom": (row.get("bottom") or {}).get("name"),
                "shoes": (row.get("shoes") or {}).get("name"),
                "dress": (row.get("dress") or {}).get("name"),
            }
        )

    prompt = f"""
You are a fashion combo evaluator.
Occasion: {occasion}
Stage: {stage}
Master type: {master_type}
Master piece: {json.dumps({'name': master_piece.get('name'), 'color': master_piece.get('color'), 'fabric': master_piece.get('fabric')}, ensure_ascii=True)}

Select best combo_ids from this list:
{json.dumps(compact, ensure_ascii=True)}

Return strict JSON object:
{{
  "selected_combo_ids": ["id1","id2","id3"]
}}
Rules:
- Keep at most 8 ids.
- Keep only ids present in the input.
- Prioritize practical wearable harmony.
"""
    try:
        parsed = ai_gateway.generate_json_object(prompt, signals={"context_mode": "styling"})
        selected = parsed.get("selected_combo_ids", []) if isinstance(parsed, dict) else []
        normalized = []
        valid = {str(c.get("combo_id")) for c in compact}
        for value in selected if isinstance(selected, list) else []:
            cid = str(value).strip()
            if cid and cid in valid and cid not in normalized:
                normalized.append(cid)
        return normalized[:8]
    except Exception:
        return []


def _rule_color_fallback(master_piece: Dict[str, Any], combos: List[Dict[str, Any]]) -> List[str]:
    master_color = str((master_piece or {}).get("color", "")).strip().lower()
    neutrals = {"black", "white", "beige", "gray", "grey", "navy", "brown"}
    selected: List[str] = []
    for combo in combos:
        palette = _combo_palette(combo)
        uniq = set(palette)
        if not uniq:
            continue
        if master_color and master_color in uniq:
            selected.append(str(combo.get("combo_id")))
            continue
        if len(uniq) <= 3 and any(c in neutrals for c in uniq):
            selected.append(str(combo.get("combo_id")))
    return selected[:8]


def _rule_pattern_fallback(combos: List[Dict[str, Any]]) -> List[str]:
    selected: List[str] = []
    for combo in combos:
        pats = [p for p in _combo_patterns(combo) if p]
        if not pats:
            selected.append(str(combo.get("combo_id")))
            continue
        standout = sum(1 for p in pats if p in {"striped", "checked", "floral", "printed"})
        if standout <= 1:
            selected.append(str(combo.get("combo_id")))
    return selected[:8]


def _select_accessories(wardrobe: Dict[str, List[Dict[str, Any]]], combo: Dict[str, Any], limit: int = 2) -> List[Dict[str, Any]]:
    accessories = wardrobe.get("accessories", []) or []
    if not accessories:
        return []
    palette = set(_combo_palette(combo))
    picked: List[Dict[str, Any]] = []
    for item in accessories:
        color = str(item.get("color", "")).strip().lower()
        if color and color in palette:
            picked.append(item)
        if len(picked) >= limit:
            return picked
    return accessories[:limit]


def generate_combinations(wardrobe: Dict[str, List[Dict[str, Any]]], max_candidates: int = 600) -> List[Dict[str, Any]]:
    tops = wardrobe.get("tops", [])
    bottoms = wardrobe.get("bottoms", [])
    shoes = wardrobe.get("shoes", [])
    outerwear = wardrobe.get("outerwear", [])

    combos: List[Dict[str, Any]] = []
    for top, bottom, shoe in product(tops, bottoms, shoes):
        base_combo = {
            "top": top,
            "bottom": bottom,
            "shoes": shoe,
            "outerwear": {},
            "combo_id": "|".join([top.get("id", ""), bottom.get("id", ""), shoe.get("id", "")]),
        }
        combos.append(base_combo)
        if outerwear:
            for layer in outerwear[:4]:
                layered = dict(base_combo)
                layered["outerwear"] = layer
                layered["combo_id"] = f"{base_combo['combo_id']}|{layer.get('id', '')}"
                combos.append(layered)
        if len(combos) >= max_candidates:
            return combos
    return combos


def validate_outfit(outfit: Dict[str, Any], context: Dict[str, Any]) -> bool:
    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}

    top_type = str(top.get("type", "")).lower()
    bottom_type = str(bottom.get("type", "")).lower()
    occasion = str(context.get("occasion", "")).lower()

    if "formal" in top_type and "short" in bottom_type:
        return False
    if occasion in ("office", "work") and "short" in bottom_type:
        return False
    return True


def _similarity_score(outfit_a: Dict[str, Any], outfit_b: Dict[str, Any]) -> float:
    if not outfit_a or not outfit_b:
        return 0.0
    score = 0.0
    checks = 0
    for part in ("top", "bottom", "dress", "shoes", "outerwear"):
        a = outfit_a.get(part, {}) or {}
        b = outfit_b.get(part, {}) or {}
        if not a or not b:
            continue
        checks += 1
        if str(a.get("type", "")).lower() == str(b.get("type", "")).lower():
            score += 0.4
        if str(a.get("color", "")).lower() == str(b.get("color", "")).lower():
            score += 0.4
        if str(a.get("fabric", "")).lower() == str(b.get("fabric", "")).lower():
            score += 0.2
    if checks == 0:
        return 0.0
    return min(1.0, score / checks)


def _color_score(colors: List[str], preferred_colors: List[str]) -> float:
    palette = [c for c in colors if c]
    if not palette:
        return 0.0
    unique = set(palette)

    score = 0.4 if len(unique) <= 2 else 0.1
    neutrals = {"black", "white", "beige", "gray", "grey", "navy", "brown"}
    if any(c in neutrals for c in unique):
        score += 0.3

    if preferred_colors:
        hits = sum(1 for c in palette if c in preferred_colors)
        score += min(0.6, hits * 0.2)

    return min(1.5, score)


def score_outfit(
    outfit: Dict[str, Any],
    context: Dict[str, Any],
    memory: Dict[str, Any],
    rules: Dict[str, Any],
    semantic_map: Dict[str, float],
) -> Dict[str, Any]:
    weather = str(context.get("weather", "")).lower()
    occasion = str(context.get("occasion", "")).lower()
    style_dna = context.get("style_dna", {}) or {}

    weather_score = 0.0
    occasion_score = 0.0
    color_intelligence = 0.0
    layering_score = 0.0
    style_graph_bonus = 0.0
    memory_score = 0.0
    feedback_adjustment = 0.0
    semantic_relevance = 0.0

    colors = []
    item_ids = []

    for part in ("master_piece", "top", "bottom", "dress", "shoes", "outerwear"):
        item = outfit.get(part, {}) or {}
        if not item:
            continue
        color = str(item.get("color", "")).lower()
        colors.append(color)

        item_id = str(item.get("id", ""))
        if item_id:
            item_ids.append(item_id)
            semantic_relevance += float(semantic_map.get(item_id, 0.0))

        weather_tags = [str(v).lower() for v in item.get("weather_tags", [])]
        occasion_tags = [str(v).lower() for v in item.get("occasion_tags", [])]

        if weather and weather in weather_tags:
            weather_score += 1.0
        if occasion and occasion in occasion_tags:
            occasion_score += 1.0

        name = str(item.get("name", "")).lower()
        fabric = str(item.get("fabric", "")).lower()
        if fabric and fabric in rules.get("preferred_fabrics", []):
            occasion_score += 0.4
        if name and name in rules.get("avoided_items", []):
            occasion_score -= 1.0

    color_intelligence = _color_score(colors, [str(c).lower() for c in style_dna.get("preferred_colors", [])])

    has_outerwear = bool(outfit.get("outerwear"))
    if weather in ("cold", "rain", "rainy", "chilly") and has_outerwear:
        layering_score += 1.0
    elif weather in ("hot", "warm") and has_outerwear:
        layering_score -= 0.4
    else:
        layering_score += 0.3

    graph = context.get("style_graph", {}) or {}
    top_id = str((outfit.get("top") or {}).get("id", ""))
    bottom_id = str((outfit.get("bottom") or {}).get("id", ""))
    shoes_id = str((outfit.get("shoes") or {}).get("id", ""))
    outer_id = str((outfit.get("outerwear") or {}).get("id", ""))

    style_graph_bonus += style_graph_engine.pair_weight(graph, top_id, bottom_id)
    style_graph_bonus += style_graph_engine.pair_weight(graph, bottom_id, shoes_id)
    if outer_id:
        style_graph_bonus += style_graph_engine.pair_weight(graph, outer_id, top_id)

    recent = memory.get("recent_outfits", [])[:20]
    liked = memory.get("liked_outfits", [])[:30]
    disliked = memory.get("disliked_outfits", [])[:30]

    repetition_penalty = sum(_similarity_score(outfit, r) * 0.9 for r in recent)
    memory_score = max(0.0, 2.0 - min(2.0, repetition_penalty))

    liked_sim = max([_similarity_score(outfit, o) for o in liked], default=0.0)
    disliked_sim = max([_similarity_score(outfit, o) for o in disliked], default=0.0)
    feedback_adjustment = (liked_sim * 1.8) - (disliked_sim * 2.2)

    semantic_relevance = semantic_relevance / max(1, len(item_ids))

    if not validate_outfit(outfit, context):
        occasion_score -= 5.0

    base_score = (
        weather_score
        + occasion_score
        + color_intelligence
        + layering_score
        + style_graph_bonus
        + memory_score
        + feedback_adjustment
        + semantic_relevance
    )

    features = {
        "occasion_rules": round(occasion_score + weather_score, 4),
        "color_intelligence": round(color_intelligence, 4),
        "layering": round(layering_score, 4),
        "style_graph": round(style_graph_bonus, 4),
        "memory": round(memory_score, 4),
        "feedback": round(feedback_adjustment, 4),
        "semantic_relevance": round(semantic_relevance, 4),
    }

    scored = deepcopy(outfit)
    scored["score"] = round(base_score, 3)
    scored["ml_features"] = features
    scored["score_breakdown"] = features
    return scored


def _explanation_for_outfit(outfit: Dict[str, Any], context: Dict[str, Any]) -> str:
    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}
    dress = outfit.get("dress", {}) or {}
    shoes = outfit.get("shoes", {}) or {}
    outer = outfit.get("outerwear", {}) or {}
    accessories = outfit.get("accessories", []) or []
    master_type = str(outfit.get("master_type", "")).lower()

    if master_type == "dress" or dress:
        lines = [
            f"Occasion fit: {context.get('occasion', 'daily')} look anchored by {dress.get('name', 'dress')}.",
            f"Color harmony: {dress.get('color', 'neutral')} pairs with {shoes.get('color', 'neutral')} footwear.",
        ]
    else:
        lines = [
            f"Occasion fit: {context.get('occasion', 'daily')} look built with {top.get('name', 'top')} and {bottom.get('name', 'bottom')}.",
            f"Color harmony: {top.get('color', 'neutral')} balances with {bottom.get('color', 'neutral')} and {shoes.get('color', 'neutral')}.",
        ]
    if outer:
        lines.append(f"Layering: {outer.get('name', 'outerwear')} adds weather-ready structure.")
    if accessories:
        lines.append(f"Accessories: {', '.join(str(x.get('name', 'accent')) for x in accessories)} complete the style board.")
    lines.append("Personalization: ranking boosted using your Style DNA, memory, and feedback signals.")
    return " ".join(lines)


def _story_title(score: float) -> str:
    if score >= 9:
        return "Hero Look"
    if score >= 7:
        return "Signature Combo"
    if score >= 5:
        return "Polished Daily"
    return "Easy Win"


def _generate_story(outfit: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    top = outfit.get("top", {}) or {}
    bottom = outfit.get("bottom", {}) or {}
    dress = outfit.get("dress", {}) or {}
    shoes = outfit.get("shoes", {}) or {}
    outer = outfit.get("outerwear", {}) or {}
    master_type = str(outfit.get("master_type", "")).lower()

    setting = str(context.get("occasion") or "the day").replace("_", " ")
    weather = str(context.get("weather") or "today")

    if master_type == "dress" or dress:
        narrative = (
            f"For {setting}, anchor the look with {dress.get('name', 'a dress')} and finish with "
            f"{shoes.get('name', 'your shoes')}."
        )
    else:
        narrative = (
            f"For {setting}, start with {top.get('name', 'a top')} to set the tone, "
            f"ground it with {bottom.get('name', 'a bottom')}, and finish with {shoes.get('name', 'your shoes')}."
        )
    if outer:
        narrative += f" Add {outer.get('name', 'an outer layer')} for {weather} comfort and extra polish."

    confidence = float(outfit.get("rank_score", outfit.get("score", 0.0)))
    return {
        "title": _story_title(confidence),
        "narrative": narrative,
        "why_it_works": _explanation_for_outfit(outfit, context),
    }


def _build_tryon_payload(outfit: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    def _safe_id(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    top_id = _safe_id((outfit.get("top") or {}).get("id"))
    bottom_id = _safe_id((outfit.get("bottom") or {}).get("id"))
    dress_id = _safe_id((outfit.get("dress") or {}).get("id"))
    shoes_id = _safe_id((outfit.get("shoes") or {}).get("id"))
    outerwear_id = _safe_id((outfit.get("outerwear") or {}).get("id"))

    # Try-on supports either top+bottom+shoes or dress+shoes.
    if not (top_id and bottom_id and shoes_id):
        if not (dress_id and shoes_id):
            return {}

    payload = {
        "mode": "virtual_try_on",
        "occasion": context.get("occasion"),
        "weather": context.get("weather"),
        "items": {
            "top_id": top_id,
            "bottom_id": bottom_id,
            "dress_id": dress_id,
            "shoes_id": shoes_id,
            "outerwear_id": outerwear_id,
        },
        "prompt": f"Try on this look for {context.get('occasion', 'daily wear')}.",
    }
    return payload


def _build_cards(outfits: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for idx, outfit in enumerate(outfits):
        story = _generate_story(outfit, context)
        tryon_payload = _build_tryon_payload(outfit, context)
        items = []
        for part in ("top", "bottom", "dress", "shoes", "outerwear"):
            value = outfit.get(part, {}) or {}
            if value:
                items.append(value)
        items.extend([x for x in (outfit.get("accessories") or []) if isinstance(x, dict)])
        cards.append(
            {
                "id": f"outfit_card_{idx + 1}",
                "title": story.get("title"),
                "score": outfit.get("rank_score", outfit.get("score", 0.0)),
                "ml_score": outfit.get("ml_score", 0.0),
                "items": items,
                "explanation": story.get("why_it_works"),
                "story": story,
                "tryon_payload": tryon_payload,
            }
        )
    return cards


def save_feedback(user_id: str, outfit: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    feedback_value = str(feedback).strip().lower()
    if feedback_value not in ("up", "down"):
        raise ValueError("feedback must be 'up' or 'down'")

    with _MEMORY_LOCK:
        user_memory = _load_user_memory(user_id)
        record = deepcopy(outfit)
        record["feedback"] = feedback_value
        record["saved_at"] = _utcnow_iso()

        if feedback_value == "up":
            user_memory["liked_outfits"] = [record] + user_memory.get("liked_outfits", [])
            user_memory["liked_outfits"] = user_memory["liked_outfits"][:100]
        else:
            user_memory["disliked_outfits"] = [record] + user_memory.get("disliked_outfits", [])
            user_memory["disliked_outfits"] = user_memory["disliked_outfits"][:100]

        _save_user_memory(user_id, user_memory)
        _index_outfit_vector(user_id=user_id, outfit=record, label=feedback_value)

    outfit_ranker.learn_from_feedback(user_id=user_id, features=outfit.get("ml_features", {}), feedback=feedback_value)
    return {"ok": True, "feedback": feedback_value}


def get_daily_outfits(user: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(user.get("user_id") or user.get("userId") or "anonymous")
    context = user.get("context", {}) or {}
    style_dna = context.get("style_dna", {}) or {}
    raw_wardrobe = user.get("wardrobe", {}) or {}

    normalized = _normalize_wardrobe(raw_wardrobe)
    semantic_items, semantic_map = _semantic_retrieval(user_id=user_id, context=context)
    wardrobe = _merge_wardrobe(normalized, semantic_items)
    occasion = str(context.get("occasion", "")).strip().lower()

    if not occasion:
        return {
            "intent": "daily_outfit",
            "context": "Need occasion clarification before styling.",
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": wardrobe,
            "pipeline": {
                "stages": [
                    "clarifying_questions",
                    "occasion_intent_capture",
                ]
            },
            "clarifying_questions": [
                "Which occasion is this for?",
                "Do you want an outfit around a top or a dress?",
            ],
        }

    occasion_filtered = _occasion_filter(wardrobe, occasion)
    master_type, master_piece = _pick_master_piece(occasion_filtered, occasion, semantic_map)
    if not master_piece:
        return {
            "intent": "daily_outfit",
            "context": f"No suitable top or dress found for occasion '{occasion}'.",
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": wardrobe,
            "pipeline": {
                "stages": [
                    "occasion_intent_capture",
                    "occasion_wardrobe_filter",
                    "master_piece_selection",
                ]
            },
        }

    combinations = _build_master_combos(occasion_filtered, master_type, master_piece, max_combos=40)
    if not combinations:
        msg = "Need bottoms + footwear for top-based styling." if master_type == "top" else "Need footwear for dress-based styling."
        return {
            "intent": "daily_outfit",
            "context": msg,
            "outfits": [],
            "cards": [],
            "boards": [],
            "normalized_wardrobe": wardrobe,
            "pipeline": {
                "stages": [
                    "occasion_intent_capture",
                    "occasion_wardrobe_filter",
                    "master_piece_selection",
                    "combo_construction",
                ]
            },
        }

    color_keep = _llm_filter_combo_ids(
        occasion=occasion,
        stage="color_combo",
        master_type=master_type,
        master_piece=master_piece,
        combos=combinations,
    )
    if not color_keep:
        color_keep = _rule_color_fallback(master_piece, combinations)
    color_filtered = [c for c in combinations if str(c.get("combo_id")) in set(color_keep)] or combinations[:8]

    pattern_keep = _llm_filter_combo_ids(
        occasion=occasion,
        stage="pattern_combo",
        master_type=master_type,
        master_piece=master_piece,
        combos=color_filtered,
    )
    if not pattern_keep:
        pattern_keep = _rule_pattern_fallback(color_filtered)
    pattern_filtered = [c for c in color_filtered if str(c.get("combo_id")) in set(pattern_keep)] or color_filtered[:5]

    merged_context = dict(context)
    merged_context["style_dna"] = style_dna
    merged_context["style_graph"] = style_graph_engine.build_graph(wardrobe)
    rules = style_engine.get_scoring_rules(style_dna, merged_context)

    with _MEMORY_LOCK:
        user_memory = _load_user_memory(user_id)

        scored = []
        for combo in pattern_filtered:
            combo["accessories"] = _select_accessories(occasion_filtered, combo, limit=2)
            scored_combo = score_outfit(combo, merged_context, user_memory, rules, semantic_map)
            scored_combo["master_type"] = master_type
            scored_combo["master_piece"] = master_piece
            scored_combo["pipeline_tags"] = ["occasion_filtered", "master_piece", "llm_color", "llm_pattern", "accessories"]
            scored.append(scored_combo)

        ranked = outfit_ranker.rank(user_id=user_id, outfits=scored, top_n=3)

        user_memory["recent_outfits"] = ranked + user_memory.get("recent_outfits", [])
        user_memory["recent_outfits"] = user_memory["recent_outfits"][:30]
        _save_user_memory(user_id, user_memory)

        for outfit in ranked:
            _index_outfit_vector(user_id=user_id, outfit=outfit, label="recent")

    cards = _build_cards(ranked, merged_context)

    board_item_ids: List[str] = []
    if ranked:
        best = ranked[0]
        for part in ("top", "bottom", "dress", "shoes", "outerwear"):
            item_id = str((best.get(part) or {}).get("id", "")).strip()
            if item_id:
                board_item_ids.append(item_id)
        for item in (best.get("accessories") or []):
            item_id = str((item or {}).get("id", "")).strip()
            if item_id:
                board_item_ids.append(item_id)

    return {
        "intent": "daily_outfit",
        "context": "Generated with occasion filtering, master-piece selection, LLM color/pattern filtering, and accessory completion.",
        "outfits": ranked,
        "cards": cards,
        "boards": cards,
        "board_item_ids": board_item_ids,
        "normalized_wardrobe": wardrobe,
        "pipeline": {
            "stages": [
                "clarifying_questions_if_needed",
                "occasion_intent_capture",
                "occasion_wardrobe_filter",
                "master_piece_selection",
                "llm_color_combo_filter",
                "llm_pattern_combo_filter",
                "accessory_completion",
                "style_board_generation",
                "scoring_engine",
                "explanation_generation",
                "tryon_payload",
                "frontend",
            ],
            "scoring_components": [
                "occasion rules",
                "color intelligence",
                "layering",
                "style graph",
                "memory",
                "feedback",
                "ml_ranker",
            ],
        },
        "memory_summary": {
            "recent_count": len(user_memory.get("recent_outfits", [])),
            "liked_count": len(user_memory.get("liked_outfits", [])),
            "disliked_count": len(user_memory.get("disliked_outfits", [])),
        },
        "premium": {
            "outfit_storytelling": True,
            "style_dna_learning": True,
            "ml_ranking": True,
        },
    }