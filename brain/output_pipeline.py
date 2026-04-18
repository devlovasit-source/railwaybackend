import random
from services.qdrant_service import QdrantService

qdrant = QdrantService()


def build_outfit_from_embeddings(items, user_id):
    """
    items = list of Qdrant payloads (already filtered by feedback)
    """

    tops = [i for i in items if i.get("category") == "top"]
    bottoms = [i for i in items if i.get("category") == "bottom"]
    shoes = [i for i in items if i.get("category") == "shoes"]

    if not tops:
        return None

    # -------------------------
    # 1. PICK ANCHOR (TOP)
    # -------------------------
    top = random.choice(tops)

    # -------------------------
    # 2. FIND BEST BOTTOM
    # -------------------------
    bottom_candidates = []

    for b in bottoms:
        score = _compatibility_score(top, b)
        bottom_candidates.append((b, score))

    bottom_candidates.sort(key=lambda x: x[1], reverse=True)
    bottom = bottom_candidates[0][0] if bottom_candidates else None

    # -------------------------
    # 3. FIND SHOES
    # -------------------------
    shoe_candidates = []

    for s in shoes:
        score = _compatibility_score(top, s)
        shoe_candidates.append((s, score))

    shoe_candidates.sort(key=lambda x: x[1], reverse=True)
    shoe = shoe_candidates[0][0] if shoe_candidates else None

    return {
        "items": [top, bottom, shoe],
        "score": sum([
            _compatibility_score(top, bottom) if bottom else 0,
            _compatibility_score(top, shoe) if shoe else 0
        ])
    }


# -------------------------
# 🔥 COMPATIBILITY (CORE AI)
# -------------------------
def _compatibility_score(a, b):

    if not a or not b:
        return 0

    score = 0

    # color harmony
    if a.get("color") == b.get("color"):
        score += 0.3

    # category balance
    if a.get("category") != b.get("category"):
        score += 0.2

    # style match
    if a.get("style") == b.get("style"):
        score += 0.3

    # fallback small boost
    score += 0.1

    return score


# -------------------------
# 🔥 MAIN FUNCTION
# -------------------------
def get_daily_outfits(input_data):

    user_id = input_data["user_id"]
    wardrobe = input_data.get("wardrobe", [])

    if not wardrobe:
        return {"outfits": []}

    outfits = []

    for _ in range(5):  # generate 5 outfits
        outfit = build_outfit_from_embeddings(wardrobe, user_id)

        if outfit:
            outfits.append(outfit)

    # sort best first
    outfits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "outfits": outfits[:3]
    }