# routers/style_engine.py

import json
import random
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from services.ai_gateway import generate_text, parse_json_object

router = APIRouter()

class StyleRequest(BaseModel):
    occasion: str
    wardrobe: List[Dict[str, Any]]

@router.post("/api/generate-outfit")
def generate_outfit(request: StyleRequest):
    occasion = request.occasion.lower()
    wardrobe = request.wardrobe

    # --- STEP 1: FIND THE MASTERPIECE ---
    potential_masterpieces = [
        item for item in wardrobe 
        if item.get("category", "").title() in ["Dresses", "Tops"] 
        and occasion in [occ.lower() for occ in item.get("occasions", [])]
    ]

    if not potential_masterpieces:
        # Fallback: Pick any Top or Dress if no occasion matches
        potential_masterpieces = [item for item in wardrobe if item.get("category", "").title() in ["Dresses", "Tops"]]
    
    if not potential_masterpieces:
        return {"status": "error", "message": "No tops or dresses found in wardrobe to build an outfit."}

    masterpiece = random.choice(potential_masterpieces)
    is_dress = masterpiece.get("category", "").title() == "Dresses"

    # --- STEP 2: GATHER CANDIDATES (Bulletproof & Case-Insensitive) ---
    available_bottoms = [i for i in wardrobe if i.get("category", "").lower() in ["bottoms", "bottom"] and i != masterpiece]
    available_shoes = [i for i in wardrobe if i.get("category", "").lower() in ["footwear", "shoes", "shoe"] and i != masterpiece]
    available_accessories = [i for i in wardrobe if i.get("category", "").lower() in ["accessories", "accessory", "bags", "bag", "jewelry"] and i != masterpiece]

    # --- STEP 3: OLLAMA STYLIST (ID-BASED PROMPT) ---
    system_prompt = (
        "You are an expert fashion stylist. I will give you a 'Masterpiece' garment, and lists of available bottoms, shoes, and accessories.\n"
        "CRITICAL RULES FOR OUTFIT COMPOSITION:\n"
        f"1. Is the Masterpiece a Dress? {str(is_dress).upper()}\n"
        "2. IF IT IS A DRESS: You MUST select EXACTLY ONE shoe and EXACTLY ONE accessory (if available). You MUST NOT select a bottom.\n"
        "3. IF IT IS A TOP: You MUST select EXACTLY ONE bottom, EXACTLY ONE shoe, and EXACTLY ONE accessory (if available).\n"
        "4. Analyze color and pattern to make the best match.\n"
        "5. Output ONLY a raw JSON object with keys: 'selected_bottom_id' (null if dress), 'selected_shoe_id', 'selected_accessory_id', and 'styling_reason'.\n"
        "6. You MUST select exact IDs from the 'id' field of the provided candidate lists. Do not use names."
    )

    user_prompt = json.dumps({
        "masterpiece": masterpiece,
        "candidates": {
            "bottoms": available_bottoms if not is_dress else [],
            "shoes": available_shoes,
            "accessories": available_accessories
        }
    })

    try:
        raw_response = generate_text(
            prompt=f"{system_prompt}\n\nDATA:\n{user_prompt}",
            options={"temperature": 0.2, "num_predict": 220},
            signals={"context_mode": "styling"},
        )
        selections = parse_json_object(raw_response) if raw_response else {}
        if not isinstance(selections, dict):
            selections = {}

        # --- STEP 4: BULLETPROOF STRICT RULE ENFORCEMENT (ID-BASED) ---
        valid_bottom_ids = [str(item.get("id")) for item in available_bottoms]
        valid_shoe_ids = [str(item.get("id")) for item in available_shoes]
        valid_accessory_ids = [str(item.get("id")) for item in available_accessories]

        # Get AI's choices
        selected_bottom_id = str(selections.get("selected_bottom_id"))
        selected_shoe_id = str(selections.get("selected_shoe_id"))
        selected_accessory_id = str(selections.get("selected_accessory_id"))

        if is_dress:
            # RULE 2: A dress MUST NOT have a bottom.
            selected_bottom_id = None
        else:
            # RULE 1: A top MUST have a bottom. 
            if (selected_bottom_id not in valid_bottom_ids) and available_bottoms:
                selected_bottom_id = str(random.choice(available_bottoms).get("id"))
            
        # Every outfit MUST have exactly 1 matching Footwear.
        if (selected_shoe_id not in valid_shoe_ids) and available_shoes:
            selected_shoe_id = str(random.choice(available_shoes).get("id"))

        # Every outfit gets exactly 1 matching Accessory if available.
        if (selected_accessory_id not in valid_accessory_ids) and available_accessories:
            selected_accessory_id = str(random.choice(available_accessories).get("id"))

        # --- STEP 5: ASSEMBLE OUTFIT ---
        final_outfit = [masterpiece]
        target_ids = [selected_bottom_id, selected_shoe_id, selected_accessory_id]
        
        for item in available_bottoms + available_shoes + available_accessories:
            if str(item.get("id")) in target_ids:
                final_outfit.append(item)

        # Extract the true Database IDs for the React Native frontend
        item_ids = [str(item.get("id")) for item in final_outfit if item.get("id")]
        style_board_tag = f"[STYLE_BOARD: {', '.join(item_ids)}]"

        return {
            "status": "success",
            "masterpiece": masterpiece.get("name"),
            "outfit": final_outfit,
            "styling_reason": selections.get("styling_reason", "A perfectly balanced look based on your wardrobe."),
            "style_board_tag": style_board_tag
        }

    except Exception:
        logging.exception("Style Engine Error")
        # Absolute Fallback
        fallback_outfit = [masterpiece]
        
        if not is_dress and available_bottoms: 
            fallback_outfit.append(random.choice(available_bottoms))
        if available_shoes: 
            fallback_outfit.append(random.choice(available_shoes))
        if available_accessories: 
            fallback_outfit.append(random.choice(available_accessories))
        
        item_ids = [str(item.get("id")) for item in fallback_outfit if item.get("id")]
        return {
            "status": "fallback",
            "outfit": fallback_outfit,
            "style_board_tag": f"[STYLE_BOARD: {', '.join(item_ids)}]"
        }
