from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List
from services import ai_gateway
from brain.outfit_pipeline import get_daily_outfits
from brain.personalization.style_dna_engine import style_dna_engine
from services.appwrite_proxy import AppwriteProxy

router = APIRouter()

# Notice we no longer accept base64 images here. We accept the CLIP output.
class ItemContextRequest(BaseModel):
    main_category: str
    sub_category: str
    color_hex: str


class OutfitPipelineRequest(BaseModel):
    user_id: str
    query: str = "What should I wear today?"
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

@router.post("/item-suggestions")
def get_item_suggestions(request: ItemContextRequest):
    system_instruction = (
        "You are Ahvi's Fashion Knowledge Engine. The user just uploaded a new garment. "
        "Based on the provided attributes, return a JSON object with: "
        "1. 'name' (A catchy, descriptive name for the item) "
        "2. 'tags' (array of 4 style keywords like 'streetwear', 'vintage') "
        "3. 'pairing_rules' (array of 2 short rules on what to wear this with). "
        "Output ONLY raw JSON."
    )
    
    user_prompt = (
        f"Item: {request.sub_category}\n"
        f"Category: {request.main_category}\n"
        f"Color Hex: {request.color_hex}"
    )
    
    try:
        messages = [{"role": "user", "content": user_prompt}]
        return ai_gateway.chat_json_object(
            messages,
            system_instruction=system_instruction,
            model="llama3.1",
        )

    except Exception as e:
        print(f"Stylist Text Engine Error: {str(e)}")
        # Safe fallback
        return {
            "name": f"{request.sub_category.title()}",
            "tags": ["versatile", "casual"],
            "pairing_rules": ["Pair with neutral basics.", "Layer depending on weather."]
        }


@router.post("/pipeline")
def run_outfit_pipeline(request: OutfitPipelineRequest):
    appwrite = AppwriteProxy()
    context = dict(request.context or {})
    context["query"] = request.query

    wardrobe = request.wardrobe
    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents("outfits", user_id=request.user_id)
        except Exception:
            wardrobe = []

    style_dna = style_dna_engine.build(
        {
            "user_id": request.user_id,
            "user_profile": request.user_profile or {},
            "history": context.get("history", []),
            "wardrobe": wardrobe,
        }
    )
    context["style_dna"] = style_dna

    try:
        result = get_daily_outfits(
            {
                "user_id": request.user_id,
                "wardrobe": wardrobe,
                "context": context,
            }
        )
        return {
            "success": True,
            "message": "Outfit pipeline generated successfully.",
            "data": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run outfit pipeline: {exc}")
