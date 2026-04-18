from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.qdrant_service import QdrantService

router = APIRouter(prefix="/api/feedback")

qdrant = QdrantService()


class FeedbackRequest(BaseModel):
    item_id: str
    feedback: str  # up / down


@router.post("/item")
def feedback_item(request: FeedbackRequest):

    fb = request.feedback.lower()

    if fb not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="feedback must be up/down")

    try:
        qdrant.update_feedback(request.item_id, fb)

        return {
            "success": True,
            "message": "Feedback recorded"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))