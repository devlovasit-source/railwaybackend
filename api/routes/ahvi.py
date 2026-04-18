# backend/api/routes/ahvi.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict

from brain.orchestrator import ahvi_orchestrator

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str | None = None
    context: Dict[str, Any] = {}


@router.post("/ahvi/chat")
def chat(req: ChatRequest):

    result = ahvi_orchestrator.run(
        text=req.message,
        user_id=req.user_id,
        context=req.context or {}
    )

    if not result.get("success", False):
        return JSONResponse(status_code=500, content=result)

    return result
