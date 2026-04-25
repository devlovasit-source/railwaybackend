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
    context: Dict[str, Any] = Field(default_factory=dict)


@router.post("/ahvi/chat")
def chat(req: ChatRequest):
    message = str(req.message or "").strip()
    if not message:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_MESSAGE",
                    "message": "message cannot be empty",
                },
            },
        )

    result = ahvi_orchestrator.run(
        text=message,
        user_id=req.user_id,
        context=req.context or {}
    )

    # Orchestrator can legitimately return success=False for business-level
    # outcomes (e.g. graceful fallback); that's not a server crash.
    return result
