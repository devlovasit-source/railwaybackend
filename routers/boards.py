from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.board_service import (
    AppwriteProxyError,
    R2StorageError,
    delete_saved_board,
    list_life_boards,
    list_saved_boards,
    save_board as save_board_service,
    save_life_board as save_life_board_service,
)

router = APIRouter(prefix="/api/boards", tags=["boards"])


class SaveBoardRequest(BaseModel):
    user_id: str
    title: str = ""
    occasion: str = "Occasion"
    description: str = ""
    image_url: str = ""
    image_base64: str = ""
    board_ids: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class SaveLifeBoardRequest(BaseModel):
    user_id: str
    title: str
    board_type: str = "daily_wear"
    description: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


@router.get("")
def list_boards(user_id: str, occasion: Optional[str] = None, limit: int = 100):
    try:
        docs = list_saved_boards(
            user_id=user_id,
            occasion=occasion,
            limit=limit,
        )
        return {"documents": docs}
    except AppwriteProxyError as exc:
        print(f"[boards.list] user_id={user_id} occasion={occasion} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/save")
def save_board(request: SaveBoardRequest):
    try:
        created = save_board_service(
            user_id=request.user_id,
            occasion=request.occasion,
            image_url=request.image_url,
            image_base64=request.image_base64,
            board_ids=request.board_ids,
            payload=request.payload,
        )
        return {"document": created}
    except R2StorageError as exc:
        raise HTTPException(status_code=500, detail=f"R2 upload failed: {exc}")
    except AppwriteProxyError as exc:
        print(f"[boards.save] user_id={request.user_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/life")
def list_life_boards(user_id: str, limit: int = 100):
    try:
        docs = list_life_boards(user_id=user_id, limit=limit)
        return {"documents": docs}
    except AppwriteProxyError as exc:
        print(f"[boards.life.list] user_id={user_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/life/save")
def save_life_board(request: SaveLifeBoardRequest):
    try:
        created = save_life_board_service(
            user_id=request.user_id,
            title=request.title,
            board_type=request.board_type,
            description=request.description,
            payload=request.payload,
        )
        return {"document": created}
    except AppwriteProxyError as exc:
        print(f"[boards.life.save] user_id={request.user_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{document_id}")
def delete_board(document_id: str):
    try:
        delete_saved_board(document_id=document_id)
        return {"ok": True}
    except AppwriteProxyError as exc:
        print(f"[boards.delete] document_id={document_id} error={exc}")
        raise HTTPException(status_code=400, detail=str(exc))
