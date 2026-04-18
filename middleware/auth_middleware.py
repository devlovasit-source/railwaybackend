import asyncio
import hashlib
import json
import time
from typing import Any, Dict

from fastapi import HTTPException, Request

from services.appwrite_service import build_account_for_jwt
from services.security_limits import get_redis_client
from services.settings import settings

_mem_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_mem_lock = asyncio.Lock()


def _extract_bearer_token(auth_header: str) -> str:
    parts = str(auth_header or "").split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return parts[1].strip()


def _token_cache_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"auth:token:{digest}"


async def _cache_get(token: str) -> Dict[str, Any] | None:
    key = _token_cache_key(token)
    redis_client = await get_redis_client()
    if redis_client is not None:
        try:
            value = await redis_client.get(key)
            if value:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass

    now = time.time()
    async with _mem_lock:
        row = _mem_cache.get(key)
        if not row:
            return None
        expires_at, payload = row
        if now >= expires_at:
            _mem_cache.pop(key, None)
            return None
        return dict(payload)


async def _cache_set(token: str, payload: Dict[str, Any]) -> None:
    key = _token_cache_key(token)
    ttl = int(settings.auth_cache_ttl_seconds)
    redis_client = await get_redis_client()
    if redis_client is not None:
        try:
            await redis_client.setex(key, ttl, json.dumps(payload))
            return
        except Exception:
            pass

    async with _mem_lock:
        _mem_cache[key] = (time.time() + ttl, dict(payload))


def _validate_token_sync(token: str) -> Dict[str, Any]:
    account = build_account_for_jwt(token)
    user = account.get()
    return {
        "user_id": user["$id"],
        "email": user.get("email"),
        "name": user.get("name"),
    }


async def get_current_user(request: Request):
    auth_header = request.headers.get("authorization", "")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = _extract_bearer_token(auth_header)
    cached = await _cache_get(token)
    if cached:
        return cached

    try:
        payload = await asyncio.to_thread(_validate_token_sync, token)
        await _cache_set(token, payload)
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

