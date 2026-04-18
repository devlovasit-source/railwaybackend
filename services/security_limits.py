import asyncio
import time
from typing import Tuple

from services.settings import settings

try:
    import redis.asyncio as redis_async
except Exception:  # pragma: no cover - optional dependency at runtime
    redis_async = None


_redis_client = None
_redis_lock = asyncio.Lock()
_local_lock = asyncio.Lock()
_local_windows: dict[str, tuple[int, float]] = {}


async def get_redis_client():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if redis_async is None:
        return None
    async with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            _redis_client = redis_async.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await _redis_client.ping()
        except Exception:
            _redis_client = None
        return _redis_client


async def is_redis_rate_limit_ready() -> bool:
    client = await get_redis_client()
    return client is not None


def extract_client_ip(headers, client_host: str | None) -> str:
    forwarded = ""
    try:
        forwarded = headers.get("x-forwarded-for", "")
    except Exception:
        forwarded = ""
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return str(client_host or "unknown")


async def _check_local_window(key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
    now = time.time()
    async with _local_lock:
        count, reset_at = _local_windows.get(key, (0, now + window_seconds))
        if now >= reset_at:
            count = 0
            reset_at = now + window_seconds
        count += 1
        _local_windows[key] = (count, reset_at)
        allowed = count <= max_requests
        remaining = max(0, max_requests - count)
        return allowed, remaining


async def check_rate_limit(
    *,
    bucket_key: str,
    max_requests: int | None = None,
    window_seconds: int | None = None,
) -> Tuple[bool, int]:
    if not settings.rate_limit_enabled:
        return True, 999999

    max_requests = int(max_requests or settings.rate_limit_max_requests)
    window_seconds = int(window_seconds or settings.rate_limit_window_seconds)

    redis_client = await get_redis_client()
    if redis_client is None:
        return await _check_local_window(bucket_key, max_requests, window_seconds)

    try:
        key = f"rl:{bucket_key}"
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, window_seconds)
        allowed = int(current) <= max_requests
        remaining = max(0, max_requests - int(current))
        return allowed, remaining
    except Exception:
        return await _check_local_window(bucket_key, max_requests, window_seconds)
