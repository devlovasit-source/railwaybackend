import asyncio
import logging
from typing import Callable
from typing import Any, Dict
from uuid import uuid4
from time import perf_counter

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import importlib.util
import os

# ?? QDRANT SERVICE
from services.qdrant_service import qdrant_service
from services.security_limits import check_rate_limit, extract_client_ip, is_redis_rate_limit_ready
from services.settings import settings
from middleware.auth_middleware import get_current_user
from services.job_tracker import job_tracker
from services.request_context import set_request_id

logger = logging.getLogger("ahvi.main")


# -------------------------
# OPTIONAL ROUTER LOADER
# -------------------------
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_optional_router(module_name: str, attr: str = "router"):
    if not _has_module(module_name):
        logger.info("router skipped module=%s reason=not_found", module_name)
        return None
    try:
        module = __import__(module_name, fromlist=[attr])
        return getattr(module, attr)
    except Exception as exc:
        logger.warning("router load failed module=%s error=%s", module_name, exc)
        return None


# -------------------------
# LOAD ALL ROUTERS (SAFE)
# -------------------------
chat_router = _load_optional_router("routers.chat")
data_router = _load_optional_router("routers.data")
utilities_router = _load_optional_router("routers.utilities")
boards_router = _load_optional_router("routers.boards")
feedback_router = _load_optional_router("routers.feedback")

# AI
ai_router = _load_optional_router("api.ai")

# Optional
stylist_router = _load_optional_router("routers.stylist")
reddit_router = _load_optional_router("routers.reddit")

# Feature-based
bg_router = None
if os.getenv("ENABLE_BG_REMOVER", "false").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["transformers", "torch", "PIL"]):
        bg_router = _load_optional_router("routers.bg_remover")

vision_router = None
if os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["cv2", "sklearn", "numpy"]):
        vision_router = _load_optional_router("routers.vision")

wardrobe_capture_router = None
if os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["numpy", "PIL"]):
        wardrobe_capture_router = _load_optional_router("routers.wardrobe_capture")

garment_router = None
if os.getenv("ENABLE_GARMENT_ANALYZER", "false").lower() in ("1", "true", "yes"):
    if all(_has_module(m) for m in ["transformers", "PIL", "cv2", "sklearn", "numpy"]):
        garment_router = _load_optional_router("routers.garment_analyzer")


# -------------------------
# OPTIONAL IMPORTS
# -------------------------
try:
    from celery.result import AsyncResult
except Exception:
    AsyncResult = None

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except Exception:
    sentry_sdk = None
    FastApiIntegration = None

try:
    from worker import celery_app
except Exception:
    celery_app = None


# -------------------------
# SENTRY
# -------------------------
_sentry_dsn = os.getenv("SENTRY_DSN")
_sentry_client_ready = False
if sentry_sdk:
    try:
        _sentry_client_ready = bool(getattr(sentry_sdk.Hub.current, "client", None))
    except Exception:
        _sentry_client_ready = False
if _sentry_dsn and sentry_sdk and FastApiIntegration and not _sentry_client_ready:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=1.0,
        integrations=[FastApiIntegration()],
    )


# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="AHVI AI Master Brain API",
    version="2.2.0"
)

logger.info("AHVI Backend Started")

class PayloadTooLargeError(Exception):
    pass


class StreamBodyLimitMiddleware:
    def __init__(self, app: Callable, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "")).upper()
        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = {}
        for k, v in scope.get("headers", []):
            try:
                headers[k.decode("latin-1").lower()] = v.decode("latin-1")
            except Exception:
                continue

        content_length = headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "error": {
                                "code": "PAYLOAD_TOO_LARGE",
                                "message": f"Upload exceeds max size {self.max_bytes} bytes",
                            },
                        },
                    )
                    await response(scope, receive, send)
                    return
            except Exception:
                pass

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                received += len(chunk)
                if received > self.max_bytes:
                    raise PayloadTooLargeError()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except PayloadTooLargeError:
            response = JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": {
                        "code": "PAYLOAD_TOO_LARGE",
                        "message": f"Upload exceeds max size {self.max_bytes} bytes",
                    },
                },
            )
            await response(scope, receive, send)


# -------------------------
# STARTUP / SHUTDOWN EVENTS
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("startup begin")

    try:
        await asyncio.to_thread(qdrant_service.init)
    except Exception as e:
        logger.exception("qdrant startup failed: %s", e)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("shutdown begin")
    try:
        client = getattr(qdrant_service, "client", None)
        if client is not None and hasattr(client, "close"):
            await asyncio.to_thread(client.close)
    except Exception as e:
        logger.exception("qdrant shutdown failed: %s", e)
    try:
        from services import appwrite_service  # local import to avoid circulars
        appwrite_client = getattr(appwrite_service, "client", None)
        if appwrite_client is not None and hasattr(appwrite_client, "close"):
            await asyncio.to_thread(appwrite_client.close)
        else:
            logger.info("appwrite shutdown skip: client.close() unavailable")
    except Exception as e:
        logger.warning("appwrite shutdown skip error=%s", e)


# -------------------------
# ERROR HANDLERS
# -------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = str(getattr(request.state, "request_id", "") or "")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "request_id": request_id,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request",
                "details": exc.errors(),
            },
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(getattr(request.state, "request_id", "") or "")
    logger.exception("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "request_id": request_id,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
            },
        },
    )


# -------------------------
# MIDDLEWARE
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(StreamBodyLimitMiddleware, max_bytes=settings.upload_max_bytes)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID")
    request_id = str(incoming or "").strip() or str(uuid4())
    set_request_id(request_id)
    request.state.request_id = request_id
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(getattr(response, "status_code", 500))
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception("request failed request_id=%s method=%s path=%s", request_id, request.method, request.url.path)
        raise
    finally:
        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "request request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


@app.middleware("http")
async def auth_guard_middleware(request: Request, call_next):
    if not settings.auth_required:
        return await call_next(request)
    path = str(request.url.path or "")
    if path in {"/", "/health"} or path.startswith("/docs") or path.startswith("/openapi"):
        return await call_next(request)
    if path.startswith("/api/tasks/"):
        return await call_next(request)
    try:
        request.state.user = await get_current_user(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not settings.rate_limit_enabled:
        return await call_next(request)
    if str(request.method or "").upper() == "OPTIONS":
        return await call_next(request)
    redis_ready = await is_redis_rate_limit_ready()
    if settings.rate_limit_require_redis and not redis_ready:
        status_code = 429 if settings.rate_limit_fail_closed else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "request_id": str(getattr(request.state, "request_id", "") or ""),
                "error": {
                    "code": "RATE_LIMIT_BACKEND_UNAVAILABLE",
                    "message": "Rate-limit backend unavailable",
                },
            },
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )
    request_id = str(getattr(request.state, "request_id", "") or "")
    ip = extract_client_ip(request.headers, request.client.host if request.client else None)
    user_id = ""
    if isinstance(getattr(request.state, "user", None), dict):
        user_id = str(request.state.user.get("$id") or request.state.user.get("id") or "")
    identity = user_id or ip
    allowed, remaining = await check_rate_limit(
        bucket_key=f"{identity}:{request.url.path}",
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "request_id": request_id,
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Too many requests. Please retry later.",
                },
            },
            headers={
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Limit": str(settings.rate_limit_max_requests),
                "X-RateLimit-Window": str(settings.rate_limit_window_seconds),
                "X-RateLimit-Backend": "redis" if redis_ready else "local",
                "Retry-After": str(settings.rate_limit_window_seconds),
            },
        )
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_max_requests)
    response.headers["X-RateLimit-Window"] = str(settings.rate_limit_window_seconds)
    response.headers["X-RateLimit-Backend"] = "redis" if redis_ready else "local"
    return response


# -------------------------
# ROUTER REGISTRATION
# -------------------------
if chat_router:
    app.include_router(chat_router, prefix="/api", tags=["Chat"])

if data_router:
    app.include_router(data_router)

if utilities_router:
    app.include_router(utilities_router)

if boards_router:
    app.include_router(boards_router)

if ai_router:
    app.include_router(ai_router, prefix="/api", tags=["AI"])

if feedback_router:
    app.include_router(feedback_router, tags=["Feedback"])

if stylist_router:
    app.include_router(stylist_router, prefix="/api/stylist")

if reddit_router:
    app.include_router(reddit_router)

if vision_router:
    app.include_router(vision_router, prefix="/api/vision")

if wardrobe_capture_router:
    app.include_router(wardrobe_capture_router)

if bg_router:
    app.include_router(bg_router, prefix="/api/background")

if garment_router:
    app.include_router(garment_router, prefix="/api")


# -------------------------
# HEALTH
# -------------------------

# -------------------------
# BG REMOVE COMPAT ROUTES
# -------------------------
class BgCompatRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)


@app.post("/api/background/remove-bg")
@app.post("/api/remove-bg")
def remove_bg_compat(payload: BgCompatRequest):
    """
    Always expose background-removal endpoint even when optional router gating
    prevents router mounting. This keeps frontend flow stable.
    """
    image_base64 = payload.image_base64

    try:
        from routers.bg_remover import remove_background_sync
    except Exception as exc:
        # Return a hard failure so clients do not treat this as a successful cutout.
        raise HTTPException(
            status_code=503,
            detail=f"BG remover unavailable: {exc}",
        )

    try:
        result = remove_background_sync(image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Background removal failed: {exc}",
        )
    if not isinstance(result, dict) or result.get("bg_removed") is not True:
        fallback = result.get("fallback_reason") if isinstance(result, dict) else "Background removal failed"
        raise HTTPException(
            status_code=503,
            detail=fallback or "Background removal failed",
        )
    logger.info(
        "bg compat result %s",
        {
            "bg_removed": result.get("bg_removed"),
            "fallback_reason": result.get("fallback_reason"),
        },
    )
    return result


class VisionCompatRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    user_id: str | None = None
    userId: str | None = None


@app.post("/api/analyze-image")
@app.post("/api/vision/analyze-image")
@app.post("/api/vision/analyze")
@app.post("/api/analyze")
def analyze_compat(payload: VisionCompatRequest):
    """
    Backward-compatible image analysis endpoints expected by older frontend builds.
    """
    try:
        from routers.vision import vision_analyze_core
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Analyze endpoint unavailable: {exc}")

    user_id = str(payload.user_id or payload.userId or "demo_user").strip() or "demo_user"
    try:
        core = vision_analyze_core(payload.image_base64, user_id)
    except Exception as exc:
        user_input_payload = {
            "name": "",
            "category": "",
            "sub_category": "",
            "pattern": "",
            "occasions": [],
            "color_code": "",
        }
        return {
            "success": False,
            "requires_user_input": True,
            "message": "Vision model unavailable. Please enter item details manually.",
            "data": user_input_payload,
            "items": [
                {
                    "id": "manual-input-required",
                    **user_input_payload,
                }
            ],
            "similar_items": [],
            "duplicate": {"is_duplicate": False, "id": None, "score": 0.0},
            "meta": {
                "llm_only": True,
                "vision_model_used": None,
                "analysis_source": "manual_input_required",
                "ollama_error": str(exc),
            },
            "questions": [
                "What is the garment type (top, bottom, dress, footwear, accessory)?",
                "What is the sub-category (for example trousers, shirt, sneakers)?",
                "What is the primary color and pattern?",
                "Which occasions does this item fit?",
            ],
        }

    if not isinstance(core, dict):
        raise HTTPException(status_code=502, detail="Vision analyzer returned invalid payload")

    analysis = core.get("data") if isinstance(core.get("data"), dict) else {}
    meta = core.get("meta") if isinstance(core.get("meta"), dict) else {}
    similar_items = core.get("similar_items") if isinstance(core.get("similar_items"), list) else []
    duplicate = {
        "is_duplicate": bool(meta.get("probable_duplicate")),
        "id": meta.get("image_duplicate_point_id") or meta.get("pixel_duplicate_point_id"),
        "score": float(meta.get("top_similarity_score") or meta.get("image_duplicate_score") or 0.0),
    }

    return {
        "success": True,
        "data": analysis,
        "processed_image_base64": core.get("processed_image_base64") or payload.image_base64,
        "items": [
            {
                "id": str(analysis.get("name") or "item"),
                "name": analysis.get("name"),
                "category": analysis.get("category"),
                "sub_category": analysis.get("sub_category"),
                "color_code": analysis.get("color_code"),
                "pattern": analysis.get("pattern"),
                "occasions": analysis.get("occasions", []),
            }
        ],
        "similar_items": similar_items,
        "duplicate": duplicate,
        "meta": {
            "vision_model_used": meta.get("vision_model_used"),
            "duplicate_threshold": meta.get("duplicate_threshold"),
            "llm_only": True,
            "analysis_source": "ollama",
            "llm_fallback": bool(meta.get("llm_fallback")),
            "bg_removed": meta.get("bg_removed"),
        },
    }
@app.get("/")
def root():
    return {"message": "AHVI backend running"}


@app.get("/health")
def health_check():
    return {"status": "online"}


# -------------------------
# CELERY STATUS
# -------------------------
@app.get("/api/tasks/{job_id}")
def get_task_status(job_id: str, request: Request):
    request_id = str(getattr(request.state, "request_id", "") or "")
    tracker_data = job_tracker.get(job_id) or {}
    if not celery_app or AsyncResult is None:
        if tracker_data:
            return {"status": tracker_data.get("status", "queued"), "state": tracker_data.get("state", "PENDING"), "job": tracker_data, "request_id": request_id}
        return {"status": "celery not configured", "request_id": request_id}

    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.state == "PENDING":
        return {
            "status": str(tracker_data.get("status") or "queued"),
            "state": "PENDING",
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "STARTED":
        return {
            "status": "processing",
            "state": "STARTED",
            "meta": task_result.info if isinstance(task_result.info, dict) else {},
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "SUCCESS":
        return {
            "status": "completed",
            "state": "SUCCESS",
            "result": task_result.result,
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "FAILURE":
        return {
            "status": "failed",
            "state": "FAILURE",
            "error": str(task_result.info),
            "job": tracker_data,
            "request_id": request_id,
        }

    if task_result.state == "RETRY":
        return {
            "status": "retrying",
            "state": "RETRY",
            "error": str(task_result.info),
            "job": tracker_data,
            "request_id": request_id,
        }

    return {
        "status": str(tracker_data.get("status") or "processing"),
        "state": task_result.state,
        "job": tracker_data,
        "request_id": request_id,
    }


@app.get("/api/jobs/recent")
def list_recent_jobs(limit: int = 25, user_id: str | None = None, request_id: str | None = None):
    return {
        "success": True,
        "jobs": job_tracker.list_recent(limit=limit, user_id=user_id, request_id=request_id),
    }
