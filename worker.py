import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
import logging
from services.job_tracker import job_tracker
from services.request_context import set_request_id

try:
    from sentry_sdk.integrations.redis import RedisIntegration
except Exception:
    RedisIntegration = None


# =========================
# SENTRY SETUP
# =========================
def _has_redis_client() -> bool:
    try:
        import redis  # noqa
        return True
    except Exception:
        return False


_sentry_integrations = [CeleryIntegration()]
if RedisIntegration is not None and _has_redis_client():
    _sentry_integrations.append(RedisIntegration())

_sentry_dsn = os.getenv("SENTRY_DSN")
_sentry_client_ready = False
try:
    _sentry_client_ready = bool(getattr(sentry_sdk.Hub.current, "client", None))
except Exception:
    _sentry_client_ready = False
if _sentry_dsn and not _sentry_client_ready:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=1.0,
        integrations=_sentry_integrations,
    )


# =========================
# CELERY INIT
# =========================
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
logger = logging.getLogger("ahvi.worker")

celery_app = Celery(
    "ahvi_tasks",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _retry_or_fail(task, exc: Exception, *, max_retries: int = 2, request_id: str | None = None):
    retries = int(getattr(task.request, "retries", 0))
    if request_id:
        set_request_id(request_id)
    if retries >= max_retries:
        try:
            job_tracker.mark_failed(str(getattr(task.request, "id", "")), error=str(exc))
        except Exception:
            pass
        raise exc
    countdown = 2 ** (retries + 1)
    try:
        job_tracker.mark_retrying(
            str(getattr(task.request, "id", "")),
            error=str(exc),
            attempt=retries + 1,
            max_retries=max_retries,
        )
    except Exception:
        pass
    raise task.retry(exc=exc, countdown=countdown, max_retries=max_retries)


def _mark_started(task, *, request_id: str | None = None, user_id: str | None = None) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        fields = {}
        if request_id:
            fields["request_id"] = str(request_id)
        if user_id:
            fields["user_id"] = str(user_id)
        if fields:
            job_tracker.update(str(getattr(task.request, "id", "")), **fields)
        attempt = int(getattr(task.request, "retries", 0)) + 1
        job_tracker.mark_started(str(getattr(task.request, "id", "")), attempt=attempt)
    except Exception:
        pass


def _mark_succeeded(task, result_meta: dict | None = None, *, request_id: str | None = None) -> None:
    try:
        if request_id:
            set_request_id(request_id)
        job_tracker.mark_succeeded(str(getattr(task.request, "id", "")), result_meta=result_meta or {})
    except Exception:
        pass


# =========================
# AUDIO TASK
# =========================
@celery_app.task(name="generate_audio", bind=True)
def run_heavy_audio_task(self, text_to_clone, lang, request_id: str = ""):
    from services import audio_service

    _mark_started(self, request_id=request_id)
    try:
        audio_base64 = audio_service.generate_cloned_audio(text_to_clone, lang)
        _mark_succeeded(self, {"task": "generate_audio", "request_id": request_id}, request_id=request_id)
        return {"status": "success", "audio_base64": audio_base64}
    except Exception as e:
        logger.exception("AUDIO TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)


# =========================
# IMAGE TASKS
# =========================
@celery_app.task(name="bg_remove_task", bind=True)
def bg_remove_task(self, image_base64: str, request_id: str = ""):
    from routers.bg_remover import remove_background_sync

    _mark_started(self, request_id=request_id)
    try:
        result = remove_background_sync(image_base64)
        _mark_succeeded(self, {"task": "bg_remove_task", "request_id": request_id}, request_id=request_id)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("BG TASK ERROR")
        _retry_or_fail(self, e, request_id=request_id)



