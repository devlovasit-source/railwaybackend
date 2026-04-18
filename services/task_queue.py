from typing import Any, Dict, Iterable
import logging

from services.job_tracker import job_tracker
from services.request_context import get_request_id

logger = logging.getLogger("ahvi.task_queue")


def enqueue_task(
    *,
    task_func,
    args: Iterable[Any] | None = None,
    kwargs: Dict[str, Any] | None = None,
    kind: str,
    user_id: str | None,
    source: str,
    request_id: str | None = None,
    meta: Dict[str, Any] | None = None,
) -> str:
    safe_args = list(args or [])
    safe_kwargs = dict(kwargs or {})
    rid = str(request_id or safe_kwargs.get("request_id") or get_request_id()).strip()
    if rid:
        safe_kwargs["request_id"] = rid

    task = task_func.delay(*safe_args, **safe_kwargs)
    logger.info(
        "task_enqueued request_id=%s task_id=%s kind=%s user_id=%s source=%s",
        rid,
        task.id,
        kind,
        str(user_id or ""),
        source,
    )

    job_tracker.create(
        job_id=task.id,
        kind=kind,
        user_id=user_id,
        request_id=rid,
        source=source,
        meta=meta or {},
    )
    return str(task.id)
