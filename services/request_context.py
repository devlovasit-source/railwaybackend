import uuid
from contextvars import ContextVar

_request_id_ctx: ContextVar[str] = ContextVar("ahvi_request_id", default="")


def new_request_id() -> str:
    return str(uuid.uuid4())


def set_request_id(request_id: str | None) -> str:
    rid = str(request_id or "").strip() or new_request_id()
    _request_id_ctx.set(rid)
    return rid


def get_request_id(default: str = "") -> str:
    rid = _request_id_ctx.get(default)
    return str(rid or "").strip()

