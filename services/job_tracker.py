import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List
import logging

try:
    import redis
except Exception:  # pragma: no cover
    redis = None

from services.appwrite_proxy import AppwriteProxy

logger = logging.getLogger("ahvi.job_tracker")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


class JobTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._redis = None
        self._appwrite = None
        self._redis_prefix = "ahvi:job:"
        self._redis_recent_global = "ahvi:job:recent:global"
        self._redis_recent_user_prefix = "ahvi:job:recent:user:"
        self._max_memory_items = 2000

    def _redis_client(self):
        if self._redis is not None:
            return self._redis
        if redis is None:
            return None
        try:
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis = redis.Redis.from_url(url, decode_responses=True)
            self._redis.ping()
            return self._redis
        except Exception:
            self._redis = None
            return None

    def _appwrite_enabled(self) -> bool:
        raw = str(os.getenv("JOB_TRACKER_APPWRITE_ENABLED", "true")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _appwrite_client(self) -> AppwriteProxy | None:
        if not self._appwrite_enabled():
            return None
        if self._appwrite is not None:
            return self._appwrite
        try:
            self._appwrite = AppwriteProxy()
            return self._appwrite
        except Exception:
            self._appwrite = None
            return None

    def _redis_key(self, job_id: str) -> str:
        return f"{self._redis_prefix}{job_id}"

    def _to_appwrite_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        created_at = _parse_iso_utc(payload.get("created_at")) or datetime.now(timezone.utc)
        updated_at = _parse_iso_utc(payload.get("updated_at")) or created_at
        duration_ms = int(payload.get("duration_ms") or 0)
        if duration_ms <= 0:
            duration_ms = max(0, int((updated_at - created_at).total_seconds() * 1000))

        attempt = int(payload.get("attempt") or 0)
        retry_count = int(payload.get("retry_count") or max(0, attempt - 1))
        status = str(payload.get("status") or "queued").strip().lower() or "queued"

        output_obj = payload.get("result_meta", {})
        output_text = ""
        try:
            output_text = json.dumps(output_obj, ensure_ascii=True)
        except Exception:
            output_text = str(output_obj or "")
        if len(output_text) > 255:
            output_text = output_text[:252] + "..."

        error_text = str(payload.get("error") or "")
        if len(error_text) > 1500:
            error_text = error_text[:1497] + "..."

        input_text = str(payload.get("source") or payload.get("kind") or "job")
        if len(input_text) > 255:
            input_text = input_text[:252] + "..."

        return {
            "userId": str(payload.get("user_id") or "system"),
            "type": str(payload.get("kind") or "job"),
            "status": status,
            "input": input_text,
            "output": output_text,
            "error": error_text,
            "retry_count": retry_count,
            "duration_ms": duration_ms,
            "request_id": str(payload.get("request_id") or ""),
        }

    def _from_appwrite_job(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return {
            "job_id": str(raw.get("$id") or raw.get("job_id") or ""),
            "kind": str(raw.get("type") or "job"),
            "user_id": str(raw.get("userId") or raw.get("user_id") or ""),
            "request_id": str(raw.get("request_id") or ""),
            "source": "",
            "status": str(raw.get("status") or "queued"),
            "state": str(raw.get("state") or "").strip() or {
                "queued": "PENDING",
                "processing": "STARTED",
                "retrying": "RETRY",
                "completed": "SUCCESS",
                "failed": "FAILURE",
            }.get(str(raw.get("status") or "").strip().lower(), "PENDING"),
            "attempt": int(raw.get("retry_count") or 0) + 1,
            "retry_count": int(raw.get("retry_count") or 0),
            "max_retries": 0,
            "created_at": str(raw.get("$createdAt") or _now_iso()),
            "updated_at": str(raw.get("$updatedAt") or _now_iso()),
            "error": str(raw.get("error") or ""),
            "result_meta": {"output": str(raw.get("output") or "")},
            "duration_ms": int(raw.get("duration_ms") or 0),
            "meta": {},
        }

    def _persist_appwrite(self, payload: Dict[str, Any]) -> None:
        client = self._appwrite_client()
        if client is None:
            return
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return
        row = self._to_appwrite_job(payload)
        try:
            client.update_document("jobs", job_id, row)
            return
        except Exception:
            pass
        try:
            client.create_document("jobs", row, document_id=job_id)
        except Exception as exc:
            logger.warning("job_tracker.appwrite_persist_failed job_id=%s error=%s", job_id, exc)

    def _write(self, payload: Dict[str, Any]) -> None:
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return
        payload["updated_at"] = _now_iso()

        client = self._redis_client()
        if client is not None:
            try:
                ts = datetime.now(timezone.utc).timestamp()
                client.setex(self._redis_key(job_id), 7 * 24 * 3600, json.dumps(payload, ensure_ascii=True))
                client.zadd(self._redis_recent_global, {job_id: ts})
                client.expire(self._redis_recent_global, 7 * 24 * 3600)
                uid = str(payload.get("user_id") or "").strip()
                if uid:
                    user_key = f"{self._redis_recent_user_prefix}{uid}"
                    client.zadd(user_key, {job_id: ts})
                    client.expire(user_key, 7 * 24 * 3600)
            except Exception:
                pass

        with self._lock:
            self._memory[job_id] = dict(payload)
            if len(self._memory) > self._max_memory_items:
                oldest = sorted(
                    self._memory.items(),
                    key=lambda kv: str(kv[1].get("updated_at") or ""),
                )[: max(0, len(self._memory) - self._max_memory_items)]
                for key, _ in oldest:
                    self._memory.pop(key, None)
        self._persist_appwrite(payload)

    def get(self, job_id: str) -> Dict[str, Any] | None:
        jid = str(job_id or "").strip()
        if not jid:
            return None

        client = self._redis_client()
        if client is not None:
            try:
                raw = client.get(self._redis_key(jid))
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass

        with self._lock:
            data = self._memory.get(jid)
            if isinstance(data, dict):
                return dict(data)

        client = self._appwrite_client()
        if client is not None:
            try:
                doc = client.get_document("jobs", jid)
                row = self._from_appwrite_job(doc)
                if row.get("job_id"):
                    with self._lock:
                        self._memory[jid] = dict(row)
                    return row
            except Exception:
                return None
        return None

    def create(
        self,
        *,
        job_id: str,
        kind: str,
        user_id: str | None = None,
        request_id: str | None = None,
        source: str | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "job_id": str(job_id),
            "kind": str(kind or "job"),
            "user_id": str(user_id or ""),
            "request_id": str(request_id or ""),
            "source": str(source or ""),
            "status": "queued",
            "state": "PENDING",
            "attempt": 0,
            "max_retries": 0,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "meta": meta or {},
        }
        self._write(payload)
        return payload

    def update(self, job_id: str, **fields: Any) -> Dict[str, Any] | None:
        current = self.get(job_id)
        if current is None:
            return None
        current.update(fields)
        self._write(current)
        return current

    def mark_started(self, job_id: str, *, attempt: int = 1) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        current.update({"status": "processing", "state": "STARTED", "attempt": int(attempt), "retry_count": max(0, int(attempt) - 1)})
        self._write(current)

    def mark_retrying(self, job_id: str, *, error: str, attempt: int, max_retries: int) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        current.update(
            {
                "status": "retrying",
                "state": "RETRY",
                "error": str(error),
                "attempt": int(attempt),
                "max_retries": int(max_retries),
            }
        )
        self._write(current)

    def mark_succeeded(self, job_id: str, *, result_meta: Dict[str, Any] | None = None) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        created = _parse_iso_utc(current.get("created_at")) or datetime.now(timezone.utc)
        duration_ms = max(0, int((datetime.now(timezone.utc) - created).total_seconds() * 1000))
        current.update(
            {
                "status": "completed",
                "state": "SUCCESS",
                "error": None,
                "result_meta": result_meta or {},
                "completed_at": _now_iso(),
                "duration_ms": duration_ms,
            }
        )
        self._write(current)

    def mark_failed(self, job_id: str, *, error: str) -> None:
        current = self.get(job_id) or {"job_id": job_id, "created_at": _now_iso(), "meta": {}}
        created = _parse_iso_utc(current.get("created_at")) or datetime.now(timezone.utc)
        duration_ms = max(0, int((datetime.now(timezone.utc) - created).total_seconds() * 1000))
        current.update(
            {
                "status": "failed",
                "state": "FAILURE",
                "error": str(error),
                "completed_at": _now_iso(),
                "duration_ms": duration_ms,
            }
        )
        self._write(current)

    def list_recent(
        self,
        *,
        limit: int = 50,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        uid = str(user_id or "").strip()
        rid = str(request_id or "").strip()

        client = self._redis_client()
        if client is not None:
            try:
                key = f"{self._redis_recent_user_prefix}{uid}" if uid else self._redis_recent_global
                job_ids = client.zrevrange(key, 0, safe_limit - 1)
                rows: List[Dict[str, Any]] = []
                for jid in job_ids:
                    row = self.get(str(jid))
                    if isinstance(row, dict):
                        rows.append(row)
                if rid:
                    rows = [row for row in rows if str(row.get("request_id") or "") == rid]
                return rows
            except Exception:
                pass

        with self._lock:
            items = list(self._memory.values())
        items.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        if uid:
            items = [row for row in items if str(row.get("user_id") or "") == uid]
        if rid:
            items = [row for row in items if str(row.get("request_id") or "") == rid]
        if items:
            return [dict(row) for row in items[:safe_limit]]

        client = self._appwrite_client()
        if client is not None:
            try:
                docs = client.list_documents("jobs", user_id=uid or None, limit=safe_limit)
                rows = [self._from_appwrite_job(doc) for doc in (docs or []) if isinstance(doc, dict)]
                if rid:
                    rows = [row for row in rows if str(row.get("request_id") or "") == rid]
                return rows[:safe_limit]
            except Exception:
                pass
        return []


job_tracker = JobTracker()
