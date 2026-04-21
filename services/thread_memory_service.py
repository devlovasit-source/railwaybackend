import json
import os
import threading
import time
from typing import Any, Dict, List

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _sanitize(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", ":"}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:160]


class ThreadMemoryService:
    def __init__(self) -> None:
        self._ttl_seconds = _env_int("THREAD_MEMORY_TTL_SECONDS", 7 * 24 * 3600, 300, 60 * 24 * 3600)
        self._max_turns = _env_int("THREAD_MEMORY_MAX_TURNS", 400, 40, 2000)
        self._retain_turns = _env_int("THREAD_MEMORY_RETAIN_TURNS", 220, 20, 1600)
        self._history_char_budget = _env_int("THREAD_MEMORY_HISTORY_MAX_CHARS", 24000, 4000, 200000)
        self._turn_char_limit = _env_int("THREAD_MEMORY_TURN_MAX_CHARS", 1200, 120, 8000)
        self._summary_limit = _env_int("THREAD_MEMORY_SUMMARY_MAX_CHARS", 5000, 400, 40000)
        self._lock = threading.Lock()
        self._local: Dict[str, Dict[str, Any]] = {}
        self._redis = self._build_redis_client()

    def _build_redis_client(self):
        if redis is None:
            return None
        try:
            url = str(os.getenv("REDIS_URL", "redis://localhost:6379/0")).strip()
            if not url:
                return None
            client = redis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            client.ping()
            return client
        except Exception:
            return None

    def _key(self, user_id: str, thread_id: str) -> str:
        return f"tm:{user_id}:{thread_id}"

    def _empty_state(self) -> Dict[str, Any]:
        return {"summary": "", "history": []}

    def _normalize_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for row in turns:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip().lower()
            text = str(row.get("text") or row.get("content") or "").strip()
            if role not in {"user", "assistant", "system"}:
                continue
            if not text:
                continue
            if len(text) > self._turn_char_limit:
                text = text[: self._turn_char_limit].strip()
            out.append({"role": role, "text": text})
        return out

    def _enforce_history_char_budget(self, history: List[Dict[str, str]], summary: str) -> tuple[List[Dict[str, str]], str]:
        if not history:
            return history, summary
        total_chars = sum(len(t.get("text", "")) for t in history)
        if total_chars <= self._history_char_budget:
            return history, summary
        # Spill oldest turns into summary until within budget.
        while history and total_chars > self._history_char_budget:
            spilled = history.pop(0)
            total_chars -= len(spilled.get("text", ""))
            summary = self._append_summary(summary, [spilled])
        return history, summary

    def _append_summary(self, current: str, turns: List[Dict[str, str]]) -> str:
        if not turns:
            return current
        lines = [f"{t['role']}: {t['text'][:240]}" for t in turns]
        extra = "\n".join(lines).strip()
        if not extra:
            return current
        merged = f"{current}\n{extra}".strip() if current else extra
        if len(merged) <= self._summary_limit:
            return merged
        return merged[-self._summary_limit :]

    def _compact_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = self._normalize_turns(state.get("history") or [])
        summary = str(state.get("summary") or "").strip()
        if len(history) > self._max_turns:
            spill = history[: max(0, len(history) - self._retain_turns)]
            history = history[-self._retain_turns :]
            summary = self._append_summary(summary, spill)
        history, summary = self._enforce_history_char_budget(history, summary)
        return {"summary": summary, "history": history}

    def _read_redis(self, key: str) -> Dict[str, Any] | None:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(key)
            if not raw:
                return None
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    def _write_redis(self, key: str, state: Dict[str, Any]) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.setex(
                key,
                self._ttl_seconds,
                json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            )
            return True
        except Exception:
            return False

    def get(self, user_id: str, thread_id: str) -> Dict[str, Any]:
        uid = _sanitize(user_id)
        tid = _sanitize(thread_id)
        if not uid or not tid:
            return self._empty_state()
        key = self._key(uid, tid)

        redis_state = self._read_redis(key)
        if isinstance(redis_state, dict):
            return self._compact_state(redis_state)

        with self._lock:
            row = self._local.get(key)
            if not isinstance(row, dict):
                return self._empty_state()
            expires_at = float(row.get("expires_at") or 0.0)
            if expires_at <= 0 or time.time() >= expires_at:
                self._local.pop(key, None)
                return self._empty_state()
            state = row.get("state") if isinstance(row.get("state"), dict) else self._empty_state()
            return self._compact_state(state)

    def set(self, user_id: str, thread_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        uid = _sanitize(user_id)
        tid = _sanitize(thread_id)
        if not uid or not tid:
            return self._empty_state()
        key = self._key(uid, tid)
        compacted = self._compact_state(state or self._empty_state())
        if not self._write_redis(key, compacted):
            with self._lock:
                self._local[key] = {
                    "expires_at": time.time() + self._ttl_seconds,
                    "state": compacted,
                }
        return compacted

    def append_turns(
        self,
        user_id: str,
        thread_id: str,
        turns: List[Dict[str, Any]],
        incoming_summary: str = "",
    ) -> Dict[str, Any]:
        base = self.get(user_id, thread_id)
        history = self._normalize_turns(base.get("history") or [])
        additions = self._normalize_turns(turns)

        for turn in additions:
            if history and history[-1]["role"] == turn["role"] and history[-1]["text"] == turn["text"]:
                continue
            history.append(turn)

        summary = str(base.get("summary") or "").strip()
        incoming = str(incoming_summary or "").strip()
        if incoming and incoming != summary:
            summary = self._append_summary(summary, [{"role": "system", "text": incoming}])

        return self.set(
            user_id=user_id,
            thread_id=thread_id,
            state={"summary": summary, "history": history},
        )


thread_memory_service = ThreadMemoryService()
