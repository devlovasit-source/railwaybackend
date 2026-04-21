import json
import logging
import os
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Tuple

import requests

from services import llm_service
from services.request_context import get_request_id

logger = logging.getLogger("ahvi.ai_gateway")


@dataclass(frozen=True)
class GatewayPolicy:
    timeout_seconds: int
    model: str | None = None


_DEFAULT_POLICY = GatewayPolicy(timeout_seconds=35, model=None)
_POLICIES: Dict[str, GatewayPolicy] = {
    "general": GatewayPolicy(timeout_seconds=35, model=None),
    "styling": GatewayPolicy(timeout_seconds=45, model=None),
    "intent": GatewayPolicy(timeout_seconds=20, model=None),
    "vision": GatewayPolicy(timeout_seconds=int(os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "1000")), model=None),
}
_BREAKER_FAIL_THRESHOLD = max(1, int(os.getenv("AI_GATEWAY_BREAKER_FAIL_THRESHOLD", "4")))
_BREAKER_COOLDOWN_SECONDS = max(3, int(os.getenv("AI_GATEWAY_BREAKER_COOLDOWN_SECONDS", "20")))
_breaker_lock = Lock()
_breaker_state: Dict[str, Dict[str, float]] = {}


def _policy(usecase: str | None) -> GatewayPolicy:
    key = str(usecase or "general").strip().lower()
    return _POLICIES.get(key, _DEFAULT_POLICY)


def _breaker_key(usecase: str | None, op: str) -> str:
    return f"{str(usecase or 'general').strip().lower()}:{op}"


def _breaker_allows(key: str) -> bool:
    now = time.monotonic()
    with _breaker_lock:
        row = _breaker_state.get(key) or {}
        opened_until = float(row.get("opened_until") or 0.0)
        return now >= opened_until


def _breaker_mark_failure(key: str) -> None:
    now = time.monotonic()
    with _breaker_lock:
        row = _breaker_state.setdefault(key, {"failures": 0.0, "opened_until": 0.0})
        row["failures"] = float(row.get("failures") or 0.0) + 1.0
        if row["failures"] >= _BREAKER_FAIL_THRESHOLD:
            row["opened_until"] = now + _BREAKER_COOLDOWN_SECONDS


def _breaker_mark_success(key: str) -> None:
    with _breaker_lock:
        _breaker_state[key] = {"failures": 0.0, "opened_until": 0.0}


def _trace(event: str, *, request_id: str, usecase: str, op: str, details: Dict[str, Any] | None = None) -> None:
    payload = {
        "event": event,
        "request_id": request_id,
        "usecase": usecase,
        "op": op,
    }
    if details:
        payload.update(details)
    logger.info("ai_gateway %s", payload)


def log_control_event(
    event: str,
    *,
    request_id: str = "",
    usecase: str = "general",
    details: Dict[str, Any] | None = None,
) -> None:
    _trace(event, request_id=str(request_id or ""), usecase=str(usecase or "general"), op="control_plane", details=details)


def generate_text(
    prompt: str,
    *,
    options: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
    model: str | None = None,
    timeout_seconds: int | None = None,
    usecase: str | None = None,
    request_id: str | None = None,
) -> str:
    rid = str(request_id or get_request_id() or "")
    case = str(usecase or (signals or {}).get("context_mode") or "general")
    p = _policy(case)
    op_key = _breaker_key(case, "generate_text")
    if not _breaker_allows(op_key):
        _trace("breaker_open", request_id=rid, usecase=case, op="generate_text")
        return "none"
    started = time.perf_counter()
    try:
        result = llm_service.generate_text(
            prompt=prompt,
            options=options,
            user_profile=user_profile,
            signals=signals,
            model=model or p.model,
            timeout_seconds=timeout_seconds or p.timeout_seconds,
        )
        _breaker_mark_success(op_key)
        _trace(
            "success",
            request_id=rid,
            usecase=case,
            op="generate_text",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        return result
    except Exception:
        _breaker_mark_failure(op_key)
        _trace(
            "error",
            request_id=rid,
            usecase=case,
            op="generate_text",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        raise


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    system_instruction: str = "",
    model: str | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
    usecase: str | None = None,
    request_id: str | None = None,
) -> str:
    rid = str(request_id or get_request_id() or "")
    case = str(usecase or (signals or {}).get("context_mode") or "general")
    p = _policy(case)
    op_key = _breaker_key(case, "chat_completion")
    if not _breaker_allows(op_key):
        _trace("breaker_open", request_id=rid, usecase=case, op="chat_completion")
        return "I'm temporarily overloaded. Please try again in a moment."
    started = time.perf_counter()
    try:
        result = llm_service.chat_completion(
            messages=messages,
            system_instruction=system_instruction,
            model=model or p.model or llm_service.DEFAULT_MODEL,
            user_profile=user_profile,
            signals=signals,
            timeout_seconds=timeout_seconds or p.timeout_seconds,
        )
        _breaker_mark_success(op_key)
        _trace(
            "success",
            request_id=rid,
            usecase=case,
            op="chat_completion",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        return result
    except Exception:
        _breaker_mark_failure(op_key)
        _trace(
            "error",
            request_id=rid,
            usecase=case,
            op="chat_completion",
            details={"latency_ms": int((time.perf_counter() - started) * 1000)},
        )
        raise


def extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty response")

    clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        return json.loads(clean)
    except Exception:
        pass

    obj_start = clean.find("{")
    obj_end = clean.rfind("}")
    arr_start = clean.find("[")
    arr_end = clean.rfind("]")

    candidates: List[str] = []
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(clean[obj_start : obj_end + 1])
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(clean[arr_start : arr_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("no valid JSON found in model response")


def parse_json_object(text: str) -> Dict[str, Any]:
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def parse_json_array(text: str) -> List[Any]:
    parsed = extract_json(text)
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON array")
    return parsed


def generate_json_object(
    prompt: str,
    *,
    options: Dict[str, Any] | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return parse_json_object(
        generate_text(
            prompt=prompt,
            options=options,
            user_profile=user_profile,
            signals=signals,
        )
    )


def chat_json_object(
    messages: List[Dict[str, Any]],
    *,
    system_instruction: str = "",
    model: str | None = None,
    user_profile: Dict[str, Any] | None = None,
    signals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return parse_json_object(
        chat_completion(
            messages=messages,
            system_instruction=system_instruction,
            model=model,
            user_profile=user_profile,
            signals=signals,
        )
    )


def _vision_model_candidates() -> List[str]:
    preferred = str(os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision:latest") or "").strip()
    fallback_raw = str(
        os.getenv(
            "OLLAMA_VISION_MODEL_FALLBACKS",
            "llama3.2-vision:latest,llama3.2-vision",
        )
        or ""
    ).strip()
    ordered: List[str] = []
    for model in [preferred, *[m.strip() for m in fallback_raw.split(",")]]:
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def _ollama_generate_url() -> str:
    # Vision Ollama endpoint. Falls back to legacy OLLAMA_URL.
    base = str(
        os.getenv(
            "OLLAMA_VISION_URL",
            os.getenv("OLLAMA_URL", "http://localhost:11434/api"),
        )
        or ""
    ).strip().rstrip("/")
    if not base:
        raise RuntimeError("OLLAMA_VISION_URL is not configured.")
    allow_local = str(os.getenv("OLLAMA_VISION_ALLOW_LOCAL", "false")).strip().lower() in {"1", "true", "yes", "on"}
    lowered = base.lower()
    if not allow_local and (
        "localhost" in lowered
        or "127.0.0.1" in lowered
        or lowered.startswith("http://0.0.0.0")
    ):
        raise RuntimeError(
            "OLLAMA_VISION_URL is pointing to localhost. Set OLLAMA_VISION_URL to your Runpod proxy URL."
        )
    return f"{base}/generate" if base.endswith("/api") else f"{base}/api/generate"


def ollama_vision_json(
    *,
    prompt: str,
    image_base64: str,
    timeout_seconds: int | None = None,
    request_id: str | None = None,
    usecase: str | None = "vision",
) -> Tuple[Dict[str, Any], str]:
    rid = str(request_id or get_request_id() or "")
    case = str(usecase or "vision")
    p = _policy(case)
    timeout = int(timeout_seconds or p.timeout_seconds or int(os.getenv("OLLAMA_VISION_TIMEOUT_SECONDS", "1000")))
    vision_num_ctx = max(256, int(os.getenv("OLLAMA_VISION_NUM_CTX", "512")))
    payload = {
        "prompt": prompt,
        "images": [str(image_base64 or "").strip()],
        "stream": False,
        "format": "json",
        "keep_alive": "0s",
        "options": {
            "num_ctx": vision_num_ctx,
            "num_predict": 512,
        },
    }

    last_error: Exception | None = None
    started = time.perf_counter()
    for model in _vision_model_candidates():
        try:
            response = requests.post(
                _ollama_generate_url(),
                json={**payload, "model": model},
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Ollama vision request failed model={model} status={response.status_code}"
                )
            raw = response.json().get("response", "{}")
            parsed = parse_json_object(raw)
            _trace(
                "success",
                request_id=rid,
                usecase=case,
                op="vision_json",
                details={
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "model_used": model,
                },
            )
            return parsed, model
        except Exception as exc:
            last_error = exc
            continue

    _trace(
        "error",
        request_id=rid,
        usecase=case,
        op="vision_json",
        details={"latency_ms": int((time.perf_counter() - started) * 1000)},
    )
    raise RuntimeError(str(last_error or "vision generation failed"))
