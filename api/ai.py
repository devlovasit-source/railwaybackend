import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Request

from brain.orchestrator import ahvi_orchestrator
from brain.response_validator import validate_orchestrator_response
from services import ai_gateway

router = APIRouter(prefix="/ai")
logger = logging.getLogger("ahvi.api.ai")

_DEFAULT_MODEL = str(os.getenv("OLLAMA_MODEL", "llama3.1:8b") or "").strip() or "llama3.1:8b"
_CHEAP_MODEL = str(os.getenv("OLLAMA_MODEL_CHEAP", _DEFAULT_MODEL) or "").strip() or _DEFAULT_MODEL
_BALANCED_MODEL = str(os.getenv("OLLAMA_MODEL_BALANCED", _DEFAULT_MODEL) or "").strip() or _DEFAULT_MODEL
_PREMIUM_MODEL = str(os.getenv("OLLAMA_MODEL_PREMIUM", _DEFAULT_MODEL) or "").strip() or _DEFAULT_MODEL


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _budget_tier(message: str, context: Dict[str, Any]) -> str:
    explicit = _norm_text((context or {}).get("budget_tier")).lower()
    if explicit in {"low", "medium", "high"}:
        return explicit
    length = len(message or "")
    if length <= 120:
        return "low"
    if length <= 700:
        return "medium"
    return "high"


def _dedupe_models(models: List[str]) -> List[str]:
    out: List[str] = []
    for m in models:
        name = _norm_text(m)
        if name and name not in out:
            out.append(name)
    return out


def _model_plan(tier: str) -> tuple[str, List[str]]:
    if tier == "low":
        primary = _CHEAP_MODEL
        fallbacks = [_BALANCED_MODEL, _DEFAULT_MODEL, _PREMIUM_MODEL]
    elif tier == "high":
        primary = _PREMIUM_MODEL
        fallbacks = [_BALANCED_MODEL, _DEFAULT_MODEL, _CHEAP_MODEL]
    else:
        primary = _BALANCED_MODEL
        fallbacks = [_DEFAULT_MODEL, _CHEAP_MODEL, _PREMIUM_MODEL]
    chain = _dedupe_models([primary, *fallbacks])
    return chain[0], chain[1:]


def _infer_usecase(context: Dict[str, Any]) -> str:
    module = _norm_text(context.get("module_context")).lower()
    if any(k in module for k in ["organize", "meal", "calendar", "bill", "workout", "skin"]):
        return "general"
    if "try" in module or "wardrobe" in module:
        return "styling"
    return "general"


def _runtime_policy(message: str, context: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    tier = _budget_tier(message, context)
    primary, fallbacks = _model_plan(tier)
    return {
        "request_id": request_id,
        "budget_tier": tier,
        "primary_model": primary,
        "fallback_models": fallbacks,
        "usecase": _infer_usecase(context),
        "max_input_chars": 4000 if tier == "low" else 8000 if tier == "medium" else 12000,
    }


def _deterministic_fallback(request_id: str, errors: List[str]) -> Dict[str, Any]:
    return {
        "success": False,
        "request_id": request_id,
        "error": {
            "code": "AI_ROUTE_FALLBACK",
            "message": "AI is temporarily unavailable. Please retry in a moment.",
            "details": {"attempt_errors": errors[-3:]},
        },
        "message": "I hit a temporary issue while generating your result. Please try again.",
        "board": "general",
        "type": "text",
        "cards": [],
        "data": {},
    }


@router.post("/run")
def run_ai(payload: dict, request: Request):
    raw_context = payload.get("context", {})
    context = dict(raw_context) if isinstance(raw_context, dict) else {}
    request_id = _norm_text(getattr(request.state, "request_id", "")) or "ai-route"
    message = _norm_text(payload.get("message"))
    user_id = _norm_text(payload.get("userId")) or None

    policy = _runtime_policy(message, context, request_id)
    max_chars = int(policy.get("max_input_chars", 8000))
    if len(message) > max_chars:
        message = message[:max_chars]
    context["request_id"] = request_id
    context["ai_runtime"] = dict(policy)

    ai_gateway.log_control_event(
        "route_selected",
        request_id=request_id,
        usecase=policy.get("usecase", "general"),
        details={"budget_tier": policy["budget_tier"], "primary_model": policy["primary_model"]},
    )
    logger.info("ai.run selected route request_id=%s policy=%s", request_id, policy)

    errors: List[str] = []
    chain = [policy["primary_model"], *policy.get("fallback_models", [])]
    for index, model_name in enumerate(chain):
        local_context = dict(context)
        local_policy = dict(policy)
        local_policy["primary_model"] = model_name
        local_policy["fallback_depth"] = index
        local_context["ai_runtime"] = local_policy
        try:
            result = ahvi_orchestrator.run(
                text=message,
                user_id=user_id,
                context=local_context,
            )
            if isinstance(result, dict):
                result = validate_orchestrator_response(result, request_id=request_id)
            if isinstance(result, dict) and result.get("success", True):
                ai_gateway.log_control_event(
                    "tier_success",
                    request_id=request_id,
                    usecase=policy.get("usecase", "general"),
                    details={"model": model_name, "depth": index},
                )
                return result
            errors.append(f"tier={index} model={model_name} soft_fail")
            ai_gateway.log_control_event(
                "tier_soft_fail",
                request_id=request_id,
                usecase=policy.get("usecase", "general"),
                details={"model": model_name, "depth": index},
            )
        except Exception as exc:
            errors.append(f"tier={index} model={model_name} error={exc}")
            ai_gateway.log_control_event(
                "tier_exception",
                request_id=request_id,
                usecase=policy.get("usecase", "general"),
                details={"model": model_name, "depth": index, "error": str(exc)},
            )
            continue

    return _deterministic_fallback(request_id=request_id, errors=errors)
