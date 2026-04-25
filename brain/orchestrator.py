# backend/brain/orchestrator.py

import traceback
import uuid
import hashlib
import logging
import json
import os
from datetime import datetime
from time import perf_counter
from typing import Any, Dict
from zoneinfo import ZoneInfo

from brain.agent_system import agent_system
from brain.context.context_engine import context_engine
from brain.execution_engine import execution_engine
from brain.outfit_pipeline import get_daily_outfits
from brain.plan_pack_flow import build_plan_pack_response
from brain.personalization.style_dna_engine import style_dna_engine
from brain.intent_engine import detect_intent
from brain.daily_dependency_engine import build_daily_dependency_response
from brain.response_validator import to_plain_text, validate_orchestrator_response

from services.appwrite_proxy import AppwriteProxy
from services.ai_gateway import generate_text
from services.settings import settings

logger = logging.getLogger("ahvi.orchestrator")

def _hash_outfit(outfit):
    try:
        canonical = json.dumps(outfit, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        canonical = str(outfit)
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _safe_log(text: str) -> None:
    try:
        print(text)
    except Exception:
        try:
            safe_text = str(text).encode("ascii", errors="ignore").decode("ascii")
            print(safe_text)
        except Exception:
            pass


class AhviOrchestrator:
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_ttl_seconds: float = 30.0

    def run(self, text: str, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        user_id = user_id or "anonymous"
        request_id = str(context.get("request_id") or uuid.uuid4())
        started_at = perf_counter()
        logger.info("orchestrator.start request_id=%s user_id=%s", request_id, user_id)

        appwrite = AppwriteProxy()
        stage = "init"

        try:
            stage = "early_organize"
            early_organize = self._resolve_organize_request(text=text, context=context, slots={}, intent="general")
            if early_organize.get("active") and ("organize" in str(text or "").lower() or "organize" in str(context.get("module_context") or "").lower()):
                return self._finalize_response(
                    self._organize_response(
                        request_id=request_id,
                        user_id=user_id,
                        appwrite=appwrite,
                        module_key=early_organize.get("module"),
                        include_counts=bool(context.get("include_counts", False)),
                    ),
                    request_id=request_id,
                )

            stage = "plan_pack_precheck"
            if self._is_plan_pack_request(text=text, context=context):
                pp = build_plan_pack_response(text=text, context=context, user_id=user_id)
                return self._finalize_response({
                    "success": True,
                    "request_id": request_id,
                    "meta": {"intent": "plan_pack", "domain": "planning"},
                    "message": pp.get("message"),
                    "board": pp.get("board", "plan_pack"),
                    "type": pp.get("type", "checklists"),
                    "cards": pp.get("cards", []),
                    "chips": pp.get("chips", []),
                    "pack_ids": pp.get("pack_ids"),
                    "data": pp.get("data", {}),
                }, request_id=request_id)

            stage = "daily_dependency_precheck"
            if self._is_daily_dependency_request(text=text, context=context):
                return self._finalize_response({
                    "success": True,
                    "request_id": request_id,
                    "meta": {"intent": "daily_dependency", "domain": "planning"},
                    **build_daily_dependency_response(
                        user_id=user_id,
                        context=context,
                        appwrite=appwrite,
                    ),
                }, request_id=request_id)

            stage = "intent_detection"
            intent_data = detect_intent(
                text,
                context.get("history"),
                model=self._runtime_model(context, "intent"),
            )
            intent = intent_data.get("intent", "general")

            stage = "daily_dependency_intent"
            if intent == "daily_dependency":
                return self._finalize_response({
                    "success": True,
                    "request_id": request_id,
                    **build_daily_dependency_response(
                        user_id=user_id,
                        context={**context, "time_slot": (intent_data.get("slots", {}) or {}).get("time")},
                        appwrite=appwrite,
                    ),
                }, request_id=request_id)

            stage = "slot_extraction"
            fallback_slots = self._extract_slots(text=text, context=context)
            llm_slots = intent_data.get("slots", {}) or {}
            slots = {**fallback_slots, **llm_slots}
            intent_data["slots"] = slots

            signals = {
                "context_mode": "styling",
                "emotion_state": self._infer_emotion_state(text),
            }

            stage = "try_on_response"
            if intent == "try_on":
                return self._finalize_response(
                    self._try_on_response(
                        request_id=request_id,
                        user_id=user_id,
                        appwrite=appwrite,
                        context=context,
                    ),
                    request_id=request_id,
                )

            stage = "wardrobe_query_response"
            if intent == "wardrobe_query" or self._is_wardrobe_count_query(text):
                return self._finalize_response(
                    self._wardrobe_query_response(
                        request_id=request_id,
                        user_id=user_id,
                        appwrite=appwrite,
                        text=text,
                    ),
                    request_id=request_id,
                )

            stage = "plan_pack_intent"
            if intent == "plan_pack" or self._is_plan_pack_request(text=text, context=context):
                pp = build_plan_pack_response(text=text, context=context, user_id=user_id)
                return self._finalize_response({
                    "success": True,
                    "request_id": request_id,
                    "meta": {"intent": "plan_pack", "domain": "planning"},
                    "message": pp.get("message"),
                    "board": pp.get("board", "plan_pack"),
                    "type": pp.get("type", "checklists"),
                    "cards": pp.get("cards", []),
                    "chips": pp.get("chips", []),
                    "pack_ids": pp.get("pack_ids"),
                    "data": pp.get("data", {}),
                }, request_id=request_id)

            stage = "organize_resolution"
            organize_signal = self._resolve_organize_request(text=text, context=context, slots=slots, intent=intent)
            if organize_signal.get("active"):
                return self._finalize_response(
                    self._organize_response(
                        request_id=request_id,
                        user_id=user_id,
                        appwrite=appwrite,
                        module_key=organize_signal.get("module"),
                        include_counts=bool(context.get("include_counts", False)),
                    ),
                    request_id=request_id,
                )

            stage = "styling_response"
            if intent in ("daily_outfit", "occasion_outfit", "explore_styles"):
                return self._finalize_response(
                    self._styling_response(
                        request_id=request_id,
                        user_id=user_id,
                        text=text,
                        context=context,
                        intent=intent,
                        intent_data=intent_data,
                        appwrite=appwrite,
                        signals=signals,
                        started_at=started_at,
                    ),
                    request_id=request_id,
                )

            stage = "general_generation"
            general_message = generate_text(
                text,
                user_profile=context.get("user_profile"),
                signals={"context_mode": "general"},
                usecase="general",
                model=self._runtime_model(context, "general"),
            )
            if not general_message or general_message == "none":
                general_message = "Tell me your occasion, weather, and vibe, and I will style you with your wardrobe."

            return self._finalize_response({
                "success": True,
                "request_id": request_id,
                "meta": {"intent": intent, "domain": "general"},
                "message": general_message,
                "board": "general",
                "type": "text",
                "cards": [],
                "data": {"intent": intent},
            }, request_id=request_id)

        except Exception:
            logger.exception("orchestrator.error request_id=%s stage=%s", request_id, stage)
            _safe_log("ERROR: Orchestrator error\n" + traceback.format_exc())
            return self._finalize_response({
                "success": False,
                "request_id": request_id,
                "error": {
                    "code": "ORCHESTRATOR_ERROR",
                    "message": f"Something went wrong at stage: {stage}",
                    "details": {
                        "request_id": request_id,
                        "stage": stage,
                    },
                },
            }, request_id=request_id)

    def _finalize_response(self, payload: Dict[str, Any], *, request_id: str) -> Dict[str, Any]:
        return validate_orchestrator_response(payload, request_id=request_id)

    # -------------------------
    # HELPERS
    # -------------------------
    def _styling_response(
        self,
        *,
        request_id: str,
        user_id: str,
        text: str,
        context: Dict[str, Any],
        intent: str,
        intent_data: Dict[str, Any],
        appwrite: AppwriteProxy,
        signals: Dict[str, Any],
        started_at: float,
    ) -> Dict[str, Any]:
        try:
            # FIX: Safely normalize the wardrobe immediately so it is an exact list of dicts
            wardrobe_docs = self._normalize_documents(appwrite.list_documents("outfits", user_id=user_id))
            context["wardrobe"] = wardrobe_docs
        except Exception as e:
            logger.warning("orchestrator.wardrobe_fetch_failed request_id=%s error=%s", request_id, e)
            context["wardrobe"] = []

        enriched_context = context_engine.build_context(
            user_id=user_id,
            intent_data=intent_data,
            wardrobe=context.get("wardrobe", []),
            user_profile=context.get("user_profile"),
            history=context.get("history", []),
            vision=context.get("vision"),
        )

        cache_key = self._cache_key(
            text=text,
            user_id=user_id,
            context={**context, **enriched_context},
        )
        cached = self._get_cache(cache_key)
        if cached is not None:
            cached_data = dict(cached)
            cached_data["meta"] = {**cached_data.get("meta", {}), "cache_hit": True}
            return cached_data

        style_dna = self._build_style_dna({**context, **enriched_context, "user_id": user_id})
        exec_result, pipeline_result = self._execute_styling_pipeline(
            intent=intent,
            user_id=user_id,
            text=text,
            context=context,
            enriched_context=enriched_context,
            style_dna=style_dna,
        )

        outfits = pipeline_result.get("outfits", [])
        boards = pipeline_result.get("boards", [])
        cards_from_pipeline = pipeline_result.get("cards", [])

        # --- THE FOOLPROOF STYLING FALLBACK ---
        # If the AI rejected the items and returned 0 outfits, manually build one!
        if not outfits and context.get("wardrobe"):
            safe_docs = self._normalize_documents(context.get("wardrobe"))
            
            def find_item(keywords):
                for doc in safe_docs:
                    cat = str(doc.get("category") or doc.get("type") or "").lower()
                    sub = str(doc.get("sub_category") or "").lower()
                    name = str(doc.get("name") or "").lower()
                    blob = f"{cat} {sub} {name}"
                    if any(k in blob for k in keywords):
                        return doc
                return {}

            f_top = find_item(["top", "shirt", "blouse", "tee", "jacket", "hoodie"])
            f_bottom = find_item(["bottom", "pant", "trouser", "jean", "skirt", "short"])
            f_shoe = find_item(["shoe", "sneaker", "boot", "heel", "sandal", "footwear"])

            if f_top and f_bottom:
                fallback_outfit = {
                    "combo_id": f"fallback_{f_top.get('$id')}_{f_bottom.get('$id')}",
                    "master_type": "top",
                    "top": f_top,
                    "bottom": f_bottom,
                    "shoes": f_shoe,
                    "score": 8.5,
                    "rank_score": 8.5,
                }
                outfits = [fallback_outfit]
        # --------------------------------------

        fallback_depth = self._fallback_depth(context)
        if fallback_depth > 0:
            logger.info(
                "orchestrator.skip_persist_on_fallback request_id=%s depth=%s outfits=%s",
                request_id,
                fallback_depth,
                len(outfits),
            )
            saved_outfit_ids = []
        else:
            saved_outfit_ids = self._persist_outfits(
                appwrite=appwrite,
                user_id=user_id,
                outfits=outfits,
                request_id=request_id,
            )

        cards = cards_from_pipeline if cards_from_pipeline else boards if boards else [
            {
                "title": f"Outfit {i+1}",
                "outfit_id": oid,
                "preview": outfits[i] if i < len(outfits) else {},
            }
            for i, oid in enumerate(saved_outfit_ids[:3])
        ]
        if not cards and outfits:
            cards = [
                {
                    "title": f"Outfit {i+1}",
                    "outfit_id": f"preview-{_hash_outfit(outfit)[:10]}",
                    "preview": outfit,
                }
                for i, outfit in enumerate(outfits[:3])
            ]

        if outfits:
            message = self._build_stylist_message(
                text=text,
                cards=cards,
                context={**context, **enriched_context},
                style_dna=style_dna,
                user_profile=context.get("user_profile"),
                signals=signals,
            )
        else:
            questions = pipeline_result.get("clarifying_questions", []) if isinstance(pipeline_result, dict) else []
            if questions:
                message = "I need one quick detail before styling: " + str(questions[0])
                cards = [{"title": "Clarify Occasion", "kind": "question", "question": str(q)} for q in questions[:3]]
            else:
                message = "I need more wardrobe items to generate outfits."

        response = {
            "success": True,
            "request_id": request_id,
            "meta": {
                "intent": intent,
                "domain": "styling",
                "cache_hit": False,
                "latency_ms": round((perf_counter() - started_at) * 1000.0, 2),
            },
            "message": message,
            "board": "outfit",
            "type": "boards" if boards else "cards",
            "cards": cards,
            "data": {
                "outfits_count": len(outfits),
                "execution": exec_result,
                "boards_available": bool(boards),
                "tryon_enabled": True,
                "pipeline": pipeline_result.get("pipeline", {}),
                "premium": pipeline_result.get("premium", {}),
            },
        }
        self._set_cache(cache_key, response)
        return response

    def _execute_styling_pipeline(
        self,
        *,
        intent: str,
        user_id: str,
        text: str,
        context: Dict[str, Any],
        enriched_context: Dict[str, Any],
        style_dna: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        plan = agent_system.plan(intent=intent, context=enriched_context)
        state: Dict[str, Any] = {"pipeline_result": {}}

        def _normalize_context():
            return {
                "slots": enriched_context.get("slots", {}),
                "meta": enriched_context.get("meta", {}),
            }

        def _build_style_graph():
            return {"status": "prepared"}

        def _generate_score_rank():
            slots_ctx = enriched_context.get("slots", {}) or {}
            
            # FIX: Ensure we use the raw context wardrobe, since enriched_context was stripping it out!
            raw_wardrobe = context.get("wardrobe") or enriched_context.get("wardrobe", [])
            safe_wardrobe = self._select_wardrobe_for_styling(raw_wardrobe, text)

            state["pipeline_result"] = get_daily_outfits(
                {
                    "user_id": user_id,
                    "wardrobe": safe_wardrobe,
                    "context": {
                        "query": text,
                        "weather": slots_ctx.get("weather"),
                        "occasion": slots_ctx.get("occasion"),
                        "location": slots_ctx.get("location"),
                        "time_of_day": slots_ctx.get("time"),
                        "history": enriched_context.get("history", []),
                        "recent_outfits": context.get("memory", {}).get("recent_outfits", []),
                        "style_dna": style_dna,
                    },
                }
            )
            return {"outfits_count": len(state["pipeline_result"].get("outfits", []))}

        exec_result = execution_engine.execute(
            plan=plan,
            handlers={
                "normalize_context": _normalize_context,
                "build_style_graph": _build_style_graph,
                "generate_score_rank": _generate_score_rank,
                "persist_and_feedback_hooks": lambda: {"feedback_endpoint": "/api/feedback/outfit"},
                "no_op": lambda: {"status": "noop"},
            },
            timeout_seconds=3.0,
            slow_step_threshold_seconds=1.5,
        )
        return exec_result, state.get("pipeline_result", {})

    def _persist_outfits(self, *, appwrite: AppwriteProxy, user_id: str, outfits: list, request_id: str = "") -> list[str]:
        saved_outfit_ids: list[str] = []
        try:
            existing_outfits = self._normalize_documents(appwrite.list_documents("outfits", user_id=user_id))
            existing_hashes = {_hash_outfit(o) for o in existing_outfits}
        except Exception as e:
            logger.warning("orchestrator.outfits_load_failed request_id=%s error=%s", request_id, e)
            existing_hashes = set()

        def _pick_first_text(node: Any, keys: list[str]) -> str:
            if not isinstance(node, dict):
                return ""
            for key in keys:
                value = str(node.get(key) or "").strip()
                if value:
                    return value
            return ""

        def _derive_outfit_image_url(outfit_obj: dict) -> str:
            keys = [
                "image_url",
                "imageUrl",
                "raw_image_url",
                "rawImageUrl",
                "masked_url",
                "maskedUrl",
                "masked_image_url",
                "maskedImageUrl",
            ]
            direct = _pick_first_text(outfit_obj, keys)
            if direct:
                return direct
            for part_key in ("top", "bottom", "shoes", "outerwear", "dress", "bag", "accessory"):
                value = outfit_obj.get(part_key)
                if isinstance(value, dict):
                    part_url = _pick_first_text(value, keys)
                    if part_url:
                        return part_url
            return ""

        for outfit in outfits:
            try:
                outfit_hash = _hash_outfit(outfit)
                if outfit_hash in existing_hashes:
                    continue
                image_url = _derive_outfit_image_url(outfit if isinstance(outfit, dict) else {})
                if not image_url:
                    logger.warning(
                        "orchestrator.outfit_save_skipped_missing_image request_id=%s user_id=%s",
                        request_id,
                        user_id,
                    )
                    continue
                safe_doc = {
                    **(outfit if isinstance(outfit, dict) else {}),
                    "userId": user_id,
                    "hash": outfit_hash,
                    "name": str((outfit or {}).get("name") or "AI Outfit").strip(),
                    "category": str((outfit or {}).get("category") or "Tops").strip(),
                    "sub_category": str((outfit or {}).get("sub_category") or "Outfit").strip(),
                    "color_code": str((outfit or {}).get("color_code") or "#000000").strip(),
                    "pattern": str((outfit or {}).get("pattern") or "plain").strip().lower(),
                    "occasions": (outfit or {}).get("occasions") if isinstance((outfit or {}).get("occasions"), list) else ["casual"],
                    "image_url": image_url,
                    "masked_url": str((outfit or {}).get("masked_url") or image_url).strip(),
                    "status": str((outfit or {}).get("status") or "active").strip(),
                    "worn": int((outfit or {}).get("worn") or 0),
                    "liked": bool((outfit or {}).get("liked") or False),
                }
                doc = appwrite.create_document(
                    "outfits",
                    safe_doc,
                )
                saved_outfit_ids.append(doc.get("$id"))
                existing_hashes.add(outfit_hash)
            except Exception as e:
                logger.warning("orchestrator.outfit_save_failed request_id=%s error=%s", request_id, e)
        return saved_outfit_ids

    def _extract_slots(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        lowered = (text or "").lower()
        slots = dict((context.get("slots") or {}))

        if not slots.get("occasion"):
            if "office" in lowered:
                slots["occasion"] = "office"
            elif "party" in lowered:
                slots["occasion"] = "party"
            elif "casual" in lowered:
                slots["occasion"] = "casual"

        if not slots.get("weather"):
            if "warm" in lowered or "hot" in lowered:
                slots["weather"] = "warm"
            elif "cold" in lowered:
                slots["weather"] = "cold"
            elif "rain" in lowered:
                slots["weather"] = "rainy"

        return slots

    def _build_style_dna(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("style_dna"):
            value = context.get("style_dna")
            return value if isinstance(value, dict) else {}
        if hasattr(style_dna_engine, "build"):
            wardrobe_for_dna = self._select_wardrobe_for_styling(
                context.get("wardrobe", []),
                str((context.get("slots") or {}).get("occasion") or ""),
            )
            return style_dna_engine.build(
                {
                    "user_id": context.get("user_id"),
                    "user_profile": context.get("user_profile"),
                    "history": context.get("history", []),
                    "signals": context.get("signals"),
                    "wardrobe": wardrobe_for_dna,
                }
            )
        return {}

    def _styling_wardrobe_limit(self) -> int:
        raw = str(os.getenv("STYLING_WARDROBE_MAX_ITEMS", "80") or "").strip()
        try:
            value = int(raw)
        except Exception:
            value = 80
        return max(20, min(200, value))

    def _select_wardrobe_for_styling(self, wardrobe: Any, query_text: str = "") -> list[dict]:
        docs = self._normalize_documents(wardrobe)
        cap = self._styling_wardrobe_limit()
        if len(docs) <= cap:
            return docs

        words = [w.strip().lower() for w in str(query_text or "").split() if len(w.strip()) >= 3]

        def _score(doc: dict) -> float:
            cat = str(doc.get("category") or doc.get("type") or "").lower()
            sub = str(doc.get("sub_category") or "").lower()
            name = str(doc.get("name") or "").lower()
            pattern = str(doc.get("pattern") or "").lower()
            occasions = " ".join(
                str(x).lower() for x in (doc.get("occasions") or []) if isinstance(x, (str, int, float))
            )
            blob = f"{cat} {sub} {name} {pattern} {occasions}"
            score = 0.0
            for w in words:
                if w in blob:
                    score += 3.0
            if doc.get("liked") is True:
                score += 0.4
            try:
                score -= min(5.0, float(doc.get("worn", 0) or 0) * 0.05)
            except Exception:
                pass
            return score

        ranked = sorted(docs, key=_score, reverse=True)
        return ranked[:cap]

    def _try_on_response(
        self,
        request_id: str,
        user_id: str,
        appwrite: AppwriteProxy,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            wardrobe_docs = self._normalize_documents(
                appwrite.list_documents("outfits", user_id=user_id, limit=100)
            )
        except Exception:
            wardrobe_docs = []

        pipeline = get_daily_outfits(
            {
                "user_id": user_id,
                "wardrobe": wardrobe_docs,
                "context": context or {},
            }
        )
        cards = pipeline.get("cards", []) if isinstance(pipeline, dict) else []
        top_card = {}
        tryon_payload = None

        for card in cards:
            if not isinstance(card, dict):
                continue
            payload = card.get("tryon_payload")
            items = (payload or {}).get("items", {}) if isinstance(payload, dict) else {}
            
            top_id = str(items.get("top_id") or "").strip()
            bottom_id = str(items.get("bottom_id") or "").strip()
            
            if top_id and bottom_id:
                top_card = card
                tryon_payload = payload
                break

        # --- THE FOOLPROOF TRY-ON FALLBACK ---
        if not tryon_payload:
            def find_item_by_keywords(docs, keywords):
                for doc in docs:
                    cat = str(doc.get("category") or doc.get("type") or "").lower()
                    sub = str(doc.get("sub_category") or "").lower()
                    name = str(doc.get("name") or "").lower()
                    blob = f"{cat} {sub} {name}"
                    if any(k in blob for k in keywords):
                        return doc.get("$id") or doc.get("id")
                return None

            fallback_top = find_item_by_keywords(wardrobe_docs, ["top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "jacket", "blazer"])
            fallback_bottom = find_item_by_keywords(wardrobe_docs, ["bottom", "pant", "trouser", "jean", "short", "skirt"])
            fallback_shoes = find_item_by_keywords(wardrobe_docs, ["shoe", "sneaker", "heel", "boot", "sandal", "footwear"])

            if fallback_top and fallback_bottom:
                tryon_payload = {
                    "items": {
                        "top_id": fallback_top,
                        "bottom_id": fallback_bottom,
                        "shoes_id": fallback_shoes or ""
                    }
                }
                cards = [{"title": "Try-On Ready", "tryon_payload": tryon_payload}]
        # -------------------------------------

        if not tryon_payload:
            return {
                "success": True,
                "request_id": request_id,
                "meta": {"intent": "try_on", "domain": "styling"},
                "message": "I need at least one Top and one Bottom in your wardrobe before try-on.",
                "board": "tryon",
                "type": "tryon",
                "cards": [],
                "data": {"action": "open_tryon", "tryon_payload": None},
            }

        allowed, remaining, used, limit = self._consume_tryon_quota(
            appwrite=appwrite,
            user_id=user_id,
            request_id=request_id,
            context=context,
        )
        if not allowed:
            return {
                "success": False,
                "request_id": request_id,
                "meta": {
                    "intent": "try_on",
                    "domain": "styling",
                    "try_on_daily_limit": limit,
                    "try_on_used_today": used,
                    "try_on_remaining": remaining,
                },
                "message": "Daily try-on limit reached. You can use try-on again tomorrow.",
                "board": "tryon",
                "type": "tryon",
                "cards": [],
                "data": {"action": "open_tryon", "tryon_payload": None},
            }

        return {
            "success": True,
            "request_id": request_id,
            "meta": {
                "intent": "try_on",
                "domain": "styling",
                "try_on_daily_limit": limit,
                "try_on_used_today": used,
                "try_on_remaining": remaining,
            },
            "message": "Try-on look prepared. Tap to launch virtual try-on.",
            "board": "tryon",
            "type": "tryon",
            "cards": cards[:3],
            "data": {
                "action": "open_tryon",
                "tryon_payload": tryon_payload,
                "board_item_ids": pipeline.get("board_item_ids", []) if isinstance(pipeline, dict) else [],
            },
        }

    def _consume_tryon_quota(
        self,
        *,
        appwrite: AppwriteProxy,
        user_id: str,
        request_id: str,
        context: Dict[str, Any],
    ) -> tuple[bool, int, int, int]:
        limit = max(1, int(getattr(settings, "try_on_daily_limit", 2) or 2))
        tz_name = self._user_timezone_name(context)
        today = self._today_in_timezone(tz_name)
        used = 0
        try:
            rows = self._normalize_documents(appwrite.list_documents("jobs", user_id=user_id, limit=300))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("type") or "").strip().lower() != "try_on":
                    continue
                created_at = str(row.get("$createdAt") or row.get("createdAt") or "")
                if created_at[:10] == today:
                    used += 1
        except Exception as exc:
            logger.warning("orchestrator.tryon_quota_read_failed request_id=%s error=%s", request_id, exc)

        if used >= limit:
            return False, 0, used, limit

        try:
            appwrite.create_document(
                "jobs",
                {
                    "userId": str(user_id),
                    "type": "try_on",
                    "status": "completed",
                    "input": f"try_on|tz={tz_name}",
                    "output": "prepared",
                    "error": "",
                    "retry_count": 0,
                    "duration_ms": 0,
                    "request_id": str(request_id),
                },
            )
        except Exception as exc:
            logger.warning("orchestrator.tryon_quota_write_failed request_id=%s error=%s", request_id, exc)

        used_after = used + 1
        return True, max(0, limit - used_after), used_after, limit

    def _user_timezone_name(self, context: Dict[str, Any]) -> str:
        profile = context.get("user_profile", {}) if isinstance(context, dict) else {}
        if not isinstance(profile, dict):
            profile = {}
        location = profile.get("location", {})
        if not isinstance(location, dict):
            location = {}
        for value in (
            profile.get("timezone"),
            profile.get("time_zone"),
            profile.get("tz"),
            location.get("timezone"),
            location.get("time_zone"),
            location.get("tz"),
        ):
            zone = str(value or "").strip()
            if zone:
                return zone
        return "UTC"

    @staticmethod
    def _today_in_timezone(tz_name: str) -> str:
        zone = str(tz_name or "").strip() or "UTC"
        try:
            return datetime.now(ZoneInfo(zone)).date().isoformat()
        except Exception:
            return datetime.now().astimezone().date().isoformat()

    def _cache_key(self, text: str, user_id: str, context: Dict[str, Any]) -> str:
        wardrobe = context.get("wardrobe", []) or []
        wardrobe_signature = "|".join(
            sorted(
                str(item.get("$id") or item.get("id") or item.get("item_id") or "")
                for item in wardrobe
                if isinstance(item, dict)
            )[:120]
        )
        slots = context.get("slots", {}) if isinstance(context.get("slots"), dict) else {}
        slot_signature = "|".join(f"{k}:{slots.get(k)}" for k in sorted(slots.keys()))
        profile = context.get("user_profile", {}) if isinstance(context.get("user_profile"), dict) else {}
        profile_colors = (profile.get("preferred_colors") or profile.get("colors") or [])
        if not isinstance(profile_colors, list):
            profile_colors = []
        profile_signature = f"{profile.get('style', '')}|{','.join([str(c) for c in profile_colors[:5]])}"
        seed = f"{user_id}|{text.strip().lower()}|{wardrobe_signature}|{slot_signature}|{profile_signature}"
        return hashlib.md5(seed.encode()).hexdigest()

    def _normalize_documents(self, payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [doc for doc in payload if isinstance(doc, dict)]
        if isinstance(payload, dict):
            docs = payload.get("documents", [])
            if isinstance(docs, list):
                return [doc for doc in docs if isinstance(doc, dict)]
        return []

    def _get_cache(self, key: str) -> Dict[str, Any] | None:
        data = self._cache.get(key)
        if not data:
            return None
        age = perf_counter() - float(data.get("_saved_at", 0.0))
        if age > self._cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        return data.get("payload")

    def _set_cache(self, key: str, payload: Dict[str, Any]) -> None:
        self._cache[key] = {"_saved_at": perf_counter(), "payload": payload}

    def _infer_emotion_state(self, text: str) -> str:
        t = str(text or "").lower()
        if any(k in t for k in ["urgent", "asap", "quick", "hurry"]):
            return "stressed"
        if any(k in t for k in ["anxious", "nervous", "confused"]):
            return "vulnerable"
        if any(k in t for k in ["excited", "date", "party", "celebrate"]):
            return "energized"
        return "neutral"

    def _runtime_model(self, context: Dict[str, Any], usecase: str) -> str | None:
        runtime = context.get("ai_runtime", {}) if isinstance(context, dict) else {}
        if not isinstance(runtime, dict):
            return None
        by_usecase = runtime.get("model_by_usecase", {})
        if isinstance(by_usecase, dict):
            selected = str(by_usecase.get(usecase) or "").strip()
            if selected:
                return selected
        selected = str(runtime.get("primary_model") or "").strip()
        return selected or None

    def _fallback_depth(self, context: Dict[str, Any]) -> int:
        runtime = context.get("ai_runtime", {}) if isinstance(context, dict) else {}
        if not isinstance(runtime, dict):
            return 0
        try:
            return max(0, int(runtime.get("fallback_depth", 0)))
        except Exception:
            return 0

    def _build_stylist_message(
        self,
        text: str,
        cards: list,
        context: Dict[str, Any],
        style_dna: Dict[str, Any],
        user_profile: Dict[str, Any],
        signals: Dict[str, Any],
    ) -> str:
        top_cards = cards[:2] if isinstance(cards, list) else []
        card_titles = [str(c.get("title", "Look")) for c in top_cards if isinstance(c, dict)]
        occasion = (context.get("slots", {}) or {}).get("occasion") or context.get("occasion") or "today"
        weather = (context.get("slots", {}) or {}).get("weather") or context.get("weather") or "current weather"
        dna_style = style_dna.get("style", "your style")

        deterministic = (
            f"I curated your looks for {occasion} in {weather}. "
            f"Top picks: {', '.join(card_titles) if card_titles else 'your best wardrobe matches'}. "
            f"They align with your style DNA ({dna_style}) and recent feedback. "
            f"Pick one and I can refine it for accessories or try-on."
        )

        try:
            model_message = generate_text(
                f"""
User request: {text}
Occasion: {occasion}
Weather: {weather}
Style DNA: {style_dna}
Top cards: {top_cards}

Write a premium stylist response in 3-4 lines:
- mention why the top look works
- mention one styling tweak option
- keep it concise and natural
""",
                user_profile=user_profile,
                signals=signals,
                usecase="styling",
                model=self._runtime_model(context, "styling"),
            )
            if model_message and model_message != "none":
                return to_plain_text(model_message, fallback=deterministic)
        except Exception:
            pass
        return to_plain_text(deterministic, fallback="I can help with styling.")

    def _resolve_organize_request(
        self,
        text: str,
        context: Dict[str, Any],
        slots: Dict[str, Any],
        intent: str,
    ) -> Dict[str, Any]:
        module_context = str(context.get("module_context") or "").lower()
        lowered = str(text or "").lower()
        module = str((slots or {}).get("module") or "").strip().lower() or None

        keyword_map = {
            "life board": "life_boards",
            "meal": "meal_planner",
            "medicine": "medicines",
            "meds": "medicines",
            "bill": "bills",
            "calendar": "calendar",
            "workout": "workout",
            "skin": "skincare",
            "contact": "contacts",
            "goal": "life_goals",
        }
        if not module:
            for key, value in keyword_map.items():
                if key in lowered or key in module_context:
                    module = value
                    break

        active = (
            intent == "organize_hub"
            or "organize" in lowered
            or "organize" in module_context
            or module is not None
        )
        return {"active": active, "module": module}

    def _build_organize_hub(self, user_id: str, appwrite: AppwriteProxy) -> Dict[str, Any]:
        module_specs = self._organize_module_specs()

        chips = []
        for module, title, resource, route in module_specs:
            count = 0
            chips.append(
                {
                    "id": module,
                    "title": title,
                    "kind": "chip",
                    "subtitle": f"{count} items",
                    "action": {
                        "type": "open_module",
                        "module": module,
                        "route": route,
                    },
                    "meta": {
                        "resource": resource,
                        "count": count,
                    },
                }
            )

        suggested_prompts = [
            "Plan my meals for this week",
            "Remind me about medicines",
            "Track this month's bills",
            "Add a life goal",
        ]

        return {"chips": chips, "suggested_prompts": suggested_prompts}

    def _organize_module_specs(self):
        return [
            ("life_boards", "Life Boards", "life_boards", "/organize/life-boards"),
            ("meal_planner", "Meal Planner", "meal_plans", "/organize/meal-planner"),
            ("medicines", "Medicines", "meds", "/organize/medicines"),
            ("bills", "Bills", "bills", "/organize/bills"),
            ("calendar", "Calendar", "plans", "/organize/calendar"),
            ("workout", "Workout", "workout_outfits", "/organize/workout"),
            ("skincare", "Skincare", "skincare_profiles", "/organize/skincare"),
            ("contacts", "Contacts", "users", "/organize/contacts"),
            ("life_goals", "Life Goals", "life_goals", "/organize/life-goals"),
        ]

    def _organize_module_spec(self, module_key: str | None):
        for module, title, resource, route in self._organize_module_specs():
            if module == (module_key or ""):
                return {
                    "module": module,
                    "title": title,
                    "resource": resource,
                    "route": route,
                }
        return {
            "module": "organize",
            "title": "Organize",
            "resource": "",
            "route": "/organize",
        }

    def _module_preview_cards(self, appwrite: AppwriteProxy, user_id: str, module_key: str | None):
        spec = self._organize_module_spec(module_key)
        resource = str(spec.get("resource") or "")
        route = str(spec.get("route") or "/organize")
        title = str(spec.get("title") or "Organize")

        if not resource:
            return {
                "cards": [],
                "count": 0,
                "route": route,
                "title": title,
                "resource": resource,
            }

        try:
            docs_raw = appwrite.list_documents(resource, user_id=user_id, limit=50)
        except Exception:
            docs_raw = []

        docs = self._normalize_documents(docs_raw)

        cards = []
        for index, doc in enumerate(docs):
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("$id") or doc.get("id") or f"{module_key}_{index}")
            doc_title = (
                str(doc.get("title") or "").strip()
                or str(doc.get("name") or "").strip()
                or str(doc.get("label") or "").strip()
                or f"{title} Item {index + 1}"
            )
            doc_subtitle = (
                str(doc.get("description") or "").strip()
                or str(doc.get("status") or "").strip()
                or str(doc.get("createdAt") or "").strip()
                or "Tap to view details"
            )
            cards.append(
                {
                    "id": doc_id,
                    "title": doc_title,
                    "kind": "item",
                    "subtitle": doc_subtitle,
                    "action": {
                        "type": "open_module",
                        "module": spec.get("module"),
                        "route": route,
                        "document_id": doc.get("$id") or doc.get("id"),
                    },
                }
            )

        return {
            "cards": cards,
            "count": len(docs),
            "route": route,
            "title": title,
            "resource": resource,
            "documents": docs,
        }

    def _organize_response(
        self,
        request_id: str,
        user_id: str,
        appwrite: AppwriteProxy,
        module_key: str | None,
        include_counts: bool,
    ) -> Dict[str, Any]:
        hub_payload = self._build_organize_hub(user_id=user_id, appwrite=appwrite)
        chips = hub_payload.get("chips", [])
        if include_counts:
            for chip in chips:
                meta = chip.get("meta", {}) if isinstance(chip.get("meta"), dict) else {}
                resource = str(meta.get("resource", ""))
                count = self._count_resource(appwrite=appwrite, resource=resource, user_id=user_id)
                meta["count"] = count
                chip["meta"] = meta
                chip["subtitle"] = f"{count} items"

        focus_card = self._build_organize_focus_card(module_key=module_key)
        message = "Choose what you want to organize."

        if module_key:
            module_preview = self._module_preview_cards(
                appwrite=appwrite,
                user_id=user_id,
                module_key=module_key,
            )
            module_title = module_preview.get("title") or focus_card.get("title", "organize module")
            route = module_preview.get("route") or focus_card.get("action", {}).get("route", "/organize")
            cards = module_preview.get("cards", [])
            count = int(module_preview.get("count") or 0)
            message = f"Showing {module_title} details."
            if count == 0:
                cards = [
                    {
                        "id": f"{module_key}_empty",
                        "title": module_title,
                        "kind": "empty",
                        "subtitle": "No items yet. Tap View More to open this board.",
                        "action": {
                            "type": "open_module",
                            "module": module_key,
                            "route": route,
                        },
                    }
                ]

            return {
                "success": True,
                "request_id": request_id,
                "meta": {
                    "intent": "organize_hub",
                    "domain": "organize",
                    "module": module_key,
                },
                "message": message,
                "board": "organize",
                "type": "cards",
                "cards": cards,
                "data": {
                    "module": module_key,
                    "resource": module_preview.get("resource"),
                    "count": count,
                    "module_documents": module_preview.get("documents", []),
                    "view_more": {
                        "type": "open_module",
                        "module": module_key,
                        "route": route,
                    },
                    "action": "open_organize_module",
                },
            }

        return {
            "success": True,
            "request_id": request_id,
            "meta": {
                "intent": "organize_hub",
                "domain": "organize",
                "module": module_key,
            },
            "message": message,
            "board": "organize",
            "type": "chips",
            "cards": chips,
            "data": {
                "hub": hub_payload,
                "module": module_key,
                "action": "open_organize_module" if module_key else "show_organize_hub",
            },
        }

    def _build_organize_focus_card(self, module_key: str | None) -> Dict[str, Any]:
        mapping = {
            "life_boards": ("Life Boards", "/organize/life-boards"),
            "meal_planner": ("Meal Planner", "/organize/meal-planner"),
            "medicines": ("Medicines", "/organize/medicines"),
            "bills": ("Bills", "/organize/bills"),
            "calendar": ("Calendar", "/organize/calendar"),
            "workout": ("Workout", "/organize/workout"),
            "skincare": ("Skincare", "/organize/skincare"),
            "contacts": ("Contacts", "/organize/contacts"),
            "life_goals": ("Life Goals", "/organize/life-goals"),
        }
        title, route = mapping.get(module_key or "", ("Organize", "/organize"))
        return {
            "id": module_key or "organize",
            "title": title,
            "kind": "chip",
            "subtitle": "Ready",
            "action": {
                "type": "open_module",
                "module": module_key or "organize",
                "route": route,
            },
        }

    def _count_resource(self, appwrite: AppwriteProxy, resource: str, user_id: str) -> int:
        try:
            docs = self._normalize_documents(appwrite.list_documents(resource, user_id=user_id, limit=50))
            return len(docs)
        except Exception:
            return 0

    def _is_wardrobe_count_query(self, text: str) -> bool:
        lowered = str(text or "").lower()
        count_words = ["how many", "count", "number of", "total", "do i have"]
        wardrobe_words = [
            "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
            "tshirt", "t-shirt", "pants", "trousers", "jeans", "bottoms", "shoes",
            "footwear", "dress", "dresses", "accessories", "jewelry", "bags", "bag",
        ]
        return any(k in lowered for k in count_words) and any(k in lowered for k in wardrobe_words)

    def _wardrobe_query_response(
        self,
        request_id: str,
        user_id: str,
        appwrite: AppwriteProxy,
        text: str,
    ) -> Dict[str, Any]:
        try:
            wardrobe_docs = self._normalize_documents(
                appwrite.list_documents("outfits", user_id=user_id, limit=100)
            )
        except Exception:
            wardrobe_docs = []

        counts = {
            "tops": 0,
            "bottoms": 0,
            "shoes": 0,
            "dresses": 0,
            "accessories": 0,
            "unknown": 0,
        }

        for doc in wardrobe_docs:
            category = str(
                doc.get("category")
                or doc.get("category_group")
                or doc.get("type")
                or ""
            ).strip().lower()
            sub = str(doc.get("sub_category") or doc.get("subcategory") or "").strip().lower()
            text_blob = f"{category} {sub}"

            if any(k in text_blob for k in ["top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "jacket", "blazer"]):
                counts["tops"] += 1
            elif any(k in text_blob for k in ["bottom", "pant", "trouser", "jean", "short", "skirt"]):
                counts["bottoms"] += 1
            elif any(k in text_blob for k in ["shoe", "sneaker", "heel", "boot", "sandal", "footwear"]):
                counts["shoes"] += 1
            elif any(k in text_blob for k in ["dress"]):
                counts["dresses"] += 1
            elif any(k in text_blob for k in ["accessory", "watch", "bag", "jewel", "necklace", "earring"]):
                counts["accessories"] += 1
            else:
                counts["unknown"] += 1

        total = len(wardrobe_docs)
        lowered = str(text or "").lower()
        requested_key = None
        requested_map = {
            "tops": ["top", "tops", "shirt", "shirts", "blouse", "blouses"],
            "bottoms": ["bottom", "bottoms", "pant", "pants", "trouser", "trousers", "jean", "jeans", "skirt", "skirts"],
            "shoes": ["shoe", "shoes", "sneaker", "sneakers", "footwear", "heels", "boots", "sandals"],
            "dresses": ["dress", "dresses"],
            "accessories": ["accessory", "accessories", "jewelry", "watch", "watches", "bag", "bags"],
        }
        for key, words in requested_map.items():
            if any(w in lowered for w in words):
                requested_key = key
                break

        if requested_key:
            message = f"You have {counts.get(requested_key, 0)} {requested_key} in your wardrobe."
        else:
            message = (
                f"You currently have {total} items: "
                f"{counts['tops']} tops, {counts['bottoms']} bottoms, {counts['shoes']} shoes, "
                f"{counts['dresses']} dresses, and {counts['accessories']} accessories."
            )

        cards = [
            {"id": "tops", "title": "Tops", "kind": "stat", "value": counts["tops"]},
            {"id": "bottoms", "title": "Bottoms", "kind": "stat", "value": counts["bottoms"]},
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {"id": "dresses", "title": "Dresses", "kind": "stat", "value": counts["dresses"]},
            {"id": "accessories", "title": "Accessories", "kind": "stat", "value": counts["accessories"]},
        ]

        return {
            "success": True,
            "request_id": request_id,
            "meta": {"intent": "wardrobe_query", "domain": "wardrobe"},
            "message": message,
            "board": "wardrobe",
            "type": "stats",
            "cards": cards,
            "data": {"counts": counts, "total_items": total},
        }

    def _is_plan_pack_request(self, text: str, context: Dict[str, Any]) -> bool:
        lowered = str(text or "").lower()
        module_context = str(context.get("module_context") or "").lower()
        keywords = [
            "plan trip", "trip plan", "packing list", "pack for", "business travel",
            "wedding checklist", "goa trip", "vacation", "travel checklist",
            "carry-on", "carry on", "cabin bag", "luggage", "baggage",
        ]
        if any(k in lowered for k in keywords) or "plan_pack" in module_context:
            return True

        history = context.get("history", []) or []
        recent_text_parts: List[str] = []
        for row in history[-8:]:
            if not isinstance(row, dict):
                continue
            piece = str(row.get("text") or row.get("content") or "").strip().lower()
            if piece:
                recent_text_parts.append(piece)
        history_blob = " ".join(recent_text_parts)
        history_pack = any(
            signal in history_blob
            for signal in [
                "pack & plan",
                "pack and plan",
                "pack for",
                "packing list",
                "travel checklist",
                "trip plan",
                "itinerary",
                "carry-on",
                "carry on",
            ]
        )
        if not history_pack:
            return False

        followup_keywords = [
            "goa",
            "chill",
            "casual",
            "formal",
            "business",
            "wedding",
            "party",
            "beach",
            "cold",
            "hot",
            "rain",
            "international",
            "domestic",
            "flight",
            "train",
            "weekend",
            "days",
            "day",
        ]
        is_short_followup = len([w for w in lowered.split() if w.strip()]) <= 4
        return is_short_followup or any(k in lowered for k in followup_keywords)

    def _is_daily_dependency_request(self, text: str, context: Dict[str, Any]) -> bool:
        lowered = str(text or "").lower()
        module_context = str(context.get("module_context") or "").lower()
        keywords = [
            "daily plan", "daily cards", "morning flow", "midday flow", "afternoon flow",
            "evening flow", "night flow", "tomorrow preview", "day planner", "daily dependency",
        ]
        return any(k in lowered for k in keywords) or "daily_dependency" in module_context


ahvi_orchestrator = AhviOrchestrator()
