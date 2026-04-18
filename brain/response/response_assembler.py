import json
import logging
import os

from brain.tone.tone_engine import tone_engine
from services.ai_gateway import chat_completion

logger = logging.getLogger("ahvi.response_assembler")


class ResponseAssembler:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile_path = os.path.join(base_dir, "config", "assembly_profiles.json")

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("assembly_profiles", {})
        except Exception as exc:
            logger.warning("Assembly profile load failed: %s", exc)
            self.config = {}

    def assemble(self, merged_output: dict, context: dict = None) -> str:
        context = context or {}
        data = merged_output.get("data", {}) if isinstance(merged_output, dict) else {}
        domains = context.get("domains") or list(data.keys())
        if not domains and context.get("domain"):
            domains = [context.get("domain")]

        is_multi_intent = bool(context.get("is_multi_intent") or len(domains) > 1)
        user_profile = context.get("user_profile", {})

        parts = [self._reaction(is_multi_intent)]

        if is_multi_intent and data:
            intelligence = self._synthesize(data, domains, user_profile)
        else:
            intelligence = (
                merged_output.get("message", "")
                or merged_output.get("context", "")
                or self._fallback_synthesis(data, domains)
            )

        if intelligence:
            parts.append(intelligence)

        if isinstance(merged_output, dict) and merged_output.get("accessories"):
            suggestion = f"Try adding {merged_output['accessories'][0]} to complete the look."
            parts.append(suggestion)

        parts.append(self._closer())
        final_text = self._apply_global_rules(parts)

        return tone_engine.apply(
            final_text,
            user_profile=user_profile,
            signals=context.get("signals"),
        )

    def _synthesize(self, data: dict, domains: list, user_profile: dict) -> str:
        llm_enabled = os.getenv("ENABLE_LLM_SYNTHESIS", "false").lower() in ("1", "true", "yes")
        if not llm_enabled:
            return self._fallback_synthesis(data, domains)

        system_prompt = (
            "You are AHVI, an AI fashion and lifestyle assistant. "
            "Combine multiple domain results into one cohesive response."
        )
        prompt = (
            f"Domains: {', '.join(domains)}\n"
            f"User Profile: {json.dumps(user_profile)}\n"
            f"Engine Data: {json.dumps(data)}\n"
            "Write a concise helpful answer."
        )

        try:
            return chat_completion(
                [{"role": "user", "content": prompt}],
                system_instruction=system_prompt,
            )
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s", exc)
            return self._fallback_synthesis(data, domains)

    def _fallback_synthesis(self, data: dict, domains: list) -> str:
        if not data:
            return ""

        chunks = []
        for domain in domains:
            domain_data = data.get(domain, {})
            if isinstance(domain_data, dict):
                msg = domain_data.get("message") or domain_data.get("summary")
                if not msg:
                    msg = f"Your {domain} plan is ready."
            else:
                msg = str(domain_data)
            chunks.append(msg)
        return "\n\n".join(chunks)

    def _reaction(self, is_multi_intent: bool) -> str:
        if is_multi_intent:
            return "Alright, I have everything queued up for you."
        return "Nice, this is coming together."

    def _closer(self) -> str:
        return "Want me to refine this further?"

    def _apply_global_rules(self, parts: list) -> str:
        rules = self.config.get("global_rules", {})
        max_q = rules.get("max_questions_per_response", 1)
        max_sent = rules.get("max_sentences_layer_1", 3)

        cleaned = []
        question_count = 0
        for part in parts:
            if not part:
                continue
            if "?" in part:
                if question_count >= max_q:
                    continue
                question_count += 1
            cleaned.append(part)

        final = "\n\n".join(cleaned)
        sentences = final.split(". ")
        if len(sentences) > max_sent:
            final = ". ".join(sentences[:max_sent])
        return final.strip()


response_assembler = ResponseAssembler()
