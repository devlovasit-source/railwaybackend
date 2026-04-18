import json
import os
import re


class ToneEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "shared", "tone", "tone_engine.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("ahvi_tone_engine_v1", {})
        except Exception as e:
            print(f"WARN: Tone engine load failed: {e}")
            self.config = {}

    def build_prompt_tone(self, user_profile: dict = None, signals: dict = None) -> dict:
        user_profile = user_profile or {}
        signals = signals or {}

        generation = self._detect_generation(user_profile)
        context_mode = str(signals.get("context_mode", "general") or "general")
        emotion = str(signals.get("emotion_state", "neutral") or "neutral")

        generation_rules = self.config.get("generation_defaults", {}).get(generation, {})
        context_rules = self.config.get("context_modes", {}).get(context_mode, {})
        emotion_rules = self.config.get("emotion_overrides", {}).get(emotion, {})
        caps = self._effective_caps(generation_rules, context_rules, emotion_rules)
        locale_key = self._detect_locale_profile(user_profile)

        sentence_style = (
            emotion_rules.get("sentence_style_prefer")
            or emotion_rules.get("sentence_style")
            or context_rules.get("sentence_style")
            or generation_rules.get("sentence_style")
            or "balanced"
        )

        instruction_parts = [
            f"Generation: {generation}.",
            f"Locale: {locale_key}.",
            f"Context mode: {context_mode}.",
            f"Emotion mode: {emotion}.",
            f"Sentence style: {sentence_style}.",
            (
                "Tone caps: "
                f"slang={caps['slang_cap']}, humor={caps['humor_cap']}, "
                f"sass={caps['sass_cap']}, emoji={caps['emoji_cap']}."
            ),
            "Response format: exactly 2 short sentences, funny + interactive + crispy.",
            "Second sentence must be a follow-up question.",
            "Keep grammar clean. Never mirror ALL CAPS, spelling errors, or chaotic punctuation.",
        ]

        if caps["slang_cap"] == 0:
            instruction_parts.append("Use zero slang.")
        if emotion == "vulnerable":
            instruction_parts.append("Validate feelings, avoid jokes, and offer one small next step.")

        return {
            "generation": generation,
            "context_mode": context_mode,
            "emotion_state": emotion,
            "locale_profile": locale_key,
            "caps": caps,
            "tone_instruction": " ".join(instruction_parts).strip(),
        }

    def apply(self, text: str, user_profile: dict = None, signals: dict = None):
        if not text:
            return text

        user_profile = user_profile or {}
        signals = signals or {}

        generation = self._detect_generation(user_profile)
        context_mode = str(signals.get("context_mode", "general") or "general")
        emotion = str(signals.get("emotion_state", "neutral") or "neutral")

        generation_rules = self.config.get("generation_defaults", {}).get(generation, {})
        context_rules = self.config.get("context_modes", {}).get(context_mode, {})
        emotion_rules = self.config.get("emotion_overrides", {}).get(emotion, {})

        caps = self._effective_caps(generation_rules, context_rules, emotion_rules)
        slang_token_cap = self._effective_slang_token_cap(generation_rules, signals)

        return self._apply_constraints(text, emotion_rules, caps, slang_token_cap)

    def _detect_generation(self, user_profile):
        if not user_profile:
            return "other"

        try:
            dob_raw = (
                user_profile.get("dob_iso")
                or user_profile.get("dob")
                or user_profile.get("date_of_birth")
            )
            if not dob_raw:
                return "other"

            match = re.search(r"(19|20)\d{2}", str(dob_raw))
            if not match:
                return "other"

            year = int(match.group(0))
        except Exception:
            return "other"

        buckets = self.config.get("generation_buckets", {})
        for name, rules in buckets.items():
            if int(rules.get("dob_year_min", 0)) <= year <= int(rules.get("dob_year_max", 9999)):
                return name
        return "other"

    def _apply_constraints(self, text: str, emotion_rules: dict, caps: dict, slang_token_cap: int) -> str:
        punctuation_cfg = (self.config.get("global_output_constraints") or {}).get("grammar_and_punctuation", {})
        max_exclamations = int(punctuation_cfg.get("max_exclamation_marks", 1))

        if max_exclamations <= 0:
            text = text.replace("!", ".")
        else:
            while text.count("!") > max_exclamations:
                text = text.replace("!", "", 1)

        if emotion_rules.get("sentence_style") == "soft":
            text = text.replace("!", ".")

        if int(caps.get("emoji_cap", 0)) == 0:
            text = re.sub(r"[\U0001F300-\U0001FAFF]", "", text)

        if int(caps.get("slang_cap", 0)) == 0 or int(slang_token_cap) == 0:
            text = self._remove_slang(text)
        else:
            text = self._clamp_slang(text, int(slang_token_cap))

        return re.sub(r"\s{2,}", " ", text).strip()

    def _remove_slang(self, text: str) -> str:
        slang_banks = self.config.get("slang_libraries", {})
        slang_list = []
        slang_list.extend((slang_banks.get("gen_z") or {}).get("approved_tokens", []))
        slang_list.extend((slang_banks.get("hinglish_optional") or {}).get("approved_tokens", []))

        for token in slang_list:
            token = str(token or "").strip()
            if not token:
                continue
            text = re.sub(re.escape(token), "", text, flags=re.IGNORECASE)

        return re.sub(r"\s{2,}", " ", text).strip()

    def _clamp_slang(self, text: str, allowed_tokens: int) -> str:
        slang_list = (self.config.get("slang_libraries", {}).get("gen_z", {}) or {}).get("approved_tokens", [])
        if not slang_list:
            return text

        remaining = allowed_tokens
        for token in slang_list:
            token = str(token or "").strip()
            if not token:
                continue

            pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
            matches = list(pattern.finditer(text))
            if not matches:
                continue

            if remaining <= 0:
                text = pattern.sub("", text)
                continue

            remaining -= 1
            if len(matches) > 1:
                first_done = False

                def _replace_once(match_obj):
                    nonlocal first_done
                    if not first_done:
                        first_done = True
                        return match_obj.group(0)
                    return ""

                text = pattern.sub(_replace_once, text)

        return re.sub(r"\s{2,}", " ", text).strip()

    def _effective_caps(self, generation_rules: dict, context_rules: dict, emotion_rules: dict) -> dict:
        slang_cap = min(int(generation_rules.get("base_slang", 10)), int(context_rules.get("slang_cap", 100)))
        humor_cap = min(int(generation_rules.get("base_humor", 20)), int(context_rules.get("humor_cap", 100)))
        sass_cap = min(int(generation_rules.get("base_sass", 10)), int(context_rules.get("sass_cap", 100)))
        emoji_cap = min(int(generation_rules.get("base_emoji", 0)), int(context_rules.get("emoji_cap", 2)))

        if "slang_cap" in emotion_rules:
            slang_cap = min(slang_cap, int(emotion_rules.get("slang_cap", slang_cap)))
        if "humor_cap" in emotion_rules:
            humor_cap = min(humor_cap, int(emotion_rules.get("humor_cap", humor_cap)))
        if "sass_cap" in emotion_rules:
            sass_cap = min(sass_cap, int(emotion_rules.get("sass_cap", sass_cap)))
        if "emoji_cap" in emotion_rules:
            emoji_cap = min(emoji_cap, int(emotion_rules.get("emoji_cap", emoji_cap)))

        slang_cap += int(emotion_rules.get("slang_boost", 0))
        humor_cap += int(emotion_rules.get("humor_boost", 0))
        sass_cap += int(emotion_rules.get("sass_boost", 0))
        emoji_cap += int(emotion_rules.get("emoji_boost", 0))

        return {
            "slang_cap": max(0, min(100, slang_cap)),
            "humor_cap": max(0, min(100, humor_cap)),
            "sass_cap": max(0, min(100, sass_cap)),
            "emoji_cap": max(0, min(3, emoji_cap)),
        }

    def _effective_slang_token_cap(self, generation_rules: dict, signals: dict) -> int:
        base = int(generation_rules.get("slang_tokens_max_per_response", 0))
        style = signals.get("user_message_style", {}) if isinstance(signals.get("user_message_style", {}), dict) else {}
        slang_presence = str(style.get("slang_presence", "none")).lower()
        mirror_cfg = ((self.config.get("mirroring_rules") or {}).get("slang_presence") or {}).get(slang_presence, {})
        mirror_max = int(mirror_cfg.get("assistant_slang_tokens_max", base))
        return min(base, mirror_max)

    def _detect_locale_profile(self, user_profile: dict) -> str:
        country = str(user_profile.get("country", "") or "").upper().strip()
        preferred_language = str(user_profile.get("preferred_language") or user_profile.get("lang") or "").lower().strip()

        locales = self.config.get("locale_profiles", {})
        for key, profile in locales.items():
            match_if = profile.get("match_if", {})
            countries = [str(c).upper().strip() for c in match_if.get("country_any_of", [])]
            langs = [str(l).lower().strip() for l in match_if.get("preferred_language_contains_any_of", [])]

            country_ok = (not countries) or ("*" in countries) or (country in countries)
            lang_ok = (not langs) or any(token and token in preferred_language for token in langs)

            if country_ok and lang_ok:
                return key

        return "en"


# Singleton
tone_engine = ToneEngine()
