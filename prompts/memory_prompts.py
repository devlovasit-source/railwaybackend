# backend/prompts/memory_prompts.py

UPDATE_MEMORY_PROMPT = """
You are a strict fashion memory extractor.

Extract ONLY long-term preferences:
- favorite colors
- style (streetwear, minimal, formal)
- dislikes
- body type

User Message: "{new_user_text}"
Current Memory: "{current_memory}"

RULES:
1. Merge new preferences into memory
2. Ignore temporary requests
3. Keep it SHORT
4. Output ONLY memory text
"""