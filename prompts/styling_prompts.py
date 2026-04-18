# backend/prompts/styling_prompts.py

STYLE_EXPLANATION_PROMPT = """
Outfit (already selected by engine):

Top: {top}
Bottom: {bottom}
Shoes: {shoes}

Context:
- Occasion: {occasion}
- Weather: {weather}
- Vibe: {vibe}

RULES:
1. DO NOT change the outfit
2. ONLY explain why it works
3. Keep it short (2–3 lines)
4. Sound stylish and confident

Add a slight personality touch.
"""

MULTI_OUTFIT_PROMPT = """
You are given 3 outfit options ranked best to worst.

Outfits:
{outfits}

RULES:
1. Recommend ONLY the best one
2. Mention why it's the best
3. Briefly hint there are other options
4. Keep it casual and stylish
"""