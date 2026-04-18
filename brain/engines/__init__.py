try:
    from .styling.style_builder import style_engine
except Exception:
    style_engine = None

try:
    from .packing.packing_engine import packing_engine
except Exception:
    packing_engine = None

# Calendar engine is optional; keep a stable name even if unavailable.
calendar_engine = None
