import os

from sentence_transformers import SentenceTransformer

# -------------------------
# GLOBAL MODEL (LAZY LOAD)
# -------------------------
_model = None
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_MODEL_DIR = os.path.abspath(
    os.getenv("EMBEDDING_MODEL_DIR", os.path.join(_PROJECT_ROOT, "local-minilm"))
)
_REMOTE_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def get_model():
    global _model

    if _model is None:
        model_source = _LOCAL_MODEL_DIR if os.path.isdir(_LOCAL_MODEL_DIR) else _REMOTE_MODEL_NAME
        print(f"Loading embedding model from: {model_source}")
        try:
            _model = SentenceTransformer(model_source)
        except Exception as exc:
            if model_source != _REMOTE_MODEL_NAME:
                print(f"Local embedding load failed ({exc}). Falling back to: {_REMOTE_MODEL_NAME}")
                _model = SentenceTransformer(_REMOTE_MODEL_NAME)
            else:
                raise

    return _model


# -------------------------
# BUILD TEXT FROM METADATA
# -------------------------
def build_text(data: dict) -> str:
    category = str(data.get("category") or "")
    sub_category = str(data.get("sub_category") or "")
    color_code = str(data.get("color_code") or "")
    pattern = str(data.get("pattern") or "")
    occasions_raw = data.get("occasions", [])
    if isinstance(occasions_raw, list):
        occasions = " ".join(str(x or "") for x in occasions_raw)
    else:
        occasions = str(occasions_raw or "")

    return " ".join([category, sub_category, color_code, pattern, occasions]).strip()


# -------------------------
# MAIN FUNCTION
# -------------------------
def encode_metadata(data: dict) -> list:
    try:
        model = get_model()

        text = build_text(data)

        embedding = model.encode(text)

        return embedding.tolist()

    except Exception as e:
        print("Embedding error:", str(e))
        return [0.0] * 384
