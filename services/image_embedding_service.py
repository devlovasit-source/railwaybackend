import base64
import io
import os
from typing import Any

import requests
from PIL import Image
try:
    import torch
except Exception:
    torch = None
try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    CLIPModel = None
    CLIPProcessor = None


_model = None
_processor = None
_device = torch.device("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu") if torch is not None else "cpu"

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_MODEL_DIR = os.path.abspath(
    os.getenv("IMAGE_EMBEDDING_MODEL_DIR", os.path.join(_PROJECT_ROOT, "local-clip-vit-base-patch32"))
)
_REMOTE_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")

_URL_VECTOR_CACHE: dict[str, list] = {}
_URL_VECTOR_CACHE_MAX = 512


def _cache_get(url: str):
    return _URL_VECTOR_CACHE.get(url)


def _cache_set(url: str, vector: list) -> None:
    if not url or not vector:
        return
    _URL_VECTOR_CACHE[url] = vector
    if len(_URL_VECTOR_CACHE) > _URL_VECTOR_CACHE_MAX:
        oldest = next(iter(_URL_VECTOR_CACHE.keys()), None)
        if oldest:
            _URL_VECTOR_CACHE.pop(oldest, None)


def _normalize_base64(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "," in text:
        text = text.split(",", 1)[1]
    return text.strip()


def _load_model():
    global _model, _processor
    if torch is None or CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError("transformers/torch for image embedding are not installed")
    if _model is not None and _processor is not None:
        return _model, _processor

    source = _LOCAL_MODEL_DIR if os.path.isdir(_LOCAL_MODEL_DIR) else _REMOTE_MODEL_NAME
    print(f"Loading image embedding model from: {source}")
    try:
        _processor = CLIPProcessor.from_pretrained(source)
        _model = CLIPModel.from_pretrained(source)
    except Exception as exc:
        if source != _REMOTE_MODEL_NAME:
            print(f"Local image embedding load failed ({exc}). Falling back to: {_REMOTE_MODEL_NAME}")
            _processor = CLIPProcessor.from_pretrained(_REMOTE_MODEL_NAME)
            _model = CLIPModel.from_pretrained(_REMOTE_MODEL_NAME)
        else:
            raise

    _model.to(_device)
    _model.eval()
    return _model, _processor


def encode_image_bytes(image_bytes: bytes) -> list:
    if not image_bytes:
        return []
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model, processor = _load_model()
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
        return features[0].detach().cpu().float().tolist()
    except Exception as exc:
        print("Image embedding error:", str(exc))
        return []


def encode_image_base64(value: Any) -> list:
    text = _normalize_base64(value)
    if not text:
        return []
    try:
        image_bytes = base64.b64decode(text, validate=True)
    except Exception:
        return []
    return encode_image_bytes(image_bytes)


def encode_image_url(url: Any, timeout_seconds: float = 8.0) -> list:
    normalized = str(url or "").strip()
    if not normalized:
        return []

    cached = _cache_get(normalized)
    if cached:
        return cached

    try:
        response = requests.get(normalized, timeout=timeout_seconds)
        response.raise_for_status()
        vector = encode_image_bytes(response.content)
        if vector:
            _cache_set(normalized, vector)
        return vector
    except Exception:
        return []
