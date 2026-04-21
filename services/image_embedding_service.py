import base64
import hashlib
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

_EMBEDDING_PROVIDER = str(os.getenv("IMAGE_EMBEDDING_PROVIDER", "clip")).strip().lower()
_OPENAI_EMBEDDING_MODEL = str(os.getenv("IMAGE_EMBEDDING_OPENAI_MODEL", "text-embedding-3-small")).strip()
_OPENAI_BASE_URL = str(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip().rstrip("/")
_OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY", "")).strip()


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


def _image_descriptor(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""
    digest = hashlib.sha256(image_bytes).hexdigest()
    size = len(image_bytes)
    width = 0
    height = 0
    mode = ""
    avg_rgb = (0, 0, 0)
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        mode = image.mode
        tiny = image.resize((16, 16))
        pixels = list(tiny.getdata())
        if pixels:
            r = int(sum(p[0] for p in pixels) / len(pixels))
            g = int(sum(p[1] for p in pixels) / len(pixels))
            b = int(sum(p[2] for p in pixels) / len(pixels))
            avg_rgb = (r, g, b)
    except Exception:
        pass
    return (
        f"image hash={digest} size={size} width={width} height={height} "
        f"mode={mode} avg_rgb={avg_rgb[0]},{avg_rgb[1]},{avg_rgb[2]}"
    )


def _encode_via_openai_text_embedding(image_bytes: bytes) -> list:
    if not image_bytes or not _OPENAI_API_KEY:
        return []
    descriptor = _image_descriptor(image_bytes)
    if not descriptor:
        return []
    try:
        response = requests.post(
            f"{_OPENAI_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {_OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": _OPENAI_EMBEDDING_MODEL,
                "input": descriptor,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        arr = data.get("data") if isinstance(data, dict) else None
        if isinstance(arr, list) and arr:
            vec = arr[0].get("embedding") if isinstance(arr[0], dict) else None
            if isinstance(vec, list):
                return [float(x) for x in vec]
    except Exception:
        return []
    return []


def encode_image_bytes(image_bytes: bytes) -> list:
    if not image_bytes:
        return []
    if _EMBEDDING_PROVIDER == "openai":
        return _encode_via_openai_text_embedding(image_bytes)
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
