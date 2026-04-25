import os
import io
import base64
import gc
import threading
import time
import tempfile
import numpy as np
import cv2
from collections import deque
from PIL import Image, ImageFilter
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, validator
from fastapi import status

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModelForImageSegmentation
except Exception:
    AutoModelForImageSegmentation = None

try:
    from huggingface_hub import snapshot_download, login as hf_login
except Exception:
    snapshot_download = None
    hf_login = None

try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task

try:
    import onnxruntime as ort
except Exception:
    ort = None

print("BG_REMOVER LOADED")

router = APIRouter()

class BGRemoveRequest(BaseModel):
    image_base64: str

    @validator("image_base64")
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid image data")
        return v

model = None
model_last_error = None
model_lock = threading.Lock()
model_fail_count = 0
model_disabled_until = 0.0
if torch is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _device_type = device.type
else:
    device = "cpu"
    _device_type = "cpu"
onnx_session = None
onnx_lock = threading.Lock()
onnx_last_error = None
onnx_runtime_disabled = False
_default_bg_use_torch = "1" if (_device_type == "cuda" and AutoModelForImageSegmentation is not None) else "0"
USE_TORCH_MODEL = (
    os.getenv("BG_USE_TORCH_MODEL", _default_bg_use_torch) == "1"
    and torch is not None
    and AutoModelForImageSegmentation is not None
)
BG_DISABLE_ONNX = os.getenv("BG_DISABLE_ONNX", "0") == "1"
BG_AUTO_DOWNLOAD = os.getenv("BG_AUTO_DOWNLOAD", "1") == "1"
BG_HF_REPO_ID = os.getenv("BG_HF_REPO_ID", "briaai/RMBG-2.0")
BG_ALLOW_TORCH_FALLBACK = os.getenv("BG_ALLOW_TORCH_FALLBACK", "1") == "1"
BG_DOWNLOAD_ONNX_ASSETS = os.getenv("BG_DOWNLOAD_ONNX_ASSETS", "0") == "1"

print(f"Using device: {_device_type}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _path_has_rmbg_assets(path: str) -> bool:
    try:
        if not path or not os.path.exists(path):
            return False
        # Torch checkpoint assets (required for AutoModelForImageSegmentation).
        if os.path.exists(os.path.join(path, "config.json")) and any(
            os.path.exists(os.path.join(path, candidate))
            for candidate in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")
        ):
            return True
        # ONNX fallback assets.
        onnx_dir = os.path.join(path, "onnx")
        if os.path.isdir(onnx_dir) and any(
            os.path.exists(os.path.join(onnx_dir, candidate))
            for candidate in (
                "model_q4f16.onnx",
                "model_q4.onnx",
                "model_int8.onnx",
                "model_uint8.onnx",
                "model_quantized.onnx",
                "model_fp16.onnx",
                "model.onnx",
            )
        ):
            return True
    except Exception:
        return False
    return False


_repo_default_model_path = os.path.join(BASE_DIR, "..", "RMBG_2_0")
_cache_default_model_path = os.path.join(tempfile.gettempdir(), "ahvi_rmbg_2_0")
_configured_model_path = str(os.getenv("BG_MODEL_PATH", "") or "").strip()
if _configured_model_path:
    MODEL_PATH = _configured_model_path
elif _path_has_rmbg_assets(_repo_default_model_path):
    MODEL_PATH = _repo_default_model_path
else:
    MODEL_PATH = _cache_default_model_path
ONNX_DIR = os.path.join(MODEL_PATH, "onnx")
ONNX_MODEL_CANDIDATES = [
    "model_q4f16.onnx",
    "model_q4.onnx",
    "model_int8.onnx",
    "model_uint8.onnx",
    "model_quantized.onnx",
    "model_fp16.onnx",
    "model.onnx",
]
ONNX_INPUT_SIZES = [1024]
BG_MAX_INPUT_BYTES = int(os.getenv("BG_MAX_INPUT_BYTES", os.getenv("UPLOAD_MAX_BYTES", str(50 * 1024 * 1024))))
BG_MAX_LONG_EDGE = max(1024, int(os.getenv("BG_MAX_LONG_EDGE", "2048")))
BG_MAX_PIXELS = max(2_000_000, int(os.getenv("BG_MAX_PIXELS", "16000000")))
BG_MODEL_RETRY_COOLDOWN_SECONDS = max(30, int(os.getenv("BG_MODEL_RETRY_COOLDOWN_SECONDS", "300")))


def _resize_for_bg(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    long_edge = max(w, h)
    pixels = w * h
    scale = 1.0
    if long_edge > BG_MAX_LONG_EDGE:
        scale = min(scale, float(BG_MAX_LONG_EDGE) / float(long_edge))
    if pixels > BG_MAX_PIXELS:
        scale = min(scale, (float(BG_MAX_PIXELS) / float(pixels)) ** 0.5)
    if scale >= 0.999:
        return image
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    print(f"BG resize: {w}x{h} -> {nw}x{nh}")
    return image.resize((nw, nh), Image.LANCZOS)

def _to_model_input_tensor(image: Image.Image, side: int = 1024):
    if torch is None:
        raise RuntimeError("torch is not installed")
    rgb = image.convert("RGB").resize((side, side), Image.BILINEAR)
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return torch.from_numpy(arr)

def _model_assets_present(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    return any(os.path.exists(os.path.join(path, c)) for c in candidates)

def _ensure_model_downloaded() -> str | None:
    if _model_assets_present(MODEL_PATH):
        return None
    if not BG_AUTO_DOWNLOAD:
        return f"Model not found at {MODEL_PATH} and BG_AUTO_DOWNLOAD=0"
    if snapshot_download is None:
        return "huggingface_hub is not installed; cannot download RMBG model"

    try:
        os.makedirs(MODEL_PATH, exist_ok=True)
        token = str(os.getenv("HF_TOKEN", "") or "").strip()
        if token and hf_login is not None:
            hf_login(token, add_to_git_credential=False)

        print(f"Downloading model from {BG_HF_REPO_ID} to {MODEL_PATH} ...")
        # Keep local startup stable by default: download only Torch assets.
        # ONNX assets in RMBG repo are very large and can cause long cold starts.
        kwargs = {
            "repo_id": BG_HF_REPO_ID,
            "local_dir": MODEL_PATH,
        }
        if not BG_DOWNLOAD_ONNX_ASSETS:
            kwargs["ignore_patterns"] = ["onnx/*", "*.onnx", "*.png"]
        snapshot_download(**kwargs)
    except Exception as exc:
        return f"Model download failed: {exc}"

    if not _model_assets_present(MODEL_PATH):
        return f"Downloaded to {MODEL_PATH} but required weights not found"
    return None

def load_model():
    global model, model_last_error, model_fail_count, model_disabled_until

    if model is not None:
        return model

    if torch is None or AutoModelForImageSegmentation is None:
        missing = []
        if torch is None:
            missing.append("torch")
        if AutoModelForImageSegmentation is None:
            missing.append("transformers")
        model_last_error = f"Torch RMBG unavailable: missing {', '.join(missing)}"
        return None

    now = time.time()
    if model_disabled_until and now < model_disabled_until:
        wait_seconds = int(model_disabled_until - now)
        model_last_error = f"Model load temporarily disabled for {wait_seconds}s after repeated failures"
        return None

    with model_lock:
        if model is not None:
            return model
        now = time.time()
        if model_disabled_until and now < model_disabled_until:
            wait_seconds = int(model_disabled_until - now)
            model_last_error = f"Model load temporarily disabled for {wait_seconds}s after repeated failures"
            return None

        print("Loading model from:", MODEL_PATH)

        download_error = _ensure_model_downloaded()
        if download_error:
            model_last_error = download_error
            print("Model load failed:", model_last_error)
            model_fail_count += 1
            if model_fail_count >= 2:
                model_disabled_until = time.time() + BG_MODEL_RETRY_COOLDOWN_SECONDS
            model = None
            return model

        attempts = [
            {"use_safetensors": True},
            {"use_safetensors": False},
        ]
        errors = []

        for attempt in attempts:
            try:
                model_local = AutoModelForImageSegmentation.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                    **attempt,
                )

                if any(p.is_meta for p in model_local.parameters()):
                    raise RuntimeError("Model contains meta tensors after load")

                model_local = model_local.to(device=device, dtype=torch.float32)
                model_local.eval()
                model = model_local
                model_last_error = None
                model_fail_count = 0
                model_disabled_until = 0.0
                print("Model ready.")
                return model
            except Exception as exc:
                errors.append(f"{attempt}: {exc}")

        model_last_error = " | ".join(errors)
        print("Model load failed:", model_last_error)
        model_fail_count += 1
        lower_error = model_last_error.lower()
        if model_fail_count >= 2 or ("out of memory" in lower_error) or ("cannot allocate" in lower_error):
            model_disabled_until = time.time() + BG_MODEL_RETRY_COOLDOWN_SECONDS
        model = None

    return model

def load_onnx_session():
    global onnx_session, onnx_last_error, onnx_runtime_disabled
    if BG_DISABLE_ONNX:
        onnx_last_error = "ONNX disabled by BG_DISABLE_ONNX=1"
        return None
    if onnx_session is not None:
        return onnx_session
    if onnx_runtime_disabled:
        onnx_last_error = "onnx runtime disabled after repeated OOM/runtime failures"
        return None
    if ort is None:
        onnx_last_error = "onnxruntime is not installed"
        return None

    with onnx_lock:
        if onnx_session is not None:
            return onnx_session
        if not os.path.exists(ONNX_DIR):
            onnx_last_error = f"ONNX directory not found: {ONNX_DIR}"
            return None

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = False
        session_options.log_severity_level = 3

        errors = []
        for name in ONNX_MODEL_CANDIDATES:
            candidate_path = os.path.join(ONNX_DIR, name)
            if not os.path.exists(candidate_path):
                continue
            try:
                onnx_session = ort.InferenceSession(
                    candidate_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                onnx_last_error = None
                print(f"ONNX fallback ready: {candidate_path}")
                return onnx_session
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        if not errors:
            onnx_last_error = (
                "No supported ONNX model file found in "
                f"{ONNX_DIR}. Tried: {', '.join(ONNX_MODEL_CANDIDATES)}"
            )
        else:
            onnx_last_error = " | ".join(errors)
        print("ONNX load failed:", onnx_last_error)
        return None

def _original_png_response(image_data: bytes, reason: str):
    try:
        original = Image.open(io.BytesIO(image_data)).convert("RGBA")
        buffer = io.BytesIO()
        original.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("BG original fallback:", reason)
        return {
            "success": True,
            "image_base64": result_base64,
            "bg_removed": False,
            "fallback_reason": reason,
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Processing failed")

def remove_background_sync(image_base64: str):
    global onnx_runtime_disabled
    model_instance = load_model() if USE_TORCH_MODEL else None
    onnx_instance = None
    input_tensor = None
    preds = None
    mask = None
    mask_u8 = None
    mask_pil = None
    final_image = None
    buffer = None
    image_data = b""

    if model_instance is None:
        onnx_instance = load_onnx_session()

    # If ONNX is unavailable on CPU-only setups, transparently fall back to
    # the Torch RMBG model so upload flow does not hard-fail.
    if model_instance is None and onnx_instance is None and BG_ALLOW_TORCH_FALLBACK:
        try:
            model_instance = load_model()
        except Exception:
            model_instance = None

    try:
        try:
            base64_data = image_base64.split(",")[-1]
            image_data = base64.b64decode(base64_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        if len(image_data) > BG_MAX_INPUT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large for background remover (max {BG_MAX_INPUT_BYTES} bytes)",
            )

        if model_instance is None and onnx_instance is None:
            detail = "Model unavailable"
            if model_last_error:
                detail = f"Model unavailable: {model_last_error}"
            if onnx_last_error:
                detail = f"{detail} | ONNX fallback failed: {onnx_last_error}"
            print("BG fallback to original:", detail)
            return _original_png_response(image_data, detail)

        orig_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        orig_image = _resize_for_bg(orig_image)
        w, h = orig_image.size

        if model_instance is not None:
            input_tensor = _to_model_input_tensor(orig_image).to(device)
            with torch.no_grad():
                preds = model_instance(input_tensor)[-1].sigmoid().cpu()
            mask = preds[0].squeeze().numpy()
        else:
            input_name = onnx_instance.get_inputs()[0].name
            input_shape = onnx_instance.get_inputs()[0].shape

            fixed_side = 1024
            if len(input_shape) >= 4:
                maybe_h = input_shape[-2]
                maybe_w = input_shape[-1]
                if (
                    isinstance(maybe_h, int)
                    and isinstance(maybe_w, int)
                    and maybe_h == maybe_w
                    and maybe_h > 0
                ):
                    fixed_side = maybe_h
            candidate_sizes = [fixed_side]
            onnx_errors = []
            mask = None

            for side in candidate_sizes:
                try:
                    img_onnx = orig_image.resize((side, side), Image.BILINEAR)
                    arr = np.asarray(img_onnx).astype(np.float32) / 255.0
                    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                        [0.229, 0.224, 0.225], dtype=np.float32
                    )
                    arr = np.transpose(arr, (2, 0, 1))[None, ...]
                    pred = onnx_instance.run(None, {input_name: arr})[0]
                    if pred.ndim == 4:
                        pred = pred[0, 0]
                    elif pred.ndim == 3:
                        pred = pred[0]
                    mask = 1.0 / (1.0 + np.exp(-pred))
                    break
                except Exception as exc:
                    onnx_errors.append(f"{side}: {exc}")

            if mask is None:
                reason = "ONNX inference failed: " + " | ".join(onnx_errors)
                onnx_runtime_disabled = True
                if any(
                    ("bad allocation" in err.lower())
                    or ("allocate" in err.lower())
                    or ("out of memory" in err.lower())
                    for err in onnx_errors
                ):
                    reason += " | ONNX disabled for this process due to memory failure"
                else:
                    reason += " | ONNX disabled for this process due to runtime failure"
                print("BG fallback to original:", reason)
                return _original_png_response(image_data, reason)

        # Let the AI mask do its job natively.
        # Removed the grabcut interference which breaks on black clothing.
        mask = np.clip(mask, 0.0, 1.0)
        mask_u8 = (mask * 255.0).astype("uint8")
        
        # Smooth the mask slightly for cleaner edges
        mask_pil = Image.fromarray(mask_u8, mode="L").resize((w, h), Image.LANCZOS)
        mask_pil = mask_pil.filter(ImageFilter.SMOOTH)

        final_image = orig_image.copy()
        final_image.putalpha(mask_pil)

        buffer = io.BytesIO()
        final_image.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        print("BG model result: success")

        return {
            "success": True,
            "image_base64": result_base64,
            "bg_removed": True,
            "fallback_reason": None,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("BG error:", e)
        try:
            base64_data = image_base64.split(",")[-1]
            image_data = base64.b64decode(base64_data)
            return _original_png_response(image_data, f"Unhandled BG error: {e}")
        except Exception:
            raise HTTPException(status_code=500, detail="Processing failed")
    finally:
        try:
            del input_tensor, preds, mask, mask_u8, mask_pil, final_image, buffer, image_data
        except Exception:
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass


@router.post("/remove-bg")
def remove_background(request: BGRemoveRequest):
    return remove_background_sync(request.image_base64)


@router.post("/remove-bg/async", status_code=status.HTTP_202_ACCEPTED)
async def remove_background_async(http_request: Request, request: BGRemoveRequest):
    try:
        from worker import bg_remove_task  # lazy import: avoid worker/celery side effects at web startup
    except Exception:
        bg_remove_task = None
    if bg_remove_task is None:
        raise HTTPException(status_code=503, detail="Celery worker not configured")
    try:
        task_id = enqueue_task(
            task_func=bg_remove_task,
            args=[request.image_base64],
            kind="bg_remove",
            user_id=None,
            request_id=str(getattr(http_request.state, "request_id", "") or ""),
            source="api:/api/background/remove-bg/async",
            meta={"task_type": "bg_remove_task"},
        )
        return {
            "success": True,
            "status": "queued",
            "task_id": task_id,
            "task_type": "bg_remove_task",
            "request_id": str(getattr(http_request.state, "request_id", "") or ""),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to queue bg removal: {exc}")
