import os
import io
import base64
import torch
import threading
import numpy as np
import cv2
from collections import deque
from PIL import Image, ImageFilter
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, validator
from transformers import AutoModelForImageSegmentation
from fastapi import status

try:
    from huggingface_hub import snapshot_download, login as hf_login
except Exception:
    snapshot_download = None
    hf_login = None

try:
    from worker import bg_remove_task
except Exception:
    bg_remove_task = None
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_session = None
onnx_lock = threading.Lock()
onnx_last_error = None
onnx_runtime_disabled = False
USE_TORCH_MODEL = os.getenv("BG_USE_TORCH_MODEL", "1") == "1"
BG_DISABLE_ONNX = os.getenv("BG_DISABLE_ONNX", "0") == "1"
BG_AUTO_DOWNLOAD = os.getenv("BG_AUTO_DOWNLOAD", "1") == "1"
BG_HF_REPO_ID = os.getenv("BG_HF_REPO_ID", "briaai/RMBG-2.0")

print(f"Using device: {device}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "RMBG_2_0")
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

def _to_model_input_tensor(image: Image.Image, side: int = 1024):
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
        snapshot_download(
            repo_id=BG_HF_REPO_ID,
            local_dir=MODEL_PATH,
        )
    except Exception as exc:
        return f"Model download failed: {exc}"

    if not _model_assets_present(MODEL_PATH):
        return f"Downloaded to {MODEL_PATH} but required weights not found"
    return None

def load_model():
    global model, model_last_error

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        print("Loading model from:", MODEL_PATH)

        download_error = _ensure_model_downloaded()
        if download_error:
            model_last_error = download_error
            print("Model load failed:", model_last_error)
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
                print("Model ready.")
                return model
            except Exception as exc:
                errors.append(f"{attempt}: {exc}")

        model_last_error = " | ".join(errors)
        print("Model load failed:", model_last_error)
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

    if model_instance is None:
        onnx_instance = load_onnx_session()

    try:
        try:
            base64_data = image_base64.split(",")[-1]
            image_data = base64.b64decode(base64_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        if len(image_data) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large")

        if model_instance is None and onnx_instance is None:
            detail = "Model unavailable"
            if model_last_error:
                detail = f"Model unavailable: {model_last_error}"
            if onnx_last_error:
                detail = f"{detail} | ONNX fallback failed: {onnx_last_error}"
            print("BG fallback to original:", detail)
            return _original_png_response(image_data, detail)

        orig_image = Image.open(io.BytesIO(image_data)).convert("RGB")
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


@router.post("/remove-bg")
def remove_background(request: BGRemoveRequest):
    return remove_background_sync(request.image_base64)


@router.post("/remove-bg/async", status_code=status.HTTP_202_ACCEPTED)
async def remove_background_async(http_request: Request, request: BGRemoveRequest):
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
