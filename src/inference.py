"""
CropCare AI — Inference engine.

Loads the best model checkpoint once (singleton pattern) and
provides predict() for single-image classification with confidence.
"""

import os
import sys
import io
import json
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODELS_DIR, FULL_DATASET_MODELS_DIR, MOBILENET_MODELS_DIR, NUM_CLASSES,
    IDX_TO_CLASS, TARGET_CLASSES, IMG_SIZE, MOBILENET_IMG_SIZE, DISEASE_INFO,
    ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
)
from src.models import build_model, ARCHITECTURES
from src.mobilenet import MODEL_NAME, build_mobilenet_v3_large
from src.dataset import get_mobilenet_eval_transform


# ──────────────────────────────────────────────────────
# Singleton model loader
# ──────────────────────────────────────────────────────

_model = None
_device = None
_arch_name = None
_img_size = IMG_SIZE
_is_mobilenet = False


def load_model(checkpoint_path: str = None,
               force_cpu: bool = False) -> Tuple[object, torch.device, str]:
    """Load model from checkpoint. Cached after first call."""
    global _model, _device, _arch_name, _img_size, _is_mobilenet

    if _model is not None:
        return _model, _device, _arch_name

    if checkpoint_path is None:
        mobile_candidate = os.path.join(MOBILENET_MODELS_DIR, f"best_{MODEL_NAME}.pth")
        if os.path.isfile(mobile_candidate):
            checkpoint_path = mobile_candidate

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "MobileNetV3-Large checkpoint not found at "
            f"{os.path.join(MOBILENET_MODELS_DIR, f'best_{MODEL_NAME}.pth')}. "
            "Train it first with: python src/train_mobilenet.py"
        )

    _device = torch.device("cpu" if force_cpu or not torch.cuda.is_available()
                           else "cuda")
    print(f"[Inference] Loading checkpoint: {checkpoint_path} on {_device}")
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=True)
    _arch_name = ckpt.get("arch", "CropCareNet_M")
    _is_mobilenet = _arch_name == MODEL_NAME
    checkpoint_classes = (
        ckpt.get("class_mapping", {}).get("classes")
        if _is_mobilenet else ckpt.get("run_config", {}).get("target_classes")
    )
    if checkpoint_classes != TARGET_CLASSES:
        raise ValueError(
            "Checkpoint class mapping does not match the current dataset configuration. "
            "Train a full-dataset checkpoint before serving predictions."
        )

    run_cfg = ckpt.get("run_config", {})
    _img_size = MOBILENET_IMG_SIZE if _is_mobilenet else run_cfg.get("img_size", IMG_SIZE)

    if _is_mobilenet:
        _model = build_mobilenet_v3_large(NUM_CLASSES, pretrained=False).to(_device)
    else:
        _model = build_model(_arch_name, num_classes=NUM_CLASSES).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    print(f"[Inference] Model ready: {_arch_name} | img_size={_img_size} | device={_device}")
    return _model, _device, _arch_name


def unload_model():
    """Release model from memory (useful for testing)."""
    global _model, _device, _arch_name, _is_mobilenet, _img_size
    _model = None
    _device = None
    _arch_name = None
    _is_mobilenet = False
    _img_size = IMG_SIZE


# ──────────────────────────────────────────────────────
# Image validation
# ──────────────────────────────────────────────────────

class ImageValidationError(Exception):
    pass


def validate_image_bytes(image_bytes: bytes,
                         filename: str = "") -> Image.Image:
    """
    Validate raw image bytes. Raises ImageValidationError with a
    user-friendly message if the image cannot be used for inference.
    """
    # Size check
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_UPLOAD_SIZE_MB} MB."
        )

    # Extension check
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}"
            )

    # Try to open and verify
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # check for corruption
    except Exception:
        raise ImageValidationError(
            "File does not appear to be a valid image or is corrupted."
        )

    # Re-open (verify() closes the file)
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ImageValidationError(f"Could not decode image: {e}")

    # Minimum size check
    if img.width < 10 or img.height < 10:
        raise ImageValidationError(
            f"Image too small ({img.width}×{img.height} px). "
            "Please provide a clear leaf photograph."
        )

    return img


# ──────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────

def predict(image_bytes: bytes,
            filename: str = "",
            checkpoint_path: str = None,
            top_k: int = 3) -> Dict:
    """
    Validate and classify a leaf image.

    Returns a dict:
      {
        "success": bool,
        "error": str | None,
        "predicted_class": str,         # canonical class name
        "display_name": str,
        "confidence": float,            # 0–1
        "confidence_pct": float,        # 0–100
        "top_predictions": [            # top-k predictions
          {"class": str, "display_name": str, "confidence": float}, ...
        ],
                "disease_info": {
                    "symptoms": str,
                    "management": str,
                    "is_healthy": bool,
                },
        "arch": str,
        "img_size": int,
      }
    """
    # Step 1: Validate image
    try:
        img = validate_image_bytes(image_bytes, filename)
    except ImageValidationError as e:
        return {"success": False, "error": str(e)}

    # Step 2: Load model (cached)
    try:
        model, device, arch_name = load_model(checkpoint_path)
    except (FileNotFoundError, ValueError) as e:
        return {"success": False, "error": str(e)}

    # Step 3: Preprocess
    preprocess = (
        get_mobilenet_eval_transform()
        if _is_mobilenet
        else transforms.Compose([
            transforms.Resize((_img_size, _img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )
    tensor = preprocess(img).unsqueeze(0).to(device)

    # Step 4: Inference
    with torch.no_grad():
        if device.type == "cuda":
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(tensor)
        else:
            logits = model(tensor)

    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    top_probs, top_idxs = probs.topk(min(top_k, NUM_CLASSES))

    predicted_idx = top_idxs[0].item()
    confidence = top_probs[0].item()
    predicted_class = IDX_TO_CLASS[predicted_idx]

    info = DISEASE_INFO.get(predicted_class, {
        "display_name": predicted_class,
        "symptoms": "No information available.",
        "management": "Consult an agricultural extension officer.",
    })

    top_predictions = []
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        cls = IDX_TO_CLASS[idx]
        d_info = DISEASE_INFO.get(cls, {"display_name": cls})
        top_predictions.append({
            "class": cls,
            "display_name": d_info["display_name"],
            "confidence": round(prob, 4),
            "confidence_pct": round(prob * 100, 2),
        })

    return {
        "success": True,
        "error": None,
        "predicted_class": predicted_class,
        "display_name": info["display_name"],
        "confidence": round(confidence, 4),
        "confidence_pct": round(confidence * 100, 2),
        "top_predictions": top_predictions,
        "disease_info": {
            "symptoms": info.get("symptoms", ""),
            "management": info.get("management", ""),
            "is_healthy": info.get("is_healthy", False),
        },
        "arch": arch_name,
        "img_size": _img_size,
    }
