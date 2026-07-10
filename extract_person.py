from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SAM_CHECKPOINT = Path(__file__).resolve().parent / "models" / "sam_vit_b_01ec64.pth"
DEFAULT_SAM_MODEL_TYPE = "vit_b"


@dataclass
class ExtractResult:
    rgba: np.ndarray
    mask: np.ndarray
    diagnostics: dict[str, object]


class SamSetupError(RuntimeError):
    """Raised when SAM dependencies or checkpoints are unavailable."""


def extract_person(
    image: np.ndarray,
    *,
    rect_margin: float = 0.08,
    feather_radius: int = 3,
    max_side: int = 1024,
    checkpoint_path: str | Path | None = None,
) -> ExtractResult:
    """Extract a central person-like subject with Segment Anything."""
    rgb = to_uint8_rgb(image)
    height, width = rgb.shape[:2]
    checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_SAM_CHECKPOINT
    predictor, device_info = load_sam_predictor(str(checkpoint))
    resized, scale = resize_for_sam(rgb, max_side)
    prompt_box = prompt_box_from_margin(resized.shape[1], resized.shape[0], rect_margin)

    predictor.set_image(resized)
    masks, scores, _ = predictor.predict(
        box=np.array(prompt_box, dtype=float),
        multimask_output=True,
    )
    mask = choose_sam_mask(masks, scores)
    if scale != 1.0:
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

    binary = (mask.astype(np.uint8) * 255)
    binary = clean_mask(binary)
    alpha = feather_mask(binary, feather_radius)
    rgba = np.dstack((rgb, alpha))
    diagnostics = {
        "method": "segment-anything",
        "model_type": DEFAULT_SAM_MODEL_TYPE,
        **device_info,
        "checkpoint": str(checkpoint),
        "width": int(width),
        "height": int(height),
        "processed_width": int(resized.shape[1]),
        "processed_height": int(resized.shape[0]),
        "prompt_x0": int(prompt_box[0]),
        "prompt_y0": int(prompt_box[1]),
        "prompt_x1": int(prompt_box[2]),
        "prompt_y1": int(prompt_box[3]),
        "sam_score": float(np.max(scores)),
        "feather_radius": int(feather_radius),
        "max_processing_side": int(max_side),
        "foreground_percent": float(np.mean(binary > 0) * 100),
    }
    return ExtractResult(rgba=rgba, mask=alpha, diagnostics=diagnostics)


@lru_cache(maxsize=1)
def load_sam_predictor(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise SamSetupError(
            f"SAM checkpoint not found: {checkpoint}. Download sam_vit_b_01ec64.pth into the models folder."
        )

    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise SamSetupError(
            "Segment Anything is not installed. Install torch and segment-anything to use the extract tool."
        ) from exc

    device_info = select_torch_device(torch)
    sam = sam_model_registry[DEFAULT_SAM_MODEL_TYPE](checkpoint=str(checkpoint))
    sam.to(device=device_info["device"])
    return SamPredictor(sam), device_info


def select_torch_device(torch_module) -> dict[str, object]:
    """Choose CUDA before CPU and report enough detail for the UI diagnostics."""
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    if cuda_available and device_count > 0:
        index = int(torch_module.cuda.current_device())
        return {
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_count": device_count,
            "cuda_device_index": index,
            "cuda_device_name": torch_module.cuda.get_device_name(index),
            "torch_version": getattr(torch_module, "__version__", "unknown"),
            "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        }

    return {
        "device": "cpu",
        "cuda_available": False,
        "cuda_device_count": device_count,
        "cuda_device_index": None,
        "cuda_device_name": None,
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
    }


def resize_for_sam(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    max_side = int(max(1, max_side))
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, 1.0
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def prompt_box_from_margin(width: int, height: int, rect_margin: float) -> np.ndarray:
    margin = float(np.clip(rect_margin, 0.01, 0.35))
    margin_x = max(1, int(width * margin))
    margin_y = max(1, int(height * margin))
    return np.array(
        [
            margin_x,
            margin_y,
            max(margin_x + 1, width - margin_x - 1),
            max(margin_y + 1, height - margin_y - 1),
        ],
        dtype=float,
    )


def choose_sam_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    areas = np.mean(masks, axis=(1, 2))
    valid = (areas > 0.01) & (areas < 0.95)
    if np.any(valid):
        valid_indices = np.flatnonzero(valid)
        return masks[valid_indices[np.argmax(scores[valid_indices])]]
    return masks[int(np.argmax(scores))]


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(max(0, radius))
    if radius == 0:
        return mask
    kernel_size = 2 * radius + 1
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]
    else:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255
    return np.clip(arr, 0, 255).astype(np.uint8)
