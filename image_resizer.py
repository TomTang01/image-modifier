from __future__ import annotations

import numpy as np
from PIL import Image


RESIZE_MODE_KEEP_ASPECT = "Keep aspect ratio"
RESIZE_MODE_FREE = "Free resize"
RESIZE_MODE_CROP = "Crop to size"
RESIZE_MODES = [RESIZE_MODE_KEEP_ASPECT, RESIZE_MODE_FREE, RESIZE_MODE_CROP]
RESIZE_SCALES = [0.25, 0.5, 1.5, 2.0, 4.0]


def scaled_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_image(image: np.ndarray, target_width: int, target_height: int, mode: str) -> np.ndarray:
    image_uint8 = _to_rgb_uint8(image)
    target_width = max(1, int(target_width))
    target_height = max(1, int(target_height))

    if mode == RESIZE_MODE_FREE:
        return _pil_resize(image_uint8, target_width, target_height)
    if mode == RESIZE_MODE_KEEP_ASPECT:
        return _resize_keep_aspect(image_uint8, target_width, target_height)
    if mode == RESIZE_MODE_CROP:
        return _resize_crop_to_size(image_uint8, target_width, target_height)

    raise ValueError(f"Unknown resize mode: {mode}")


def _resize_keep_aspect(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    scale = min(target_width / src_width, target_height / src_height)
    resized_width, resized_height = scaled_dimensions(src_width, src_height, scale)
    return _pil_resize(image, resized_width, resized_height)


def _resize_crop_to_size(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    scale = max(target_width / src_width, target_height / src_height)
    resized_width, resized_height = scaled_dimensions(src_width, src_height, scale)
    resized = _pil_resize(image, resized_width, resized_height)

    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return resized[top : top + target_height, left : left + target_width]


def _pil_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((width, height), _lanczos_resample())
    return np.asarray(resized, dtype=np.uint8)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[2] == 4:
        alpha = arr[..., 3:4].astype(float)
        rgb = arr[..., :3].astype(float)
        if alpha.max(initial=0) > 1.0:
            alpha = alpha / 255.0
        if rgb.max(initial=0) > 1.0:
            rgb = rgb / 255.0
        arr = rgb * alpha + (1 - alpha)
    else:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(float)
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255
    return np.clip(arr, 0, 255).astype(np.uint8)


def _lanczos_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS
