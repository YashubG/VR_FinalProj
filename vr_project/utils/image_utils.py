"""
utils/image_utils.py
---------------------
Pure image-processing helpers used across the pipeline.
No model state lives here — only stateless functions.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps


# ─────────────────────────────────────────────────────────────────────────────
# Basic I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(str(path)).convert("RGB")
    return img


def save_image(img: Image.Image, path: str | Path, quality: int = 95) -> None:
    """Save a PIL image to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=quality)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image (H×W×3, uint8) → float tensor (1×3×H×W, [0,1])."""
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a float tensor (1×3×H×W or 3×H×W) back to a PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP pre-processing transform
# ─────────────────────────────────────────────────────────────────────────────

def get_clip_transform(image_size: int = 224) -> T.Compose:
    """
    Returns the standard CLIP pre-processing transform.

    Rationale: CLIP was trained with centre-crop + specific normalisation stats.
    Using exactly these stats keeps embeddings in the same distribution as
    those seen during CLIP pre-training.
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Crop helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_box(
    img: Image.Image,
    box: Tuple[float, float, float, float],
    padding: float = 0.05,
) -> Image.Image:
    """
    Crop a PIL image to a bounding box with optional proportional padding.

    Parameters
    ----------
    img     : Source PIL image.
    box     : (x1, y1, x2, y2) in pixel coordinates.
    padding : Fractional padding added to each side (relative to box size).
              A value of 0.05 adds 5 % of the box dimensions on each edge.
              Padding helps retain some context around the detected product.

    Returns
    -------
    Cropped PIL image.
    """
    W, H = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def safe_crop_or_full(
    img: Image.Image,
    box: Optional[Tuple[float, float, float, float]],
) -> Image.Image:
    """
    Crop if a valid box is provided; otherwise return the full image.
    This is the fallback used when YOLO finds no detection.
    """
    if box is None:
        return img
    return crop_box(img, box)


# ─────────────────────────────────────────────────────────────────────────────
# Normalise / denormalise embeddings
# ─────────────────────────────────────────────────────────────────────────────

def l2_normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a 2-D array (or a 1-D vector)."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-10)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Resize utility used by Streamlit
# ─────────────────────────────────────────────────────────────────────────────

def resize_for_display(img: Image.Image, max_size: int = 512) -> Image.Image:
    """Proportionally resize an image so its longest side ≤ max_size."""
    W, H = img.size
    if max(W, H) <= max_size:
        return img
    scale = max_size / max(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
