import os
from pathlib import Path
from typing import Optional

import numpy as np
import cv2


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sample_points_inside_mask(mask: np.ndarray, n: int = 5, seed: int = 0) -> np.ndarray:
    """
    Return n (x, y) points sampled uniformly inside the binary mask.
    Falls back to fewer points if the mask is very small.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
        return np.array([[cx, cy]])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)


def sample_negative_points_near_boundary(
    mask: np.ndarray,
    n: int = 3,
    dilation_px: int = 15,
    seed: int = 0,
) -> np.ndarray:
    """
    Return n (x, y) negative points just outside the object boundary.
    Strategy: dilate mask, subtract original mask, sample from the ring.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    ring = (dilated - mask.astype(np.uint8)).clip(0, 1)
    ys, xs = np.where(ring > 0)
    if len(xs) == 0:
        # fallback: corners of image
        h, w = mask.shape
        return np.array([[0, 0], [w - 1, 0], [0, h - 1]][:n])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
