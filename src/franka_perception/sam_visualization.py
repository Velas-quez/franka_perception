#!/usr/bin/env python3
"""Visualization helpers for SAM RGB-D debugging."""

from typing import Iterable

import numpy as np


def overlay_masks(rgb_image: np.ndarray, masks: Iterable[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    """Return RGB image with colored binary masks overlaid."""
    out = rgb_image.copy().astype(np.float32)
    rng = np.random.default_rng(7)
    for mask in masks:
        color = rng.integers(0, 255, size=3, dtype=np.int32).astype(np.float32)
        m = np.asarray(mask, dtype=bool)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def show_masks_window(
    rgb_image: np.ndarray,
    masks: Iterable[np.ndarray],
    *,
    window_name: str = "SAM Masks",
    alpha: float = 0.45,
    wait_ms: int = 1,
) -> None:
    """Render one OpenCV window with the RGB image overlaid by SAM masks."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("opencv-python (cv2) is required for SAM debug window") from exc

    overlay = overlay_masks(rgb_image, masks, alpha=alpha)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, bgr)
    cv2.waitKey(wait_ms)
