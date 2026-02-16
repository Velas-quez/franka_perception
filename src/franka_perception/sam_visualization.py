#!/usr/bin/env python3
"""Visualization helpers for SAM masks in a separate OpenCV window."""

from typing import List, Optional

import numpy as np


def build_mask_overlay(rgb_image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """Create a color overlay where each mask has a stable pseudo-random color."""
    overlay = rgb_image.copy()
    rng = np.random.default_rng(7)
    for mask in masks:
        color = rng.integers(32, 255, size=3, dtype=np.uint8)
        alpha = 0.45
        blend = np.round((1.0 - alpha) * overlay[mask] + alpha * color).astype(np.uint8)
        overlay[mask] = blend
    return overlay


def show_rgb_and_masks(rgb_image: np.ndarray,
                       masks: List[np.ndarray],
                       *,
                       wait_ms: int = 1,
                       title_prefix: str = "SAM") -> Optional[np.ndarray]:
    """Render RGB and SAM-mask overlay in two OpenCV windows."""
    try:
        import cv2
    except ImportError:
        return None

    overlay = build_mask_overlay(rgb_image, masks)
    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"{title_prefix} RGB", rgb_bgr)
    cv2.imshow(f"{title_prefix} Masks", overlay_bgr)
    cv2.waitKey(max(1, int(wait_ms)))
    return overlay
