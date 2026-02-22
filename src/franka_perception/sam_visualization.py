#!/usr/bin/env python3
"""Visualization helpers for SAM masks in a separate OpenCV window."""

from typing import List, Optional, Tuple

import numpy as np


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 RGB for stable OpenCV display."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.uint8)

    mn = float(np.min(arr[finite]))
    mx = float(np.max(arr[finite]))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)

    # Common camera case: float in [0, 1]
    if mn >= 0.0 and mx <= 1.0:
        scaled = np.clip(arr * 255.0, 0.0, 255.0)
        return scaled.astype(np.uint8)

    scaled = (arr - mn) * (255.0 / (mx - mn))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _fit_mask_to_image(mask: np.ndarray, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Fit mask to image shape (H, W), handling transpose/resize when needed."""
    h, w = int(target_hw[0]), int(target_hw[1])
    m = np.asarray(mask)
    if m.ndim > 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        return None

    if m.shape == (h, w):
        return m.astype(bool)
    if m.shape == (w, h):
        return m.T.astype(bool)

    try:
        import cv2
    except ImportError:
        return None

    resized = cv2.resize(
        m.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def build_mask_overlay(rgb_image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """Create a color overlay where each mask has a stable pseudo-random color."""
    base = _to_uint8_rgb(rgb_image)
    overlay = base.copy()
    rng = np.random.default_rng(7)
    rendered = 0
    skipped = 0
    target_hw = overlay.shape[:2]
    for mask in masks:
        fitted = _fit_mask_to_image(mask, target_hw)
        if fitted is None:
            skipped += 1
            continue
        color = rng.integers(32, 255, size=3, dtype=np.uint8)
        alpha = 0.45
        blend = np.round((1.0 - alpha) * overlay[fitted] + alpha * color).astype(np.uint8)
        overlay[fitted] = blend
        rendered += 1
    print(f"SAM overlay masks: input={len(masks)} rendered={rendered} skipped={skipped}")
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

    rgb_uint8 = _to_uint8_rgb(rgb_image)
    overlay = build_mask_overlay(rgb_uint8, masks)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"{title_prefix} RGB", rgb_bgr)
    cv2.imshow(f"{title_prefix} Masks", overlay_bgr)
    cv2.waitKey(max(15, int(wait_ms)))
    return overlay


def show_dino_and_sam(rgb_image: np.ndarray,
                      dino_boxes: Optional[np.ndarray],
                      masks: List[np.ndarray],
                      *,
                      wait_ms: int = 1,
                      title_prefix: str = "SAM") -> Optional[np.ndarray]:
    """Render DINO detections and SAM masks in two dedicated windows."""
    try:
        import cv2
    except ImportError:
        return None

    cv2.startWindowThread()

    rgb_window = f"{title_prefix} DINO"
    mask_window = f"{title_prefix} SAM"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)

    rgb_uint8 = np.ascontiguousarray(_to_uint8_rgb(rgb_image)).copy()
    dino_img = rgb_uint8.copy()
    boxes = np.asarray(dino_boxes if dino_boxes is not None else np.zeros((0, 4)))
    if boxes.ndim == 1 and boxes.size == 4:
        boxes = boxes.reshape(1, 4)

    for box in boxes:
        if box.size < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        cv2.rectangle(dino_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    overlay = build_mask_overlay(rgb_uint8, masks)
    dino_bgr = np.ascontiguousarray(cv2.cvtColor(dino_img, cv2.COLOR_RGB2BGR))
    sam_bgr = np.ascontiguousarray(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Push a few UI cycles so windows paint reliably before Open3D blocks.
    for _ in range(4):
        cv2.imshow(rgb_window, dino_bgr)
        cv2.imshow(mask_window, sam_bgr)
        cv2.waitKey(max(15, int(wait_ms)))
    return overlay
