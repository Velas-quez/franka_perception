#!/usr/bin/env python3
"""SAM-based segmentation utilities for RGB frames."""

from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np


@dataclass
class SamMask:
    """Container for one SAM mask and its metadata."""
    mask: np.ndarray
    area: int
    score: float
    bbox_xywh: tuple


class SamSegmenter:
    """Thin wrapper over Segment Anything automatic mask generation."""

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        verbose: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.verbose = verbose
        self._generator = None

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:  # pragma: no cover - depends on external install
            raise ImportError(
                "segment_anything is required. "
                "Install it before using SamSegmenter."
            ) from exc

        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        model.to(device=self.device)
        self._generator = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )
        if self.verbose:
            print(f"[SAM] model={self.model_type} device={self.device} loaded")

    def generate_masks(
        self,
        rgb_image: np.ndarray,
        *,
        min_area_pixels: int = 200,
        max_masks: Optional[int] = None,
    ) -> List[SamMask]:
        """Generate masks sorted by area (largest first)."""
        self._ensure_generator()
        image = _validate_rgb(rgb_image)
        t0 = time.perf_counter()
        raw_masks = self._generator.generate(image)
        if self.verbose:
            print(f"[SAM] raw masks={len(raw_masks)} in {time.perf_counter() - t0:.3f}s")

        masks: List[SamMask] = []
        for item in raw_masks:
            mask = np.asarray(item["segmentation"], dtype=bool)
            area = int(item.get("area", int(mask.sum())))
            if area < min_area_pixels:
                continue
            masks.append(
                SamMask(
                    mask=mask,
                    area=area,
                    score=float(item.get("predicted_iou", 0.0)),
                    bbox_xywh=tuple(item.get("bbox", (0, 0, 0, 0))),
                )
            )

        masks.sort(key=lambda m: m.area, reverse=True)
        if max_masks is not None:
            masks = masks[:max_masks]
        if self.verbose:
            print(f"[SAM] filtered masks={len(masks)} min_area={min_area_pixels} max_masks={max_masks}")
        return masks


def _validate_rgb(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (H, W, 3)")
    if rgb_image.dtype == np.uint8:
        return rgb_image
    clipped = np.clip(rgb_image, 0, 255)
    return clipped.astype(np.uint8)
