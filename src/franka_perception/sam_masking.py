#!/usr/bin/env python3
"""SAM helpers for RGB mask generation."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SamMask:
    """Normalized SAM mask payload used by the RGB-D pipeline."""
    segmentation: np.ndarray
    area: int
    score: float


class SamAutomaticSegmenter:
    """Wrapper around Segment Anything automatic mask generator."""

    def __init__(self,
                 checkpoint_path: str,
                 model_type: str = "vit_b",
                 device: str = "auto",
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.92,
                 min_mask_region_area: int = 150) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self._generator = None

    def _build_generator(self):
        if self._generator is not None:
            return self._generator
        if not self.checkpoint_path:
            raise ValueError(
                "Missing SAM checkpoint. Set ~sam_checkpoint_path to a valid .pth file.")

        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "SAM dependencies missing. Install torch and segment-anything.") from exc

        if self.model_type not in sam_model_registry:
            valid = ", ".join(sorted(sam_model_registry.keys()))
            raise ValueError(f"Invalid sam_model_type '{self.model_type}'. Valid: {valid}")

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=device)
        self._generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=int(self.points_per_side),
            pred_iou_thresh=float(self.pred_iou_thresh),
            stability_score_thresh=float(self.stability_score_thresh),
            min_mask_region_area=int(self.min_mask_region_area),
        )
        return self._generator

    def generate(self,
                 rgb_image: np.ndarray,
                 *,
                 max_masks: int = 8,
                 min_mask_pixels: int = 1200) -> List[SamMask]:
        """Generate and rank SAM masks."""
        generator = self._build_generator()
        raw_masks = generator.generate(rgb_image)

        masks: List[SamMask] = []
        for entry in raw_masks:
            seg = np.asarray(entry.get("segmentation"), dtype=bool)
            area = int(seg.sum())
            if area < int(min_mask_pixels):
                continue
            score = float(entry.get("predicted_iou", 0.0)) * float(
                entry.get("stability_score", 0.0))
            masks.append(SamMask(segmentation=seg, area=area, score=score))

        masks.sort(key=lambda m: m.score, reverse=True)
        if max_masks > 0:
            masks = masks[:int(max_masks)]
        return masks


def erode_mask(mask: np.ndarray,
               kernel_size: int = 3,
               iterations: int = 1) -> np.ndarray:
    """Apply binary erosion to reduce boundary noise from masks."""
    if kernel_size <= 1 or iterations <= 0:
        return mask.astype(bool)

    try:
        import cv2
    except ImportError:
        return mask.astype(bool)

    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=int(iterations))
    return eroded.astype(bool)


def select_mask_candidates(
    masks: List[SamMask],
    depth_m: np.ndarray,
    *,
    max_masks: int,
    min_depth_pixels: int,
) -> List[SamMask]:
    """Filter masks by valid depth support so only 3D-usable masks remain."""
    valid_masks: List[SamMask] = []
    depth_valid = np.isfinite(depth_m) & (depth_m > 0.0)
    for mask in masks:
        depth_pixels = int(np.count_nonzero(mask.segmentation & depth_valid))
        if depth_pixels < int(min_depth_pixels):
            continue
        valid_masks.append(mask)
        if max_masks > 0 and len(valid_masks) >= int(max_masks):
            break
    return valid_masks
