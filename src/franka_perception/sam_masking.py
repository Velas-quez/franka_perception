#!/usr/bin/env python3
"""SAM3 helpers for prompt-based RGB mask generation."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SamMask:
    """Normalized SAM mask payload used by the RGB-D pipeline."""
    segmentation: np.ndarray
    area: int
    score: float


class Sam3PromptSegmenter:
    """Wrapper around SAM3 prompt segmentation."""

    def __init__(self,
                 model_id: str = "facebook/sam3.1-hiera-large",
                 device: str = "auto",
                 score_threshold: float = 0.0) -> None:
        self.model_id = model_id
        self.device = device
        self.score_threshold = score_threshold
        self._processor = None
        self._model = None
        self._torch = None
        self._runtime_device = None

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            from PIL import Image
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise ImportError(
                "SAM3 dependencies missing. Install torch, transformers and pillow.") from exc

        if self.device == "auto":
            runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            runtime_device = self.device

        model_dtype = torch.float16 if runtime_device == "cuda" else torch.float32
        self._processor = Sam3Processor.from_pretrained(self.model_id)
        self._model = Sam3Model.from_pretrained(self.model_id, torch_dtype=model_dtype)
        self._model.to(runtime_device)
        self._model.eval()
        self._torch = torch
        self._runtime_device = runtime_device
        self._pil_image_cls = Image

    def generate(self,
                 rgb_image: np.ndarray,
                 *,
                 prompt: str = "cube",
                 max_masks: int = 8,
                 min_mask_pixels: int = 1200) -> List[SamMask]:
        """Generate and rank SAM3 masks from text prompt(s)."""
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._runtime_device is not None

        prompts = [p.strip() for p in str(prompt).split(",") if p.strip()]
        if not prompts:
            prompts = ["cube"]

        image = self._pil_image_cls.fromarray(rgb_image)
        image_inputs = self._processor(images=image, return_tensors="pt")
        original_sizes = image_inputs["original_sizes"].cpu()
        reshaped_sizes = image_inputs["reshaped_input_sizes"].cpu()
        pixel_values = image_inputs["pixel_values"].to(self._runtime_device)
        with self._torch.inference_mode():
            image_embeddings = self._model.get_image_embeddings(pixel_values)

        masks: List[SamMask] = []
        for text_prompt in prompts:
            self._processor.set_image_embeddings(image_embeddings)
            self._processor.set_text_prompt([text_prompt])
            prompt_inputs = self._processor()
            prompt_inputs = {
                key: value.to(self._runtime_device) if hasattr(value, "to") else value
                for key, value in prompt_inputs.items()
            }
            with self._torch.inference_mode():
                outputs = self._model(**prompt_inputs, multimask_output=False)

            pred_masks = outputs.pred_masks.detach().cpu()
            iou_scores = outputs.iou_scores.detach().cpu().numpy()
            processed = self._processor.post_process_masks(
                pred_masks,
                original_sizes,
                reshaped_sizes,
            )
            if not processed:
                continue

            raw = processed[0]
            if hasattr(raw, "numpy"):
                raw = raw.numpy()
            raw = np.asarray(raw)
            if raw.ndim == 2:
                raw = raw[None, ...]

            score_values = np.asarray(iou_scores).reshape(-1)
            for idx, mask_arr in enumerate(raw):
                seg = np.asarray(mask_arr > 0, dtype=bool)
                area = int(seg.sum())
                if area < int(min_mask_pixels):
                    continue
                score = float(score_values[idx]) if idx < score_values.size else 0.0
                if score < float(self.score_threshold):
                    continue
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
