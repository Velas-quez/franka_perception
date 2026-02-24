#!/usr/bin/env python3
"""SAM helpers for RGB mask generation."""

from dataclasses import dataclass
from typing import List, Tuple

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
        self.last_boxes = np.zeros((0, 4), dtype=np.float32)
        self.last_scores = np.zeros((0,), dtype=np.float32)

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
        self.last_boxes = np.zeros((0, 4), dtype=np.float32)
        self.last_scores = np.zeros((0,), dtype=np.float32)

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


class SamPromptedSegmenter:
    """Prompted segmentation using grounding + SAM models."""

    def __init__(self,
                 prompt_text: str = "cube.",
                 device: str = "auto",
                 grounding_model_id: str = "IDEA-Research/grounding-dino-base",
                 sam_model_id: str = "facebook/sam2-hiera-large",
                 box_threshold: float = 0.25,
                 text_threshold: float = 0.25) -> None:
        self.prompt_text = prompt_text
        self.device = device
        self.grounding_model_id = grounding_model_id
        self.sam_model_id = sam_model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self._torch = None
        self._det_processor = None
        self._det_model = None
        self._sam_processor = None
        self._sam_model = None
        self._sam_post_mode = "sam"
        self.last_boxes = np.zeros((0, 4), dtype=np.float32)
        self.last_scores = np.zeros((0,), dtype=np.float32)

    def _select_device(self) -> str:
        if self.device != "auto":
            return self.device
        if self._torch is None:
            raise RuntimeError("Torch not loaded before selecting device.")
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _build_models(self):
        if all(v is not None for v in (
                self._det_processor, self._det_model, self._sam_processor, self._sam_model)):
            return

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            try:
                from transformers import Sam2Model as _SamModel
                from transformers import Sam2Processor as _SamProcessor
                self._sam_post_mode = "sam2"
            except ImportError:
                from transformers import SamModel as _SamModel
                from transformers import SamProcessor as _SamProcessor
                self._sam_post_mode = "sam"
        except ImportError as exc:
            raise ImportError(
                "Prompted SAM dependencies missing. Install torch, transformers and pillow."
            ) from exc

        self._torch = torch
        device = self._select_device()

        self._det_processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        self._det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_id).to(device)

        sam_model_id = self.sam_model_id
        if self._sam_post_mode == "sam" and "sam2" in sam_model_id.lower():
            sam_model_id = "facebook/sam-vit-huge"
            print("SAM2 classes unavailable in transformers; falling back to facebook/sam-vit-huge")

        self._sam_processor = _SamProcessor.from_pretrained(sam_model_id)
        self._sam_model = _SamModel.from_pretrained(sam_model_id).to(device)

    @staticmethod
    def _rgb_to_pil(rgb_image: np.ndarray):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for prompted SAM mode.") from exc
        return Image.fromarray(rgb_image.astype(np.uint8), mode="RGB")

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return "cube."
        normalized = []
        for part in parts:
            normalized.append(part if part.endswith(".") else f"{part}.")
        return " ".join(normalized)

    def _grounded_boxes(self, rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._build_models()
        assert self._torch is not None
        assert self._det_processor is not None and self._det_model is not None

        image_pil = self._rgb_to_pil(rgb_image)
        prompt = self._normalize_prompt(self.prompt_text)
        device = self._select_device()

        inputs = self._det_processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(device)
        with self._torch.no_grad():
            outputs = self._det_model(**inputs)

        target_sizes = [image_pil.size[::-1]]
        results = self._det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=float(self.box_threshold),
            text_threshold=float(self.text_threshold),
            target_sizes=target_sizes,
        )[0]
        boxes = results.get("boxes")
        scores = results.get("scores")
        if boxes is None or scores is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return boxes.detach().cpu().numpy(), scores.detach().cpu().numpy()

    def _boxes_to_masks(self, rgb_image: np.ndarray, boxes_xyxy: np.ndarray) -> List[np.ndarray]:
        if boxes_xyxy.size == 0:
            return []
        assert self._sam_processor is not None and self._sam_model is not None
        assert self._torch is not None
        device = self._select_device()
        image_pil = self._rgb_to_pil(rgb_image)

        input_boxes = [boxes_xyxy.tolist()]
        sam_inputs = self._sam_processor(
            images=image_pil,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(device)
        with self._torch.no_grad():
            outputs = self._sam_model(**sam_inputs, multimask_output=False)

        original_sizes = sam_inputs.get("original_sizes")
        reshaped_input_sizes = sam_inputs.get("reshaped_input_sizes")

        if self._sam_post_mode == "sam2":
            masks_batch = self._sam_processor.post_process_masks(
                outputs.pred_masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
            )
        else:
            masks_batch = self._sam_processor.image_processor.post_process_masks(
                outputs.pred_masks,
                original_sizes=original_sizes,
                reshaped_input_sizes=reshaped_input_sizes,
            )

        if not masks_batch:
            return []

        def _to_bool_masks(tensor_like) -> List[np.ndarray]:
            arr = tensor_like
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            else:
                arr = np.asarray(arr)
            if arr.ndim < 2:
                return []
            if arr.ndim == 2:
                arr = arr[None, ...]
            elif arr.ndim > 3:
                arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))

            # Some processors return probabilities [0, 1], others logits.
            if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
                bool_arr = arr > 0.5
            else:
                bool_arr = arr > 0.0

            return [np.asarray(mask, dtype=bool) for mask in bool_arr]

        masks: List[np.ndarray] = []
        for item in masks_batch:
            masks.extend(_to_bool_masks(item))
        return masks

    def generate(self,
                 rgb_image: np.ndarray,
                 *,
                 max_masks: int = 8,
                 min_mask_pixels: int = 1200) -> List[SamMask]:
        """Generate and rank prompted masks (text -> boxes -> masks)."""
        boxes, scores = self._grounded_boxes(rgb_image)
        self.last_boxes = boxes.copy()
        self.last_scores = scores.copy()
        print(f"DINO boxes: generated={boxes.shape[0]}")
        if boxes.shape[0] == 0:
            return []

        masks_arr = self._boxes_to_masks(rgb_image, boxes)
        masks: List[SamMask] = []
        for idx, seg in enumerate(masks_arr):
            area = int(np.count_nonzero(seg))
            if area < int(min_mask_pixels):
                continue
            score = float(scores[idx]) if idx < len(scores) else 0.0
            masks.append(SamMask(segmentation=seg, area=area, score=score))

        masks.sort(key=lambda m: m.score, reverse=True)
        if max_masks > 0:
            masks = masks[:int(max_masks)]
        return masks


class SamSegmenter:
    """Unified segmenter wrapper for automatic or prompted SAM modes."""

    def __init__(self,
                 mode: str = "sam3",
                 checkpoint_path: str = "",
                 model_type: str = "vit_b",
                 device: str = "auto",
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.92,
                 min_mask_region_area: int = 150,
                 prompt_text: str = "cube.",
                 prompt_box_threshold: float = 0.25,
                 prompt_text_threshold: float = 0.25,
                 grounding_model_id: str = "IDEA-Research/grounding-dino-base",
                 sam_model_id: str = "facebook/sam2-hiera-large") -> None:
        self.mode = mode.strip().lower()
        if self.mode in {"sam3", "prompted", "text_prompt"}:
            self._impl = SamPromptedSegmenter(
                prompt_text=prompt_text,
                device=device,
                grounding_model_id=grounding_model_id,
                sam_model_id=sam_model_id,
                box_threshold=prompt_box_threshold,
                text_threshold=prompt_text_threshold,
            )
        elif self.mode in {"sam1", "automatic", "auto"}:
            self._impl = SamAutomaticSegmenter(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                device=device,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
        else:
            raise ValueError(
                "Invalid sam_mode. Use sam3|prompted|text_prompt or sam1|automatic|auto.")

    def generate(self,
                 rgb_image: np.ndarray,
                 *,
                 max_masks: int = 8,
                 min_mask_pixels: int = 1200) -> List[SamMask]:
        return self._impl.generate(
            rgb_image,
            max_masks=max_masks,
            min_mask_pixels=min_mask_pixels,
        )

    def get_last_boxes(self) -> np.ndarray:
        boxes = getattr(self._impl, "last_boxes", None)
        if boxes is None:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32)


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
