#!/usr/bin/env python3
"""High-level SAM + RGB-D clustering pipeline."""

from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np
import open3d as o3d

from .pipeline import CubeDetectionPipeline, CubeDetectionResult
from .rgbd_masking import (
    CameraIntrinsics,
    erode_binary_masks,
    masked_depth_to_point_cloud_clusters,
)
from .sam_segmentation import SamMask, SamSegmenter


@dataclass
class SamRgbdDebug:
    raw_masks: List[SamMask]
    eroded_masks: List[np.ndarray]
    clusters: List[o3d.geometry.PointCloud]


class SamRgbdCubePipeline:
    """Run SAM masks on RGB, project with depth, then reuse cube fitting."""

    def __init__(self,
                 cube_pipeline: Optional[CubeDetectionPipeline] = None,
                 verbose: bool = False) -> None:
        self.cube_pipeline = cube_pipeline or CubeDetectionPipeline()
        self.verbose = verbose

    def process(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: CameraIntrinsics,
        segmenter: SamSegmenter,
        *,
        stop_after: str = "all",
        min_area_pixels: int = 200,
        max_masks: Optional[int] = None,
        erosion_kernel: int = 3,
        erosion_iterations: int = 1,
        depth_scale: float = 1.0,
        depth_trunc: float = 3.0,
        min_cluster_points: int = 30,
        flip: bool = True,
    ):
        """Execute RGB-D segmentation pipeline and return detection result + debug."""
        stage = stop_after.lower()
        if stage not in {"cluster", "all"}:
            raise ValueError(f"Invalid stop_after '{stop_after}'. Choose from: cluster, all.")

        t0 = time.perf_counter()
        if self.verbose:
            print(f"[SAM-RGBD] input rgb={rgb_image.shape} depth={depth_image.shape} stage={stage}")

        raw_masks = segmenter.generate_masks(
            rgb_image,
            min_area_pixels=min_area_pixels,
            max_masks=max_masks,
        )
        if self.verbose:
            print(f"[SAM-RGBD] masks after SAM={len(raw_masks)}")

        terode = time.perf_counter()
        eroded_masks = erode_binary_masks(
            [m.mask for m in raw_masks],
            kernel_size=erosion_kernel,
            iterations=erosion_iterations,
        )
        if self.verbose:
            print(f"[SAM-RGBD] erosion masks={len(eroded_masks)} in {time.perf_counter() - terode:.3f}s")

        tproj = time.perf_counter()
        clusters, _ = masked_depth_to_point_cloud_clusters(
            depth_image,
            eroded_masks,
            intrinsics,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            min_points=min_cluster_points,
            flip=flip,
        )
        if self.verbose:
            points_per_cluster = [len(c.points) for c in clusters]
            print(f"[SAM-RGBD] 3D clusters={len(clusters)} points={points_per_cluster} "
                  f"in {time.perf_counter() - tproj:.3f}s")
        cluster_boxes = _make_cluster_boxes(clusters)
        result = self.cube_pipeline.process_preclustered_clusters(
            clusters=clusters,
            stop_after=stage,
            cluster_boxes=cluster_boxes,
        )
        if self.verbose:
            print(f"[SAM-RGBD] cubes={len(result.cubes)} total={time.perf_counter() - t0:.3f}s")
        debug = SamRgbdDebug(raw_masks=raw_masks, eroded_masks=eroded_masks, clusters=clusters)
        return result, debug


def _make_cluster_boxes(clusters: List[o3d.geometry.PointCloud]) -> list:
    boxes = []
    for cluster in clusters:
        if len(cluster.points) == 0:
            continue
        bbox = cluster.get_oriented_bounding_box()
        bbox.color = (0.2, 0.8, 0.2)
        boxes.append(bbox)
    return boxes
