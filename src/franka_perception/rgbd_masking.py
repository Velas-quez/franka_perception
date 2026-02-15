#!/usr/bin/env python3
"""Mask post-processing and RGB-D projection helpers."""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def erode_binary_masks(
    masks: Iterable[np.ndarray],
    kernel_size: int = 3,
    iterations: int = 1,
) -> List[np.ndarray]:
    """Apply morphological erosion to binary masks."""
    if kernel_size <= 1 or iterations <= 0:
        return [np.asarray(m, dtype=bool) for m in masks]

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("opencv-python (cv2) is required for mask erosion") from exc

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = []
    for mask in masks:
        m = np.asarray(mask, dtype=np.uint8)
        e = cv2.erode(m, kernel, iterations=iterations)
        eroded.append(e.astype(bool))
    return eroded


def masked_depth_to_point_cloud_clusters(
    depth_image: np.ndarray,
    masks: Iterable[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float = 1.0,
    depth_trunc: float = 3.0,
    min_points: int = 30,
    flip: bool = True,
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
    """Project each mask to a 3D point cluster using the aligned depth image."""
    depth = _validate_depth(depth_image)
    clusters: List[o3d.geometry.PointCloud] = []
    cluster_points: List[np.ndarray] = []
    for mask in masks:
        points = masked_depth_to_points(
            depth,
            np.asarray(mask, dtype=bool),
            intrinsics,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            flip=flip,
        )
        if points.shape[0] < min_points:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        clusters.append(pcd)
        cluster_points.append(points)
    return clusters, cluster_points


def masked_depth_to_points(
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float = 1.0,
    depth_trunc: float = 3.0,
    flip: bool = True,
) -> np.ndarray:
    """Project one binary mask from image space to 3D camera coordinates."""
    if depth_scale <= 0.0:
        raise ValueError("depth_scale must be > 0")
    if depth_image.shape != mask.shape:
        raise ValueError("depth_image and mask must have the same shape")
    if depth_image.shape[0] != intrinsics.height or depth_image.shape[1] != intrinsics.width:
        raise ValueError("intrinsics dimensions must match image shape")

    depth_m = depth_image.astype(np.float64) / depth_scale
    valid = mask & np.isfinite(depth_m) & (depth_m > 1e-6) & (depth_m < depth_trunc)
    v, u = np.where(valid)
    if u.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_m[v, u]
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))

    if flip:
        points[:, 1] *= -1.0
        points[:, 2] *= -1.0
    return points


def camera_info_to_intrinsics(camera_info) -> CameraIntrinsics:
    """Convert a ROS CameraInfo-like object to CameraIntrinsics."""
    return CameraIntrinsics(
        fx=float(camera_info.K[0]),
        fy=float(camera_info.K[4]),
        cx=float(camera_info.K[2]),
        cy=float(camera_info.K[5]),
        width=int(camera_info.width),
        height=int(camera_info.height),
    )


def _validate_depth(depth_image: np.ndarray) -> np.ndarray:
    if depth_image.ndim != 2:
        raise ValueError("depth_image must have shape (H, W)")
    return depth_image
