#!/usr/bin/env python3
"""Helpers for Open3D tensor operations with automatic device selection."""

from typing import Tuple

import numpy as np
import open3d as o3d


def resolve_open3d_device(device: str = "auto") -> o3d.core.Device:
    """Resolve an Open3D device string with CUDA auto-detection."""
    text = str(device).strip()
    if not text or text.lower() == "auto":
        if o3d.core.cuda.is_available():
            return o3d.core.Device("CUDA:0")
        return o3d.core.Device("CPU:0")
    return o3d.core.Device(text)


def intrinsics_tensor(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: o3d.core.Device,
) -> o3d.core.Tensor:
    """Build a 3x3 intrinsics tensor on the requested device."""
    return o3d.core.Tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=o3d.core.float32,
        device=device,
    )


def tensor_image(array: np.ndarray, device: o3d.core.Device) -> o3d.t.geometry.Image:
    """Create a tensor image from a numpy array on the requested device."""
    image = np.ascontiguousarray(array)
    if image.ndim == 2:
        image = image[..., None]
    return o3d.t.geometry.Image(o3d.core.Tensor(image, device=device))


def tensor_point_cloud_from_legacy(
    point_cloud: o3d.geometry.PointCloud,
    device: o3d.core.Device,
    dtype: o3d.core.Dtype = o3d.core.float32,
) -> o3d.t.geometry.PointCloud:
    """Move a legacy point cloud into the tensor backend."""
    return o3d.t.geometry.PointCloud.from_legacy(point_cloud, dtype, device)


def voxel_downsample_legacy_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    device: str = "auto",
) -> o3d.geometry.PointCloud:
    """Downsample a legacy point cloud through the tensor backend."""
    if voxel_size <= 0.0 or len(point_cloud.points) == 0:
        return o3d.geometry.PointCloud(point_cloud)
    t_pcd = tensor_point_cloud_from_legacy(
        point_cloud,
        resolve_open3d_device(device),
    )
    return t_pcd.voxel_down_sample(float(voxel_size)).to_legacy()


def tensor_distance_filter(
    point_cloud: o3d.t.geometry.PointCloud,
    max_dist: float,
) -> o3d.t.geometry.PointCloud:
    """Keep only points closer than max_dist to the origin."""
    if max_dist <= 0.0 or int(point_cloud.point.positions.shape[0]) == 0:
        return point_cloud
    positions = point_cloud.point.positions
    distances = (positions * positions).sum(1).sqrt()
    return point_cloud.select_by_mask(distances < float(max_dist))


def run_tensor_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    device: str = "auto",
    coarse_threshold: float = 0.008,
    fine_threshold: float = 0.003,
) -> Tuple[np.ndarray, float, float]:
    """Run point-to-point ICP through the tensor backend."""
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        return init_T.copy(), 0.0, float("inf")

    resolved_device = resolve_open3d_device(device)
    source_t = tensor_point_cloud_from_legacy(source_pcd, resolved_device)
    target_t = tensor_point_cloud_from_legacy(target_pcd, resolved_device)
    init_tensor = o3d.core.Tensor(
        np.asarray(init_T, dtype=np.float64),
        dtype=o3d.core.float64,
        device=resolved_device,
    )
    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPoint()

    current_init = init_tensor
    if coarse_threshold > 0.0 and coarse_threshold > fine_threshold:
        coarse_result = o3d.t.pipelines.registration.icp(
            source_t,
            target_t,
            float(coarse_threshold),
            current_init,
            estimation,
        )
        current_init = coarse_result.transformation.to(resolved_device)

    fine_distance = float(fine_threshold if fine_threshold > 0.0 else coarse_threshold)
    fine_result = o3d.t.pipelines.registration.icp(
        source_t,
        target_t,
        fine_distance,
        current_init,
        estimation,
    )
    return (
        np.asarray(fine_result.transformation.cpu().numpy(), dtype=np.float64),
        float(fine_result.fitness),
        float(fine_result.inlier_rmse),
    )
