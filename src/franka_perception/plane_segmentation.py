#!/usr/bin/env python3
"""Plane segmentation utilities for clustered point clouds."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d


@dataclass
class PlaneDetection:
    cloud: o3d.geometry.PointCloud
    model: np.ndarray  # plane coefficients a, b, c, d
    center: np.ndarray
    normal: np.ndarray
    axis_index: Optional[int] = None
    sign: Optional[float] = None


def segment_planes(pcd: o3d.geometry.PointCloud,
                   distance_threshold: float = 0.001,
                   ransac_n: int = 3,
                   num_iterations: int = 300,
                   min_inliers: int = 20,
                   min_ratio: float = 0.02,
                   max_planes: int = 6) -> List[PlaneDetection]:
    """Extract dominant planes from a point cloud using iterative RANSAC."""
    planes: List[PlaneDetection] = []
    if pcd.is_empty():
        return planes

    remaining = pcd
    total_points = len(remaining.points)
    for _ in range(max_planes):
        if len(remaining.points) < max(min_inliers, ransac_n):
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations)
        if len(inliers) < min_inliers or len(inliers) < min_ratio * total_points:
            break
        plane_cloud = remaining.select_by_index(inliers)
        points = np.asarray(plane_cloud.points)
        if points.size == 0:
            break
        center = points.mean(axis=0)
        normal = np.asarray(plane_model[:3], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-9
        planes.append(
            PlaneDetection(
                cloud=plane_cloud,
                model=np.asarray(plane_model, dtype=float),
                center=center,
                normal=normal,
            )
        )
        remaining = remaining.select_by_index(inliers, invert=True)
        if len(remaining.points) < min_inliers:
            break

    return planes
