#!/usr/bin/env python3
"""Open3D visualization helpers."""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from franka_perception.pipelines.result_type import CubeDetectionResult


def _painted_cloud_from_points(points: np.ndarray,
                               color: Sequence[float]) -> o3d.geometry.PointCloud:
    """Create a colored point cloud from an Nx3 numpy array."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color(list(color))
    return cloud


def build_geometries(result: CubeDetectionResult,
                     axis_size: float = 0.1,
                     paint_cloud: bool = True,
                     show_original_cloud: bool = False,
                     extra_clouds: Optional[Iterable[Tuple[np.ndarray, Sequence[float]]]] = None):
    """Create list of Open3D geometries for rendering."""
    geometries = []
    has_masked = result.masked_cloud is not None and len(result.masked_cloud.points) > 0
    if has_masked:
        full_pcd = o3d.geometry.PointCloud(result.original_cloud)
        masked_pcd = o3d.geometry.PointCloud(result.masked_cloud)
        if paint_cloud:
            full_pcd.paint_uniform_color([0.55, 0.55, 0.55])
            masked_pcd.paint_uniform_color([0.95, 0.2, 0.2])
        geometries.append(full_pcd)
        geometries.append(masked_pcd)
    else:
        pcd = result.original_cloud if show_original_cloud else result.filtered_cloud
        if paint_cloud:
            pcd = o3d.geometry.PointCloud(pcd)
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geometries.append(pcd)

    for extra_cloud in extra_clouds or ():
        points, color = extra_cloud
        if points is None or points.size == 0:
            continue
        geometries.append(_painted_cloud_from_points(points, color))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(axis_size),
        origin=[0.0, 0.0, 0.0],
    )
    geometries.append(axis)
    geometries.extend(result.cluster_boxes)
    # geometries.extend(result.plane_obbs)
    # geometries.extend(result.failed_initial_meshes)
    # geometries.extend([c.initial_mesh for c in result.cubes if c.initial_mesh is not None])
    geometries.extend([c.mesh for c in result.cubes])
    return geometries


def draw(result: CubeDetectionResult,
         axis_size: float = 0.1,
         show_original_cloud: bool = False,
         extra_clouds: Optional[Iterable[Tuple[np.ndarray, Sequence[float]]]] = None) -> None:
    geoms = build_geometries(
        result,
        axis_size=axis_size,
        show_original_cloud=show_original_cloud,
        extra_clouds=extra_clouds,
    )
    o3d.visualization.draw_geometries(
        geoms,
        window_name="ZED Point Clouds",
        width=960,
        height=540,
    )
