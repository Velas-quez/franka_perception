#!/usr/bin/env python3
"""Open3D visualization helpers."""

import open3d as o3d

from .pipeline import CubeDetectionResult


def build_geometries(result: CubeDetectionResult,
                     axis_size: float = 0.1,
                     paint_cloud: bool = True,
                     show_original_cloud: bool = False):
    """Create list of Open3D geometries for rendering."""
    geometries = []
    pcd = result.original_cloud if show_original_cloud else result.filtered_cloud
    # pcd = result.original_cloud
    if paint_cloud:
        pcd = o3d.geometry.PointCloud(pcd)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geometries.append(pcd)

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
         show_original_cloud: bool = False) -> None:
    geoms = build_geometries(
        result,
        axis_size=axis_size,
        show_original_cloud=show_original_cloud,
    )
    o3d.visualization.draw_geometries(
        geoms,
        window_name="ZED Point Cloud",
        width=960,
        height=540,
    )
