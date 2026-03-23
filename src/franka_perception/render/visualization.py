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
    """Create non-cube and cube geometry lists for rendering."""
    geometries = []
    has_masked = result.masked_cloud is not None and len(result.masked_cloud.points) > 0
    if has_masked:
        masked_pcd = o3d.geometry.PointCloud(result.masked_cloud)
        if paint_cloud:
            masked_pcd.paint_uniform_color([0.95, 0.2, 0.2])
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

    if result.plane_inlier_cloud is not None and len(result.plane_inlier_cloud.points) > 0:
        plane_pcd = o3d.geometry.PointCloud(result.plane_inlier_cloud)
        plane_pcd.paint_uniform_color([0.0, 0.0, 0.0])
        geometries.append(plane_pcd)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(axis_size),
        origin=[0.0, 0.0, 0.0],
    )
    geometries.append(axis)
    geometries.extend(result.cluster_boxes)
    # geometries.extend(result.plane_obbs)
    initial_meshes = list(result.failed_initial_meshes)
    initial_meshes.extend([c.initial_mesh for c in result.cubes if c.initial_mesh is not None])
    cube_meshes = [c.mesh for c in result.cubes]
    return geometries, cube_meshes, initial_meshes


def draw(result: CubeDetectionResult,
         axis_size: float = 0.1,
         show_original_cloud: bool = False,
         extra_clouds: Optional[Iterable[Tuple[np.ndarray, Sequence[float]]]] = None) -> None:
    base_geoms, cube_geoms, initial_geoms = build_geometries(
        result,
        axis_size=axis_size,
        show_original_cloud=show_original_cloud,
        extra_clouds=extra_clouds,
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="ZED Point Clouds", width=960, height=540)

    for geom in base_geoms:
        vis.add_geometry(geom, reset_bounding_box=False)
    for geom in cube_geoms:
        vis.add_geometry(geom, reset_bounding_box=False)

    if base_geoms or cube_geoms:
        vis.reset_view_point(True)

    state = {"cubes_visible": True, "initial_visible": False}

    def _toggle_cubes(_vis):
        state["cubes_visible"] = not state["cubes_visible"]
        for geom in cube_geoms:
            if state["cubes_visible"]:
                _vis.add_geometry(geom, reset_bounding_box=False)
            else:
                _vis.remove_geometry(geom, reset_bounding_box=False)
        return False

    def _toggle_initial(_vis):
        state["initial_visible"] = not state["initial_visible"]
        for geom in initial_geoms:
            if state["initial_visible"]:
                _vis.add_geometry(geom, reset_bounding_box=False)
            else:
                _vis.remove_geometry(geom, reset_bounding_box=False)
        return False

    vis.register_key_callback(ord("C"), _toggle_cubes)
    vis.register_key_callback(ord("c"), _toggle_cubes)
    vis.register_key_callback(ord("I"), _toggle_initial)
    vis.register_key_callback(ord("i"), _toggle_initial)
    vis.run()
    vis.destroy_window()
