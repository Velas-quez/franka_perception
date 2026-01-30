#!/usr/bin/env python3
"""High-level pipeline to detect cubes from point clouds."""

from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np
import open3d as o3d

from .clustering import cluster_point_cloud
from .cube_fitting import CubeEstimate, fit_cubes_in_cluster
from .filtering import filter_point_cloud


@dataclass
class CubeDetectionResult:
    filtered_cloud: o3d.geometry.PointCloud
    cluster_boxes: list
    plane_obbs: list
    cubes: List[CubeEstimate]
    failed_initial_meshes: list


class CubeDetectionPipeline:
    """Shared pipeline used by the ROS nodes."""

    def __init__(self,
                 cube_side_length: float = 0.045,
                 voxel_size: float = 0.002,
                 base_plane_distance: float = 0.01,
                 cluster_eps: float = 0.005,
                 cluster_min_points: int = 10,
                 max_cubes_per_cluster: int = 2,
                 clearance: float = 0.015) -> None:
        self.cube_side_length = cube_side_length
        self.voxel_size = voxel_size
        self.base_plane_distance = base_plane_distance
        self.cluster_eps = cluster_eps
        self.cluster_min_points = cluster_min_points
        self.max_cubes_per_cluster = max_cubes_per_cluster
        self.clearance = clearance

    def process(self, points: np.ndarray) -> CubeDetectionResult:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Apply basic filtering and table plane removal
        print("Filtering point cloud...")
        t0 = time.perf_counter()
        filtered = filter_point_cloud(pcd, voxel_size=self.voxel_size)

        _, inliers = filtered.segment_plane(distance_threshold=self.base_plane_distance,
                                            ransac_n=3,
                                            num_iterations=1000)
        filtered = filtered.select_by_index(inliers, invert=True)

        cluster_boxes, clusters = cluster_point_cloud(
            filtered, eps=self.cluster_eps, min_points=self.cluster_min_points, render_boxes=False)
        print(f"filter_point_cloud: {time.perf_counter() - t0:.3f}s")

        plane_obbs = []
        cubes: List[CubeEstimate] = []
        failed_initial_meshes: list = []
        for cluster in clusters:
            print("Clustering...")
            tcluster = time.perf_counter()
            estimates, obbs, failed_inits = fit_cubes_in_cluster(
                cluster,
                cube_side_length=self.cube_side_length,
                max_cubes=self.max_cubes_per_cluster,
                clearance=self.clearance,
                plane_distance=0.0005,
                plane_min_inliers=20,
            )
            cubes.extend(estimates)
            plane_obbs.extend(obbs)
            failed_initial_meshes.extend(failed_inits)
            print(f"cube_fitting: {time.perf_counter() - tcluster:.3f}s")

        return CubeDetectionResult(
            filtered_cloud=filtered,
            cluster_boxes=cluster_boxes,
            plane_obbs=plane_obbs,
            cubes=cubes,
            failed_initial_meshes=failed_initial_meshes,
        )
