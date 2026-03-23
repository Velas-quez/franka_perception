#!/usr/bin/env python3
"""High-level pipeline to detect cubes from point clouds."""

from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np
import open3d as o3d

from franka_perception.pipelines.result_type import CubeDetectionResult

from ..geometry.clustering import cluster_point_cloud
from ..geometry.cube_fitting import CubeEstimate, fit_cubes_in_cluster, select_best_cubes
from ..geometry.filtering import filter_point_cloud


class CubeDetectionPipeline:
    """Shared pipeline used by the ROS nodes."""

    def __init__(self,
                 cube_side_length: float = 0.045,
                 voxel_size: float = 0.002,
                 base_plane_distance: float = 0.01,
                 cluster_eps: float = 0.005,
                 cluster_min_points: int = 10,
                 max_cubes_per_cluster: int = 2,
                 num_best_cubes: int = 2,
                 clearance: float = 0.015,
                 max_cluster_distance_from_plane_inliers: float = 0.08,
                 below_plane_tolerance: float = 0.002,
                 support_plane_constraint: bool = True) -> None:
        self.cube_side_length = cube_side_length
        self.voxel_size = voxel_size
        self.base_plane_distance = base_plane_distance
        self.cluster_eps = cluster_eps
        self.cluster_min_points = cluster_min_points
        self.max_cubes_per_cluster = max_cubes_per_cluster
        self.num_best_cubes = num_best_cubes
        self.clearance = clearance
        self.max_cluster_distance_from_plane_inliers = max_cluster_distance_from_plane_inliers
        self.below_plane_tolerance = below_plane_tolerance
        self.support_plane_constraint = support_plane_constraint

    def process(self, points: np.ndarray, stop_after: str = "all") -> CubeDetectionResult:
        """Run the detection pipeline up to a chosen stage.

        stop_after options:
            - "none": return the raw cloud without any processing.
            - "filter": stop after filtering/plane removal.
            - "cluster": stop after clustering.
            - "all": run the full pipeline (default).
        """
        stage = stop_after.lower()
        if stage not in {"none", "filter", "cluster", "all"}:
            raise ValueError(f"Invalid stop_after '{stop_after}'. "
                             "Choose from: none, filter, cluster, all.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if stage == "none":
            return CubeDetectionResult(
                original_cloud=pcd,
                filtered_cloud=pcd,
                masked_cloud=None,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=None,
                plane_inlier_indices=None,
                plane_inlier_cloud=None,
            )

        # Apply basic filtering and table plane removal
        print("Filtering point cloud...")
        t0 = time.perf_counter()
        filtered = filter_point_cloud(pcd, voxel_size=self.voxel_size)

        plane_model, inliers = filtered.segment_plane(distance_threshold=self.base_plane_distance,
                                                      ransac_n=3,
                                                      num_iterations=1000)
        plane_inlier_cloud = filtered.select_by_index(inliers)
        filtered = filtered.select_by_index(inliers, invert=True)

        if stage == "filter":
            return CubeDetectionResult(
                original_cloud=pcd,
                filtered_cloud=filtered,
                masked_cloud=None,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=np.asarray(plane_model, dtype=float),
                plane_inlier_indices=np.asarray(inliers, dtype=int),
                plane_inlier_cloud=plane_inlier_cloud,
            )

        render_boxes = stage in {"cluster", "all"}
        cluster_boxes, clusters = cluster_point_cloud(
            filtered,
            eps=self.cluster_eps,
            min_points=self.cluster_min_points,
            render_boxes=render_boxes,
            plane_model=np.asarray(plane_model, dtype=float),
            plane_inlier_points=np.asarray(plane_inlier_cloud.points),
            below_plane_tolerance=self.below_plane_tolerance,
            max_distance_from_inliers=self.max_cluster_distance_from_plane_inliers,
        )
        print(f"filter_point_cloud: {time.perf_counter() - t0:.3f}s")

        if stage == "cluster":
            cluster_cloud = o3d.geometry.PointCloud()
            if clusters:
                cluster_points = np.vstack([np.asarray(c.points) for c in clusters])
                cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            else:
                cluster_cloud = filtered
            return CubeDetectionResult(
                original_cloud=pcd,
                filtered_cloud=cluster_cloud,
                masked_cloud=None,
                cluster_boxes=cluster_boxes,
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=np.asarray(plane_model, dtype=float),
                plane_inlier_indices=np.asarray(inliers, dtype=int),
                plane_inlier_cloud=plane_inlier_cloud,
            )

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
                support_plane_model=np.asarray(plane_model, dtype=float),
                support_plane_constraint=self.support_plane_constraint,
            )
            cubes.extend(estimates)
            plane_obbs.extend(obbs)
            failed_initial_meshes.extend(failed_inits)
            print(f"cube_fitting: {time.perf_counter() - tcluster:.3f}s")

        # Select only the best num_best_cubes based on ICP fitness
        selected_cubes = select_best_cubes(cubes, self.num_best_cubes)
        print(f"Selected {len(selected_cubes)} best cubes out of {len(cubes)} total")

        return CubeDetectionResult(
            original_cloud=pcd,
            filtered_cloud=filtered,
            masked_cloud=None,
            cluster_boxes=cluster_boxes,
            plane_obbs=plane_obbs,
            cubes=selected_cubes,
            failed_initial_meshes=failed_initial_meshes,
            plane_model=np.asarray(plane_model, dtype=float),
            plane_inlier_indices=np.asarray(inliers, dtype=int),
            plane_inlier_cloud=plane_inlier_cloud,
        )
