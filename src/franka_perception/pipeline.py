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
                filtered_cloud=pcd,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
            )

        # Apply basic filtering and table plane removal
        print("Filtering point cloud...")
        t0 = time.perf_counter()
        filtered = filter_point_cloud(pcd, voxel_size=self.voxel_size)

        _, inliers = filtered.segment_plane(distance_threshold=self.base_plane_distance,
                                            ransac_n=3,
                                            num_iterations=1000)
        filtered = filtered.select_by_index(inliers, invert=True)

        if stage == "filter":
            return CubeDetectionResult(
                filtered_cloud=filtered,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
            )

        render_boxes = stage in {"cluster", "all"}
        cluster_boxes, clusters = cluster_point_cloud(
            filtered,
            eps=self.cluster_eps,
            min_points=self.cluster_min_points,
            render_boxes=render_boxes,
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
                filtered_cloud=cluster_cloud,
                cluster_boxes=cluster_boxes,
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
            )

        return self.process_preclustered_clusters(
            clusters,
            stop_after=stage,
            filtered_cloud=filtered,
            cluster_boxes=cluster_boxes,
        )

    def process_preclustered_clusters(
        self,
        clusters: List[o3d.geometry.PointCloud],
        stop_after: str = "all",
        filtered_cloud: Optional[o3d.geometry.PointCloud] = None,
        cluster_boxes: Optional[list] = None,
    ) -> CubeDetectionResult:
        """Run the back half of the pipeline on precomputed clusters.

        This is useful when clusters come from 2D segmentation masks projected
        through depth (for example, SAM masks on RGB-D frames).
        """
        stage = stop_after.lower()
        if stage not in {"cluster", "all"}:
            raise ValueError(f"Invalid stop_after '{stop_after}' for preclustered input. "
                             "Choose from: cluster, all.")

        boxes = cluster_boxes if cluster_boxes is not None else []
        cloud = filtered_cloud if filtered_cloud is not None else self._merge_clusters(clusters)
        if stage == "cluster":
            return CubeDetectionResult(
                filtered_cloud=cloud,
                cluster_boxes=boxes,
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
            )

        plane_obbs, cubes, failed_initial_meshes = self._fit_clusters(clusters)
        return CubeDetectionResult(
            filtered_cloud=cloud,
            cluster_boxes=boxes,
            plane_obbs=plane_obbs,
            cubes=cubes,
            failed_initial_meshes=failed_initial_meshes,
        )

    def _fit_clusters(self, clusters: List[o3d.geometry.PointCloud]):
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
        return plane_obbs, cubes, failed_initial_meshes

    @staticmethod
    def _merge_clusters(clusters: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud()
        if not clusters:
            return cloud
        cluster_points = []
        for cluster in clusters:
            points = np.asarray(cluster.points)
            if points.size:
                cluster_points.append(points)
        if cluster_points:
            cloud.points = o3d.utility.Vector3dVector(np.vstack(cluster_points))
        return cloud
