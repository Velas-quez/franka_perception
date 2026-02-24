#!/usr/bin/env python3
"""DBSCAN clustering utilities."""

from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


def cluster_point_cloud(pcd: o3d.geometry.PointCloud,
                        eps: float = 0.02,
                        min_points: int = 10,
                        max_points: Optional[int] = 4000,
                        render_boxes: bool = True,
                        plane_model: Optional[np.ndarray] = None,
                        plane_inlier_points: Optional[np.ndarray] = None,
                        below_plane_tolerance: float = 0.002,
                        max_distance_from_inliers: Optional[float] = 0.08
                        ) -> Tuple[list, List[o3d.geometry.PointCloud]]:
    """Cluster the point cloud using DBSCAN."""
    plane_normal = None
    plane_d = None
    if plane_model is not None:
        plane_model = np.asarray(plane_model, dtype=float)
        if plane_model.shape == (4,):
            normal_norm = np.linalg.norm(plane_model[:3])
            if normal_norm > 1e-9:
                plane_normal = plane_model[:3] / normal_norm
                plane_d = float(plane_model[3] / normal_norm)
                cloud_points = np.asarray(pcd.points)
                if cloud_points.size > 0:
                    mean_signed = float(np.mean(cloud_points @ plane_normal + plane_d))
                    if mean_signed < 0.0:
                        plane_normal *= -1.0
                        plane_d *= -1.0

    inlier_cloud = None
    if plane_inlier_points is not None and len(plane_inlier_points) > 0:
        inlier_cloud = o3d.geometry.PointCloud()
        inlier_cloud.points = o3d.utility.Vector3dVector(plane_inlier_points)

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        cluster = pcd.select_by_index(indices)
        cluster_size = len(cluster.points)
        if max_points is not None and cluster_size > max_points:
            # used to ignore large objects such as the robot arm
            continue
        if cluster_size < 100:
            # used to ignore small noise clusters
            continue

        if plane_normal is not None and plane_d is not None:
            centroid = np.mean(np.asarray(cluster.points), axis=0)
            signed_dist = float(np.dot(plane_normal, centroid) + plane_d)
            if signed_dist < -abs(float(below_plane_tolerance)):
                # reject clusters that are below the support plane
                continue

        if inlier_cloud is not None and max_distance_from_inliers is not None:
            centroid = np.mean(np.asarray(cluster.points), axis=0)
            centroid_cloud = o3d.geometry.PointCloud()
            centroid_cloud.points = o3d.utility.Vector3dVector([centroid])
            nn_dist = np.asarray(centroid_cloud.compute_point_cloud_distance(inlier_cloud))[0]
            if float(nn_dist) > float(max_distance_from_inliers):
                # reject clusters disconnected from the table inlier region
                continue

        clusters.append(cluster)
        print(f"Cluster {i}: {cluster_size} points")

    cluster_boxes = []
    if render_boxes:
        for cluster in clusters:
            bbox = cluster.get_oriented_bounding_box()
            bbox.color = (0.2, 0.8, 0.2)
            cluster_boxes.append(bbox)

    return cluster_boxes, clusters
