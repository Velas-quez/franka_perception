#!/usr/bin/env python3
"""DBSCAN clustering utilities."""

from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


def cluster_point_cloud(pcd: o3d.geometry.PointCloud,
                        eps: float = 0.02,
                        min_points: int = 10,
                        max_points: Optional[int] = 7000,
                        render_boxes: bool = False) -> Tuple[list, List[o3d.geometry.PointCloud]]:
    """Cluster the point cloud using DBSCAN."""
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
        clusters.append(cluster)
        print(f"Cluster {i}: {cluster_size} points")

    cluster_boxes = []
    if render_boxes:
        for cluster in clusters:
            bbox = cluster.get_oriented_bounding_box()
            bbox.color = (0.2, 0.8, 0.2)
            cluster_boxes.append(bbox)

    return cluster_boxes, clusters
