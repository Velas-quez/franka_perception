#!/usr/bin/env python3
"""Filtering helpers for point clouds."""

import numpy as np
import open3d as o3d


def filter_point_cloud(point_cloud: o3d.geometry.PointCloud,
                       voxel_size: float = 0.003,
                       nb_neighbors: int = 20,
                       std_ratio: float = 2.0,
                       max_dist: float = 1.2) -> o3d.geometry.PointCloud:
    """Voxel downsample + distance filter + statistical outlier removal."""
    downsampled_pc = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    distances = np.asarray(downsampled_pc.points)
    dists = np.linalg.norm(distances, axis=1)
    mask = dists < max_dist
    downsampled_pc = downsampled_pc.select_by_index(np.where(mask)[0])

    _, ind = downsampled_pc.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                       std_ratio=std_ratio)
    filtered_pc = downsampled_pc.select_by_index(ind)
    return filtered_pc
