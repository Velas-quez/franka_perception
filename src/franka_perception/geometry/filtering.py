#!/usr/bin/env python3
"""Filtering helpers for point clouds."""

import numpy as np
import open3d as o3d

from ..core.open3d_backend import (
    resolve_open3d_device,
    tensor_distance_filter,
    tensor_point_cloud_from_legacy,
)


def filter_point_cloud(point_cloud: o3d.geometry.PointCloud,
                       voxel_size: float = 0.003,
                       nb_neighbors: int = 20,
                       std_ratio: float = 2.0,
                       max_dist: float = 1,
                       open3d_device: str = "auto") -> o3d.geometry.PointCloud:
    """Voxel downsample + distance filter + statistical outlier removal."""
    if len(point_cloud.points) == 0:
        return o3d.geometry.PointCloud()

    if voxel_size > 0.0:
        tensor_pcd = tensor_point_cloud_from_legacy(
            point_cloud,
            resolve_open3d_device(open3d_device),
        )
        downsampled_pc = tensor_distance_filter(
            tensor_pcd.voxel_down_sample(voxel_size=float(voxel_size)),
            max_dist=float(max_dist),
        ).to_legacy()
    else:
        downsampled_pc = o3d.geometry.PointCloud(point_cloud)

        distances = np.asarray(downsampled_pc.points)
        dists = np.linalg.norm(distances, axis=1)
        mask = dists < max_dist
        downsampled_pc = downsampled_pc.select_by_index(np.where(mask)[0])

    _, ind = downsampled_pc.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                       std_ratio=std_ratio)
    filtered_pc = downsampled_pc.select_by_index(ind)
    return filtered_pc
