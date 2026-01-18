#!/usr/bin/env python3
import numpy as np

def filter_point_cloud(point_cloud, voxel_size=0.003, nb_neighbors=20, std_ratio=2.0, max_dist=1.2):
    """
    Filters the input point cloud using voxel downsampling and statistical outlier removal.

    Parameters:
    - point_cloud (o3d.geometry.PointCloud): The input point cloud to be filtered.
    - voxel_size (float): The size of the voxel grid for downsampling.
    - nb_neighbors (int): Number of neighbors to analyze for each point in statistical outlier removal.
    - std_ratio (float): Standard deviation ratio for statistical outlier removal.

    Returns:
    - o3d.geometry.PointCloud: The filtered point cloud.
    """
    # Voxel Downsampling
    downsampled_pc = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    
    # Distance Filtering
    distances = np.asarray(downsampled_pc.points)
    dists = np.linalg.norm(distances, axis=1)
    mask = dists < max_dist
    downsampled_pc = downsampled_pc.select_by_index(np.where(mask)[0])

    # Statistical Outlier Removal
    cl, ind = downsampled_pc.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                        std_ratio=std_ratio)
    filtered_pc = downsampled_pc.select_by_index(ind)

    return filtered_pc