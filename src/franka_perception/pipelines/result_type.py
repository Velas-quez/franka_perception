from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d

from ..geometry.cube_fitting import CubeEstimate

@dataclass
class CubeDetectionResult:
    original_cloud: o3d.geometry.PointCloud
    filtered_cloud: o3d.geometry.PointCloud
    masked_cloud: Optional[o3d.geometry.PointCloud]
    cluster_boxes: list
    plane_obbs: list
    cubes: List[CubeEstimate]
    failed_initial_meshes: list
    plane_model: Optional[np.ndarray] = None
    plane_inlier_indices: Optional[np.ndarray] = None
    plane_inlier_cloud: Optional[o3d.geometry.PointCloud] = None
    sam_rgb_image: Optional[np.ndarray] = None
    sam_masks: Optional[List[np.ndarray]] = None
    sam_dino_boxes: Optional[np.ndarray] = None
    sam_overlay: Optional[np.ndarray] = None