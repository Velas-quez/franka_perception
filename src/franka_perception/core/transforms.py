#!/usr/bin/env python3
"""Transform helpers for ROS/Open3D perception outputs."""

import copy
from typing import Iterable, List, Optional

import numpy as np
import tf.transformations as tf_trans

from ..geometry.cube_fitting import CubeEstimate
from ..pipelines.result_type import CubeDetectionResult


def transform_to_matrix(tf_msg) -> np.ndarray:
    """Convert TransformStamped to a 4x4 transform matrix."""
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    matrix = tf_trans.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    matrix[0, 3] = trans.x
    matrix[1, 3] = trans.y
    matrix[2, 3] = trans.z
    return matrix


def transform_points(points: Optional[np.ndarray], matrix: np.ndarray) -> Optional[np.ndarray]:
    if points is None:
        return None
    if points.size == 0:
        return points.copy()
    return (points @ matrix[:3, :3].T) + matrix[:3, 3]


def transform_geometry_copy(geometry, matrix: np.ndarray):
    if geometry is None:
        return None

    transformed = copy.deepcopy(geometry)
    if hasattr(transformed, "transform"):
        transformed.transform(matrix)
        return transformed

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    if hasattr(transformed, "rotate"):
        transformed.rotate(rotation, center=(0.0, 0.0, 0.0))
    if hasattr(transformed, "translate"):
        transformed.translate(translation)
    return transformed


def transform_cubes(cubes: Iterable[CubeEstimate], matrix: np.ndarray) -> List[CubeEstimate]:
    transformed = []
    for cube in cubes:
        transformed.append(
            CubeEstimate(
                transform=matrix @ cube.transform,
                mesh=transform_geometry_copy(cube.mesh, matrix),
                initial_mesh=transform_geometry_copy(cube.initial_mesh, matrix),
                icp_fitness=cube.icp_fitness,
            )
        )
    return transformed


def transform_plane_model(plane_model: Optional[np.ndarray],
                          matrix: np.ndarray) -> Optional[np.ndarray]:
    if plane_model is None:
        return None

    plane = np.asarray(plane_model, dtype=float).reshape(-1)
    if plane.shape[0] != 4:
        return plane.copy()

    normal = plane[:3]
    offset = float(plane[3])
    rotation = np.asarray(matrix[:3, :3], dtype=float)
    translation = np.asarray(matrix[:3, 3], dtype=float)

    new_normal = rotation @ normal
    new_offset = offset - float(np.dot(new_normal, translation))
    return np.array([new_normal[0], new_normal[1], new_normal[2], new_offset], dtype=float)


def transform_detection_result(result: CubeDetectionResult,
                               matrix: np.ndarray) -> CubeDetectionResult:
    return CubeDetectionResult(
        original_cloud=transform_geometry_copy(result.original_cloud, matrix),
        filtered_cloud=transform_geometry_copy(result.filtered_cloud, matrix),
        masked_cloud=transform_geometry_copy(result.masked_cloud, matrix),
        cluster_boxes=[transform_geometry_copy(box, matrix) for box in result.cluster_boxes],
        plane_obbs=[transform_geometry_copy(box, matrix) for box in result.plane_obbs],
        cubes=transform_cubes(result.cubes, matrix),
        failed_initial_meshes=[
            transform_geometry_copy(mesh, matrix) for mesh in result.failed_initial_meshes
        ],
        plane_model=transform_plane_model(result.plane_model, matrix),
        plane_inlier_indices=(
            result.plane_inlier_indices.copy()
            if result.plane_inlier_indices is not None else None
        ),
        plane_inlier_cloud=transform_geometry_copy(result.plane_inlier_cloud, matrix),
        sam_rgb_image=result.sam_rgb_image,
        sam_masks=result.sam_masks,
        sam_dino_boxes=result.sam_dino_boxes,
        sam_overlay=result.sam_overlay,
    )
