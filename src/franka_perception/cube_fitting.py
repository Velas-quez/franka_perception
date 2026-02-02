#!/usr/bin/env python3
"""Cube fitting utilities (ICP + cleanup)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from .cube_pose import estimate_cube_pose
from .plane_segmentation import PlaneDetection, segment_planes


@dataclass
class CubeEstimate:
    transform: np.ndarray  # 4x4
    mesh: o3d.geometry.TriangleMesh
    initial_mesh: Optional[o3d.geometry.TriangleMesh] = None
    face_colors: Optional[Dict[str, np.ndarray]] = None
    face_counts: Optional[Dict[str, int]] = None


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def fit_cubes_in_cluster(cluster_pcd: o3d.geometry.PointCloud,
                         cube_side_length: float,
                         max_cubes: int = 2,
                         clearance: float = 0.015,
                         plane_distance: float = 0.0005,
                         plane_min_inliers: int = 20) -> Tuple[List[CubeEstimate], List[o3d.geometry.OrientedBoundingBox], List[o3d.geometry.TriangleMesh]]:
    """Fit up to max_cubes into the cluster using plane detection + ICP.

    Returns:
        estimates: Successful cube fits.
        obbs: Plane bounding boxes for debugging.
        failed_initial_meshes: Initial ICP guesses for fits that failed, rendered for debugging.
    """
    remaining_pcd = cluster_pcd
    obbs: List[o3d.geometry.OrientedBoundingBox] = []
    estimates: List[CubeEstimate] = []
    failed_initial_meshes: List[o3d.geometry.TriangleMesh] = []

    for _ in range(max_cubes):
        if len(remaining_pcd.points) < 30:
            break

        planes: List[PlaneDetection] = segment_planes(
            remaining_pcd,
            distance_threshold=plane_distance,
            ransac_n=3,
            num_iterations=1000,
            min_inliers=plane_min_inliers,
            min_ratio=0.02,
            max_planes=6,
        )
        for plane in planes:
            obb = plane.cloud.get_oriented_bounding_box()
            obb.color = (1, 0, 0)
            obbs.append(obb)

        if len(planes) < 1:
            break

        pose = estimate_cube_pose(planes, cube_side_length)
        if pose is None:
            break

        R_init, t_init = pose
        init_T = np.eye(4)
        init_T[:3, :3] = R_init
        init_T[:3, 3] = t_init

        # Initial ICP cube position mesh for rendering
        cube_mesh_init = o3d.geometry.TriangleMesh.create_box(
            cube_side_length,
            cube_side_length,
            cube_side_length
        )
        cube_mesh_init.translate(-cube_mesh_init.get_center())
        cube_mesh_init.transform(init_T)
        cube_mesh_init.paint_uniform_color([1.0, 0.6, 0.0])  # orange for "attempted"

        cube_mesh = o3d.geometry.TriangleMesh.create_box(
            cube_side_length,
            cube_side_length,
            cube_side_length
        )
        cube_mesh.translate(-cube_mesh.get_center())
        source_pcd = cube_mesh.sample_points_poisson_disk(1500)
        target_pcd = remaining_pcd.voxel_down_sample(voxel_size=0.002)
        if len(target_pcd.points) == 0:
            failed_initial_meshes.append(cube_mesh_init)
            break
        # target_pcd.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=50)
        # )
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.008,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            0.003,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        used_estimator = "point_to_point"

        if icp_result.fitness <= 0.2 or icp_result.inlier_rmse > 0.004:
            print(f"ICP {used_estimator} coarse: fitness={icp_coarse.fitness:.3f}, rmse={icp_coarse.inlier_rmse:.4f}")
            print(f"ICP {used_estimator} poor: fitness={icp_result.fitness:.3f}, rmse={icp_result.inlier_rmse:.4f}")
            failed_initial_meshes.append(cube_mesh_init)
            break

        T_final = icp_result.transformation.copy()
        T_final[:3, :3] = _orthonormalize(T_final[:3, :3])

        cube_mesh_est = o3d.geometry.TriangleMesh.create_box(
            cube_side_length,
            cube_side_length,
            cube_side_length
        )
        cube_mesh_est.translate(-cube_mesh_est.get_center())
        cube_mesh_est.transform(T_final)
        cube_mesh_est.paint_uniform_color([0.1, 0.4, 0.9])

        face_colors, face_counts = _extract_face_colors(
            cluster_pcd, T_final, cube_side_length)
        estimates.append(
            CubeEstimate(
                transform=T_final,
                mesh=cube_mesh_est,
                initial_mesh=cube_mesh_init,
                face_colors=face_colors,
                face_counts=face_counts,
            )
        )

        cube_pcd = cube_mesh_est.sample_points_uniformly(400)
        distances = remaining_pcd.compute_point_cloud_distance(cube_pcd)
        keep_indices = [i for i, d in enumerate(distances) if d > clearance]
        remaining_pcd = remaining_pcd.select_by_index(keep_indices)
        if len(remaining_pcd.points) < 30:
            break

    return estimates, obbs, failed_initial_meshes


def _extract_face_colors(cluster_pcd: o3d.geometry.PointCloud,
                         transform: np.ndarray,
                         cube_side_length: float,
                         distance_tol: float = 0.008,
                         min_face_points: int = 20):
    """Assign cluster points to cube faces and compute mean color."""
    colors = np.asarray(cluster_pcd.colors)
    if colors.shape[0] != len(cluster_pcd.points) or colors.size == 0:
        return {}, {}

    points = np.asarray(cluster_pcd.points)
    if points.size == 0:
        return {}, {}

    T_inv = np.linalg.inv(transform)
    homog = np.hstack([points, np.ones((len(points), 1))])
    local = (homog @ T_inv.T)[:, :3]
    half = cube_side_length / 2.0

    buckets: Dict[str, List[np.ndarray]] = {
        "px": [], "nx": [], "py": [], "ny": [], "pz": [], "nz": []
    }

    for pt_local, color in zip(local, colors):
        axis = int(np.argmax(np.abs(pt_local)))
        sign = 1 if pt_local[axis] >= 0 else -1
        dist = abs(pt_local[axis]) - half
        if dist > distance_tol:
            continue
        # ensure point lies within the face boundaries (with tolerance)
        in_face = np.all(np.abs(np.delete(pt_local, axis)) <= half + distance_tol)
        if not in_face:
            continue
        face_key = ("p" if sign > 0 else "n") + ("xyz"[axis])
        buckets[face_key].append(color)

    face_colors: Dict[str, np.ndarray] = {}
    face_counts: Dict[str, int] = {}
    for key, cols in buckets.items():
        if len(cols) < min_face_points:
            continue
        cols_arr = np.asarray(cols)
        face_colors[key] = cols_arr.mean(axis=0)
        face_counts[key] = len(cols)

    return face_colors, face_counts
