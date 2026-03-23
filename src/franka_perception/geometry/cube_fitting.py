#!/usr/bin/env python3
"""Cube fitting utilities (ICP + cleanup)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

from .cube_pose import estimate_cube_pose
from .plane_segmentation import PlaneDetection, segment_planes

_SUPPORT_PLANE_CONSTRAINT_MODES = {"fix_icp", "ajust", "none"}


@dataclass
class CubeEstimate:
    transform: np.ndarray  # 4x4
    mesh: o3d.geometry.TriangleMesh
    initial_mesh: Optional[o3d.geometry.TriangleMesh] = None
    icp_fitness: float = 0.0  # ICP fitness score (higher is better)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def _normalize_plane(plane_model: Optional[np.ndarray],
                     reference_point: Optional[np.ndarray] = None):
    if plane_model is None:
        return None
    plane = np.asarray(plane_model, dtype=float).reshape(-1)
    if plane.shape[0] != 4:
        return None
    normal = plane[:3]
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm
    offset = float(plane[3] / norm)
    if reference_point is not None:
        if float(np.dot(normal, reference_point) + offset) < 0.0:
            normal = -normal
            offset = -offset
    return normal, offset


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return np.eye(3)
    axis = axis / axis_norm
    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_between_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src /= np.linalg.norm(src) + 1e-12
    dst /= np.linalg.norm(dst) + 1e-12
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cross_norm < 1e-9:
        if dot > 0.0:
            return np.eye(3)
        aux = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(aux, src))) > 0.9:
            aux = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, aux)
        axis /= np.linalg.norm(axis) + 1e-12
        return _rotation_about_axis(axis, np.pi)
    axis = cross / cross_norm
    angle = float(np.arccos(dot))
    return _rotation_about_axis(axis, angle)


def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vec - normal * float(np.dot(vec, normal))


def _make_supported_transform(params: np.ndarray,
                              base_center: np.ndarray,
                              plane_u: np.ndarray,
                              plane_v: np.ndarray,
                              plane_normal: np.ndarray,
                              plane_offset: float,
                              half_side: float,
                              R_base: np.ndarray,
                              support_axis: np.ndarray) -> np.ndarray:
    dx, dy, yaw = [float(v) for v in params]
    center = base_center + dx * plane_u + dy * plane_v
    signed_distance = float(np.dot(plane_normal, center) + plane_offset)
    center = center + (half_side - signed_distance) * plane_normal
    R = _rotation_about_axis(support_axis, yaw) @ R_base
    R = _orthonormalize(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    return T


def _cloud_distance_cost(source_points: np.ndarray,
                         target_pcd: o3d.geometry.PointCloud,
                         T: np.ndarray) -> float:
    transformed = (source_points @ T[:3, :3].T) + T[:3, 3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(transformed)
    distances = np.asarray(cloud.compute_point_cloud_distance(target_pcd), dtype=float)
    if distances.size == 0:
        return float("inf")
    return float(np.mean(distances ** 2))


def _evaluate_transform(source_points: np.ndarray,
                        target_pcd: o3d.geometry.PointCloud,
                        T: np.ndarray,
                        threshold: float = 0.003) -> Tuple[float, float]:
    transformed = (source_points @ T[:3, :3].T) + T[:3, 3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(transformed)
    distances = np.asarray(cloud.compute_point_cloud_distance(target_pcd), dtype=float)
    fitness = float(np.mean(distances <= threshold)) if distances.size else 0.0
    rmse = float(np.sqrt(np.mean(distances ** 2))) if distances.size else float("inf")
    return fitness, rmse


def _normalize_support_plane_constraint(mode) -> str:
    if isinstance(mode, bool):
        return "fix_icp" if mode else "none"

    text = str(mode).strip().lower()
    aliases = {
        "true": "fix_icp",
        "false": "none",
        "adjust": "ajust",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in _SUPPORT_PLANE_CONSTRAINT_MODES else "fix_icp"


def _adjust_transform_to_support_plane(T: np.ndarray,
                                       cube_side_length: float,
                                       support_plane_model: Optional[np.ndarray]) -> np.ndarray:
    plane = _normalize_plane(support_plane_model, T[:3, 3])
    if plane is None:
        return T.copy()

    plane_normal, plane_offset = plane
    adjusted_T = T.copy()
    R = _orthonormalize(adjusted_T[:3, :3])

    support_axis_index = int(np.argmax(np.abs(R.T @ plane_normal)))
    support_axis = R[:, support_axis_index]
    if float(np.dot(support_axis, plane_normal)) < 0.0:
        support_axis = -support_axis

    align_R = _rotation_between_vectors(support_axis, plane_normal)
    adjusted_R = _orthonormalize(align_R @ R)
    adjusted_T[:3, :3] = adjusted_R

    half_side = 0.5 * float(cube_side_length)
    center = adjusted_T[:3, 3].copy()
    signed_distance = float(np.dot(plane_normal, center) + plane_offset)
    center = center + (half_side - signed_distance) * plane_normal
    adjusted_T[:3, 3] = center
    return adjusted_T


def _supported_icp(source_pcd: o3d.geometry.PointCloud,
                   target_pcd: o3d.geometry.PointCloud,
                   init_T: np.ndarray,
                   cube_side_length: float,
                   support_plane_model: Optional[np.ndarray]):
    plane = _normalize_plane(support_plane_model, init_T[:3, 3])
    if plane is None or len(target_pcd.points) == 0 or len(source_pcd.points) == 0:
        return None

    plane_normal, plane_offset = plane
    R_init = init_T[:3, :3]
    center_init = init_T[:3, 3]
    half_side = 0.5 * float(cube_side_length)

    support_axis_index = int(np.argmax(np.abs(R_init.T @ plane_normal)))
    support_axis = R_init[:, support_axis_index]
    if float(np.dot(support_axis, plane_normal)) < 0.0:
        support_axis = -plane_normal
    else:
        support_axis = plane_normal

    align_R = _rotation_between_vectors(R_init[:, support_axis_index], support_axis)
    R_base = _orthonormalize(align_R @ R_init)

    center_signed_distance = float(np.dot(plane_normal, center_init) + plane_offset)
    base_center = center_init + (half_side - center_signed_distance) * plane_normal

    remaining_indices = [idx for idx in range(3) if idx != support_axis_index]
    best_idx = None
    best_norm = -1.0
    for idx in remaining_indices:
        projected = _project_to_plane(R_base[:, idx], plane_normal)
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm > best_norm:
            best_norm = projected_norm
            best_idx = idx

    if best_idx is None or best_norm < 1e-9:
        aux = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(aux, plane_normal))) > 0.9:
            aux = np.array([0.0, 1.0, 0.0])
        plane_u = np.cross(plane_normal, aux)
    else:
        plane_u = _project_to_plane(R_base[:, best_idx], plane_normal)
    plane_u /= np.linalg.norm(plane_u) + 1e-12
    plane_v = np.cross(plane_normal, plane_u)
    plane_v /= np.linalg.norm(plane_v) + 1e-12

    source_points = np.asarray(source_pcd.points, dtype=float)
    params = np.zeros(3, dtype=float)
    best_T = _make_supported_transform(
        params,
        base_center,
        plane_u,
        plane_v,
        plane_normal,
        plane_offset,
        half_side,
        R_base,
        support_axis,
    )
    best_cost = _cloud_distance_cost(source_points, target_pcd, best_T)

    step_xy = max(float(cube_side_length) * 0.2, 0.002)
    step_yaw = np.deg2rad(15.0)
    for _ in range(6):
        improved = True
        while improved:
            improved = False
            candidate_best = params
            candidate_T = best_T
            candidate_cost = best_cost
            for dim, step in enumerate((step_xy, step_xy, step_yaw)):
                for sign in (-1.0, 1.0):
                    trial = params.copy()
                    trial[dim] += sign * step
                    T_trial = _make_supported_transform(
                        trial,
                        base_center,
                        plane_u,
                        plane_v,
                        plane_normal,
                        plane_offset,
                        half_side,
                        R_base,
                        support_axis,
                    )
                    cost = _cloud_distance_cost(source_points, target_pcd, T_trial)
                    if cost + 1e-12 < candidate_cost:
                        candidate_cost = cost
                        candidate_best = trial
                        candidate_T = T_trial
            if candidate_cost + 1e-12 < best_cost:
                params = candidate_best
                best_T = candidate_T
                best_cost = candidate_cost
                improved = True
        step_xy *= 0.5
        step_yaw *= 0.5

    fitness, rmse = _evaluate_transform(source_points, target_pcd, best_T)
    return best_T, fitness, rmse


def fit_cubes_in_cluster(cluster_pcd: o3d.geometry.PointCloud,
                         cube_side_length: float,
                         max_cubes: int = 2,
                         clearance: float = 0.015,
                         plane_distance: float = 0.0005,
                         plane_min_inliers: int = 20,
                         support_plane_model: Optional[np.ndarray] = None,
                         support_plane_constraint: str = "fix_icp") -> Tuple[List[CubeEstimate], List[o3d.geometry.OrientedBoundingBox], List[o3d.geometry.TriangleMesh]]:
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
    support_plane_mode = _normalize_support_plane_constraint(support_plane_constraint)

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

        cube_mesh_init = o3d.geometry.TriangleMesh.create_box(
            cube_side_length,
            cube_side_length,
            cube_side_length
        )
        cube_mesh_init.translate(-cube_mesh_init.get_center())
        cube_mesh_init.transform(init_T)
        cube_mesh_init.paint_uniform_color([1.0, 0.6, 0.0])

        cube_mesh = o3d.geometry.TriangleMesh.create_box(
            cube_side_length,
            cube_side_length,
            cube_side_length
        )
        cube_mesh.translate(-cube_mesh.get_center())
        source_pcd = cube_mesh.sample_points_poisson_disk(1500)
        source_points = np.asarray(source_pcd.points, dtype=float)
        target_pcd = remaining_pcd.voxel_down_sample(voxel_size=0.002)
        if len(target_pcd.points) == 0:
            failed_initial_meshes.append(cube_mesh_init)
            break

        constrained_result = None
        if support_plane_mode == "fix_icp":
            constrained_result = _supported_icp(
                source_pcd,
                target_pcd,
                init_T,
                cube_side_length,
                support_plane_model,
            )

        if constrained_result is not None:
            T_final, icp_fitness, _ = constrained_result
        else:
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
            T_final = icp_result.transformation.copy()
            icp_fitness = float(icp_result.fitness)

        if support_plane_mode == "ajust":
            T_final = _adjust_transform_to_support_plane(
                T_final,
                cube_side_length,
                support_plane_model,
            )
            icp_fitness, _ = _evaluate_transform(source_points, target_pcd, T_final)

        T_final[:3, :3] = _orthonormalize(T_final[:3, :3])

        cube_mesh_est = o3d.geometry.TriangleMesh.create_box(
            cube_side_length,
            cube_side_length,
            cube_side_length
        )
        cube_mesh_est.translate(-cube_mesh_est.get_center())
        cube_mesh_est.transform(T_final)
        cube_mesh_est.paint_uniform_color([0.1, 0.4, 0.9])

        estimates.append(
            CubeEstimate(
                transform=T_final,
                mesh=cube_mesh_est,
                initial_mesh=cube_mesh_init,
                icp_fitness=icp_fitness,
            )
        )

        cube_pcd = cube_mesh_est.sample_points_uniformly(400)
        distances = remaining_pcd.compute_point_cloud_distance(cube_pcd)
        keep_indices = [i for i, d in enumerate(distances) if d > clearance]
        remaining_pcd = remaining_pcd.select_by_index(keep_indices)
        if len(remaining_pcd.points) < 30:
            break

    return estimates, obbs, failed_initial_meshes


def select_best_cubes(estimates: List[CubeEstimate], num_best: int) -> List[CubeEstimate]:
    """Select the top num_best cubes by ICP fitness score.
    
    Args:
        estimates: List of CubeEstimate objects to filter
        num_best: Number of best cubes to keep
        
    Returns:
        List of the best cubes sorted by fitness (highest first)
    """
    if len(estimates) == 0:
        return []
    
    if num_best <= 0:
        return []
    
    sorted_estimates = sorted(estimates, key=lambda c: c.icp_fitness, reverse=True)
    result = sorted_estimates[:min(num_best, len(sorted_estimates))]
    
    for i, cube in enumerate(result):
        print(f"Selected cube {i+1}/{len(result)}: fitness={cube.icp_fitness:.4f}")
    
    return result
