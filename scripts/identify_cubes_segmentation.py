#!/usr/bin/env python3
"""Plane segmentation utilities for clustered point clouds."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d


@dataclass
class PlaneDetection:
    cloud: o3d.geometry.PointCloud
    model: np.ndarray  # plane coefficients a, b, c, d
    center: np.ndarray
    normal: np.ndarray
    axis_index: Optional[int] = None
    sign: Optional[float] = None


def segment_planes(pcd: o3d.geometry.PointCloud,
                   distance_threshold: float = 0.001,
                   ransac_n: int = 3,
                   num_iterations: int = 300,
                   min_inliers: int = 20,
                   min_ratio: float = 0.02,
                   max_planes: int = 6) -> List[PlaneDetection]:
    """Extract dominant planes from a point cloud using iterative RANSAC.

    Args:
        pcd: Nuvem de pontos a ser segmentada em planos.
        distance_threshold: Distância máxima de um ponto ao plano para ser contado como inlier.
        ransac_n: Quantos pontos são amostrados em cada iteração do RANSAC.
        num_iterations: Máximo de iterações do RANSAC para encontrar um plano.
        min_inliers: Número mínimo de inliers para aceitar um plano detectado.
        min_ratio: Fração mínima (em relação ao total inicial) de inliers para aceitar o plano.
        max_planes: Quantidade máxima de planos a serem extraídos antes de parar.

    Returns:
        Lista de planos detectados com nuvem, modelo, normal e centro.
    """
    planes: List[PlaneDetection] = []
    if pcd.is_empty():
        return planes

    remaining = pcd
    total_points = len(remaining.points)
    for idx in range(max_planes):
        if len(remaining.points) < max(min_inliers, ransac_n):
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations)
        if len(inliers) < min_inliers or len(inliers) < min_ratio * total_points:
            break
        plane_cloud = remaining.select_by_index(inliers)
        points = np.asarray(plane_cloud.points)
        if points.size == 0:
            break
        center = points.mean(axis=0)
        normal = np.asarray(plane_model[:3], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-9
        planes.append(
            PlaneDetection(
                cloud=plane_cloud,
                model=np.asarray(plane_model, dtype=float),
                center=center,
                normal=normal,
            )
        )
        remaining = remaining.select_by_index(inliers, invert=True)
        if len(remaining.points) < min_inliers:
            break

    return planes


def estimate_cube_from_planes(planes: List[PlaneDetection],
                              side_length: float) -> Optional[dict]:
    """Estimate cube center/orientation from detected planes."""
    if not planes:
        return None

    normals = np.stack([p.normal for p in planes], axis=0)
    cov = normals.T @ normals
    axes, _, _ = np.linalg.svd(cov)
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    # Agrupar planos por eixo dominante
    offsets_per_axis = {0: [], 1: [], 2: []}
    for plane in planes:
        alignment = axes.T @ plane.normal
        axis_idx = int(np.argmax(np.abs(alignment)))
        sign = np.sign(alignment[axis_idx]) or 1.0
        axis_vec = axes[:, axis_idx] * sign
        offset = float(axis_vec @ plane.center)
        offsets_per_axis[axis_idx].append(offset)
        plane.axis_index = axis_idx
        plane.sign = sign

    # Para cada eixo, usa menor e maior offset como faces opostas
    center_components = np.zeros(3)
    for axis_idx in range(3):
        offsets = offsets_per_axis[axis_idx]
        if offsets:
            min_off = min(offsets)
            max_off = max(offsets)
            center_components[axis_idx] = 0.5 * (min_off + max_off)
        else:
            # Sem planos neste eixo: não estimamos deslocamento
            center_components[axis_idx] = 0.0

    center = axes @ center_components

    # Centros das faces estimados ± side_length/2 ao longo de cada eixo
    face_centers = []
    half = 0.5 * side_length
    for axis_idx in range(3):
        axis_vec = axes[:, axis_idx]
        face_centers.append(center + axis_vec * half)
        face_centers.append(center - axis_vec * half)

    bbox = o3d.geometry.OrientedBoundingBox(center, axes, [side_length] * 3)
    return {
        "center": center,
        "axes": axes,
        "face_centers": face_centers,
        "bbox": bbox,
    }
