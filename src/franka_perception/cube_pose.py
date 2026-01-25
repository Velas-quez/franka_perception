#!/usr/bin/env python3
"""Cube pose estimation from plane observations."""

from itertools import combinations, permutations, product
from typing import List, Optional, Tuple

import numpy as np

from .plane_segmentation import PlaneDetection


def estimate_cube_pose(planes: List[PlaneDetection],
                       cube_side_length: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Estimate rotation (3x3) and center (3,) from up to 3 visible faces."""
    if not planes:
        return None

    if len(planes) <= 3:
        chosen_planes = planes
    else:
        best = None
        for triplet in combinations(planes, 3):
            normals = [p.normal / np.linalg.norm(p.normal) for p in triplet]
            dots = [
                abs(float(np.dot(normals[0], normals[1]))),
                abs(float(np.dot(normals[0], normals[2]))),
                abs(float(np.dot(normals[1], normals[2])))
            ]
            score = max(dots)
            if best is None or score < best[0]:
                best = (score, triplet)
        chosen_planes = list(best[1]) if best else planes[:3]

    normals = [p.normal / (np.linalg.norm(p.normal) + 1e-12) for p in chosen_planes]
    centroids = [p.center for p in chosen_planes]
    half_side = 0.5 * cube_side_length

    def _rotation_from_normals(ns):
        ns = [n / (np.linalg.norm(n) + 1e-12) for n in ns]
        m = len(ns)
        best = None
        for perm in permutations(range(m), m):
            x = ns[perm[0]]
            if m >= 2:
                y_raw = ns[perm[1]]
                y = y_raw - x * float(np.dot(x, y_raw))
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-6:
                    continue
                y /= y_norm
            else:
                aux = np.array([1.0, 0.0, 0.0])
                if abs(float(np.dot(aux, x))) > 0.9:
                    aux = np.array([0.0, 1.0, 0.0])
                y = np.cross(x, aux)
                y /= (np.linalg.norm(y) + 1e-12)

            z = np.cross(x, y)
            z_norm = np.linalg.norm(z)
            if z_norm < 1e-6:
                continue
            z /= z_norm

            if m == 3:
                n3 = ns[perm[2]]
                if float(np.dot(z, n3)) < 0:
                    z *= -1.0
                    y *= -1.0

            R_candidate = np.stack([x, y, z], axis=1)
            score = 0.0
            score += 1.0 - abs(float(np.dot(R_candidate[:, 0], ns[perm[0]])))
            if m >= 2:
                score += 1.0 - abs(float(np.dot(R_candidate[:, 1], ns[perm[1]])))
            if m == 3:
                score += 1.0 - abs(float(np.dot(R_candidate[:, 2], ns[perm[2]])))
            if best is None or score < best[0]:
                best = (score, R_candidate)
        return None if best is None else best[1]

    best = None
    for signs in product([-1.0, 1.0], repeat=len(normals)):
        oriented_normals = []
        candidate_centers = []
        for n, c, s in zip(normals, centroids, signs):
            oriented_n = s * n
            oriented_normals.append(oriented_n)
            candidate_centers.append(c - oriented_n * half_side)

        centers_arr = np.stack(candidate_centers)
        center_mean = centers_arr.mean(axis=0)
        spread = np.linalg.norm(centers_arr - center_mean, axis=1).mean()

        R_candidate = _rotation_from_normals(oriented_normals)
        if R_candidate is None:
            continue

        align_err = 0.0
        for n in oriented_normals:
            dots = np.abs(R_candidate.T @ n)
            align_err += 1.0 - float(np.max(dots))

        score = (spread, align_err)
        if best is None or score < best[0]:
            best = (score, R_candidate, center_mean)

    if best is None:
        return None

    R, center = best[1], best[2]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R, center
