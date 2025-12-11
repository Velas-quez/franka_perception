#!/usr/bin/env python3
import numpy as np


def _group_face_indices(normals: np.ndarray, angle_threshold_deg: float = 20.0):
    """Agrupa índices de pontos em faces com base na similaridade das normais."""
    if normals.size == 0:
        return []

    angle_thr = np.deg2rad(angle_threshold_deg)
    cos_thr = np.cos(angle_thr)

    groups = []  # lista de listas de índices
    mean_normals = []  # normal média de cada grupo

    for idx, n in enumerate(normals):
        n = n / (np.linalg.norm(n) + 1e-9)
        assigned = False
        for g_id, m in enumerate(mean_normals):
            if abs(np.dot(n, m)) >= cos_thr:
                groups[g_id].append(idx)
                # atualiza normal média do grupo
                mean_normals[g_id] = (
                    mean_normals[g_id] * (len(groups[g_id]) - 1) + n
                ) / len(groups[g_id])
                assigned = True
                break
        if not assigned:
            groups.append([idx])
            mean_normals.append(n)

    # remove grupos muito pequenos
    min_points_per_face = 30
    filtered_groups = [
        g for g in groups if len(g) >= min_points_per_face
    ]
    return filtered_groups


def _estimate_face_properties(points: np.ndarray, normals: np.ndarray):
    """
    Calcula centro, normal e dimensões (altura, largura) de uma face.
    Ajusta para quadrado estendendo a menor dimensão.
    """
    if points.shape[0] < 3:
        return None

    center = points.mean(axis=0)
    normal = normals.mean(axis=0)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-6:
        return None
    normal = normal / n_norm

    # PCA nos pontos projetados no plano da face para obter direções principais
    pts_centered = points - center
    cov = pts_centered.T @ pts_centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # ordenar por autovalor decrescente
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # garantir que o primeiro eixo esteja no plano (aproximadamente ortogonal à normal)
    u = eigvecs[:, 0]
    if abs(np.dot(u, normal)) > 0.5:
        u = eigvecs[:, 1]
    u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-9)

    # coordenadas no plano
    u_coords = pts_centered @ u
    v_coords = pts_centered @ v

    width = u_coords.max() - u_coords.min()
    height = v_coords.max() - v_coords.min()

    # garante quadrado
    size = max(width, height)
    width = height = size

    return {
        "center": center,
        "normal": normal,
        "width": width,
        "height": height,
        "size": size,
    }


def identify_cubes(point_cloud, labels):
    """
    Identifica cubos em uma nuvem de pontos segmentada por rótulos.

    Lógica:
    1. Assume que 1 label = 1 cubo.
    2. Para cada label, separa faces usando as normais dos pontos.
    3. Calcula centro, normal, altura e largura de cada face; ajusta para quadrado.
    4. Para cada face, estima a posição central do cubo; a posição final é a média
       das estimativas de todas as faces visíveis.
    5. Define a orientação do cubo com base na geometria do cluster (PCA).
    6. Retorna a lista de cubos com posição, orientação e tamanho.

    Parameters
    ----------
    point_cloud : o3d.geometry.PointCloud
        Nuvem de pontos com `points` e `normals` definidos.
    labels : np.ndarray
        Vetor de rótulos de cluster (msm tamanho que número de pontos).

    Returns
    -------
    List[dict]
        Cada item contém:
        - center: np.ndarray shape (3,)
        - orientation: np.ndarray shape (3, 3) (matriz de rotação)
        - size: float (aresta do cubo)
    """
    points = np.asarray(point_cloud.points)
    if len(point_cloud.normals) == 0:
        raise ValueError("PointCloud não possui normais estimadas.")
    normals = np.asarray(point_cloud.normals)

    if points.shape[0] != labels.shape[0]:
        raise ValueError("Tamanho de labels não corresponde ao número de pontos.")

    cubes = []
    max_label = labels.max()
    if max_label < 0:
        return cubes

    for current_label in range(0, max_label + 1):
        indices = np.where(labels == current_label)[0]
        if indices.size == 0:
            continue

        cluster_points = points[indices]
        cluster_normals = normals[indices]

        # Separar faces pelo agrupamento de normais
        face_groups = _group_face_indices(cluster_normals)
        if not face_groups:
            continue

        face_centers_estimates = []
        face_sizes = []

        for g in face_groups:
            face_pts = cluster_points[g]
            face_nrm = cluster_normals[g]
            props = _estimate_face_properties(face_pts, face_nrm)
            if props is None:
                continue

            face_center = props["center"]
            normal = props["normal"]
            size = props["size"]

            # Estima o centro do cubo deslocando o centro da face pela metade da aresta
            cube_center_est = face_center + normal * (size / 2.0)
            face_centers_estimates.append(cube_center_est)
            face_sizes.append(size)

        if not face_centers_estimates:
            continue

        cube_center = np.mean(np.stack(face_centers_estimates, axis=0), axis=0)
        cube_size = float(np.median(face_sizes))

        # Orientação do cubo via PCA do cluster inteiro
        pts_centered = cluster_points - cluster_points.mean(axis=0)
        cov = pts_centered.T @ pts_centered / max(cluster_points.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]

        # Garante orientação ortonormal e mão direita
        u = R[:, 0] / (np.linalg.norm(R[:, 0]) + 1e-9)
        v = R[:, 1] / (np.linalg.norm(R[:, 1]) + 1e-9)
        w = np.cross(u, v)
        w = w / (np.linalg.norm(w) + 1e-9)
        v = np.cross(w, u)
        v = v / (np.linalg.norm(v) + 1e-9)
        R = np.stack([u, v, w], axis=1)

        cubes.append(
            {
                "center": cube_center,
                "orientation": R,
                "size": cube_size,
                "label": int(current_label),
            }
        )

    return cubes
