#!/usr/bin/env python3
"""Render the first received ZED2 point cloud with Open3D."""

from itertools import combinations, product
import threading
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

from utils.filter import filter_point_cloud
from utils.cluster import cluster_point_cloud


class SingleCloudViewer:
    """Subscribe to a point cloud and render only the first valid message."""

    def __init__(self) -> None:
        self._cloud_topic = rospy.get_param(
            "~cloud_topic", "/zed2/zed_node/point_cloud/cloud_registered")

        self._cloud_sub = rospy.Subscriber(self._cloud_topic, PointCloud2,
                                           self._cloud_cb, queue_size=1)

        self._cube_side_length = rospy.get_param("~cube_side_length", 0.045)
        self._camera_position = np.asarray(
            rospy.get_param("~camera_position", [0.0, 0.0, 0.0]),
            dtype=np.float64
        )

        self._points: Optional[np.ndarray] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()
        rospy.loginfo("Waiting for point cloud on %s", self._cloud_topic)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._cloud_ready.is_set():
            return

        try:
            points = self._msg_to_xyz(msg)
        except Exception as exc:
            rospy.logerr("Could not parse point cloud: %s", exc)
            return

        if points.size == 0:
            rospy.logwarn("Received empty point cloud; ignoring")
            return

        with self._lock:
            self._points = points
            self._cloud_ready.set()
        rospy.loginfo("Captured first cloud with %d points", points.shape[0])

        try:
            self._cloud_sub.unregister()
        except Exception:
            pass

    @staticmethod
    def _msg_to_xyz(msg: PointCloud2) -> np.ndarray:
        points = [(x, y, z) for x, y, z in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)]
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        arr = np.asarray(points, dtype=np.float64)
        if not np.isfinite(arr).all():
            arr = arr[np.all(np.isfinite(arr), axis=1)]
        return arr

    def _estimate_cube_pose(self, planes):
        """Estimate cube pose from up to 3 visible faces (no parallel requirement)."""
        if not planes:
            return None

        # Pick the best subset of up to 3 planes with near-orthogonal normals.
        if len(planes) <= 3:
            chosen_planes = planes
        else:
            best = None
            for triplet in combinations(planes, 3):
                normals = [p["normal"] / np.linalg.norm(p["normal"]) for p in triplet]
                dots = [
                    abs(float(np.dot(normals[0], normals[1]))),
                    abs(float(np.dot(normals[0], normals[2]))),
                    abs(float(np.dot(normals[1], normals[2])))
                ]
                score = max(dots)
                if best is None or score < best[0]:
                    best = (score, triplet)
            chosen_planes = list(best[1]) if best else planes[:3]

        normals = [p["normal"] / (np.linalg.norm(p["normal"]) + 1e-12) for p in chosen_planes]
        centroids = [p["centroid"] for p in chosen_planes]
        half_side = 0.5 * self._cube_side_length

        # Try all sign combinations to find the most consistent center estimate.
        best_center = None
        best_normals = None
        best_spread = np.inf
        for signs in product([-1.0, 1.0], repeat=len(normals)):
            candidate_normals = []
            candidate_centers = []
            for n, c, s in zip(normals, centroids, signs):
                oriented_n = s * n
                candidate_normals.append(oriented_n)
                candidate_centers.append(c - oriented_n * half_side)
            centers_arr = np.stack(candidate_centers)
            center_mean = centers_arr.mean(axis=0)
            spread = np.linalg.norm(centers_arr - center_mean, axis=1).mean()
            if spread < best_spread:
                best_spread = spread
                best_center = center_mean
                best_normals = candidate_normals

        if best_center is None or best_normals is None:
            return None

        axes = list(best_normals)
        # Complete an orthonormal basis if fewer than 3 planes were seen.
        if len(axes) == 1:
            aux = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(aux, axes[0]))) > 0.9:
                aux = np.array([0.0, 1.0, 0.0])
            axes.append(np.cross(axes[0], aux))
        if len(axes) == 2:
            cross_axis = np.cross(axes[0], axes[1])
            if np.linalg.norm(cross_axis) < 1e-6:
                cross_axis = np.array([0.0, 0.0, 1.0])
            axes.append(cross_axis)

        if len(axes) == 1:
            to_cam = self._camera_position - best_center
            if np.linalg.norm(to_cam) > 1e-9 and float(np.dot(axes[0], to_cam)) < 0:
                axes[0] *= -1.0

        axis_matrix = np.stack([a / (np.linalg.norm(a) + 1e-12) for a in axes], axis=1)
        U, _, Vt = np.linalg.svd(axis_matrix)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R, best_center

    def run(self) -> None:
        rospy.loginfo("Waiting for the first valid cloud...")
        while not rospy.is_shutdown():
            if self._cloud_ready.wait(timeout=0.2):
                break

        if rospy.is_shutdown():
            return

        with self._lock:
            points = self._points.copy() if self._points is not None else None

        if points is None or points.size == 0:
            rospy.logerr("No valid point cloud received before shutdown")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        geometries = []

        # Filtering
        pcd = filter_point_cloud(pcd, voxel_size=0.002)
        
        # Plane removal
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=1000)
        pcd = pcd.select_by_index(inliers, invert=True)
          
        # Clustering the point cloud
        cluster_boxes, clusters = cluster_point_cloud(pcd, eps=0.005, min_points=10, render_boxes=False)
        cluster_count = len(clusters)
        
        
        ### Cubes identification
        # selected_clusters = [clusters[5], clusters[7], clusters[8]]  # Manually select clusters corresponding to cubes
        selected_clusters = clusters
        
        # Plane clustering - Detect planes in each cluster
        obbs = []
        estimated_cubes = []
        for cluster_idx, selected_cluster in enumerate(selected_clusters):
            remaining_pcd = selected_cluster
            max_cubes = 2
            clearance = 0.015  # meters to carve out points after a cube is fitted
            for cube_idx in range(max_cubes):
                if len(remaining_pcd.points) < 30:
                    rospy.loginfo("Cluster %d, cube %d: not enough points (%d) left", cluster_idx, cube_idx, len(remaining_pcd.points))
                    break

                plane_infos = []
                temp_pcd = remaining_pcd
                while len(temp_pcd.points) > 20:
                    plane_model, inliers = temp_pcd.segment_plane(distance_threshold=0.0005,
                                                        ransac_n=3,
                                                        num_iterations=1000)
                inlier_cloud = remaining_pcd.select_by_index(inliers)
                outlier_cloud = remaining_pcd.select_by_index(inliers, invert=True)
                remaining_pcd = outlier_cloud
                
                if len(inlier_cloud.points) < 20:
                    continue
                
                # Compute oriented bounding box
                obb = inlier_cloud.get_oriented_bounding_box()
                obb.color = (1, 0, 0)
                obbs.append(obb)
                normal = np.asarray(plane_model[:3], dtype=np.float64)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-9:
                    continue
                normal /= normal_norm
                centroid = np.asarray(inlier_cloud.points).mean(axis=0)
                plane_infos.append({"normal": normal, "centroid": centroid})

            # Estimate cube pose from planes
            if len(plane_infos) >= 3:
                cube_pose = self._estimate_cube_pose(plane_infos)
                if cube_pose is None:
                    rospy.loginfo("Cluster %d, cube %d: cube_pose estimation failed", cluster_idx, cube_idx)
                    break

                # Fit cube with ICP when pose exists
                R_init, t_init = cube_pose
                cube_mesh = o3d.geometry.TriangleMesh.create_box(
                    self._cube_side_length,
                    self._cube_side_length,
                    self._cube_side_length
                )
                cube_mesh.translate(-cube_mesh.get_center())
                source_pcd = cube_mesh.sample_points_poisson_disk(1500)
                target_pcd = remaining_pcd.voxel_down_sample(voxel_size=0.002)
                if len(target_pcd.points) == 0:
                    rospy.loginfo("Cluster %d, cube %d: target_pcd empty after downsample", cluster_idx, cube_idx)
                    break
                target_pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=50)
                )
                init_T = np.eye(4)
                init_T[:3, :3] = R_init
                init_T[:3, 3] = t_init
                icp_coarse = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    0.008,
                    init_T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                icp_result = o3d.pipelines.registration.registration_icp(
                    source_pcd,
                    target_pcd,
                    0.003,
                    icp_coarse.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
                if icp_result.fitness <= 0.1 or icp_result.inlier_rmse > 0.005:
                    icp_wide = o3d.pipelines.registration.registration_icp(
                        source_pcd,
                        target_pcd,
                        0.01,
                        icp_coarse.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane()
                    )
                    if icp_wide.fitness > icp_result.fitness:
                        icp_result = icp_wide
                best_T = icp_result.transformation
                best_fit = icp_result.fitness
                if icp_result.fitness <= 0.08:
                    if icp_coarse.fitness > best_fit:
                        best_fit = icp_coarse.fitness
                        best_T = icp_coarse.transformation
                use_fallback_pose = False
                if best_fit <= 0.03:
                    rospy.loginfo("Cluster %d, cube %d: weak cube fit (fitness=%.3f), using init pose", cluster_idx, cube_idx, best_fit)
                    T_final = init_T.copy()
                    use_fallback_pose = True
                else:
                    T_final = best_T.copy()

                R_fit = T_final[:3, :3]
                U, _, Vt = np.linalg.svd(R_fit)
                R_ortho = U @ Vt
                if np.linalg.det(R_ortho) < 0:
                    U[:, -1] *= -1
                    R_ortho = U @ Vt
                T_final[:3, :3] = R_ortho

                cube_mesh_est = o3d.geometry.TriangleMesh.create_box(
                    self._cube_side_length,
                    self._cube_side_length,
                    self._cube_side_length
                )
                cube_mesh_est.translate(-cube_mesh_est.get_center())
                cube_mesh_est.transform(T_final)
                cube_mesh_est.paint_uniform_color([0.1, 0.4, 0.9])
                estimated_cubes.append(cube_mesh_est)

                # Remove points near the fitted cube to search for a second one
                cube_pcd = cube_mesh_est.sample_points_uniformly(400)
                distances = remaining_pcd.compute_point_cloud_distance(cube_pcd)
                keep_indices = [i for i, d in enumerate(distances) if d > clearance]
                remaining_pcd = remaining_pcd.select_by_index(keep_indices)
                if len(remaining_pcd.points) < 30:
                    break

        # Rendering
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geometries = [pcd]
        geometries.extend(cluster_boxes)
        geometries.extend(obbs)
        geometries.extend(estimated_cubes)

        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name="ZED Point Cloud",
                width=960,
                height=540,
            )
        except Exception as exc:
            rospy.logerr("Failed to render point cloud: %s", exc)


def main() -> None:
    rospy.init_node("listener", anonymous=False)
    viewer = SingleCloudViewer()
    viewer.run()


if __name__ == "__main__":
    main()
