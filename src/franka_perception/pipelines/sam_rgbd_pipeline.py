#!/usr/bin/env python3
"""RGB-D cube pipeline that uses SAM masks as pre-clustered cube candidates."""

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import open3d as o3d

from franka_perception.pipelines.result_type import CubeDetectionResult

from ..core.cloud_io import depth_to_xyz, rgbd_msgs_to_numpy
from ..geometry.cube_fitting import CubeEstimate, fit_cubes_in_cluster, select_best_cubes
from ..geometry.sam_masking import SamSegmenter, erode_mask, select_mask_candidates
from ..render.sam_visualization import build_mask_overlay


class SamRgbdCubeDetectionPipeline:
    """Cube detection where SAM replaces 3D filtering+clustering."""

    def __init__(self,
                 cube_side_length: float = 0.045,
                 voxel_size: float = 0.002,
                 max_cubes_per_cluster: int = 2,
                 num_best_cubes: int = 2,
                 clearance: float = 0.015,
                 sam_mode: str = "sam3",
                 sam_checkpoint_path: str = "",
                 sam_model_type: str = "vit_b",
                 sam_device: str = "auto",
                 sam_points_per_side: int = 32,
                 sam_pred_iou_thresh: float = 0.86,
                 sam_stability_score_thresh: float = 0.92,
                 sam_min_mask_region_area: int = 150,
                 sam_prompt_text: str = "cube.",
                 sam_prompt_box_threshold: float = 0.25,
                 sam_prompt_text_threshold: float = 0.25,
                 sam_grounding_model_id: str = "IDEA-Research/grounding-dino-base",
                 sam_segmentor_model_id: str = "facebook/sam2-hiera-large",
                 sam_max_masks: int = 8,
                 sam_min_mask_pixels: int = 1200,
                 sam_min_depth_pixels: int = 600,
                 sam_mask_erosion_kernel: int = 3,
                 sam_mask_erosion_iterations: int = 1,
                 sam_min_points_per_cluster: int = 120,
                 sam_max_mask_area_ratio: float = 0.35,
                 sam_plane_ransac_distance: float = 0.006,
                 sam_near_plane_distance: float = 0.006,
                 sam_max_near_plane_ratio: float = 0.85,
                 sam_min_mask_plane_height: float = 0.012,
                 sam_max_cluster_extent_multiplier: float = 2.8,
                 sam_max_cluster_volume_multiplier: float = 7.0,
                 support_plane_constraint: str = "fix_icp",
                 n_stack_cube_cloud: int = 1) -> None:
        self.cube_side_length = cube_side_length
        self.voxel_size = voxel_size
        self.max_cubes_per_cluster = max_cubes_per_cluster
        self.num_best_cubes = num_best_cubes
        self.clearance = clearance
        self.sam_max_masks = sam_max_masks
        self.sam_min_mask_pixels = sam_min_mask_pixels
        self.sam_min_depth_pixels = sam_min_depth_pixels
        self.sam_mask_erosion_kernel = sam_mask_erosion_kernel
        self.sam_mask_erosion_iterations = sam_mask_erosion_iterations
        self.sam_min_points_per_cluster = sam_min_points_per_cluster
        self.sam_max_mask_area_ratio = sam_max_mask_area_ratio
        self.sam_plane_ransac_distance = sam_plane_ransac_distance
        self.sam_near_plane_distance = sam_near_plane_distance
        self.sam_max_near_plane_ratio = sam_max_near_plane_ratio
        self.sam_min_mask_plane_height = sam_min_mask_plane_height
        self.sam_max_cluster_extent_multiplier = sam_max_cluster_extent_multiplier
        self.sam_max_cluster_volume_multiplier = sam_max_cluster_volume_multiplier
        self.support_plane_constraint = support_plane_constraint
        self.n_stack_cube_cloud = max(1, int(n_stack_cube_cloud))
        self._stacked_cluster_history: Deque[List[Tuple[np.ndarray, np.ndarray]]] = deque()
        self.segmenter = SamSegmenter(
            mode=sam_mode,
            checkpoint_path=sam_checkpoint_path,
            model_type=sam_model_type,
            device=sam_device,
            points_per_side=sam_points_per_side,
            pred_iou_thresh=sam_pred_iou_thresh,
            stability_score_thresh=sam_stability_score_thresh,
            min_mask_region_area=sam_min_mask_region_area,
            prompt_text=sam_prompt_text,
            prompt_box_threshold=sam_prompt_box_threshold,
            prompt_text_threshold=sam_prompt_text_threshold,
            grounding_model_id=sam_grounding_model_id,
            sam_model_id=sam_segmentor_model_id,
        )

    @staticmethod
    def _to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        if points.size:
            pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    @staticmethod
    def _merge_clouds(clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        if not clouds:
            return o3d.geometry.PointCloud()
        all_points = [np.asarray(c.points) for c in clouds if len(c.points) > 0]
        if not all_points:
            return o3d.geometry.PointCloud()
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        return merged

    @staticmethod
    def _cluster_points(cluster: o3d.geometry.PointCloud) -> np.ndarray:
        return np.asarray(cluster.points, dtype=np.float64).copy()

    @staticmethod
    def _cluster_centroid(points: np.ndarray) -> np.ndarray:
        return np.mean(points, axis=0, dtype=np.float64)

    def _match_historical_clusters(self,
                                   current_centroids: List[np.ndarray],
                                   historical_frame: List[Tuple[np.ndarray, np.ndarray]],
                                   max_centroid_distance: float) -> List[Optional[np.ndarray]]:
        matches: List[Optional[np.ndarray]] = [None] * len(current_centroids)
        if not historical_frame:
            return matches

        candidate_pairs = []
        for current_idx, current_centroid in enumerate(current_centroids):
            for history_idx, (history_points, history_centroid) in enumerate(historical_frame):
                if history_points.size == 0:
                    continue
                distance = float(np.linalg.norm(current_centroid - history_centroid))
                if distance <= max_centroid_distance:
                    candidate_pairs.append((distance, current_idx, history_idx))

        if not candidate_pairs:
            return matches

        candidate_pairs.sort(key=lambda item: item[0])
        used_currents = set()
        used_history = set()
        for _, current_idx, history_idx in candidate_pairs:
            if current_idx in used_currents or history_idx in used_history:
                continue
            matches[current_idx] = historical_frame[history_idx][0]
            used_currents.add(current_idx)
            used_history.add(history_idx)
        return matches

    def _fuse_clusters_with_history(self,
                                    clusters: List[o3d.geometry.PointCloud]) -> List[o3d.geometry.PointCloud]:
        if (
            self.n_stack_cube_cloud <= 1
            or not clusters
            or not self._stacked_cluster_history
        ):
            return clusters

        current_points = [self._cluster_points(cluster) for cluster in clusters]
        current_centroids = [self._cluster_centroid(points) for points in current_points]
        fused_point_sets = [[points] for points in current_points]
        max_centroid_distance = max(
            float(self.cube_side_length) * 0.75,
            float(self.voxel_size) * 4.0,
        )

        for historical_frame in self._stacked_cluster_history:
            matches = self._match_historical_clusters(
                current_centroids,
                historical_frame,
                max_centroid_distance,
            )
            for cluster_idx, history_points in enumerate(matches):
                if history_points is not None and history_points.size > 0:
                    fused_point_sets[cluster_idx].append(history_points)

        fused_clusters = []
        min_cluster_points = max(1, int(self.sam_min_points_per_cluster))
        for cluster, point_set in zip(clusters, fused_point_sets):
            fused_points = np.vstack(point_set)
            fused_cluster = self._to_pcd(fused_points)
            if self.voxel_size > 0.0 and len(fused_cluster.points) > 0:
                fused_cluster = fused_cluster.voxel_down_sample(self.voxel_size)
            try:
                fused_cluster, _ = fused_cluster.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0)
            except RuntimeError:
                pass
            if len(fused_cluster.points) < min_cluster_points:
                fused_clusters.append(cluster)
            else:
                fused_clusters.append(fused_cluster)
        return fused_clusters

    def _build_cluster_boxes(self,
                             clusters: List[o3d.geometry.PointCloud]) -> List[o3d.geometry.OrientedBoundingBox]:
        boxes = []
        for cluster in clusters:
            if len(cluster.points) == 0:
                continue
            box = cluster.get_oriented_bounding_box()
            box.color = (0.0, 1.0, 0.0)
            boxes.append(box)
        return boxes

    def reset_stack_history(self) -> None:
        self._stacked_cluster_history.clear()

    def _update_cluster_history(self, clusters: List[o3d.geometry.PointCloud]) -> None:
        if self.n_stack_cube_cloud <= 1:
            self._stacked_cluster_history.clear()
            return

        history_frame = []
        for cluster in clusters:
            points = self._cluster_points(cluster)
            if points.size == 0:
                continue
            history_frame.append((points, self._cluster_centroid(points)))
        self._stacked_cluster_history.append(history_frame)

        max_history = self.n_stack_cube_cloud - 1
        while len(self._stacked_cluster_history) > max_history:
            self._stacked_cluster_history.popleft()

    def process(self,
                rgb_msg,
                depth_msg,
                camera_info_msg,
                *,
                depth_scale: Optional[float] = None,
                depth_trunc: float = 3.0,
                flip: bool = True,
                stop_after: str = "all") -> CubeDetectionResult:
        """Run SAM-first RGB-D pipeline up to the selected stage."""
        stage = stop_after.lower()
        if stage not in {"none", "filter", "cluster", "all"}:
            raise ValueError(f"Invalid stop_after '{stop_after}'. "
                             "Choose from: none, filter, cluster, all.")

        rgb_image, depth_m = rgbd_msgs_to_numpy(
            rgb_msg, depth_msg, depth_scale=depth_scale)
        original_points = depth_to_xyz(
            depth_m,
            camera_info_msg,
            depth_trunc=depth_trunc,
            flip=flip,
        )
        original_cloud = self._to_pcd(original_points)
        if stage == "none":
            return CubeDetectionResult(
                original_cloud=original_cloud,
                filtered_cloud=original_cloud,
                masked_cloud=None,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=None,
                plane_inlier_indices=None,
                plane_inlier_cloud=None,
                sam_rgb_image=rgb_image,
                sam_masks=[],
                sam_dino_boxes=np.zeros((0, 4), dtype=np.float32),
                sam_overlay=None,
            )

        plane_model = None
        plane_inliers = None
        plane_inlier_cloud = None
        if len(original_cloud.points) >= 300:
            try:
                model, inliers = original_cloud.segment_plane(
                    distance_threshold=max(0.001, float(self.sam_plane_ransac_distance)),
                    ransac_n=3,
                    num_iterations=1000,
                )
                plane_model = np.asarray(model, dtype=np.float64)
                plane_inliers = np.asarray(inliers, dtype=np.int32)
                plane_inlier_cloud = original_cloud.select_by_index(inliers)
            except RuntimeError:
                plane_model = None
                plane_inliers = None
                plane_inlier_cloud = None

        sam_masks = self.segmenter.generate(
            rgb_image,
            max_masks=self.sam_max_masks,
            min_mask_pixels=self.sam_min_mask_pixels,
        )
        dino_boxes = self.segmenter.get_last_boxes()
        selected_masks = select_mask_candidates(
            sam_masks,
            depth_m,
            max_masks=self.sam_max_masks,
            min_depth_pixels=self.sam_min_depth_pixels,
        )
        print(f"SAM masks: generated={len(sam_masks)} kept={len(selected_masks)}")

        clusters: List[o3d.geometry.PointCloud] = []
        viz_masks: List[np.ndarray] = []
        for mask in selected_masks:
            clean_mask = erode_mask(
                mask.segmentation,
                kernel_size=self.sam_mask_erosion_kernel,
                iterations=self.sam_mask_erosion_iterations,
            )
            viz_masks.append(clean_mask)
            area_ratio = float(clean_mask.sum()) / float(clean_mask.size)
            if area_ratio > float(self.sam_max_mask_area_ratio):
                print(f"Rejecting SAM mask by area ratio={area_ratio:.3f}")
                continue

            points = depth_to_xyz(
                depth_m,
                camera_info_msg,
                mask=clean_mask,
                depth_trunc=depth_trunc,
                flip=flip,
            )
            if points.shape[0] < self.sam_min_points_per_cluster:
                continue

            if plane_model is not None:
                n = plane_model[:3]
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-9:
                    distances = np.abs((points @ n + plane_model[3]) / n_norm)
                    near_ratio = float(np.mean(
                        distances < float(self.sam_near_plane_distance)))
                    p95_height = float(np.percentile(distances, 95.0))
                    if near_ratio > float(self.sam_max_near_plane_ratio):
                        print(f"Rejecting SAM mask by near-plane ratio={near_ratio:.3f}")
                        continue
                    if p95_height < float(self.sam_min_mask_plane_height):
                        print(f"Rejecting SAM mask by low height p95={p95_height:.4f}m")
                        continue
                    # Remove table/plane points from accepted masks.
                    points = points[distances >= float(self.sam_near_plane_distance)]
                    if points.shape[0] < self.sam_min_points_per_cluster:
                        continue

            cluster = self._to_pcd(points)
            if self.voxel_size > 0.0 and len(cluster.points) > 0:
                cluster = cluster.voxel_down_sample(self.voxel_size)
            if len(cluster.points) < self.sam_min_points_per_cluster:
                continue

            # Light denoising on each mask-cluster, not global cloud filtering.
            try:
                cluster, _ = cluster.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0)
            except RuntimeError:
                pass
            if len(cluster.points) < self.sam_min_points_per_cluster:
                continue

            obb = cluster.get_oriented_bounding_box()
            extents = np.asarray(obb.extent, dtype=np.float64)
            max_extent = float(np.max(extents))
            volume = float(np.prod(extents))
            cube_extent_limit = float(self.cube_side_length) * float(
                self.sam_max_cluster_extent_multiplier)
            cube_volume_limit = float(self.cube_side_length ** 3) * float(
                self.sam_max_cluster_volume_multiplier)
            if max_extent > cube_extent_limit:
                print(f"Rejecting SAM mask by extent={max_extent:.4f}m")
                continue
            if volume > cube_volume_limit:
                print(f"Rejecting SAM mask by volume={volume:.6f}m^3")
                continue

            clusters.append(cluster)

        estimation_clusters = self._fuse_clusters_with_history(clusters)
        cluster_cloud = self._merge_clouds(estimation_clusters)
        cluster_boxes = self._build_cluster_boxes(estimation_clusters)
        self._update_cluster_history(clusters)

        if len(cluster_cloud.points) == 0:
            cluster_cloud = original_cloud

        if stage == "filter":
            return CubeDetectionResult(
                original_cloud=original_cloud,
                filtered_cloud=cluster_cloud,
                masked_cloud=cluster_cloud,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=plane_model,
                plane_inlier_indices=plane_inliers,
                plane_inlier_cloud=plane_inlier_cloud,
                sam_rgb_image=rgb_image,
                sam_masks=viz_masks,
                sam_dino_boxes=dino_boxes,
                sam_overlay=build_mask_overlay(rgb_image, viz_masks) if viz_masks else None,
            )

        if stage == "cluster":
            return CubeDetectionResult(
                original_cloud=original_cloud,
                filtered_cloud=cluster_cloud,
                masked_cloud=cluster_cloud,
                cluster_boxes=cluster_boxes,
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=plane_model,
                plane_inlier_indices=plane_inliers,
                plane_inlier_cloud=plane_inlier_cloud,
                sam_rgb_image=rgb_image,
                sam_masks=viz_masks,
                sam_dino_boxes=dino_boxes,
                sam_overlay=build_mask_overlay(rgb_image, viz_masks) if viz_masks else None,
            )

        plane_obbs = []
        cubes: List[CubeEstimate] = []
        failed_initial_meshes = []
        for cluster in estimation_clusters:
            estimates, obbs, failed_inits = fit_cubes_in_cluster(
                cluster,
                cube_side_length=self.cube_side_length,
                max_cubes=self.max_cubes_per_cluster,
                clearance=self.clearance,
                plane_distance=0.0005,
                plane_min_inliers=20,
                support_plane_model=plane_model,
                support_plane_constraint=self.support_plane_constraint,
            )
            cubes.extend(estimates)
            plane_obbs.extend(obbs)
            failed_initial_meshes.extend(failed_inits)

        selected_cubes = select_best_cubes(cubes, self.num_best_cubes)
        print(f"Selected {len(selected_cubes)} best cubes out of {len(cubes)} total")
        for idx, cube in enumerate(selected_cubes):
            center = cube.transform[:3, 3]
            print(
                f"Cube[{idx}] center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
                f"fitness={cube.icp_fitness:.4f}"
            )

        return CubeDetectionResult(
            original_cloud=original_cloud,
            filtered_cloud=cluster_cloud,
            masked_cloud=cluster_cloud,
            cluster_boxes=cluster_boxes,
            plane_obbs=plane_obbs,
            cubes=selected_cubes,
            failed_initial_meshes=failed_initial_meshes,
            plane_model=plane_model,
            plane_inlier_indices=plane_inliers,
            plane_inlier_cloud=plane_inlier_cloud,
            sam_rgb_image=rgb_image,
            sam_masks=viz_masks,
            sam_dino_boxes=dino_boxes,
            sam_overlay=build_mask_overlay(rgb_image, viz_masks) if viz_masks else None,
        )
