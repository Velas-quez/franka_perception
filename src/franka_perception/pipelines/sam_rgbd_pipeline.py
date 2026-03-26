#!/usr/bin/env python3
"""RGB-D cube pipeline that uses SAM masks as pre-clustered cube candidates."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d

from franka_perception.pipelines.result_type import CubeDetectionResult

from ..core.cloud_io import depth_to_xyz, rgbd_msgs_to_numpy
from ..core.open3d_backend import voxel_downsample_legacy_point_cloud
from ..geometry.cube_fitting import CubeEstimate, fit_cubes_in_cluster, select_best_cubes
from ..geometry.sam_masking import SamSegmenter, erode_mask, select_mask_candidates
from ..render.sam_visualization import build_mask_overlay


@dataclass
class _SamMaskCandidate:
    segmentation: np.ndarray
    cluster: o3d.geometry.PointCloud
    centroid: np.ndarray


@dataclass
class _SamFrameData:
    original_cloud: o3d.geometry.PointCloud
    plane_model: Optional[np.ndarray]
    plane_inliers: Optional[np.ndarray]
    plane_inlier_cloud: Optional[o3d.geometry.PointCloud]
    rgb_image: np.ndarray
    viz_masks: List[np.ndarray]
    dino_boxes: np.ndarray
    candidates: List[_SamMaskCandidate]


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
                 n_stack_cube_cloud: int = 1,
                 sam_batch_consistency_ratio: float = 1.0,
                 open3d_device: str = "auto") -> None:
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
        self.sam_batch_consistency_ratio = min(
            1.0, max(0.0, float(sam_batch_consistency_ratio)))
        self.open3d_device = open3d_device
        self.sam_batch_mask_iou_threshold = 0.35
        self.sam_batch_mask_dilation_kernel = 3
        self.sam_batch_mask_dilation_iterations = 1
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

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel_size = int(self.sam_batch_mask_dilation_kernel)
        iterations = int(self.sam_batch_mask_dilation_iterations)
        if kernel_size <= 1 or iterations <= 0:
            return mask.astype(bool)

        try:
            import cv2
        except ImportError:
            return mask.astype(bool)

        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
        return dilated.astype(bool)

    def _mask_iou(self, first: np.ndarray, second: np.ndarray) -> float:
        first_mask = self._dilate_mask(first)
        second_mask = self._dilate_mask(second)
        union = int(np.count_nonzero(first_mask | second_mask))
        if union == 0:
            return 0.0
        intersection = int(np.count_nonzero(first_mask & second_mask))
        return float(intersection) / float(union)

    def _build_none_result(self,
                           rgb_msg,
                           depth_msg,
                           camera_info_msg,
                           *,
                           depth_scale: Optional[float],
                           depth_trunc: float,
                           flip: bool) -> CubeDetectionResult:
        rgb_image, depth_m = rgbd_msgs_to_numpy(
            rgb_msg, depth_msg, depth_scale=depth_scale)
        original_points = depth_to_xyz(
            depth_m,
            camera_info_msg,
            depth_trunc=depth_trunc,
            flip=flip,
        )
        original_cloud = self._to_pcd(original_points)
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

    def _extract_frame_data(self,
                            rgb_msg,
                            depth_msg,
                            camera_info_msg,
                            *,
                            depth_scale: Optional[float],
                            depth_trunc: float,
                            flip: bool) -> _SamFrameData:
        rgb_image, depth_m = rgbd_msgs_to_numpy(
            rgb_msg, depth_msg, depth_scale=depth_scale)
        original_points = depth_to_xyz(
            depth_m,
            camera_info_msg,
            depth_trunc=depth_trunc,
            flip=flip,
        )
        original_cloud = self._to_pcd(original_points)

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

        candidates: List[_SamMaskCandidate] = []
        viz_masks: List[np.ndarray] = []
        for mask in selected_masks:
            clean_mask = erode_mask(
                mask.segmentation,
                kernel_size=self.sam_mask_erosion_kernel,
                iterations=self.sam_mask_erosion_iterations,
            )
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
                    points = points[distances >= float(self.sam_near_plane_distance)]
                    if points.shape[0] < self.sam_min_points_per_cluster:
                        continue

            cluster = self._to_pcd(points)
            if self.voxel_size > 0.0 and len(cluster.points) > 0:
                cluster = voxel_downsample_legacy_point_cloud(
                    cluster,
                    self.voxel_size,
                    device=self.open3d_device,
                )
            if len(cluster.points) < self.sam_min_points_per_cluster:
                continue

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

            viz_masks.append(clean_mask)
            candidates.append(_SamMaskCandidate(
                segmentation=clean_mask,
                cluster=cluster,
                centroid=self._cluster_centroid(np.asarray(cluster.points, dtype=np.float64)),
            ))

        return _SamFrameData(
            original_cloud=original_cloud,
            plane_model=plane_model,
            plane_inliers=plane_inliers,
            plane_inlier_cloud=plane_inlier_cloud,
            rgb_image=rgb_image,
            viz_masks=viz_masks,
            dino_boxes=dino_boxes,
            candidates=candidates,
        )

    def _merge_candidate_frames(self,
                                candidate_frames: Sequence[List[_SamMaskCandidate]]) -> List[o3d.geometry.PointCloud]:
        if not candidate_frames:
            return []

        min_cluster_points = max(1, int(self.sam_min_points_per_cluster))
        required_frames = max(
            1,
            int(np.ceil(float(len(candidate_frames)) * self.sam_batch_consistency_ratio)),
        )
        tracks = []

        for frame_idx, frame_candidates in enumerate(candidate_frames):
            used_tracks = set()
            for candidate in frame_candidates:
                best_track_idx = None
                best_iou = 0.0
                for track_idx, track in enumerate(tracks):
                    if track_idx in used_tracks or frame_idx in track["frame_indices"]:
                        continue
                    iou = self._mask_iou(candidate.segmentation, track["mask"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track_idx = track_idx

                if best_track_idx is not None and best_iou >= self.sam_batch_mask_iou_threshold:
                    track = tracks[best_track_idx]
                    track["mask"] = np.logical_or(track["mask"], candidate.segmentation)
                    track["point_sets"].append(self._cluster_points(candidate.cluster))
                    track["centroid"] = track["centroid"] + (
                        candidate.centroid - track["centroid"]
                    ) / float(len(track["point_sets"]))
                    track["frame_indices"].add(frame_idx)
                    used_tracks.add(best_track_idx)
                else:
                    tracks.append({
                        "mask": candidate.segmentation.copy(),
                        "point_sets": [self._cluster_points(candidate.cluster)],
                        "centroid": candidate.centroid.copy(),
                        "frame_indices": {frame_idx},
                    })
                    used_tracks.add(len(tracks) - 1)

        tracks = [
            track for track in tracks
            if len(track["frame_indices"]) >= required_frames
        ]
        tracks.sort(key=lambda track: tuple(np.round(track["centroid"], 6)))

        merged_clusters = []
        for track in tracks:
            merged_points = np.vstack(track["point_sets"])
            merged_cluster = self._to_pcd(merged_points)
            if self.voxel_size > 0.0 and len(merged_cluster.points) > 0:
                merged_cluster = voxel_downsample_legacy_point_cloud(
                    merged_cluster,
                    self.voxel_size,
                    device=self.open3d_device,
                )
            try:
                merged_cluster, _ = merged_cluster.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0)
            except RuntimeError:
                pass
            if len(merged_cluster.points) >= min_cluster_points:
                merged_clusters.append(merged_cluster)

        print(
            "SAM temporal consistency: "
            f"kept={len(merged_clusters)} tracks with min_frames={required_frames}/{len(candidate_frames)}"
        )
        return merged_clusters

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

    def _build_result(self,
                      frame_data: _SamFrameData,
                      estimation_clusters: List[o3d.geometry.PointCloud],
                      *,
                      stop_after: str) -> CubeDetectionResult:
        cluster_cloud = self._merge_clouds(estimation_clusters)
        if len(cluster_cloud.points) == 0:
            cluster_cloud = frame_data.original_cloud
        cluster_boxes = self._build_cluster_boxes(estimation_clusters)

        if stop_after == "filter":
            return CubeDetectionResult(
                original_cloud=frame_data.original_cloud,
                filtered_cloud=cluster_cloud,
                masked_cloud=cluster_cloud,
                cluster_boxes=[],
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=frame_data.plane_model,
                plane_inlier_indices=frame_data.plane_inliers,
                plane_inlier_cloud=frame_data.plane_inlier_cloud,
                sam_rgb_image=frame_data.rgb_image,
                sam_masks=frame_data.viz_masks,
                sam_dino_boxes=frame_data.dino_boxes,
                sam_overlay=build_mask_overlay(frame_data.rgb_image, frame_data.viz_masks)
                if frame_data.viz_masks else None,
            )

        if stop_after == "cluster":
            return CubeDetectionResult(
                original_cloud=frame_data.original_cloud,
                filtered_cloud=cluster_cloud,
                masked_cloud=cluster_cloud,
                cluster_boxes=cluster_boxes,
                plane_obbs=[],
                cubes=[],
                failed_initial_meshes=[],
                plane_model=frame_data.plane_model,
                plane_inlier_indices=frame_data.plane_inliers,
                plane_inlier_cloud=frame_data.plane_inlier_cloud,
                sam_rgb_image=frame_data.rgb_image,
                sam_masks=frame_data.viz_masks,
                sam_dino_boxes=frame_data.dino_boxes,
                sam_overlay=build_mask_overlay(frame_data.rgb_image, frame_data.viz_masks)
                if frame_data.viz_masks else None,
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
                support_plane_model=frame_data.plane_model,
                support_plane_constraint=self.support_plane_constraint,
                open3d_device=self.open3d_device,
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
            original_cloud=frame_data.original_cloud,
            filtered_cloud=cluster_cloud,
            masked_cloud=cluster_cloud,
            cluster_boxes=cluster_boxes,
            plane_obbs=plane_obbs,
            cubes=selected_cubes,
            failed_initial_meshes=failed_initial_meshes,
            plane_model=frame_data.plane_model,
            plane_inlier_indices=frame_data.plane_inliers,
            plane_inlier_cloud=frame_data.plane_inlier_cloud,
            sam_rgb_image=frame_data.rgb_image,
            sam_masks=frame_data.viz_masks,
            sam_dino_boxes=frame_data.dino_boxes,
            sam_overlay=build_mask_overlay(frame_data.rgb_image, frame_data.viz_masks)
            if frame_data.viz_masks else None,
        )

    def process_batch(self,
                      frames: Sequence[Tuple[object, object, object]],
                      *,
                      depth_scale: Optional[float] = None,
                      depth_trunc: float = 3.0,
                      flip: bool = True,
                      stop_after: str = "all") -> CubeDetectionResult:
        stage = stop_after.lower()
        if stage not in {"none", "filter", "cluster", "all"}:
            raise ValueError(f"Invalid stop_after '{stop_after}'. "
                             "Choose from: none, filter, cluster, all.")
        if not frames:
            raise ValueError("process_batch requires at least one RGB-D frame")

        last_rgb_msg, last_depth_msg, last_camera_info_msg = frames[-1]
        if stage == "none":
            return self._build_none_result(
                last_rgb_msg,
                last_depth_msg,
                last_camera_info_msg,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                flip=flip,
            )

        frame_data_list = [
            self._extract_frame_data(
                rgb_msg,
                depth_msg,
                camera_info_msg,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                flip=flip,
            )
            for rgb_msg, depth_msg, camera_info_msg in frames
        ]
        last_frame_data = frame_data_list[-1]
        estimation_clusters = self._merge_candidate_frames(
            [frame_data.candidates for frame_data in frame_data_list]
        )
        return self._build_result(
            last_frame_data,
            estimation_clusters,
            stop_after=stage,
        )

    def process(self,
                rgb_msg,
                depth_msg,
                camera_info_msg,
                *,
                depth_scale: Optional[float] = None,
                depth_trunc: float = 3.0,
                flip: bool = True,
                stop_after: str = "all") -> CubeDetectionResult:
        return self.process_batch(
            [(rgb_msg, depth_msg, camera_info_msg)],
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            flip=flip,
            stop_after=stop_after,
        )
