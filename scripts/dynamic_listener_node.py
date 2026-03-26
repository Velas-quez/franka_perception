#!/usr/bin/env python3
"""Process point clouds continuously and publish cube poses/markers."""

import threading
import traceback
from typing import Iterable, List, Optional, Sequence, Tuple

import message_filters
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from franka_perception_thiago.msg import TrackedCubeArray
from franka_perception.core.cloud_io import has_points, msg_to_xyz, rgbd_msgs_to_xyz
from franka_perception.core.publishers import (
    publish_markers,
    publish_point_clouds,
    publish_poses,
    publish_safe_area,
    publish_tracked_cubes,
    publish_tracked_labels,
    publish_tracked_markers,
)
from franka_perception.core.transforms import transform_cubes, transform_to_matrix
from franka_perception.geometry.cube_fitting import CubeEstimate
from franka_perception.params import load_params
from franka_perception.pipelines.pipeline import CubeDetectionPipeline
from franka_perception.pipelines.sam_rgbd_pipeline import SamRgbdCubeDetectionPipeline
from franka_perception.safe_area import SAFE_AREA_FRAME, safe_area_keep_mask
from franka_perception.tracking.cube_tracker import CubeTracker, DetectionObservation


class DynamicListenerNode:
    """Subscribe to a point cloud, process each frame, and publish results."""

    def __init__(self) -> None:
        self.params = load_params()
        self.pipeline_mode = self.params.pipeline_mode.strip().lower()
        if self.pipeline_mode not in {"classic", "sam_rgbd"}:
            raise ValueError("Invalid ~pipeline_mode. Use 'classic' or 'sam_rgbd'.")
        if self.pipeline_mode == "sam_rgbd" and not self.params.use_rgbd:
            raise ValueError("pipeline_mode=sam_rgbd requires ~use_rgbd:=true")
        self._target_rgbd_frames = (
            self.params.n_stack_cube_cloud if self.pipeline_mode == "sam_rgbd" else 1
        )
        self.enable_tracking = bool(self.params.enable_tracking)

        self.pipeline = None
        self.sam_pipeline = None
        self._build_detection_pipeline()

        self._tracker = None
        self._tracking_frame_id: Optional[str] = None
        if self.enable_tracking:
            self._tracker = CubeTracker(
                max_match_distance=self.params.tracking_max_match_distance,
                max_missed_frames=self.params.tracking_max_missed_frames,
                mask_max_distance=self.params.tracking_mask_max_distance,
                position_weight=self.params.tracking_position_weight,
                mask_weight=self.params.tracking_mask_weight,
                velocity_alpha=self.params.tracking_velocity_alpha,
            )

        self._points: Optional[np.ndarray] = None
        self._rgb_msg: Optional[Image] = None
        self._depth_msg: Optional[Image] = None
        self._camera_info_msg: Optional[CameraInfo] = None
        self._rgbd_frames: List[Tuple[Image, Image, CameraInfo, Header]] = []
        self._latest_header: Optional[Header] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._pose_pub = rospy.Publisher("~cube_poses", PoseArray, queue_size=5)
        self._marker_pub = rospy.Publisher("~cube_markers", MarkerArray, queue_size=5)
        self._safe_area_pub = rospy.Publisher("~safe_area", MarkerArray, queue_size=1, latch=True)
        self._reconstructed_cloud_pub = rospy.Publisher("~reconstructed_cloud", PointCloud2, queue_size=1)
        self._tracked_pub = None
        self._tracked_marker_pub = None
        self._tracked_label_pub = None
        if self.enable_tracking:
            self._tracked_pub = rospy.Publisher("~tracked_cubes", TrackedCubeArray, queue_size=5)
            self._tracked_marker_pub = rospy.Publisher("~tracked_cube_markers", MarkerArray, queue_size=5)
            self._tracked_label_pub = rospy.Publisher("~tracked_cube_labels", MarkerArray, queue_size=5)
        self._cloud_pubs = {}

        self._cloud_sub = None
        self._rgb_sub = None
        self._depth_sub = None
        self._info_sub = None
        self._sync = None

        if self.params.use_rgbd:
            self._rgb_sub = message_filters.Subscriber(self.params.rgb_topic, Image)
            self._depth_sub = message_filters.Subscriber(self.params.depth_topic, Image)
            self._info_sub = message_filters.Subscriber(self.params.camera_info_topic, CameraInfo)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub, self._info_sub], queue_size=5, slop=0.2)
            self._sync.registerCallback(self._rgbd_cb)
            rospy.loginfo(
                "Listening for RGB-D on %s + %s + %s (mode=%s, batch_frames=%d, tracking=%s)",
                self.params.rgb_topic,
                self.params.depth_topic,
                self.params.camera_info_topic,
                self.pipeline_mode,
                self._target_rgbd_frames,
                self.enable_tracking,
            )
        else:
            self._cloud_sub = rospy.Subscriber(
                self.params.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
            rospy.loginfo(
                "Listening for point clouds on %s (tracking=%s)",
                self.params.cloud_topic,
                self.enable_tracking,
            )


    def _build_detection_pipeline(self) -> None:
        self.pipeline = None
        self.sam_pipeline = None
        if self.pipeline_mode == "sam_rgbd":
            self.sam_pipeline = SamRgbdCubeDetectionPipeline(
                cube_side_length=self.params.cube_side_length,
                voxel_size=self.params.voxel_size,
                max_cubes_per_cluster=self.params.max_cubes_per_cluster,
                num_best_cubes=self.params.num_best_cubes,
                clearance=self.params.clearance,
                sam_mode=self.params.sam_mode,
                sam_checkpoint_path=self.params.sam_checkpoint_path,
                sam_model_type=self.params.sam_model_type,
                sam_device=self.params.sam_device,
                sam_points_per_side=self.params.sam_points_per_side,
                sam_pred_iou_thresh=self.params.sam_pred_iou_thresh,
                sam_stability_score_thresh=self.params.sam_stability_score_thresh,
                sam_min_mask_region_area=self.params.sam_min_mask_region_area,
                sam_prompt_text=self.params.sam_prompt_text,
                sam_prompt_box_threshold=self.params.sam_prompt_box_threshold,
                sam_prompt_text_threshold=self.params.sam_prompt_text_threshold,
                sam_grounding_model_id=self.params.sam_grounding_model_id,
                sam_segmentor_model_id=self.params.sam_segmentor_model_id,
                sam_max_masks=self.params.sam_max_masks,
                sam_min_mask_pixels=self.params.sam_min_mask_pixels,
                sam_min_depth_pixels=self.params.sam_min_depth_pixels,
                sam_mask_erosion_kernel=self.params.sam_mask_erosion_kernel,
                sam_mask_erosion_iterations=self.params.sam_mask_erosion_iterations,
                sam_min_points_per_cluster=self.params.sam_min_points_per_cluster,
                sam_max_mask_area_ratio=self.params.sam_max_mask_area_ratio,
                sam_plane_ransac_distance=self.params.sam_plane_ransac_distance,
                sam_near_plane_distance=self.params.sam_near_plane_distance,
                sam_max_near_plane_ratio=self.params.sam_max_near_plane_ratio,
                sam_min_mask_plane_height=self.params.sam_min_mask_plane_height,
                sam_max_cluster_extent_multiplier=self.params.sam_max_cluster_extent_multiplier,
                sam_max_cluster_volume_multiplier=self.params.sam_max_cluster_volume_multiplier,
                support_plane_constraint=self.params.support_plane_constraint,
                n_stack_cube_cloud=self.params.n_stack_cube_cloud,
                sam_batch_consistency_ratio=self.params.sam_batch_consistency_ratio,
                open3d_device=self.params.open3d_device,
            )
        else:
            self.pipeline = CubeDetectionPipeline(
                cube_side_length=self.params.cube_side_length,
                voxel_size=self.params.voxel_size,
                base_plane_distance=self.params.base_plane_distance,
                cluster_eps=self.params.cluster_eps,
                cluster_min_points=self.params.cluster_min_points,
                max_cubes_per_cluster=self.params.max_cubes_per_cluster,
                num_best_cubes=self.params.num_best_cubes,
                clearance=self.params.clearance,
                max_cluster_distance_from_plane_inliers=self.params.max_cluster_distance_from_plane_inliers,
                below_plane_tolerance=self.params.below_plane_tolerance,
                support_plane_constraint=self.params.support_plane_constraint,
                open3d_device=self.params.open3d_device,
            )

    def _restart_pipeline(self) -> None:
        rospy.logwarn("Restarting %s pipeline after processing failure", self.pipeline_mode)
        self._build_detection_pipeline()
        if self._tracker is not None:
            self._tracker.reset()
            self._tracking_frame_id = None

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._cloud_ready.is_set():
            return
        points = msg_to_xyz(msg)
        if not has_points(points):
            rospy.logwarn("Received empty/invalid point cloud; ignoring")
            return
        with self._lock:
            self._points = points
            self._latest_header = msg.header
            self._cloud_ready.set()
        rospy.loginfo("Captured cloud with %d points", points.shape[0])

    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if self._cloud_ready.is_set():
            return
        depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
        reconstructed_points = rgbd_msgs_to_xyz(
            rgb_msg,
            depth_msg,
            info_msg,
            depth_scale=depth_scale,
            depth_trunc=self.params.depth_trunc,
            flip=self.params.rgbd_flip,
            open3d_device=self.params.open3d_device,
        )
        if has_points(reconstructed_points):
            self._publish_reconstructed_cloud(reconstructed_points, depth_msg.header)

        if self.pipeline_mode == "sam_rgbd":
            with self._lock:
                if len(self._rgbd_frames) >= self._target_rgbd_frames:
                    return
                self._rgb_msg = rgb_msg
                self._depth_msg = depth_msg
                self._camera_info_msg = info_msg
                self._latest_header = depth_msg.header
                self._rgbd_frames.append((rgb_msg, depth_msg, info_msg, depth_msg.header))
                captured_frames = len(self._rgbd_frames)
                if captured_frames >= self._target_rgbd_frames:
                    self._cloud_ready.set()
            rospy.loginfo(
                "Captured RGB-D cloud %d/%d for SAM batch",
                captured_frames,
                self._target_rgbd_frames,
            )
            return

        if not has_points(reconstructed_points):
            rospy.logwarn("Received empty/invalid RGB-D; ignoring")
            return
        with self._lock:
            self._points = reconstructed_points
            self._latest_header = depth_msg.header
            self._cloud_ready.set()
        rospy.loginfo("Captured RGB-D cloud with %d points", reconstructed_points.shape[0])

    def _publish_reconstructed_cloud(self, points: np.ndarray, header: Optional[Header]) -> None:
        """Publish RGB-D reconstructed cloud for RViz debugging."""
        if header is None or not has_points(points):
            return

        cloud_header = Header()
        cloud_header.frame_id = header.frame_id
        cloud_header.stamp = header.stamp
        cloud_msg = pc2.create_cloud_xyz32(cloud_header, points.astype(np.float32, copy=False))
        self._reconstructed_cloud_pub.publish(cloud_msg)

    def run(self) -> None:
        if self.pipeline_mode == "sam_rgbd":
            rospy.loginfo("Waiting for RGB-D batches with %d synchronized frames...",
                          self._target_rgbd_frames)
        else:
            rospy.loginfo("Waiting for point clouds...")
        while not rospy.is_shutdown():
            if not self._cloud_ready.wait(timeout=0.2):
                continue

            with self._lock:
                points = self._points.copy() if self._points is not None else None
                rgb_msg = self._rgb_msg
                depth_msg = self._depth_msg
                info_msg = self._camera_info_msg
                header = self._latest_header
                rgbd_frames = list(self._rgbd_frames)
                self._rgbd_frames.clear()
                self._cloud_ready.clear()

            try:
                if self.pipeline_mode == "sam_rgbd":
                    if not rgbd_frames:
                        rospy.logwarn("No valid synchronized RGB-D batch received; waiting for next")
                        continue
                    depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
                    rospy.loginfo("Processing SAM batch with %d frames", len(rgbd_frames))
                    batch_frames = [
                        (frame_rgb_msg, frame_depth_msg, frame_info_msg)
                        for frame_rgb_msg, frame_depth_msg, frame_info_msg, _ in rgbd_frames
                    ]
                    result = self.sam_pipeline.process_batch(
                        batch_frames,
                        depth_scale=depth_scale,
                        depth_trunc=self.params.depth_trunc,
                        flip=self.params.rgbd_flip,
                        stop_after="all",
                    )
                    header = rgbd_frames[-1][3]
                    info_msg = rgbd_frames[-1][2]
                else:
                    if not has_points(points):
                        rospy.logwarn("No valid point cloud received; waiting for next")
                        continue
                    result = self.pipeline.process(points)

                source_cubes = list(result.cubes)
                self._publish_safe_area(header)
                filtered_source_cubes, cubes_base, header_base = self._filter_and_transform_cubes(
                    source_cubes,
                    header,
                )

                publish_poses(cubes_base, header_base, self.params.cube_side_length, self._pose_pub)
                publish_markers(cubes_base, header_base, self.params.cube_side_length, self._marker_pub)
                self._publish_cube_point_clouds(cubes_base, header_base)

                if self.enable_tracking:
                    observations = self._build_tracking_observations(
                        filtered_source_cubes,
                        cubes_base,
                        result.sam_masks,
                        info_msg,
                    )
                    tracked_cubes = self._update_tracker(observations, header_base)
                    publish_tracked_cubes(tracked_cubes, header_base, self._tracked_pub)
                    publish_tracked_markers(
                        tracked_cubes,
                        header_base,
                        self.params.cube_side_length,
                        self._tracked_marker_pub,
                    )
                    publish_tracked_labels(
                        tracked_cubes,
                        header_base,
                        self.params.cube_side_length,
                        self._tracked_label_pub,
                    )
            except Exception as exc:
                rospy.logerr(
                    "Dynamic listener processing crashed: %s\n%s",
                    exc,
                    traceback.format_exc(),
                )
                self._restart_pipeline()

    def _build_tracking_observations(self,
                                     source_cubes: Sequence[CubeEstimate],
                                     transformed_cubes: Sequence[CubeEstimate],
                                     sam_masks,
                                     camera_info_msg: Optional[CameraInfo]) -> List[DetectionObservation]:
        mask_centroids = self._associate_masks_to_cubes(source_cubes, sam_masks, camera_info_msg)
        observations = []
        for cube, mask_centroid in zip(transformed_cubes, mask_centroids):
            observations.append(DetectionObservation(cube=cube, mask_centroid=mask_centroid))
        return observations

    def _associate_masks_to_cubes(self,
                                  cubes: Sequence[CubeEstimate],
                                  sam_masks,
                                  camera_info_msg: Optional[CameraInfo]) -> List[Optional[np.ndarray]]:
        if not cubes:
            return []
        if camera_info_msg is None or not sam_masks or self.params.rgbd_flip:
            return [None] * len(cubes)

        mask_centroids = []
        for mask in sam_masks:
            centroid = self._mask_centroid(mask)
            if centroid is not None:
                mask_centroids.append(centroid)
        if not mask_centroids:
            return [None] * len(cubes)

        candidate_pairs = []
        for cube_idx, cube in enumerate(cubes):
            pixel = self._project_cube_center(cube, camera_info_msg)
            if pixel is None:
                continue
            for mask_idx, centroid in enumerate(mask_centroids):
                distance = float(np.linalg.norm(pixel - centroid))
                if distance <= float(self.params.tracking_mask_max_distance):
                    candidate_pairs.append((distance, cube_idx, mask_idx))

        assignments: List[Optional[np.ndarray]] = [None] * len(cubes)
        used_cubes = set()
        used_masks = set()
        for _, cube_idx, mask_idx in sorted(candidate_pairs, key=lambda item: item[0]):
            if cube_idx in used_cubes or mask_idx in used_masks:
                continue
            assignments[cube_idx] = mask_centroids[mask_idx]
            used_cubes.add(cube_idx)
            used_masks.add(mask_idx)
        return assignments

    @staticmethod
    def _mask_centroid(mask) -> Optional[np.ndarray]:
        mask_array = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(mask_array)
        if xs.size == 0:
            return None
        return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=float)

    @staticmethod
    def _project_cube_center(cube: CubeEstimate,
                             camera_info_msg: CameraInfo) -> Optional[np.ndarray]:
        center = np.asarray(cube.transform[:3, 3], dtype=float)
        z = float(center[2])
        if not np.isfinite(z) or z <= 1e-6:
            return None

        K = np.asarray(camera_info_msg.K, dtype=float).reshape(3, 3)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            return None

        u = fx * float(center[0]) / z + cx
        v = fy * float(center[1]) / z + cy
        if not (np.isfinite(u) and np.isfinite(v)):
            return None

        width = int(getattr(camera_info_msg, "width", 0) or 0)
        height = int(getattr(camera_info_msg, "height", 0) or 0)
        if width > 0 and height > 0 and (u < 0.0 or u >= width or v < 0.0 or v >= height):
            return None
        return np.array([u, v], dtype=float)

    def _update_tracker(self,
                        observations: Sequence[DetectionObservation],
                        header: Optional[Header]):
        if self._tracker is None:
            return []

        frame_id = header.frame_id if header is not None else None
        if frame_id:
            if self._tracking_frame_id is None:
                self._tracking_frame_id = frame_id
            elif frame_id != self._tracking_frame_id:
                rospy.logwarn(
                    "Tracking frame changed from %s to %s; resetting tracker.",
                    self._tracking_frame_id,
                    frame_id,
                )
                self._tracker.reset()
                self._tracking_frame_id = frame_id

        timestamp = None
        if header is not None:
            try:
                timestamp = float(header.stamp.to_sec())
            except AttributeError:
                timestamp = None
        return self._tracker.update(observations, timestamp=timestamp)

    def _publish_cube_point_clouds(self,
                                   cubes: Iterable[CubeEstimate],
                                   header: Optional[Header]):
        """Publish point clouds sampled from each detected cube."""
        cubes_list = list(cubes)

        active_cube_indices = set(range(len(cubes_list)))
        inactive_indices = set(self._cloud_pubs.keys()) - active_cube_indices
        for idx in inactive_indices:
            if idx in self._cloud_pubs:
                del self._cloud_pubs[idx]

        for idx in active_cube_indices:
            if idx not in self._cloud_pubs:
                pub_name = f"~cube_cloud_{idx}"
                self._cloud_pubs[idx] = rospy.Publisher(pub_name, PointCloud2, queue_size=5)
                rospy.loginfo("Created publisher for %s", pub_name)

        for idx, cube in enumerate(cubes_list):
            if idx in self._cloud_pubs:
                publish_point_clouds([cube], header, num_samples=1500,
                                     publisher=self._cloud_pubs[idx])

    def _publish_safe_area(self, header: Optional[Header]) -> None:
        safe_area_header = Header()
        safe_area_header.frame_id = SAFE_AREA_FRAME
        safe_area_header.stamp = header.stamp if header is not None else rospy.Time.now()
        publish_safe_area(
            safe_area_header,
            self.params.safe_area_width,
            self.params.safe_area_length,
            self.params.safe_area_length_offset,
            self._safe_area_pub,
        )

    def _filter_and_transform_cubes(self,
                                    cubes: Sequence[CubeEstimate],
                                    header: Optional[Header]):
        cubes = list(cubes)
        cubes_world, header_world, world_ok = self._transform_cubes_to_frame(
            cubes,
            header,
            SAFE_AREA_FRAME,
        )
        if world_ok:
            keep_mask = safe_area_keep_mask(
                cubes_world,
                self.params.safe_area_width,
                self.params.safe_area_length,
                self.params.safe_area_length_offset,
            )
            filtered_source = [cube for cube, keep in zip(cubes, keep_mask) if keep]
            filtered_world = [cube for cube, keep in zip(cubes_world, keep_mask) if keep]
            discarded = len(cubes) - len(filtered_source)
            if discarded > 0:
                rospy.loginfo("Discarded %d cube estimate(s) outside the safe area", discarded)
        else:
            filtered_source = cubes
            filtered_world = cubes_world
            if cubes:
                rospy.logwarn_throttle(
                    5.0,
                    "Could not transform cube estimates to world; safe-area filtering skipped.",
                )

        target_frame = str(self.params.target_frame).strip() if self.params.target_frame is not None else ""
        if target_frame == "" or target_frame.lower() == "none":
            return filtered_source, filtered_source, header
        if target_frame == SAFE_AREA_FRAME and world_ok:
            return filtered_source, filtered_world, header_world

        cubes_target, header_target, _ = self._transform_cubes_to_frame(
            filtered_source,
            header,
            target_frame,
        )
        return filtered_source, cubes_target, header_target

    def _transform_cubes_to_frame(self,
                                  cubes: Iterable[CubeEstimate],
                                  header: Optional[Header],
                                  target_frame: Optional[str]):
        cubes = list(cubes)
        if header is None or target_frame is None:
            return cubes, header, False

        target_frame = str(target_frame).strip()
        if target_frame == "" or target_frame.lower() == "none":
            return cubes, header, False
        if header.frame_id == target_frame:
            return cubes, header, True

        try:
            tf_msg = self._tf_buffer.lookup_transform(
                target_frame,
                header.frame_id,
                header.stamp,
                rospy.Duration(0.2),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as exc:
            rospy.logwarn("TF lookup failed (%s -> %s): %s",
                          header.frame_id, target_frame, exc)
            return cubes, header, False

        transformed = transform_cubes(cubes, transform_to_matrix(tf_msg))

        new_header = Header()
        new_header.frame_id = target_frame
        new_header.stamp = header.stamp
        return transformed, new_header, True


def main() -> None:
    rospy.init_node("dynamic_listener", anonymous=False)
    node = DynamicListenerNode()
    node.run()


if __name__ == "__main__":
    main()
