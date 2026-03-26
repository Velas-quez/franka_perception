#!/usr/bin/env python3
"""Render point-cloud and RGB-D inputs with Open3D using the shared pipeline."""

import threading
from typing import List, Optional, Tuple

import message_filters
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header

from franka_perception.core.cloud_io import has_points, msg_to_xyz, rgbd_msgs_to_xyz
from franka_perception.core.transforms import (
    transform_detection_result,
    transform_points,
    transform_to_matrix,
)
from franka_perception.params import load_params
from franka_perception.pipelines.pipeline import CubeDetectionPipeline
from franka_perception.pipelines.sam_rgbd_pipeline import SamRgbdCubeDetectionPipeline
from franka_perception.render.sam_visualization import show_dino_and_sam
from franka_perception.render.visualization import draw
from franka_perception.safe_area import (
    SAFE_AREA_FRAME,
    build_safe_area_geometries,
    safe_area_keep_mask,
)


class SingleCloudNode:
    """Capture point-cloud and RGB-D inputs once, process one, and render both."""

    def __init__(self, stage: Optional[str] = None, show_original_cloud: Optional[bool] = None) -> None:
        self.params = load_params()
        self.stage = (stage or rospy.get_param("~stage", "all")).strip().lower()
        if self.stage not in {"none", "filter", "cluster", "all"}:
            raise ValueError("Invalid ~stage. Use: none|filter|cluster|all")
        self.pipeline_mode = self.params.pipeline_mode.strip().lower()
        if self.pipeline_mode not in {"classic", "sam_rgbd"}:
            raise ValueError("Invalid ~pipeline_mode. Use 'classic' or 'sam_rgbd'.")
        if show_original_cloud is None:
            self.show_original_cloud = bool(rospy.get_param("~show_original_cloud", False))
        else:
            self.show_original_cloud = bool(show_original_cloud)
        self.show_input_clouds = bool(rospy.get_param("~show_input_clouds", False))
        if self.pipeline_mode == "sam_rgbd" and not self.params.use_rgbd:
            raise ValueError("pipeline_mode=sam_rgbd requires ~use_rgbd:=true")
        self._target_rgbd_frames = (
            self.params.n_stack_cube_cloud if self.pipeline_mode == "sam_rgbd" else 1
        )

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
                sam_rerun_on_extent_rejection=self.params.sam_rerun_on_extent_rejection,
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
        self._cloud_topic_points: Optional[np.ndarray] = None
        self._rgbd_points: Optional[np.ndarray] = None
        self._rgb_msg: Optional[Image] = None
        self._depth_msg: Optional[Image] = None
        self._camera_info_msg: Optional[CameraInfo] = None
        self._rgbd_frames: List[Tuple[Image, Image, CameraInfo, Header]] = []
        self._cloud_header: Optional[Header] = None
        self._rgbd_header: Optional[Header] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._cloud_sub = None
        self._rgb_sub = None
        self._depth_sub = None
        self._info_sub = None
        self._sync = None
        self._latest_info: Optional[CameraInfo] = None

        self._cloud_sub = rospy.Subscriber(
            self.params.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        self._rgb_sub = message_filters.Subscriber(self.params.rgb_topic, Image)
        self._depth_sub = message_filters.Subscriber(self.params.depth_topic, Image)
        self._info_sub = rospy.Subscriber(
            self.params.camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], queue_size=5, slop=0.2)
        self._sync.registerCallback(self._rgbd_cb)
        rospy.loginfo(
            "Waiting for point cloud on %s and RGB-D on %s + %s (mode=%s, pipeline_source=%s)",
            self.params.cloud_topic,
            self.params.rgb_topic,
            self.params.depth_topic,
            self.pipeline_mode,
            "rgbd" if self.params.use_rgbd else "point_cloud",
        )

    def _ready_locked(self) -> bool:
        has_cloud_topic = has_points(self._cloud_topic_points)
        has_rgbd_cloud = has_points(self._rgbd_points)
        if self.pipeline_mode == "sam_rgbd":
            return has_cloud_topic and len(self._rgbd_frames) >= self._target_rgbd_frames
        return has_cloud_topic and has_rgbd_cloud

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._cloud_ready.is_set():
            return
        points = msg_to_xyz(msg)
        if not has_points(points):
            rospy.logwarn("Received empty/invalid point cloud; ignoring")
            return
        with self._lock:
            if has_points(self._cloud_topic_points):
                return
            self._cloud_topic_points = points
            self._cloud_header = msg.header
            if self._ready_locked():
                self._cloud_ready.set()
        rospy.loginfo("Captured point-cloud topic with %d points", points.shape[0])
        self._unregister_cloud_subscriber()

    def _unregister_cloud_subscriber(self) -> None:
        try:
            if self._cloud_sub is not None:
                self._cloud_sub.unregister()
        except Exception:
            pass

    def _info_cb(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_info = msg

    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._cloud_ready.is_set():
            return
        with self._lock:
            info_msg = self._latest_info
        if info_msg is None:
            rospy.logwarn_throttle(2.0, "Waiting for camera_info on %s",
                                   self.params.camera_info_topic)
            return
        depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
        points = rgbd_msgs_to_xyz(
            rgb_msg,
            depth_msg,
            info_msg,
            depth_scale=depth_scale,
            depth_trunc=self.params.depth_trunc,
            flip=self.params.rgbd_flip,
            open3d_device=self.params.open3d_device,
        )
        if not has_points(points):
            rospy.logwarn("Received empty/invalid RGB-D; ignoring")
            return

        captured_frames = 1
        should_unregister = False
        with self._lock:
            if self.pipeline_mode == "sam_rgbd":
                if len(self._rgbd_frames) >= self._target_rgbd_frames:
                    return
                self._rgbd_points = points
                self._rgb_msg = rgb_msg
                self._depth_msg = depth_msg
                self._camera_info_msg = info_msg
                self._rgbd_header = depth_msg.header
                self._rgbd_frames.append((rgb_msg, depth_msg, info_msg, depth_msg.header))
                captured_frames = len(self._rgbd_frames)
                should_unregister = captured_frames >= self._target_rgbd_frames
            else:
                if has_points(self._rgbd_points):
                    return
                self._rgbd_points = points
                self._rgbd_header = depth_msg.header
                captured_frames = 1
                should_unregister = True

            if self._ready_locked():
                self._cloud_ready.set()

        if self.pipeline_mode == "sam_rgbd":
            rospy.loginfo(
                "Captured RGB-D cloud %d/%d with %d points",
                captured_frames,
                self._target_rgbd_frames,
                points.shape[0],
            )
        else:
            rospy.loginfo("Captured RGB-D cloud with %d points", points.shape[0])

        if should_unregister:
            self._unregister_rgbd_subscribers()

    def _unregister_rgbd_subscribers(self) -> None:
        for sub in (self._rgb_sub, self._depth_sub):
            try:
                if sub is not None:
                    sub.unregister()
            except Exception:
                pass
        try:
            if self._info_sub is not None:
                self._info_sub.unregister()
        except Exception:
            pass

    def run(self) -> None:
        if self.pipeline_mode == "sam_rgbd":
            rospy.loginfo(
                "Waiting for one valid point-cloud topic frame and %d synchronized RGB-D frames...",
                self._target_rgbd_frames,
            )
        else:
            rospy.loginfo("Waiting for one valid point-cloud topic frame and one valid RGB-D cloud...")
        while not rospy.is_shutdown():
            if self._cloud_ready.wait(timeout=0.2):
                break

        if rospy.is_shutdown():
            return

        with self._lock:
            cloud_topic_points = (
                self._cloud_topic_points.copy()
                if self._cloud_topic_points is not None else None
            )
            rgbd_points = self._rgbd_points.copy() if self._rgbd_points is not None else None
            rgb_msg = self._rgb_msg
            depth_msg = self._depth_msg
            info_msg = self._camera_info_msg
            rgbd_frames = list(self._rgbd_frames)
            cloud_header = self._cloud_header
            rgbd_header = self._rgbd_header

        rospy.loginfo("Rendering pipeline stage: %s", self.stage)
        result_header = rgbd_header if self.params.use_rgbd else cloud_header
        if self.pipeline_mode == "sam_rgbd":
            if not rgbd_frames:
                rospy.logerr("No valid synchronized RGB-D frame received before shutdown")
                return
            depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
            rospy.loginfo("Processing SAM batch with %d frames", len(rgbd_frames))
            result = self.sam_pipeline.process_batch(
                [(frame_rgb, frame_depth, frame_info) for frame_rgb, frame_depth, frame_info, _ in rgbd_frames],
                depth_scale=depth_scale,
                depth_trunc=self.params.depth_trunc,
                flip=self.params.rgbd_flip,
                stop_after=self.stage,
            )
            rgb_msg = rgbd_frames[-1][0]
            result_header = rgbd_frames[-1][3]

            if self.params.sam_show_windows and result.sam_rgb_image is not None:
                rgb_arr = np.asarray(result.sam_rgb_image)
                finite = np.isfinite(rgb_arr)
                if np.any(finite):
                    rgb_min = float(np.min(rgb_arr[finite]))
                    rgb_max = float(np.max(rgb_arr[finite]))
                else:
                    rgb_min = float("nan")
                    rgb_max = float("nan")
                rospy.loginfo(
                    "SAM preview image: encoding=%s shape=%s dtype=%s min=%.3f max=%.3f",
                    getattr(rgb_msg, "encoding", "unknown"),
                    tuple(rgb_arr.shape),
                    rgb_arr.dtype,
                    rgb_min,
                    rgb_max,
                )
                overlay = show_dino_and_sam(
                    result.sam_rgb_image,
                    result.sam_dino_boxes,
                    result.sam_masks or [],
                    wait_ms=self.params.sam_window_wait_ms,
                    title_prefix="SAM",
                )
                if overlay is not None:
                    result.sam_overlay = overlay
        else:
            points = rgbd_points if self.params.use_rgbd else cloud_topic_points
            if not has_points(points):
                rospy.logerr("No valid point cloud received before shutdown")
                return
            result = self.pipeline.process(points, stop_after=self.stage)

        extra_clouds = None
        if self.show_input_clouds:
            extra_clouds = []
            if has_points(cloud_topic_points):
                extra_cloud = self._transform_points_to_world(cloud_topic_points, cloud_header)
                extra_clouds.append((extra_cloud, [0.15, 0.45, 0.95]))
            if has_points(rgbd_points):
                extra_cloud = self._transform_points_to_world(rgbd_points, rgbd_header)
                extra_clouds.append((extra_cloud, [0.95, 0.55, 0.15]))

        result, world_ok = self._transform_result_to_world(result, result_header)
        safe_area_geometries = []
        if world_ok:
            keep_mask = safe_area_keep_mask(
                result.cubes,
                self.params.safe_area_width,
                self.params.safe_area_length,
            )
            kept_cubes = [cube for cube, keep in zip(result.cubes, keep_mask) if keep]
            discarded = len(result.cubes) - len(kept_cubes)
            if discarded > 0:
                rospy.loginfo("Discarded %d cube estimate(s) outside the safe area", discarded)
            result.cubes = kept_cubes
            safe_area_geometries = build_safe_area_geometries(
                self.params.safe_area_width,
                self.params.safe_area_length,
            )
        elif result.cubes:
            rospy.logwarn(
                "Could not transform the visualization result to world; safe-area filtering was skipped."
            )

        try:
            draw(
                result,
                axis_size=self.params.axis_size,
                show_original_cloud=self.show_original_cloud,
                extra_clouds=extra_clouds,
                extra_geometries=safe_area_geometries,
            )
        except Exception as exc:
            rospy.logerr("Failed to render point cloud: %s", exc)

    def _transform_points_to_world(self,
                                   points: Optional[np.ndarray],
                                   header: Optional[Header]) -> Optional[np.ndarray]:
        if not has_points(points):
            return points
        transform_matrix = self._lookup_transform_matrix(SAFE_AREA_FRAME, header)
        if transform_matrix is None:
            return points
        return transform_points(points, transform_matrix)

    def _transform_result_to_world(self,
                                   result,
                                   header: Optional[Header]):
        transform_matrix = self._lookup_transform_matrix(SAFE_AREA_FRAME, header)
        if transform_matrix is None:
            return result, False
        return transform_detection_result(result, transform_matrix), True

    def _lookup_transform_matrix(self,
                                 target_frame: str,
                                 header: Optional[Header]) -> Optional[np.ndarray]:
        if header is None:
            return None

        source_frame = str(header.frame_id).strip()
        target_frame = str(target_frame).strip()
        if not source_frame or not target_frame:
            return None
        if source_frame == target_frame:
            return np.eye(4)

        try:
            tf_msg = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                header.stamp,
                rospy.Duration(0.2),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as exc:
            rospy.logwarn("TF lookup failed (%s -> %s): %s", source_frame, target_frame, exc)
            return None
        return transform_to_matrix(tf_msg)


def main() -> None:
    rospy.init_node("listener", anonymous=False)
    node = SingleCloudNode()
    node.run()


if __name__ == "__main__":
    main()
