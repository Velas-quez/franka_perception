#!/usr/bin/env python3
"""Process point clouds continuously and publish cube poses/markers."""

import copy
import threading
from typing import Optional, Iterable

import numpy as np
import rospy
import message_filters
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from franka_perception.cloud_io import msg_to_xyz, has_points, rgbd_msgs_to_xyz
from franka_perception.params import load_params
from franka_perception.pipeline import CubeDetectionPipeline
from franka_perception.sam_rgbd_pipeline import SamRgbdCubeDetectionPipeline
from franka_perception.publishers import publish_markers, publish_poses, publish_point_clouds
from franka_perception.cube_fitting import CubeEstimate


class DynamicListenerNode:
    """Subscribe to a point cloud, process each frame, and publish results."""

    def __init__(self) -> None:
        self.params = load_params()
        self.pipeline_mode = self.params.pipeline_mode.strip().lower()
        if self.pipeline_mode not in {"classic", "sam_rgbd"}:
            raise ValueError("Invalid ~pipeline_mode. Use 'classic' or 'sam_rgbd'.")
        if self.pipeline_mode == "sam_rgbd" and not self.params.use_rgbd:
            raise ValueError("pipeline_mode=sam_rgbd requires ~use_rgbd:=true")

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
            )
        self._points: Optional[np.ndarray] = None
        self._rgb_msg: Optional[Image] = None
        self._depth_msg: Optional[Image] = None
        self._camera_info_msg: Optional[CameraInfo] = None
        self._latest_header: Optional[Header] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._pose_pub = rospy.Publisher("~cube_poses", PoseArray, queue_size=5)
        self._marker_pub = rospy.Publisher("~cube_markers", MarkerArray, queue_size=5)
        self._cloud_pubs = {}  # Dictionary to store publishers for each cube's point cloud

        self._cloud_sub = None
        self._rgb_sub = None
        self._depth_sub = None
        self._info_sub = None
        self._sync = None
        self._latest_info: Optional[CameraInfo] = None

        if self.params.use_rgbd:
            self._rgb_sub = message_filters.Subscriber(self.params.rgb_topic, Image)
            self._depth_sub = message_filters.Subscriber(self.params.depth_topic, Image)
            self._info_sub = rospy.Subscriber(
                self.params.camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub], queue_size=5, slop=0.2)
            self._sync.registerCallback(self._rgbd_cb)
            rospy.loginfo("Listening for RGB-D on %s + %s (mode=%s)",
                          self.params.rgb_topic, self.params.depth_topic, self.pipeline_mode)
        else:
            self._cloud_sub = rospy.Subscriber(
                self.params.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
            rospy.loginfo("Listening for point clouds on %s", self.params.cloud_topic)

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
        if self.pipeline_mode == "sam_rgbd":
            with self._lock:
                self._rgb_msg = rgb_msg
                self._depth_msg = depth_msg
                self._camera_info_msg = info_msg
                self._latest_header = rgb_msg.header
                self._cloud_ready.set()
            return

        depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
        points = rgbd_msgs_to_xyz(
            rgb_msg,
            depth_msg,
            info_msg,
            depth_scale=depth_scale,
            depth_trunc=self.params.depth_trunc,
            flip=self.params.rgbd_flip,
        )
        if not has_points(points):
            rospy.logwarn("Received empty/invalid RGB-D; ignoring")
            return
        with self._lock:
            self._points = points
            self._latest_header = info_msg.header
            self._cloud_ready.set()
        rospy.loginfo("Captured RGB-D cloud with %d points", points.shape[0])

    def run(self) -> None:
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
                self._cloud_ready.clear()

            if self.pipeline_mode == "sam_rgbd":
                if rgb_msg is None or depth_msg is None or info_msg is None:
                    rospy.logwarn("No valid synchronized RGB-D frame received; waiting for next")
                    continue
                depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
                result = self.sam_pipeline.process(
                    rgb_msg,
                    depth_msg,
                    info_msg,
                    depth_scale=depth_scale,
                    depth_trunc=self.params.depth_trunc,
                    flip=self.params.rgbd_flip,
                    stop_after="all",
                )
            else:
                if not has_points(points):
                    rospy.logwarn("No valid point cloud received; waiting for next")
                    continue
                result = self.pipeline.process(points)

            cubes_base, header_base = self._transform_cubes_to_target(result.cubes, header)
            publish_poses(cubes_base, header_base, self.params.cube_side_length, self._pose_pub)
            publish_markers(cubes_base, header_base, self.params.cube_side_length, self._marker_pub)
            self._publish_cube_point_clouds(cubes_base, header_base)

    def _publish_cube_point_clouds(self,
                                   cubes: Iterable[CubeEstimate],
                                   header: Optional[Header]):
        """Publish point clouds sampled from each detected cube."""
        cubes_list = list(cubes)
        
        # Create publishers for new cubes and clean up old ones
        active_cube_indices = set(range(len(cubes_list)))
        inactive_indices = set(self._cloud_pubs.keys()) - active_cube_indices
        
        for idx in inactive_indices:
            if idx in self._cloud_pubs:
                del self._cloud_pubs[idx]
        
        # Create publishers for new cubes if needed
        for idx in active_cube_indices:
            if idx not in self._cloud_pubs:
                pub_name = f"~cube_cloud_{idx}"
                self._cloud_pubs[idx] = rospy.Publisher(pub_name, PointCloud2, queue_size=5)
                rospy.loginfo(f"Created publisher for {pub_name}")
        
        # Publish point clouds for each cube
        for idx, cube in enumerate(cubes_list):
            if idx in self._cloud_pubs:
                publish_point_clouds([cube], header, num_samples=1500, 
                                   publisher=self._cloud_pubs[idx])

    def _transform_cubes_to_target(self,
                                   cubes: Iterable[CubeEstimate],
                                   header: Optional[Header]):
        """Transform estimated cube poses to target frame only (cheap)."""
        if header is None:
            return cubes, header

        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self.params.target_frame,
                header.frame_id,
                header.stamp,
                rospy.Duration(0.2),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as exc:
            rospy.logwarn("TF lookup failed (%s -> %s): %s",
                          header.frame_id, self.params.target_frame, exc)
            return cubes, header

        T = _transform_to_matrix(tf_msg)
        transformed = []
        for cube in cubes:
            new_T = T @ cube.transform
            # Transform mesh to target frame
            transformed_mesh = copy.deepcopy(cube.mesh)
            transformed_mesh.transform(T)
            transformed.append(
                CubeEstimate(
                    transform=new_T,
                    mesh=transformed_mesh,
                    initial_mesh=cube.initial_mesh,
                )
            )

        new_header = Header()
        new_header.frame_id = self.params.target_frame
        new_header.stamp = header.stamp
        return transformed, new_header


def _transform_to_matrix(tf_msg):
    """Convert TransformStamped to 4x4 matrix."""
    trans = tf_msg.transform.translation
    rot = tf_msg.transform.rotation
    T = tf_trans.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    T[0, 3] = trans.x
    T[1, 3] = trans.y
    T[2, 3] = trans.z
    return T


def main() -> None:
    rospy.init_node("dynamic_listener", anonymous=False)
    node = DynamicListenerNode()
    node.run()


if __name__ == "__main__":
    main()
