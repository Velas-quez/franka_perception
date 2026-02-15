#!/usr/bin/env python3
"""Render the first received ZED2 point cloud with Open3D using shared pipeline."""

import argparse
import sys
import threading
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import message_filters

from franka_perception.cloud_io import msg_to_xyz, has_points, rgbd_msgs_to_xyz
from franka_perception.params import load_params
from franka_perception.pipeline import CubeDetectionPipeline
from franka_perception.visualization import draw


class SingleCloudNode:
    """Subscribe to a point cloud, process once, then render."""

    def __init__(self, stage: str = "all", show_original_cloud: Optional[bool] = None) -> None:
        self.params = load_params()
        self.stage = stage
        if show_original_cloud is None:
            self.show_original_cloud = bool(rospy.get_param("~show_original_cloud", False))
        else:
            self.show_original_cloud = bool(show_original_cloud)
        self.pipeline = CubeDetectionPipeline(
            cube_side_length=self.params.cube_side_length,
            voxel_size=self.params.voxel_size,
            base_plane_distance=self.params.base_plane_distance,
            cluster_eps=self.params.cluster_eps,
            cluster_min_points=self.params.cluster_min_points,
            max_cubes_per_cluster=self.params.max_cubes_per_cluster,
            clearance=self.params.clearance,
            max_cluster_distance_from_plane_inliers=self.params.max_cluster_distance_from_plane_inliers,
            below_plane_tolerance=self.params.below_plane_tolerance,
        )
        self._points: Optional[np.ndarray] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()

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
            rospy.loginfo("Waiting for RGB-D on %s + %s",
                          self.params.rgb_topic, self.params.depth_topic)
        else:
            self._cloud_sub = rospy.Subscriber(
                self.params.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
            rospy.loginfo("Waiting for point cloud on %s", self.params.cloud_topic)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._cloud_ready.is_set():
            return
        points = msg_to_xyz(msg)
        if not has_points(points):
            rospy.logwarn("Received empty/invalid point cloud; ignoring")
            return
        with self._lock:
            self._points = points
            self._cloud_ready.set()
        rospy.loginfo("Captured cloud with %d points", points.shape[0])
        try:
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
        )
        if not has_points(points):
            rospy.logwarn("Received empty/invalid RGB-D; ignoring")
            return
        with self._lock:
            self._points = points
            self._cloud_ready.set()
        rospy.loginfo("Captured RGB-D cloud with %d points", points.shape[0])
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
        rospy.loginfo("Waiting for the first valid cloud...")
        while not rospy.is_shutdown():
            if self._cloud_ready.wait(timeout=0.2):
                break

        if rospy.is_shutdown():
            return

        with self._lock:
            points = self._points.copy() if self._points is not None else None

        if not has_points(points):
            rospy.logerr("No valid point cloud received before shutdown")
            return

        rospy.loginfo("Rendering pipeline stage: %s", self.stage)
        result = self.pipeline.process(points, stop_after=self.stage)
        try:
            draw(
                result,
                axis_size=self.params.axis_size,
                show_original_cloud=self.show_original_cloud,
            )
        except Exception as exc:
            rospy.logerr("Failed to render point cloud: %s", exc)


def _parse_args():
    parser = argparse.ArgumentParser(description="Render a ZED point cloud through the cube pipeline.")
    parser.add_argument(
        "--stage",
        choices=["none", "filter", "cluster", "all"],
        default="all",
        help="Stop the pipeline after this stage for visualization.",
    )
    parser.add_argument(
        "--show-original-cloud",
        action="store_true",
        help="Render the original cloud instead of the filtered cloud.",
    )
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main() -> None:
    args = _parse_args()
    rospy.init_node("listener", anonymous=False)
    node = SingleCloudNode(stage=args.stage, show_original_cloud=args.show_original_cloud)
    node.run()


if __name__ == "__main__":
    main()
