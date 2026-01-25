#!/usr/bin/env python3
"""Process point clouds continuously and publish cube poses/markers."""

import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from franka_perception.cloud_io import msg_to_xyz, has_points
from franka_perception.params import load_params
from franka_perception.pipeline import CubeDetectionPipeline
from franka_perception.publishers import publish_markers, publish_poses


class DynamicListenerNode:
    """Subscribe to a point cloud, process each frame, and publish results."""

    def __init__(self) -> None:
        self.params = load_params()
        self.pipeline = CubeDetectionPipeline(
            cube_side_length=self.params.cube_side_length,
            voxel_size=self.params.voxel_size,
            base_plane_distance=self.params.base_plane_distance,
            cluster_eps=self.params.cluster_eps,
            cluster_min_points=self.params.cluster_min_points,
            max_cubes_per_cluster=self.params.max_cubes_per_cluster,
            clearance=self.params.clearance,
        )
        self._points: Optional[np.ndarray] = None
        self._latest_header: Optional[Header] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()

        self._pose_pub = rospy.Publisher("~cube_poses", PoseArray, queue_size=5)
        self._marker_pub = rospy.Publisher("~cube_markers", MarkerArray, queue_size=5)

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

    def run(self) -> None:
        rospy.loginfo("Waiting for point clouds...")
        while not rospy.is_shutdown():
            if not self._cloud_ready.wait(timeout=0.2):
                continue

            with self._lock:
                points = self._points.copy() if self._points is not None else None
                header = self._latest_header
                self._cloud_ready.clear()

            if not has_points(points):
                rospy.logwarn("No valid point cloud received; waiting for next")
                continue

            result = self.pipeline.process(points)
            publish_poses(result.cubes, header, self.params.cube_side_length, self._pose_pub)
            publish_markers(result.cubes, header, self.params.cube_side_length, self._marker_pub)


def main() -> None:
    rospy.init_node("dynamic_listener", anonymous=False)
    node = DynamicListenerNode()
    node.run()


if __name__ == "__main__":
    main()
