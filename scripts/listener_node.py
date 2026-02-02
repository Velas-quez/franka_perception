#!/usr/bin/env python3
"""Render the first received ZED2 point cloud with Open3D using shared pipeline."""

import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

from franka_perception.cloud_io import msg_to_xyzrgb, has_points
from franka_perception.params import load_params
from franka_perception.pipeline import CubeDetectionPipeline
from franka_perception.visualization import draw


class SingleCloudNode:
    """Subscribe to a point cloud, process once, then render."""

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
        self._colors: Optional[np.ndarray] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()

        self._cloud_sub = rospy.Subscriber(
            self.params.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        rospy.loginfo("Waiting for point cloud on %s", self.params.cloud_topic)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._cloud_ready.is_set():
            return
        points, colors = msg_to_xyzrgb(msg)
        if not has_points(points):
            rospy.logwarn("Received empty/invalid point cloud; ignoring")
            return
        with self._lock:
            self._points = points
            self._colors = colors
            self._cloud_ready.set()
        rospy.loginfo("Captured cloud with %d points", points.shape[0])
        try:
            self._cloud_sub.unregister()
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
            colors = self._colors.copy() if getattr(self, "_colors", None) is not None else None

        if not has_points(points):
            rospy.logerr("No valid point cloud received before shutdown")
            return

        result = self.pipeline.process(points, colors=colors)
        try:
            draw(result, axis_size=self.params.axis_size, paint_cloud=False)
        except Exception as exc:
            rospy.logerr("Failed to render point cloud: %s", exc)


def main() -> None:
    rospy.init_node("listener", anonymous=False)
    node = SingleCloudNode()
    node.run()


if __name__ == "__main__":
    main()
