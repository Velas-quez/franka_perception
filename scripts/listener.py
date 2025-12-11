#!/usr/bin/env python3
"""Render the first received ZED2 point cloud with Open3D."""

import threading
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

from filter import filter_point_cloud
from identify_cubes import identify_cubes


class SingleCloudViewer:
    """Subscribe to a point cloud and render only the first valid message."""

    def __init__(self) -> None:
        self._cloud_topic = rospy.get_param(
            "~cloud_topic", "/zed2/zed_node/point_cloud/cloud_registered")

        self._cloud_sub = rospy.Subscriber(self._cloud_topic, PointCloud2,
                                           self._cloud_cb, queue_size=1)

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
        pcd = filter_point_cloud(pcd)
        
        # Plane removal
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                        ransac_n=3,
                                        num_iterations=1000)

        pcd = pcd.select_by_index(inliers, invert=True)
          
        # Clustering the point cloud
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            
        if True:
            # Estimating normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        if True:
            # Identificar cubos a partir dos clusters e das normais
            try:
                cubes = identify_cubes(pcd, labels)
            except Exception as exc:
                rospy.logerr("Erro ao identificar cubos: %s", exc)
                cubes = []

            geometries = [pcd]

            # Renderizar cubos estimados como OrientedBoundingBox
            for cube in cubes:
                center = cube["center"]
                R = cube["orientation"]
                size = cube["size"]

                obb = o3d.geometry.OrientedBoundingBox()
                obb.center = center
                obb.R = R
                obb.extent = np.array([size, size, size], dtype=float)
                obb.color = (1.0, 0.0, 0.0)
                geometries.append(obb)

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
