#!/usr/bin/env python3
"""Visualize the point cloud coming from the simulated ZED2 camera."""

import threading
from typing import Optional

import numpy as np
import open3d as o3d
import rospy
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


class ZedListener:
    """Subscribes to the registered point cloud and renders it via Open3D."""

    def __init__(self):
        cloud_topic = rospy.get_param("~cloud_topic",
                                      "/zed2/zed_node/point_cloud/cloud_registered")

        self._cloud_sub = rospy.Subscriber(cloud_topic, PointCloud2, self._cloud_cb,
                                           queue_size=1)

        self._latest_points: Optional[np.ndarray] = None
        self._cloud_lock = threading.Lock()
        self._cloud_ready = threading.Event()
        self._stop_event = threading.Event()
        rospy.on_shutdown(self._shutdown_visualizer)

        rospy.loginfo("Listening to point cloud on %s", cloud_topic)

    def _shutdown_visualizer(self) -> None:
        """Signal the visualization loop to exit."""
        self._stop_event.set()
        self._cloud_ready.set()

    def _cloud_cb(self, msg: PointCloud2) -> None:
        rospy.loginfo_throttle(1.0, "Point cloud in %s (%d points), stamp %.3f",
                               msg.header.frame_id, msg.width * msg.height,
                               msg.header.stamp.to_sec())

        try:
            points = self._pointcloud_to_array(msg)
        except Exception as exc:  # defensive logging
            rospy.logerr("Failed to parse point cloud: %s", exc)
            return

        rospy.logdebug("Parsed point cloud with shape %s", points.shape)

        if points.size == 0:
            rospy.logwarn_throttle(5.0, "Received an empty point cloud, skipping render")
            return
        rospy.logdebug("Cloud XYZ stats min %s max %s",
                       np.min(points, axis=0), np.max(points, axis=0))
        if rospy.get_param("~log_first_points", False):
            rospy.logdebug("First 3 points sample: %s", points[:3])
        with self._cloud_lock:
            self._latest_points = points
        self._cloud_ready.set()
        rospy.logdebug("Queued point cloud for visualization")

    def run(self) -> None:
        """Run the Open3D visualizer in the main thread."""
        vis = o3d.visualization.Visualizer()
        created = vis.create_window(window_name="ZED Point Cloud", width=960, height=540)
        rospy.loginfo("Created Open3D window: %s", created)
        if not created:
            rospy.logerr("Failed to create Open3D window; visualization will not run")
        try:
            render_opt = vis.get_render_option()
            render_opt.point_size = 2.0
            rospy.loginfo("Render options set: point_size=%.1f", render_opt.point_size)
        except Exception as exc:
            rospy.logwarn("Failed to adjust render options: %s", exc)
        pcd = o3d.geometry.PointCloud()
        vis.add_geometry(pcd)

        camera_configured = False
        rate = rospy.Rate(100)
        try:
            while not rospy.is_shutdown() and not self._stop_event.is_set():
                has_new = self._cloud_ready.wait(timeout=0.01)
                if has_new and not self._stop_event.is_set():
                    with self._cloud_lock:
                        points = (self._latest_points.copy()
                                  if self._latest_points is not None else None)
                        self._cloud_ready.clear()

                    if points is not None and points.size:
                        rospy.logdebug("Updating geometry with %d points", points.shape[0])
                        rospy.logdebug("Update XYZ stats min %s max %s",
                                       np.min(points, axis=0), np.max(points, axis=0))
                        pcd.points = o3d.utility.Vector3dVector(points)
                        if not camera_configured:
                            self._configure_camera(vis, points)
                            camera_configured = True
                            rospy.loginfo("Camera configured based on first cloud")
                        vis.update_geometry(pcd)
                    else:
                        rospy.logwarn_throttle(2.0, "Visualizer woken with no points")

                vis.poll_events()
                vis.update_renderer()
                rate.sleep()
        finally:
            vis.destroy_window()
            rospy.loginfo("Shutting down Open3D visualizer")

    @staticmethod
    def _pointcloud_to_array(msg: PointCloud2) -> np.ndarray:
        """Convert a PointCloud2 message into an XYZ numpy array."""
        points = [(x, y, z) for x, y, z in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)]
        rospy.logdebug("Read %d finite points from message", len(points))
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        arr = np.asarray(points, dtype=np.float64)
        if not np.isfinite(arr).all():
            rospy.logwarn("Cloud contains non-finite values; filtering")
            arr = arr[np.all(np.isfinite(arr), axis=1)]
        return arr

    @staticmethod
    def _configure_camera(vis: o3d.visualization.Visualizer,
                          points: np.ndarray) -> None:
        """Aim the virtual camera at the cloud bounds the first time we draw."""
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(points))
        center = bbox.get_center()
        extent = np.linalg.norm(bbox.get_extent())
        radius = max(extent, 1e-3)

        eye = center + np.array([0.0, 0.0, radius * 1.5])
        front_vec = (center - eye)
        if np.linalg.norm(front_vec) < 1e-6:
            front_vec = np.array([0.0, 0.0, -1.0])
        front = (front_vec / np.linalg.norm(front_vec)).tolist()
        up = [0.0, 1.0, 0.0]

        rospy.loginfo("Camera center %s eye %s front %s up %s radius %.3f extent %.3f",
                      center, eye, front, up, radius, extent)

        controller = vis.get_view_control()
        controller.set_lookat(center.tolist())
        controller.set_front(front)
        controller.set_up(up)
        controller.set_zoom(0.8)


def main():
    rospy.init_node("zed_listener")
    listener = ZedListener()
    listener.run()


if __name__ == "__main__":
    main()
