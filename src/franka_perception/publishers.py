#!/usr/bin/env python3
"""ROS publishers for detected cubes."""

from typing import Iterable, Optional

import numpy as np
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .cube_fitting import CubeEstimate


def publish_poses(cubes: Iterable[CubeEstimate],
                  header: Optional[Header],
                  cube_side_length: float,
                  publisher) -> None:
    if publisher is None:
        return

    pose_array = PoseArray()
    if header:
        pose_array.header.frame_id = header.frame_id
        pose_array.header.stamp = header.stamp
    else:
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = rospy.Time.now()

    for cube in cubes:
        T = cube.transform
        pose = Pose()
        pose.position.x = float(T[0, 3])
        pose.position.y = float(T[1, 3])
        pose.position.z = float(T[2, 3])
        quat = tf_trans.quaternion_from_matrix(T)
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])
        pose_array.poses.append(pose)

    publisher.publish(pose_array)


def publish_markers(cubes: Iterable[CubeEstimate],
                    header: Optional[Header],
                    cube_side_length: float,
                    publisher) -> None:
    if publisher is None:
        return

    hdr = Header()
    if header:
        hdr.frame_id = header.frame_id
        hdr.stamp = header.stamp
    else:
        hdr.frame_id = "map"
        hdr.stamp = rospy.Time.now()

    marker_array = MarkerArray()
    cubes = list(cubes)
    if not cubes:
        delete_msg = Marker()
        delete_msg.header = hdr
        delete_msg.action = Marker.DELETEALL
        marker_array.markers.append(delete_msg)
        publisher.publish(marker_array)
        return

    for idx, cube in enumerate(cubes):
        T = cube.transform
        quat = tf_trans.quaternion_from_matrix(T)

        marker = Marker()
        marker.header = hdr
        marker.ns = "estimated_cubes"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(T[0, 3])
        marker.pose.position.y = float(T[1, 3])
        marker.pose.position.z = float(T[2, 3])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])
        marker.scale.x = cube_side_length
        marker.scale.y = cube_side_length
        marker.scale.z = cube_side_length
        marker.color.r = 0.1
        marker.color.g = 0.4
        marker.color.b = 0.9
        marker.color.a = 0.8
        marker.lifetime = rospy.Duration(0)
        marker_array.markers.append(marker)

    publisher.publish(marker_array)


def publish_point_clouds(cubes: Iterable[CubeEstimate],
                         header: Optional[Header],
                         num_samples: int = 1000,
                         publisher=None) -> None:
    """Publish point clouds sampled from each detected cube mesh in world reference frame."""
    if publisher is None:
        return

    hdr = Header()
    if header:
        hdr.frame_id = header.frame_id
        hdr.stamp = header.stamp
    else:
        hdr.frame_id = "world"
        hdr.stamp = rospy.Time.now()

    cubes_list = list(cubes)
    # Only publish the first cube (when called with single cube from dynamic_listener)
    if cubes_list:
        cube = cubes_list[0]
        # Sample points uniformly from the cube mesh (already in world frame)
        sampled_pcd = cube.mesh.sample_points_uniformly(num_samples)
        points = np.asarray(sampled_pcd.points, dtype=np.float32)

        # Create PointCloud2 message
        pc2_msg = pc2.create_cloud_xyz32(hdr, points)
        
        # Publish
        publisher.publish(pc2_msg)

