#!/usr/bin/env python3
"""ROS publishers for detected cubes."""

from typing import Iterable, Optional

import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, PoseArray
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
        marker.color.a = 0.3  # translucent base cube
        marker.lifetime = rospy.Duration(0)
        marker_array.markers.append(marker)

        face_markers = _face_markers_for_cube(
            cube, cube_side_length, hdr, base_id=idx * 10)
        marker_array.markers.extend(face_markers)

    publisher.publish(marker_array)


def _face_markers_for_cube(cube: CubeEstimate,
                           cube_side_length: float,
                           header: Header,
                           base_id: int,
                           thickness: float = 0.004):
    """Create thin cube-face markers with the detected dominant color."""
    if not cube.face_colors:
        return []

    face_markers = []
    half = cube_side_length / 2.0
    T = cube.transform
    quat = tf_trans.quaternion_from_matrix(T)

    offsets = {
        "px": (half, 0, 0),
        "nx": (-half, 0, 0),
        "py": (0, half, 0),
        "ny": (0, -half, 0),
        "pz": (0, 0, half),
        "nz": (0, 0, -half),
    }
    for face_idx, (face, color) in enumerate(cube.face_colors.items()):
        offset = offsets.get(face)
        if offset is None:
            continue
        marker = Marker()
        marker.header = header
        marker.ns = "cube_faces"
        marker.id = base_id + face_idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(T[0, 3] + offset[0] * T[0, 0] + offset[1] * T[0, 1] + offset[2] * T[0, 2])
        marker.pose.position.y = float(T[1, 3] + offset[0] * T[1, 0] + offset[1] * T[1, 1] + offset[2] * T[1, 2])
        marker.pose.position.z = float(T[2, 3] + offset[0] * T[2, 0] + offset[1] * T[2, 1] + offset[2] * T[2, 2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])
        # thin slab aligned with the face
        if face[1] == "x":
            marker.scale.x = thickness
            marker.scale.y = cube_side_length
            marker.scale.z = cube_side_length
        elif face[1] == "y":
            marker.scale.x = cube_side_length
            marker.scale.y = thickness
            marker.scale.z = cube_side_length
        else:  # z
            marker.scale.x = cube_side_length
            marker.scale.y = cube_side_length
            marker.scale.z = thickness
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 0.95
        marker.lifetime = rospy.Duration(0)
        face_markers.append(marker)
    return face_markers
