#!/usr/bin/env python3
"""ROS publishers for detected and tracked cubes."""

from typing import Iterable, Optional, Sequence

import numpy as np
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from franka_perception_thiago.msg import TrackedCube, TrackedCubeArray

from ..geometry.cube_fitting import CubeEstimate
from ..tracking.cube_tracker import TrackedCubeState


def _resolved_header(header: Optional[Header], default_frame: str) -> Header:
    resolved = Header()
    if header:
        resolved.frame_id = header.frame_id
        resolved.stamp = header.stamp
    else:
        resolved.frame_id = default_frame
        resolved.stamp = rospy.Time.now()
    return resolved


def _pose_from_transform(transform: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(transform[0, 3])
    pose.position.y = float(transform[1, 3])
    pose.position.z = float(transform[2, 3])
    quat = tf_trans.quaternion_from_matrix(transform)
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def publish_poses(cubes: Iterable[CubeEstimate],
                  header: Optional[Header],
                  cube_side_length: float,
                  publisher) -> None:
    del cube_side_length
    if publisher is None:
        return

    pose_array = PoseArray()
    pose_array.header = _resolved_header(header, default_frame="map")
    for cube in cubes:
        pose_array.poses.append(_pose_from_transform(cube.transform))
    publisher.publish(pose_array)


def publish_markers(cubes: Iterable[CubeEstimate],
                    header: Optional[Header],
                    cube_side_length: float,
                    publisher) -> None:
    if publisher is None:
        return

    hdr = _resolved_header(header, default_frame="map")
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
        marker = Marker()
        marker.header = hdr
        marker.ns = "estimated_cubes"
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = _pose_from_transform(cube.transform)
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


def publish_tracked_cubes(tracks: Sequence[TrackedCubeState],
                          header: Optional[Header],
                          publisher) -> None:
    if publisher is None:
        return

    msg = TrackedCubeArray()
    msg.header = _resolved_header(header, default_frame="map")
    for track in tracks:
        item = TrackedCube()
        item.id = int(track.track_id)
        item.pose = _pose_from_transform(track.cube.transform)
        item.is_occluded = bool(track.is_occluded)
        msg.cubes.append(item)
    publisher.publish(msg)


def publish_tracked_markers(tracks: Sequence[TrackedCubeState],
                            header: Optional[Header],
                            cube_side_length: float,
                            publisher) -> None:
    if publisher is None:
        return

    hdr = _resolved_header(header, default_frame="map")
    marker_array = MarkerArray()
    tracks = list(tracks)
    if not tracks:
        delete_msg = Marker()
        delete_msg.header = hdr
        delete_msg.action = Marker.DELETEALL
        marker_array.markers.append(delete_msg)
        publisher.publish(marker_array)
        return

    for track in tracks:
        marker = Marker()
        marker.header = hdr
        marker.ns = "tracked_cubes"
        marker.id = int(track.track_id)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = _pose_from_transform(track.cube.transform)
        marker.scale.x = cube_side_length
        marker.scale.y = cube_side_length
        marker.scale.z = cube_side_length
        if track.is_occluded:
            marker.color.r = 0.6
            marker.color.g = 0.6
            marker.color.b = 0.6
            marker.color.a = 0.35
        else:
            marker.color.r = 0.15
            marker.color.g = 0.8
            marker.color.b = 0.35
            marker.color.a = 0.85
        marker.lifetime = rospy.Duration(0)
        marker_array.markers.append(marker)

    publisher.publish(marker_array)


def publish_tracked_labels(tracks: Sequence[TrackedCubeState],
                           header: Optional[Header],
                           cube_side_length: float,
                           publisher) -> None:
    if publisher is None:
        return

    hdr = _resolved_header(header, default_frame="map")
    marker_array = MarkerArray()
    tracks = list(tracks)
    if not tracks:
        delete_msg = Marker()
        delete_msg.header = hdr
        delete_msg.action = Marker.DELETEALL
        marker_array.markers.append(delete_msg)
        publisher.publish(marker_array)
        return

    text_height = max(0.012, float(cube_side_length) * 0.45)
    z_offset = 0.65 * float(cube_side_length)
    for track in tracks:
        marker = Marker()
        marker.header = hdr
        marker.ns = "tracked_cube_labels"
        marker.id = int(track.track_id)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose = _pose_from_transform(track.cube.transform)
        marker.pose.position.z += z_offset
        marker.scale.z = text_height
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = f"id={int(track.track_id)}"
        if track.is_occluded:
            marker.text += " [occluded]"
            marker.color.r = 0.85
            marker.color.g = 0.85
            marker.color.b = 0.85
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

    hdr = _resolved_header(header, default_frame="world")
    cubes_list = list(cubes)
    if cubes_list:
        cube = cubes_list[0]
        sampled_pcd = cube.mesh.sample_points_uniformly(num_samples)
        points = np.asarray(sampled_pcd.points, dtype=np.float32)
        pc2_msg = pc2.create_cloud_xyz32(hdr, points)
        publisher.publish(pc2_msg)
