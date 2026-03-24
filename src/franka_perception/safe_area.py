#!/usr/bin/env python3
"""Helpers for safe-area filtering and visualization."""

from typing import Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .geometry.cube_fitting import CubeEstimate

SAFE_AREA_FRAME = "world"
_SAFE_AREA_THICKNESS = 0.002
_SAFE_AREA_OUTLINE_Z = 0.004
_SAFE_AREA_OUTLINE_WIDTH = 0.006


def _normalized_dimensions(width: float, length: float) -> Tuple[float, float]:
    return max(0.0, float(width)), max(0.0, float(length))


def has_safe_area(width: float, length: float) -> bool:
    width, length = _normalized_dimensions(width, length)
    return width > 0.0 and length > 0.0


def cube_in_safe_area(cube: CubeEstimate,
                      width: float,
                      length: float,
                      length_offset: float = 0.0) -> bool:
    if not has_safe_area(width, length):
        return True

    width, length = _normalized_dimensions(width, length)
    center = np.asarray(cube.transform[:3, 3], dtype=float)
    x = float(center[0])
    y = float(center[1])
    x_min = float(length_offset)
    x_max = x_min + length
    half_width = 0.5 * width
    return x_min <= x <= x_max and -half_width <= y <= half_width


def safe_area_keep_mask(cubes: Iterable[CubeEstimate],
                        width: float,
                        length: float,
                        length_offset: float = 0.0) -> List[bool]:
    return [cube_in_safe_area(cube, width, length, length_offset) for cube in cubes]


def build_safe_area_marker_array(header: Optional[Header],
                                 width: float,
                                 length: float,
                                 length_offset: float = 0.0) -> MarkerArray:
    marker_array = MarkerArray()
    resolved_header = Header()
    if header is not None:
        resolved_header.frame_id = header.frame_id
        resolved_header.stamp = header.stamp
    else:
        resolved_header.frame_id = SAFE_AREA_FRAME
        resolved_header.stamp = rospy.Time.now()

    if not has_safe_area(width, length):
        delete_msg = Marker()
        delete_msg.header = resolved_header
        delete_msg.action = Marker.DELETEALL
        marker_array.markers.append(delete_msg)
        return marker_array

    width, length = _normalized_dimensions(width, length)
    half_width = 0.5 * width

    fill_marker = Marker()
    fill_marker.header = resolved_header
    fill_marker.ns = "safe_area"
    fill_marker.id = 0
    fill_marker.type = Marker.CUBE
    fill_marker.action = Marker.ADD
    fill_marker.pose.position.x = float(length_offset) + 0.5 * length
    fill_marker.pose.position.y = 0.0
    fill_marker.pose.position.z = 0.5 * _SAFE_AREA_THICKNESS
    fill_marker.pose.orientation.w = 1.0
    fill_marker.scale.x = length
    fill_marker.scale.y = width
    fill_marker.scale.z = _SAFE_AREA_THICKNESS
    fill_marker.color.r = 0.1
    fill_marker.color.g = 0.75
    fill_marker.color.b = 0.2
    fill_marker.color.a = 0.2
    fill_marker.lifetime = rospy.Duration(0)
    marker_array.markers.append(fill_marker)

    outline_marker = Marker()
    outline_marker.header = resolved_header
    outline_marker.ns = "safe_area"
    outline_marker.id = 1
    outline_marker.type = Marker.LINE_STRIP
    outline_marker.action = Marker.ADD
    outline_marker.pose.orientation.w = 1.0
    outline_marker.scale.x = _SAFE_AREA_OUTLINE_WIDTH
    outline_marker.color.r = 0.05
    outline_marker.color.g = 0.95
    outline_marker.color.b = 0.15
    outline_marker.color.a = 0.9
    outline_marker.lifetime = rospy.Duration(0)
    for x, y in (
        (length_offset, -half_width),
        (length_offset + length, -half_width),
        (length_offset + length, half_width),
        (length_offset, half_width),
        (length_offset, -half_width),
    ):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = _SAFE_AREA_OUTLINE_Z
        outline_marker.points.append(point)
    marker_array.markers.append(outline_marker)
    return marker_array


def build_safe_area_geometries(width: float,
                               length: float,
                               length_offset: float = 0.0) -> List[object]:
    if not has_safe_area(width, length):
        return []

    width, length = _normalized_dimensions(width, length)
    half_width = 0.5 * width

    area_mesh = o3d.geometry.TriangleMesh.create_box(
        width=length,
        height=width,
        depth=_SAFE_AREA_THICKNESS,
    )
    area_mesh.translate([float(length_offset), -half_width, 0.0])
    area_mesh.paint_uniform_color([0.1, 0.75, 0.2])

    outline_points = np.array([
        [length_offset, -half_width, _SAFE_AREA_OUTLINE_Z],
        [length_offset + length, -half_width, _SAFE_AREA_OUTLINE_Z],
        [length_offset + length, half_width, _SAFE_AREA_OUTLINE_Z],
        [length_offset, half_width, _SAFE_AREA_OUTLINE_Z],
    ], dtype=float)
    outline_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    outline_colors = np.tile(np.array([[0.05, 0.95, 0.15]], dtype=float), (4, 1))
    outline = o3d.geometry.LineSet()
    outline.points = o3d.utility.Vector3dVector(outline_points)
    outline.lines = o3d.utility.Vector2iVector(outline_lines)
    outline.colors = o3d.utility.Vector3dVector(outline_colors)
    return [area_mesh, outline]
