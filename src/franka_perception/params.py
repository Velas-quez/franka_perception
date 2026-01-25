#!/usr/bin/env python3
"""Parameter helpers."""

from dataclasses import dataclass

import rospy


@dataclass
class PerceptionParams:
    cloud_topic: str
    cube_side_length: float
    axis_size: float
    voxel_size: float
    cluster_eps: float
    cluster_min_points: int
    base_plane_distance: float
    max_cubes_per_cluster: int
    clearance: float


def load_params(ns: str = "~") -> PerceptionParams:
    """Load parameters with defaults."""
    def _p(name, default):
        return rospy.get_param(f"{ns}{name}", default)

    return PerceptionParams(
        cloud_topic=_p("cloud_topic", "/zed2/zed_node/point_cloud/cloud_registered"),
        cube_side_length=float(_p("cube_side_length", 0.045)),
        axis_size=float(_p("axis_size", 0.1)),
        voxel_size=float(_p("voxel_size", 0.002)),
        cluster_eps=float(_p("cluster_eps", 0.005)),
        cluster_min_points=int(_p("cluster_min_points", 10)),
        base_plane_distance=float(_p("base_plane_distance", 0.01)),
        max_cubes_per_cluster=int(_p("max_cubes_per_cluster", 2)),
        clearance=float(_p("clearance", 0.015)),
    )
