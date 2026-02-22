#!/usr/bin/env python3
"""Parameter helpers."""

from dataclasses import dataclass

import rospy


@dataclass
class PerceptionParams:
    cloud_topic: str
    use_rgbd: bool
    pipeline_mode: str
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    depth_scale: float
    depth_trunc: float
    rgbd_flip: bool
    sam_model_id: str
    sam_device: str
    sam_prompt: str
    sam_score_threshold: float
    sam_max_masks: int
    sam_min_mask_pixels: int
    sam_min_depth_pixels: int
    sam_mask_erosion_kernel: int
    sam_mask_erosion_iterations: int
    sam_min_points_per_cluster: int
    sam_max_mask_area_ratio: float
    sam_plane_ransac_distance: float
    sam_near_plane_distance: float
    sam_max_near_plane_ratio: float
    sam_min_mask_plane_height: float
    sam_max_cluster_extent_multiplier: float
    sam_max_cluster_volume_multiplier: float
    sam_show_windows: bool
    sam_window_wait_ms: int
    target_frame: str
    cube_side_length: float
    axis_size: float
    voxel_size: float
    cluster_eps: float
    cluster_min_points: int
    base_plane_distance: float
    below_plane_tolerance: float
    max_cluster_distance_from_plane_inliers: float
    max_cubes_per_cluster: int
    num_best_cubes: int
    clearance: float


def load_params(ns: str = "~") -> PerceptionParams:
    """Load parameters with defaults."""
    def _p(name, default):
        return rospy.get_param(f"{ns}{name}", default)

    return PerceptionParams(
        cloud_topic=_p("cloud_topic", "/zed2/zed_node/point_cloud/cloud_registered"),
        use_rgbd=bool(_p("use_rgbd", False)),
        pipeline_mode=_p("pipeline_mode", "classic"),
        rgb_topic=_p("rgb_topic", "/zed2/zed_node/rgb/image_rect_color"),
        depth_topic=_p("depth_topic", "/zed2/zed_node/depth/depth_registered"),
        camera_info_topic=_p("camera_info_topic", "/zed2/zed_node/rgb/camera_info"),
        depth_scale=float(_p("depth_scale", 0.0)),
        depth_trunc=float(_p("depth_trunc", 3.0)),
        rgbd_flip=bool(_p("rgbd_flip", True)),
        sam_model_id=_p("sam_model_id", "facebook/sam3.1-hiera-large"),
        sam_device=_p("sam_device", "auto"),
        sam_prompt=_p("sam_prompt", "cube"),
        sam_score_threshold=float(_p("sam_score_threshold", 0.0)),
        sam_max_masks=int(_p("sam_max_masks", 8)),
        sam_min_mask_pixels=int(_p("sam_min_mask_pixels", 1200)),
        sam_min_depth_pixels=int(_p("sam_min_depth_pixels", 600)),
        sam_mask_erosion_kernel=int(_p("sam_mask_erosion_kernel", 3)),
        sam_mask_erosion_iterations=int(_p("sam_mask_erosion_iterations", 1)),
        sam_min_points_per_cluster=int(_p("sam_min_points_per_cluster", 120)),
        sam_max_mask_area_ratio=float(_p("sam_max_mask_area_ratio", 0.35)),
        sam_plane_ransac_distance=float(_p("sam_plane_ransac_distance", 0.006)),
        sam_near_plane_distance=float(_p("sam_near_plane_distance", 0.006)),
        sam_max_near_plane_ratio=float(_p("sam_max_near_plane_ratio", 0.85)),
        sam_min_mask_plane_height=float(_p("sam_min_mask_plane_height", 0.012)),
        sam_max_cluster_extent_multiplier=float(
            _p("sam_max_cluster_extent_multiplier", 2.8)),
        sam_max_cluster_volume_multiplier=float(
            _p("sam_max_cluster_volume_multiplier", 7.0)),
        sam_show_windows=bool(_p("sam_show_windows", True)),
        sam_window_wait_ms=int(_p("sam_window_wait_ms", 1)),
        target_frame=_p("target_frame", "world"),
        cube_side_length=float(_p("cube_side_length", 0.045)),
        axis_size=float(_p("axis_size", 0.1)),
        voxel_size=float(_p("voxel_size", 0.002)),
        cluster_eps=float(_p("cluster_eps", 0.005)),
        cluster_min_points=int(_p("cluster_min_points", 10)),
        base_plane_distance=float(_p("base_plane_distance", 0.01)),
        below_plane_tolerance=float(_p("below_plane_tolerance", 0.002)),
        max_cluster_distance_from_plane_inliers=float(
            _p("max_cluster_distance_from_plane_inliers", 0.08)),
        max_cubes_per_cluster=int(_p("max_cubes_per_cluster", 2)),
        num_best_cubes=int(_p("num_best_cubes", 2)),
        clearance=float(_p("clearance", 0.015)),
    )
