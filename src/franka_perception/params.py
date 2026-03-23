#!/usr/bin/env python3
"""Parameter helpers."""

from dataclasses import dataclass

import rospy


_TOPICS_BY_ENVIROMENT = {
    "poseidon": {
        "cloud_topic": "/group1/zed2/zed_node/point_cloud/cloud_registered",
        "rgb_topic": "/group1/zed2/zed_node/rgb/image_rect_color",
        "depth_topic": "/group1/zed2/zed_node/depth/depth_registered",
        "camera_info_topic": "/group1/zed2/zed_node/rgb/camera_info",
    },
    "atena": {
        "cloud_topic": "/group1/zed2i/zed_node/point_cloud/cloud_registered",
        "rgb_topic": "/group1/zed2i/zed_node/left/image_rect_color",
        "depth_topic": "/group1/zed2i/zed_node/depth/depth_registered",
        "camera_info_topic": "/group1/zed2i/zed_node/left/camera_info",
    },
    "simulation": {
        "cloud_topic": "/zed2/zed_node/point_cloud/cloud_registered",
        "rgb_topic": "/zed2/zed_node/left/image_rect_color",
        "depth_topic": "/zed2/zed_node/depth/depth_registered",
        "camera_info_topic": "/zed2/zed_node/depth/camera_info",
    },
}

_SUPPORT_PLANE_CONSTRAINT_MODES = {"fix_icp", "ajust", "none"}


def _normalize_support_plane_constraint(value) -> str:
    if isinstance(value, bool):
        return "fix_icp" if value else "none"

    mode = str(value).strip().lower()
    aliases = {
        "true": "fix_icp",
        "false": "none",
        "adjust": "ajust",
    }
    mode = aliases.get(mode, mode)
    if mode not in _SUPPORT_PLANE_CONSTRAINT_MODES:
        rospy.logwarn(
            "Unknown support_plane_constraint='%s'; falling back to 'fix_icp'. "
            "Valid values: fix_icp|ajust|none",
            value,
        )
        return "fix_icp"
    return mode


@dataclass
class PerceptionParams:
    # Deployment selection and high-level runtime mode.
    enviroment: str
    use_rgbd: bool
    pipeline_mode: str
    target_frame: str

    # Input topics (resolved from enviroment, with optional ROS param overrides).
    cloud_topic: str
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str

    # RGB-D to point-cloud conversion.
    depth_scale: float
    depth_trunc: float
    rgbd_flip: bool

    # Classic geometry pipeline.
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
    support_plane_constraint: str

    # SAM backend and model/prompt configuration.
    sam_mode: str
    sam_checkpoint_path: str
    sam_model_type: str
    sam_device: str
    sam_points_per_side: int
    sam_pred_iou_thresh: float
    sam_stability_score_thresh: float
    sam_min_mask_region_area: int
    sam_prompt_text: str
    sam_prompt_box_threshold: float
    sam_prompt_text_threshold: float
    sam_grounding_model_id: str
    sam_segmentor_model_id: str

    # SAM mask filtering and 3D gating.
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

    # SAM debug visualization.
    sam_show_windows: bool
    sam_window_wait_ms: int


def load_params(ns: str = "~") -> PerceptionParams:
    """Load parameters with defaults."""
    def _p(name, default):
        return rospy.get_param(f"{ns}{name}", default)

    enviroment = str(_p("enviroment", "simulation")).strip().lower()
    if enviroment not in _TOPICS_BY_ENVIROMENT:
        rospy.logwarn(
            "Unknown ~enviroment='%s'; falling back to 'simulation'. "
            "Valid values: poseidon|atena|simulation",
            enviroment,
        )
        enviroment = "simulation"
    topic_defaults = _TOPICS_BY_ENVIROMENT[enviroment]

    return PerceptionParams(
        # Deployment selection and high-level runtime mode.
        enviroment=enviroment,
        use_rgbd=bool(_p("use_rgbd", False)),
        pipeline_mode=_p("pipeline_mode", "classic"),
        target_frame=_p("target_frame", "world"),

        # Input topics (resolved by enviroment, overridable via ROS params).
        cloud_topic=_p("cloud_topic", topic_defaults["cloud_topic"]),
        rgb_topic=_p("rgb_topic", topic_defaults["rgb_topic"]),
        depth_topic=_p("depth_topic", topic_defaults["depth_topic"]),
        camera_info_topic=_p("camera_info_topic", topic_defaults["camera_info_topic"]),

        # RGB-D to point-cloud conversion.
        depth_scale=float(_p("depth_scale", 0.0)),
        depth_trunc=float(_p("depth_trunc", 3.0)),
        rgbd_flip=bool(_p("rgbd_flip", False)),

        # Classic geometry pipeline.
        cube_side_length=float(_p("cube_side_length", 0.045)),
        axis_size=float(_p("axis_size", 0.1)),
        voxel_size=float(_p("voxel_size", 0.0)),
        cluster_eps=float(_p("cluster_eps", 0.005)),
        cluster_min_points=int(_p("cluster_min_points", 10)),
        base_plane_distance=float(_p("base_plane_distance", 0.01)),
        below_plane_tolerance=float(_p("below_plane_tolerance", 0.002)),
        max_cluster_distance_from_plane_inliers=float(
            _p("max_cluster_distance_from_plane_inliers", 0.08)),
        max_cubes_per_cluster=int(_p("max_cubes_per_cluster", 1)),
        num_best_cubes=int(_p("num_best_cubes", 100)),
        clearance=float(_p("clearance", 0.015)),
        support_plane_constraint=_normalize_support_plane_constraint(
            _p("support_plane_constraint", "fix_icp")),

        # SAM backend and model/prompt configuration.
        sam_mode=_p("sam_mode", "sam3"),
        sam_checkpoint_path=_p("sam_checkpoint_path", ""),
        sam_model_type=_p("sam_model_type", "vit_b"),
        sam_device=_p("sam_device", "auto"),
        sam_points_per_side=int(_p("sam_points_per_side", 32)),
        sam_pred_iou_thresh=float(_p("sam_pred_iou_thresh", 0.86)),
        sam_stability_score_thresh=float(_p("sam_stability_score_thresh", 0.92)),
        sam_min_mask_region_area=int(_p("sam_min_mask_region_area", 0)),
        sam_prompt_text=_p("sam_prompt_text", "cube."),
        sam_prompt_box_threshold=float(_p("sam_prompt_box_threshold", 0.25)),
        sam_prompt_text_threshold=float(_p("sam_prompt_text_threshold", 0.25)),
        sam_grounding_model_id=_p("sam_grounding_model_id", "IDEA-Research/grounding-dino-base"),
        sam_segmentor_model_id=_p("sam_segmentor_model_id", "facebook/sam-vit-base"), # alternatives: facebook/sam2-hiera-large facebook/sam-vit-huge

        # SAM mask filtering and 3D gating.
        sam_max_masks=int(_p("sam_max_masks", 100)),
        sam_min_mask_pixels=int(_p("sam_min_mask_pixels", 0)),
        sam_min_depth_pixels=int(_p("sam_min_depth_pixels", 0)),
        sam_mask_erosion_kernel=int(_p("sam_mask_erosion_kernel", 2)),
        sam_mask_erosion_iterations=int(_p("sam_mask_erosion_iterations", 1)),
        sam_min_points_per_cluster=int(_p("sam_min_points_per_cluster", 0)),
        sam_max_mask_area_ratio=float(_p("sam_max_mask_area_ratio", 0.35)),
        sam_plane_ransac_distance=float(_p("sam_plane_ransac_distance", 0.006)),
        sam_near_plane_distance=float(_p("sam_near_plane_distance", 0.006)),
        sam_max_near_plane_ratio=float(_p("sam_max_near_plane_ratio", 0.85)),
        sam_min_mask_plane_height=float(_p("sam_min_mask_plane_height", 0.012)),
        sam_max_cluster_extent_multiplier=float(
            _p("sam_max_cluster_extent_multiplier", 2.8)),
        sam_max_cluster_volume_multiplier=float(
            _p("sam_max_cluster_volume_multiplier", 7.0)),

        # SAM debug visualization.
        sam_show_windows=bool(_p("sam_show_windows", True)),
        sam_window_wait_ms=int(_p("sam_window_wait_ms", 1)),
    )
