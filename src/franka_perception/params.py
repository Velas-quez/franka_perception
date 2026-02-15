#!/usr/bin/env python3
"""Parameter helpers."""

from dataclasses import dataclass

import rospy


@dataclass
class PerceptionParams:
    cloud_topic: str
    use_rgbd: bool
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    depth_scale: float
    depth_trunc: float
    rgbd_flip: bool
    target_frame: str
    cube_side_length: float
    axis_size: float
    voxel_size: float
    cluster_eps: float
    cluster_min_points: int
    base_plane_distance: float
    max_cubes_per_cluster: int
    clearance: float
    use_sam_segmentation: bool
    sam_checkpoint_path: str
    sam_model_type: str
    sam_device: str
    sam_points_per_side: int
    sam_pred_iou_thresh: float
    sam_stability_score_thresh: float
    sam_min_mask_region_area: int
    sam_max_masks: int
    mask_min_area_pixels: int
    mask_erosion_kernel: int
    mask_erosion_iterations: int
    mask_min_points: int
    sam_debug_logs: bool
    sam_show_masks_window: bool


def load_params(ns: str = "~") -> PerceptionParams:
    """Load parameters with defaults."""
    def _p(name, default):
        return rospy.get_param(f"{ns}{name}", default)

    return PerceptionParams(
        cloud_topic=_p("cloud_topic", "/zed2/zed_node/point_cloud/cloud_registered"),
        use_rgbd=bool(_p("use_rgbd", False)),
        rgb_topic=_p("rgb_topic", "/zed2/zed_node/rgb/image_rect_color"),
        depth_topic=_p("depth_topic", "/zed2/zed_node/depth/depth_registered"),
        camera_info_topic=_p("camera_info_topic", "/zed2/zed_node/rgb/camera_info"),
        depth_scale=float(_p("depth_scale", 0.0)),
        depth_trunc=float(_p("depth_trunc", 3.0)),
        rgbd_flip=bool(_p("rgbd_flip", True)),
        target_frame=_p("target_frame", "world"),
        cube_side_length=float(_p("cube_side_length", 0.045)),
        axis_size=float(_p("axis_size", 0.1)),
        voxel_size=float(_p("voxel_size", 0.002)),
        cluster_eps=float(_p("cluster_eps", 0.005)),
        cluster_min_points=int(_p("cluster_min_points", 10)),
        base_plane_distance=float(_p("base_plane_distance", 0.01)),
        max_cubes_per_cluster=int(_p("max_cubes_per_cluster", 2)),
        clearance=float(_p("clearance", 0.015)),
        use_sam_segmentation=bool(_p("use_sam_segmentation", False)),
        sam_checkpoint_path=_p("sam_checkpoint_path", ""),
        sam_model_type=_p("sam_model_type", "vit_b"),
        sam_device=_p("sam_device", "cuda"),
        sam_points_per_side=int(_p("sam_points_per_side", 32)),
        sam_pred_iou_thresh=float(_p("sam_pred_iou_thresh", 0.86)),
        sam_stability_score_thresh=float(_p("sam_stability_score_thresh", 0.92)),
        sam_min_mask_region_area=int(_p("sam_min_mask_region_area", 100)),
        sam_max_masks=int(_p("sam_max_masks", 20)),
        mask_min_area_pixels=int(_p("mask_min_area_pixels", 200)),
        mask_erosion_kernel=int(_p("mask_erosion_kernel", 3)),
        mask_erosion_iterations=int(_p("mask_erosion_iterations", 1)),
        mask_min_points=int(_p("mask_min_points", 30)),
        sam_debug_logs=bool(_p("sam_debug_logs", True)),
        sam_show_masks_window=bool(_p("sam_show_masks_window", False)),
    )
