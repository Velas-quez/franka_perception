#!/usr/bin/env python3
"""Render the first received ZED2 point cloud with Open3D using shared pipeline."""

import argparse
import sys
import threading
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import message_filters

from franka_perception.cloud_io import msg_to_xyz, has_points, rgbd_msgs_to_xyz
from franka_perception.params import load_params
from franka_perception.pipeline import CubeDetectionPipeline
from franka_perception.rgbd_inputs import image_msgs_to_numpy
from franka_perception.rgbd_masking import camera_info_to_intrinsics
from franka_perception.sam_rgbd_pipeline import SamRgbdCubePipeline
from franka_perception.sam_segmentation import SamSegmenter
from franka_perception.sam_visualization import show_masks_window
from franka_perception.visualization import draw


class SingleCloudNode:
    """Subscribe to a point cloud, process once, then render."""

    def __init__(self, stage: str = "all") -> None:
        self.params = load_params()
        self.stage = stage
        self.pipeline = CubeDetectionPipeline(
            cube_side_length=self.params.cube_side_length,
            voxel_size=self.params.voxel_size,
            base_plane_distance=self.params.base_plane_distance,
            cluster_eps=self.params.cluster_eps,
            cluster_min_points=self.params.cluster_min_points,
            max_cubes_per_cluster=self.params.max_cubes_per_cluster,
            clearance=self.params.clearance,
        )
        self._points: Optional[np.ndarray] = None
        self._rgb_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self._inferred_depth_scale: Optional[float] = None
        self._cloud_ready = threading.Event()
        self._lock = threading.Lock()

        self._cloud_sub = None
        self._rgb_sub = None
        self._depth_sub = None
        self._info_sub = None
        self._sync = None
        self._latest_info: Optional[CameraInfo] = None
        self._sam_rgbd_pipeline: Optional[SamRgbdCubePipeline] = None
        self._sam_segmenter: Optional[SamSegmenter] = None

        if self.params.use_sam_segmentation:
            if not self.params.use_rgbd:
                raise RuntimeError("use_sam_segmentation requires use_rgbd=true")
            if not self.params.sam_checkpoint_path:
                raise RuntimeError("sam_checkpoint_path must be set when use_sam_segmentation=true")
            self._sam_segmenter = SamSegmenter(
                checkpoint_path=self.params.sam_checkpoint_path,
                model_type=self.params.sam_model_type,
                device=self.params.sam_device,
                points_per_side=self.params.sam_points_per_side,
                pred_iou_thresh=self.params.sam_pred_iou_thresh,
                stability_score_thresh=self.params.sam_stability_score_thresh,
                min_mask_region_area=self.params.sam_min_mask_region_area,
                verbose=self.params.sam_debug_logs,
            )
            self._sam_rgbd_pipeline = SamRgbdCubePipeline(
                self.pipeline,
                verbose=self.params.sam_debug_logs,
            )
            rospy.loginfo("Pipeline mode: SAM+RGBD segmentation")
            rospy.loginfo("SAM checkpoint: %s", self.params.sam_checkpoint_path)
        else:
            rospy.loginfo("Pipeline mode: classic point-cloud filtering+clustering")

        if self.params.use_rgbd:
            self._rgb_sub = message_filters.Subscriber(self.params.rgb_topic, Image)
            self._depth_sub = message_filters.Subscriber(self.params.depth_topic, Image)
            self._info_sub = rospy.Subscriber(
                self.params.camera_info_topic, CameraInfo, self._info_cb, queue_size=1)
            self._sync = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_sub, self._depth_sub], queue_size=5, slop=0.2)
            self._sync.registerCallback(self._rgbd_cb)
            rospy.loginfo("Waiting for RGB-D on %s + %s",
                          self.params.rgb_topic, self.params.depth_topic)
        else:
            self._cloud_sub = rospy.Subscriber(
                self.params.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
            rospy.loginfo("Waiting for point cloud on %s", self.params.cloud_topic)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._cloud_ready.is_set():
            return
        points = msg_to_xyz(msg)
        if not has_points(points):
            rospy.logwarn("Received empty/invalid point cloud; ignoring")
            return
        with self._lock:
            self._points = points
            self._cloud_ready.set()
        rospy.loginfo("Captured cloud with %d points", points.shape[0])
        try:
            self._cloud_sub.unregister()
        except Exception:
            pass

    def _info_cb(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_info = msg

    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._cloud_ready.is_set():
            return
        with self._lock:
            info_msg = self._latest_info
        if info_msg is None:
            rospy.logwarn_throttle(2.0, "Waiting for camera_info on %s",
                                   self.params.camera_info_topic)
            return
        if self.params.use_sam_segmentation:
            rgb_np, depth_np, inferred_depth_scale = image_msgs_to_numpy(rgb_msg, depth_msg)
            with self._lock:
                self._rgb_image = rgb_np
                self._depth_image = depth_np
                self._inferred_depth_scale = inferred_depth_scale
                self._cloud_ready.set()
            rospy.loginfo("Captured RGB-D frame for SAM segmentation (%dx%d)",
                          rgb_np.shape[1], rgb_np.shape[0])
        else:
            depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else None
            points = rgbd_msgs_to_xyz(
                rgb_msg,
                depth_msg,
                info_msg,
                depth_scale=depth_scale,
                depth_trunc=self.params.depth_trunc,
                flip=self.params.rgbd_flip,
            )
            if not has_points(points):
                rospy.logwarn("Received empty/invalid RGB-D; ignoring")
                return
            with self._lock:
                self._points = points
                self._cloud_ready.set()
            rospy.loginfo("Captured RGB-D cloud with %d points", points.shape[0])
        for sub in (self._rgb_sub, self._depth_sub):
            try:
                if sub is not None:
                    sub.unregister()
            except Exception:
                pass
        try:
            if self._info_sub is not None:
                self._info_sub.unregister()
        except Exception:
            pass

    def run(self) -> None:
        rospy.loginfo("Waiting for the first valid cloud...")
        while not rospy.is_shutdown():
            if self._cloud_ready.wait(timeout=0.2):
                break

        if rospy.is_shutdown():
            return

        with self._lock:
            points = self._points.copy() if self._points is not None else None
            rgb = self._rgb_image.copy() if self._rgb_image is not None else None
            depth = self._depth_image.copy() if self._depth_image is not None else None
            info = self._latest_info
            inferred_depth_scale = self._inferred_depth_scale

        rospy.loginfo("Rendering pipeline stage: %s", self.stage)
        if self.params.use_sam_segmentation:
            if self.stage not in {"cluster", "all"}:
                raise ValueError("With use_sam_segmentation=true, --stage must be 'cluster' or 'all'")
            if rgb is None or depth is None or info is None:
                rospy.logerr("Missing RGB-D data/camera_info for SAM pipeline")
                return
            intrinsics = camera_info_to_intrinsics(info)
            depth_scale = self.params.depth_scale if self.params.depth_scale > 0.0 else inferred_depth_scale
            result, debug = self._sam_rgbd_pipeline.process(
                rgb_image=rgb,
                depth_image=depth,
                intrinsics=intrinsics,
                segmenter=self._sam_segmenter,
                stop_after=self.stage,
                min_area_pixels=self.params.mask_min_area_pixels,
                max_masks=self.params.sam_max_masks,
                erosion_kernel=self.params.mask_erosion_kernel,
                erosion_iterations=self.params.mask_erosion_iterations,
                depth_scale=depth_scale,
                depth_trunc=self.params.depth_trunc,
                min_cluster_points=self.params.mask_min_points,
                flip=self.params.rgbd_flip,
            )
            rospy.loginfo("SAM masks=%d, clusters=%d, cubes=%d",
                          len(debug.raw_masks), len(debug.clusters), len(result.cubes))
            if self.params.sam_show_masks_window:
                try:
                    show_masks_window(rgb, debug.eroded_masks, window_name="SAM Masks (Eroded)")
                except Exception as exc:
                    rospy.logwarn("Failed to render SAM mask window: %s", exc)
        else:
            if not has_points(points):
                rospy.logerr("No valid point cloud received before shutdown")
                return
            result = self.pipeline.process(points, stop_after=self.stage)
        try:
            draw(result, axis_size=self.params.axis_size)
        except Exception as exc:
            rospy.logerr("Failed to render point cloud: %s", exc)


def _parse_args():
    parser = argparse.ArgumentParser(description="Render a ZED point cloud through the cube pipeline.")
    parser.add_argument(
        "--stage",
        choices=["none", "filter", "cluster", "all"],
        default="all",
        help="Stop the pipeline after this stage for visualization.",
    )
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main() -> None:
    args = _parse_args()
    rospy.init_node("listener", anonymous=False)
    node = SingleCloudNode(stage=args.stage)
    node.run()


if __name__ == "__main__":
    main()
