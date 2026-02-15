#!/usr/bin/env python3
"""Point cloud IO utilities."""

from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


def msg_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 to Nx3 numpy array, dropping NaNs/inf."""
    points = [(x, y, z) for x, y, z in pc2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True)]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = arr[np.all(np.isfinite(arr), axis=1)]
    return arr


def has_points(arr: Optional[np.ndarray]) -> bool:
    """Check whether array is non-empty and finite."""
    return arr is not None and arr.size > 0 and np.isfinite(arr).all()


def camera_info_to_intrinsics(camera_info: CameraInfo) -> o3d.camera.PinholeCameraIntrinsic:
    """Build Open3D intrinsics from a ROS CameraInfo message."""
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]
    return o3d.camera.PinholeCameraIntrinsic(
        width=camera_info.width,
        height=camera_info.height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def _o3d_images_from_msgs(
    color_msg: Image,
    depth_msg: Image,
) -> Tuple[o3d.geometry.Image, o3d.geometry.Image, float]:
    """Convert ROS Image messages to Open3D images and infer depth scale."""
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:  # pragma: no cover - depends on ROS env
        raise ImportError("cv_bridge is required for RGB-D conversion") from exc

    bridge = CvBridge()

    # Color: always convert to RGB8 for Open3D
    color = bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")

    # Depth: keep native encoding
    depth_encoding = depth_msg.encoding.lower()
    if depth_encoding in {"16uc1", "mono16"}:
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        depth_scale = 1000.0  # typical depth in millimeters
    elif depth_encoding in {"32fc1"}:
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        depth_scale = 1.0  # already meters
    else:
        # Let cv_bridge try a sane conversion; default to meter scale
        depth = bridge.imgmsg_to_cv2(depth_msg)
        depth_scale = 1.0

    return o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale


def rgbd_msgs_to_xyz(
    color_msg: Image,
    depth_msg: Image,
    camera_info: CameraInfo,
    *,
    depth_scale: Optional[float] = None,
    depth_trunc: float = 3.0,
    convert_rgb_to_intensity: bool = False,
    flip: bool = True,
) -> np.ndarray:
    """Create a point cloud from ROS RGB-D messages using Open3D.

    Returns Nx3 numpy array in the camera frame (or flipped if requested).
    """
    color_o3d, depth_o3d, inferred_scale = _o3d_images_from_msgs(
        color_msg, depth_msg)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=inferred_scale if depth_scale is None else depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity,
    )
    intrinsics = camera_info_to_intrinsics(camera_info)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    if flip:
        pcd.transform([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    points = np.asarray(pcd.points, dtype=np.float64)
    if not points.size:
        return np.empty((0, 3), dtype=np.float64)
    if not np.isfinite(points).all():
        points = points[np.all(np.isfinite(points), axis=1)]
    return points
