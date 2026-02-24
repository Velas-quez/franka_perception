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
    fx, fy, cx, cy = _intrinsics_from_camera_info(camera_info)
    return o3d.camera.PinholeCameraIntrinsic(
        width=camera_info.width,
        height=camera_info.height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def _intrinsics_from_camera_info(camera_info: CameraInfo) -> Tuple[float, float, float, float]:
    """Return fx, fy, cx, cy preferring projection matrix P for rectified streams."""
    if len(camera_info.P) >= 7:
        fx_p = float(camera_info.P[0])
        fy_p = float(camera_info.P[5])
        cx_p = float(camera_info.P[2])
        cy_p = float(camera_info.P[6])
        if fx_p > 0.0 and fy_p > 0.0:
            return fx_p, fy_p, cx_p, cy_p

    return (
        float(camera_info.K[0]),
        float(camera_info.K[4]),
        float(camera_info.K[2]),
        float(camera_info.K[5]),
    )


def _get_cv_bridge():
    """Instantiate CvBridge lazily to keep non-RGBD workflows lightweight."""
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:  # pragma: no cover - depends on ROS env
        raise ImportError("cv_bridge is required for RGB-D conversion") from exc
    return CvBridge()


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8 while preserving relative contrast."""
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr_f)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)
    mn = float(np.min(arr_f[finite]))
    mx = float(np.max(arr_f[finite]))
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    if mn >= 0.0 and mx <= 1.0:
        return np.clip(arr_f * 255.0, 0.0, 255.0).astype(np.uint8)
    scaled = (arr_f - mn) * (255.0 / (mx - mn))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _color_msg_to_rgb8(color_msg: Image) -> np.ndarray:
    """Decode ROS color image robustly across common camera encodings."""
    bridge = _get_cv_bridge()
    try:
        color = bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")
    except Exception:
        return bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")

    enc = (color_msg.encoding or "").lower()
    arr = np.asarray(color)
    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = _to_uint8_image(arr)
        return np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1)).copy()

    if arr.ndim != 3:
        fallback = bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        return np.ascontiguousarray(fallback).copy()

    ch = arr.shape[2]
    if ch == 1:
        single = arr[:, :, 0]
        if single.dtype != np.uint8:
            single = _to_uint8_image(single)
        return np.ascontiguousarray(np.stack([single, single, single], axis=-1)).copy()

    if ch == 3:
        if enc in {"bgr8", "8uc3"}:
            arr = arr[:, :, ::-1]
        # enc rgb8 and unknown 3-ch are kept as-is.
        if arr.dtype != np.uint8:
            arr = _to_uint8_image(arr)
        return np.ascontiguousarray(arr).copy()

    if ch >= 4:
        arr4 = arr[:, :, :4]
        if enc == "bgra8":
            rgb = arr4[:, :, [2, 1, 0]]
        else:
            # rgba8 and unknown 4-channel encodings.
            rgb = arr4[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = _to_uint8_image(rgb)
        return np.ascontiguousarray(rgb).copy()

    fallback = bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
    return np.ascontiguousarray(fallback).copy()


def rgbd_msgs_to_numpy(
    color_msg: Image,
    depth_msg: Image,
    *,
    depth_scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ROS RGB-D messages to numpy arrays (RGB8 + depth in meters)."""
    bridge = _get_cv_bridge()
    color = _color_msg_to_rgb8(color_msg)

    depth_encoding = depth_msg.encoding.lower()
    if depth_encoding in {"16uc1", "mono16"}:
        depth_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        scale = 1000.0 if depth_scale is None or depth_scale <= 0.0 else float(depth_scale)
        depth_m = depth_raw.astype(np.float32) / scale
    elif depth_encoding in {"32fc1"}:
        depth_m = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1").astype(np.float32)
        if depth_scale is not None and depth_scale > 0.0 and depth_scale != 1.0:
            depth_m = depth_m / float(depth_scale)
    else:
        depth_raw = bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)
        scale = 1.0 if depth_scale is None or depth_scale <= 0.0 else float(depth_scale)
        depth_m = depth_raw / scale

    return color, depth_m


def depth_to_xyz(
    depth_m: np.ndarray,
    camera_info: CameraInfo,
    *,
    mask: Optional[np.ndarray] = None,
    depth_trunc: float = 3.0,
    flip: bool = True,
) -> np.ndarray:
    """Project a depth map (meters) to Nx3 points using CameraInfo intrinsics."""
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a HxW array")

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if depth_trunc > 0.0:
        valid &= depth_m <= float(depth_trunc)

    if mask is not None:
        if mask.shape != depth_m.shape:
            raise ValueError("mask shape must match depth shape")
        valid &= mask.astype(bool)

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = depth_m[rows, cols].astype(np.float64)
    fx, fy, cx, cy = _intrinsics_from_camera_info(camera_info)

    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))

    if flip and points.size:
        points[:, 1] *= -1.0
        points[:, 2] *= -1.0

    return points


def _o3d_images_from_msgs(
    color_msg: Image,
    depth_msg: Image,
) -> Tuple[o3d.geometry.Image, o3d.geometry.Image, float]:
    """Convert ROS Image messages to Open3D images and infer depth scale."""
    bridge = _get_cv_bridge()

    # Color: decode robustly to RGB8 for Open3D
    color = _color_msg_to_rgb8(color_msg)

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
