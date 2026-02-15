#!/usr/bin/env python3
"""RGB-D input conversion helpers for ROS and local files."""

from typing import Tuple

import numpy as np


def image_msgs_to_numpy(rgb_msg, depth_msg) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert ROS Image messages to numpy arrays + inferred depth scale."""
    try:
        from cv_bridge import CvBridge
    except ImportError as exc:  # pragma: no cover - depends on ROS env
        raise ImportError("cv_bridge is required for ROS Image conversion") from exc

    bridge = CvBridge()
    rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

    depth_encoding = depth_msg.encoding.lower()
    if depth_encoding in {"16uc1", "mono16"}:
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        depth_scale = 1000.0
    elif depth_encoding == "32fc1":
        depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        depth_scale = 1.0
    else:
        depth = bridge.imgmsg_to_cv2(depth_msg)
        depth_scale = 1.0

    return np.asarray(rgb), np.asarray(depth), depth_scale


def load_rgb_image(path: str) -> np.ndarray:
    """Load RGB image from disk as uint8 RGB array."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("opencv-python (cv2) is required to load images") from exc
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth_image(path: str) -> np.ndarray:
    """Load depth image from disk preserving native dtype."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("opencv-python (cv2) is required to load images") from exc
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    if depth.ndim != 2:
        raise ValueError("Depth image must be single-channel")
    return np.asarray(depth)
