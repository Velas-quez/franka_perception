#!/usr/bin/env python3
"""Point cloud IO utilities."""

from typing import Optional

import numpy as np
from sensor_msgs import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


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


def msg_to_xyzrgb(msg: PointCloud2):
    """Convert PointCloud2 to XYZ + normalized RGB (if available)."""
    # Attempt to read packed rgb/rgba; fall back to xyz only.
    try:
        data = np.array(list(pc2.read_points(
            msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)), dtype=np.float32)
    except (ValueError, AssertionError):
        return msg_to_xyz(msg), None

    if data.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    xyz = data[:, :3].astype(np.float64)
    # Ensure contiguous buffer before byte view to avoid reshape errors
    rgb_uint32 = np.ascontiguousarray(data[:, 3]).astype(np.uint32, copy=False)
    rgb_bytes = rgb_uint32.view(np.uint8).reshape(-1, 4)
    colors = rgb_bytes[:, :3].astype(np.float32) / 255.0

    # Drop rows with invalid values
    finite_mask = np.isfinite(xyz).all(axis=1)
    if not finite_mask.all():
        xyz = xyz[finite_mask]
        colors = colors[finite_mask]
    return xyz, colors


def has_points(arr: Optional[np.ndarray]) -> bool:
    """Check whether array is non-empty and finite."""
    return arr is not None and arr.size > 0 and np.isfinite(arr).all()
