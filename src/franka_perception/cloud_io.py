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


def has_points(arr: Optional[np.ndarray]) -> bool:
    """Check whether array is non-empty and finite."""
    return arr is not None and arr.size > 0 and np.isfinite(arr).all()
