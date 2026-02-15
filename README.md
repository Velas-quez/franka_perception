# franka_perception

ROS pipeline for 3D perception with a ZED2, focused on detecting cubes and estimating their poses for a Franka Emika arm. Provides a single-frame viewer (`listener_node`) and a continuous processor that publishes poses/markers (`dynamic_listener_node`).

## Requirements
- ROS (catkin)
- Python 3 + Open3D
- ZED2 camera publishing to `/zed2/zed_node/point_cloud/cloud_registered`

## Layout
- `src/franka_perception/src/franka_perception/` – pipeline: filtering, plane segmentation, clustering, cube fitting.
- `scripts/` – ROS nodes (`listener_node.py`, `dynamic_listener_node.py`, etc.).
- `launch/` – launch files (`listener.launch`, `dynamic_listener.launch`).
- `params.py` – default parameters loaded by the nodes.

## Build
```bash
catkin_make          # or catkin build
source devel/setup.bash
```

## Run
[Testing] Visualize a single cloud (Open3D):
```bash
roslaunch franka_perception listener.launch
```

[Real] Continuous processing + pose/marker publication:
```bash
roslaunch franka_perception dynamic_listener.launch
```

## Pipeline stage control (listener)
`listener_node.py` accepts `--stage` to stop early and render only up to that stage. `listener.launch` exposes `stage` (default `all`).
- `none`   – raw cloud.
- `filter` – filtered cloud (voxel + table removal).
- `cluster`– selected clusters only (with boxes).
- `all`    – full pipeline with fitted cubes.

Example:
```bash
roslaunch franka_perception listener.launch stage:=cluster
```

## Key parameters (via launch/param server)
- `cloud_topic`: PointCloud2 topic (default `/zed2/zed_node/point_cloud/cloud_registered`). -> needs to be changed to /group1/zed2/... on real robot
- `cube_side_length`: cube edge (m).
- `voxel_size`: downsample size.
- `cluster_eps`, `cluster_min_points`: DBSCAN.
- `base_plane_distance`: threshold to remove the table plane.
- `below_plane_tolerance`: rejects clusters below the segmented plane.
- `max_cluster_distance_from_plane_inliers`: rejects clusters too far from plane inliers.
- `max_cubes_per_cluster`, `clearance`: fitting limits and clearance.
- `axis_size`: coordinate frame size in Open3D.

## Published topics (dynamic listener)
- `~cube_poses` (`geometry_msgs/PoseArray`)
- `~cube_markers` (`visualization_msgs/MarkerArray`)

## Development notes
- Core pipeline in `pipeline.py`, visualization helpers in `visualization.py`.
- Tune parameters in `params.py` or via launch.
