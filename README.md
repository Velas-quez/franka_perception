# franka_perception

ROS pipeline for 3D perception with a ZED2, focused on detecting cubes and estimating their poses for a Franka Emika arm. Provides a single-frame viewer (`listener_node`) and a continuous processor that publishes poses/markers (`dynamic_listener_node`).

## Requirements
- ROS (catkin)
- Python 3 + Open3D
- ZED2 camera publishing to `/zed2/zed_node/point_cloud/cloud_registered`
- For SAM pipeline (listener only): `torch`, `segment-anything`, `opencv-python`

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

[Testing] Visualize using SAM RGB-D masks (listener only):
```bash
roslaunch franka_perception listener.launch \
  pipeline_mode:=sam_rgbd \
  sam_checkpoint_path:=/absolute/path/to/sam_vit_b_01ec64.pth
```

[Real] Continuous processing + pose/marker publication:
```bash
roslaunch franka_perception dynamic_listener.launch
```

[Real] Continuous processing with SAM masking (no visualization windows):
```bash
roslaunch franka_perception dynamic_listener.launch \
  pipeline_mode:=sam_rgbd \
  use_rgbd:=true \
  sam_checkpoint_path:=/absolute/path/to/sam_vit_b_01ec64.pth
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

## SAM pipeline setup (listener.launch only)
Install Python packages in the same environment used by ROS:
```bash
pip install opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or cpu wheel
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Download a SAM checkpoint (example `sam_vit_b_01ec64.pth`) and pass its path in launch:
```bash
roslaunch franka_perception listener.launch \
  pipeline_mode:=sam_rgbd \
  sam_model_type:=vit_b \
  sam_device:=auto \
  sam_checkpoint_path:=/absolute/path/to/sam_vit_b_01ec64.pth
```

When `pipeline_mode:=sam_rgbd`, the listener:
- consumes RGB + depth + camera_info;
- segments masks with SAM;
- erodes masks and projects each mask to its own 3D cluster;
- rejects table-like masks by area and plane-distance heuristics;
- runs cube fitting on each mask cluster;
- renders full cloud + masked cloud together in Open3D (different colors);
- opens extra windows for RGB image and SAM masks.

Useful SAM filtering parameters (listener.launch):
- `sam_max_mask_area_ratio`: rejects huge masks (table/background).
- `sam_near_plane_distance`: distance (m) to consider a point near table plane.
- `sam_max_near_plane_ratio`: if too many points are near plane, reject mask.
- `sam_min_mask_plane_height`: minimum p95 height (m) above plane to accept mask.
- `sam_max_cluster_extent_multiplier`: rejects clusters much larger than cube edge.
- `sam_max_cluster_volume_multiplier`: rejects clusters with excessive 3D volume.

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
