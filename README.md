# franka_perception

ROS pipeline for 3D perception with a ZED2, focused on detecting cubes and estimating their poses for a Franka Emika arm. Provides a single-frame viewer (`listener_node`) and a continuous processor that publishes poses/markers (`dynamic_listener_node`).

## Requirements
- ROS (catkin)
- Python 3 + Open3D
- ZED2 camera publishing to `/zed2/zed_node/point_cloud/cloud_registered`
- For SAM3 pipeline: `torch`, `transformers`, `pillow`, `huggingface_hub`
- SAM3 official requirement: Python `>=3.10` (the current container is Python 3.8)

## Dependency setup (mounted repo inside container)
Run dependency installation inside the container (where ROS nodes run):

```bash
cd /opt/ros_ws
source /opt/ros/$ROS_DISTRO/setup.bash

rosdep update
rosdep install --from-paths src --ignore-src -r -y

python3 -m pip install --upgrade pip
python3 -m pip install -e src/franka_perception
# with SAM extras:
# python3 -m pip install -e "src/franka_perception[sam]"
```

When dependencies change in `package.xml` or `setup.py`, update with:

```bash
cd /opt/ros_ws
source /opt/ros/$ROS_DISTRO/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -e src/franka_perception
# or with SAM extras:
# python3 -m pip install -e "src/franka_perception[sam]"
catkin_make
source devel/setup.bash
```

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
  sam_model_id:=facebook/sam3.1-hiera-large \
  sam_prompt:=cube
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
  sam_model_id:=facebook/sam3.1-hiera-large \
  sam_prompt:=cube
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

## SAM3 pipeline setup
Install SAM3 extras in the same environment used by ROS:
```bash
python3 -m pip install -e "src/franka_perception[sam]"
```

Run with prompt-driven SAM3 segmentation:
```bash
roslaunch franka_perception listener.launch \
  pipeline_mode:=sam_rgbd \
  sam_model_id:=facebook/sam3.1-hiera-large \
  sam_prompt:=cube \
  sam_device:=auto \
  sam_score_threshold:=0.0
```

When `pipeline_mode:=sam_rgbd`, the listener:
- consumes RGB + depth + camera_info;
- segments masks with SAM3 text prompt;
- erodes masks and projects each mask to its own 3D cluster;
- rejects table-like masks by area and plane-distance heuristics;
- runs cube fitting on each mask cluster;
- renders full cloud + masked cloud together in Open3D (different colors);
- opens extra windows for RGB image and SAM masks.

Useful SAM3 parameters (listener.launch):
- `sam_prompt`: prompt text. You can pass a CSV list, e.g. `cube, red cube`.
- `sam_model_id`: model from Hugging Face (default `facebook/sam3.1-hiera-large`).
- `sam_score_threshold`: minimum SAM3 IoU confidence to keep mask.
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
