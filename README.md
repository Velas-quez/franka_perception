
roslaunch franka_perception_thiago dynamic_listener.launch enviroment:=atena n_stack_cube_cloud:=4
pip install --force-reinstall --no-cache-dir open3d
python3 -m pip install -e "src/local_src/my_package/franka_perception-master/franka_perception-master[sam]"
python3 -m pip install -e src/local_src/my_package/franka_perception-master/franka_perception-master
pip install open3d
rosdep install --from-paths src --ignore-src -r -y

roslaunch zed_wrapper zed2i.launch

roslaunch zed_wrapper zed2i.launch \
    camera_name:=zed2 \
    node_name:=zed_node \
    confidence:=50 \
    texture_confidence:=10 \
    depth_stabilization:=1

roslaunch panda_moveit_config franka_control.launch robot_ip:=10.10.10.10

# franka_perception

ROS pipeline for 3D perception with ZED cameras, focused on cube detection and pose estimation for Franka.
It provides:
- `listener_node.py`: processes one frame and renders with Open3D.
- `dynamic_listener_node.py`: processes continuously and publishes poses/markers.

## Requirements
- ROS (catkin)
- Python 3 + Open3D
- For SAM RGB-D pipeline: `torch`, `transformers`, `pillow`

## Dependency setup
```bash
cd /opt/ros_ws
source /opt/ros/$ROS_DISTRO/setup.bash

rosdep update
rosdep install --from-paths src --ignore-src -r -y

python3 -m pip install --upgrade pip
python3 -m pip install -e src/franka_perception
# with SAM extras (needed for main pipeline):
python3 -m pip install -e "src/franka_perception[sam]"
```

## Build
```bash
cd /opt/ros_ws
catkin_make
source devel/setup.bash
```

## Package layout
- `src/franka_perception/src/franka_perception/core/` – IO and ROS publishing helpers.
- `src/franka_perception/src/franka_perception/geometry/` – filtering, clustering, plane/cube geometry.
- `src/franka_perception/src/franka_perception/pipelines/` – classic pipeline and SAM RGB-D pipeline.
- `src/franka_perception/src/franka_perception/render/` – visualization helpers.
- `src/franka_perception/src/franka_perception/params.py` – centralized defaults for both nodes.
- `src/franka_perception/scripts/` – ROS executable nodes.
- `src/franka_perception/launch/` – launch files.

## Launch arguments (only relevant runtime controls)

### `listener.launch`
- `stage`: `none|filter|cluster|all`
- `pipeline_mode`: `classic|sam_rgbd`
- `show_original_cloud`: `true|false`
- `show_input_clouds`: `true|false` (render both raw input clouds as debug overlay)
- `sam_mode`: `sam3|sam1`
- `sam_checkpoint_path`: SAM1 checkpoint path (optional)
- `sam_prompt_text`: text prompt used in SAM3
- `use_rgbd`: `true|false`
- `support_plane_constraint`: `fix_icp|adjust|stack|none` (default `adjust`; constrain during ICP, adjust only after ICP, stack cubes on the top face of lower cubes when applicable, or disable support-plane handling)
- `rgbd_flip`: `true|false` (default `false`; keep `false` when using TF to world)
- `enviroment`: `poseidon|atena|simulation`
- `enable_tracking`: `true|false` (default `false`; publish persistent tracked IDs on dedicated topics)

### `dynamic_listener.launch`
- `pipeline_mode`: `classic|sam_rgbd`
- `sam_mode`: `sam3|sam1`
- `sam_checkpoint_path`: SAM1 checkpoint path (optional)
- `sam_prompt_text`: text prompt used in SAM3
- `use_rgbd`: `true|false`
- `support_plane_constraint`: `fix_icp|adjust|stack|none` (default `adjust`)
- `rgbd_flip`: `true|false` (default `false`; keep `false` when using TF to world)
- `enviroment`: `poseidon|atena|simulation`

## Environment-based topic selection
`enviroment` selects default topics in `params.py`:
- `poseidon`
  - `/group1/zed2i/zed_node/point_cloud/cloud_registered`
  - `/group1/zed2i/zed_node/rgb/image_rect_color`
  - `/group1/zed2i/zed_node/depth/depth_registered`
  - `/group1/zed2i/zed_node/rgb/camera_info`
- `atena`
  - `/group1/zed2/zed_node/point_cloud/cloud_registered`
  - `/group1/zed2/zed_node/left/image_rect_color`
  - `/group1/zed2/zed_node/depth/depth_registered`
  - `/group1/zed2/zed_node/left/camera_info`
- `simulation`
  - `/zed2/zed_node/point_cloud/cloud_registered`
  - `/zed2/zed_node/rgb/image_rect_color`
  - `/zed2/zed_node/depth/depth_registered`
  - `/zed2/zed_node/rgb/camera_info`

You can still override any topic directly via ROS params if needed.

## Parameter policy
Most tunable values (voxel size, clustering, SAM thresholds, etc.) are intentionally configured in `params.py`, so `listener` and `dynamic_listener` stay aligned.

## Run examples

Single-frame classic pipeline in simulation:
```bash
roslaunch franka_perception listener.launch \
  stage:=all \
  pipeline_mode:=classic \
  use_rgbd:=false \
  enviroment:=simulation
```

Single-frame SAM RGB-D in Poseidon:
```bash
roslaunch franka_perception listener.launch \
  stage:=all \
  pipeline_mode:=sam_rgbd \
  use_rgbd:=true \
  sam_mode:=sam3 \
  sam_prompt_text:="cube, block" \
  enviroment:=poseidon
```

Continuous SAM RGB-D in Atena:
```bash
roslaunch franka_perception dynamic_listener.launch \
  pipeline_mode:=sam_rgbd \
  use_rgbd:=true \
  sam_mode:=sam3 \
  sam_prompt_text:="cube, block" \
  enviroment:=atena
```

## Published topics (dynamic listener)
- `~cube_poses` (`geometry_msgs/PoseArray`)
- `~cube_markers` (`visualization_msgs/MarkerArray`)
- `~reconstructed_cloud` (`sensor_msgs/PointCloud2`) - nuvem reconstruida a partir de RGB-D
- `~tracked_cubes` (`franka_perception_thiago/TrackedCubeArray`) when `enable_tracking:=true`
- `~tracked_cube_markers` (`visualization_msgs/MarkerArray`) when `enable_tracking:=true`
- `~tracked_cube_labels` (`visualization_msgs/MarkerArray`) when `enable_tracking:=true`
