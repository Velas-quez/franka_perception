source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --upgrade pip
python3 -m pip install -e src/local_src/my_package/franka_perception-master/franka_perception-master
python3 -m pip install -e "src/local_src/my_package/franka_perception-master/franka_perception-master[sam]"
python3 -m pip install open3d
catkin_make
source devel/setup.bash