source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --upgrade pip
python3 -m pip install -e "src/local_src/my_package/franka_perception-master/franka_perception-master[sam]"
catkin_make
source devel/setup.bash