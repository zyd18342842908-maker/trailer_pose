cmake_minimum_required(VERSION 3.0.2)
project(trailer_pose)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs geometry_msgs std_msgs tf
)

catkin_install_python(PROGRAMS
  scripts/trailer_pose_node.py
  scripts/pcd_publisher.py
  scripts/preprocess_bags.py
  scripts/inference.py
  scripts/prepare_dataset.py
  scripts/analyze_sessions.py
  scripts/annotate_sessions.py
  scripts/annotate_pose_gui.py
  scripts/train_bev.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(FILES scripts/trailer_types.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)
