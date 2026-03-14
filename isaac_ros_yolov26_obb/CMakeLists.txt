cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_yolov26_obb LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)

ament_auto_find_build_dependencies()

# OBB Decoder node
ament_auto_add_library(yolov26_obb_decoder_node SHARED src/yolov26_obb_decoder_node.cpp)
rclcpp_components_register_nodes(
  yolov26_obb_decoder_node "nvidia::isaac_ros::yolov26_obb::YoloV26OBBDecoderNode")

set_target_properties(yolov26_obb_decoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/isaac_ros_yolov26_obb_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()
