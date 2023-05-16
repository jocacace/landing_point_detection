cmake_minimum_required(VERSION 2.8.3)
project(landing_point_detection)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  tf
  cv_bridge 
  image_geometry 
  image_transport
)

include_directories(/usr/include/eigen3)

#find_package( PCL 1.8 REQUIRED )
find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(landing_point_detection src/landing_point_detection.cpp )
target_link_libraries(landing_point_detection ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${realsense2_LIBRARY})

