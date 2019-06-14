cmake_minimum_required(VERSION 2.8.3)
project(object_detect)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  mavros_msgs
  sensor_msgs
  dynamic_reconfigure
  cv_bridge
  image_transport
  image_geometry
  message_generation
  mrs_msgs
  mrs_lib
  mrs_uav_manager
  tf2
  tf2_ros
  tf2_msgs
  tf2_sensor_msgs
  tf2_eigen
  tf2_geometry_msgs
  nodelet
)

find_package(OpenCV 3.2 REQUIRED)

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

generate_dynamic_reconfigure_options(
  config/DetectionParams.cfg
)

catkin_package(
  CATKIN_DEPENDS message_runtime image_transport # roscpp rospy std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -Wall -g -O0")

# add_executable(display_tracking src/display_tracking.cpp)
# add_dependencies(display_tracking
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )
# target_link_libraries(display_tracking
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )

# add_executable(object_track src/object_track.cpp)
# add_dependencies(object_track
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )
# target_link_libraries(object_track
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )

add_library(ObjectDetector src/ObjectDetector.cpp src/BlobDetector.cpp src/utility_fcs.cpp)
add_dependencies(ObjectDetector
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ObjectDetector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
