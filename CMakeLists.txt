cmake_minimum_required(VERSION 3.1.2)
# shut up some fckin cmake warning
cmake_policy(SET CMP0048 NEW)

# new cmake requires to specify project version for some reason
project(object_detect VERSION 666)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
  tf2
  tf2_ros
  tf2_msgs
  tf2_sensor_msgs
  tf2_eigen
  tf2_geometry_msgs
  nodelet
)

find_package(OpenCV 4.2 REQUIRED)

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

generate_dynamic_reconfigure_options(
  config/DetectionParams.cfg
)

#Generate messages in the 'msg' folder
add_message_files(
   FILES
   BallDetection.msg
   BallDetections.msg
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
   sensor_msgs
 )

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS image_transport mrs_lib mrs_msgs roscpp std_msgs
  LIBRARIES color_mapping
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

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -O3 -g -DUSE_OPENCL -fopenmp -DNDEBUG -lstdc++fs")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wall -O3 -g -DUSE_OPENCL -fopenmp -DNDEBUG -lstdc++fs")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -Wall -O3 -g -DUSE_OPENCL -fopenmp -DNDEBUG -lstdc++fs")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++17 -Wall -O3 -g -DUSE_OPENCL -fopenmp -DNDEBUG -lstdc++fs")
# set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -g -O0 -DUSE_OPENCL")
# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wall -g -O0 -DUSE_OPENCL")
# set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -Wall -g -O0 -DUSE_OPENCL")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-std=c++17 -Wall -g -O0 -DUSE_OPENCL")

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

add_library(ObjectDetector src/ObjectDetector.cpp src/BlobDetector.cpp src/BallCandidate.cpp src/utility_fcs.cpp src/lut.cpp src/lut_fcs.cpp src/color_mapping.cpp)
add_dependencies(ObjectDetector
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(ObjectDetector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_library(color_mapping src/color_mapping.cpp)
add_dependencies(color_mapping
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(color_mapping
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_executable(backproject_display src/backproject_display.cpp)
# add_dependencies(backproject_display
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )
# target_link_libraries(backproject_display
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
# )

add_executable(cpicker src/cpicker/cpicker.cpp)
add_dependencies(cpicker
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(cpicker
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# add_library(color_mapping src/color_mapping.cpp)
# target_link_libraries(color_mapping
#   ${catkin_LIBRARIES}
#   ${Eigen_LIBRARIES}
#   )
