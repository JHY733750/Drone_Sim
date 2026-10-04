cmake_minimum_required(VERSION 3.8)
project(dsim_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
# Optimised by default: the node has a real-time budget (a 424x240 cloud
# every 33 ms) and an unoptimised Eigen build is several times too slow.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)

# --- core library: no ROS, so it can be unit tested in isolation -----------
# scripts/mutation_check.sh compiles exactly these sources with g++ on the
# host; a source added here must be added there too, or the harness aborts.
add_library(dsim_slam_core SHARED
  src/voxel_map.cpp
  src/registration.cpp
  src/gyro_buffer.cpp
  src/scan_odometry.cpp
)
target_include_directories(dsim_slam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(dsim_slam_core PUBLIC Eigen3::Eigen)

# --- node ------------------------------------------------------------------
add_executable(slam_node src/slam_node.cpp)
target_link_libraries(slam_node dsim_slam_core)
ament_target_dependencies(slam_node
  rclcpp nav_msgs sensor_msgs geometry_msgs diagnostic_msgs tf2 tf2_ros tf2_eigen)

install(TARGETS dsim_slam_core
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS slam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # The flight tests ray-cast ~100k pixels per frame for ~1500 frames.
  ament_add_gtest(test_slam test/test_slam.cpp TIMEOUT 600)
  target_link_libraries(test_slam dsim_slam_core)
  ament_add_gtest(test_cloud_conversion test/test_cloud_conversion.cpp)
  target_include_directories(test_cloud_conversion PRIVATE include)
  target_link_libraries(test_cloud_conversion Eigen3::Eigen)
  ament_target_dependencies(test_cloud_conversion sensor_msgs)
endif()

ament_export_include_directories(include)
ament_export_libraries(dsim_slam_core)
ament_export_dependencies(Eigen3)
ament_package()
