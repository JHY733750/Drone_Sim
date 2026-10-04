cmake_minimum_required(VERSION 3.8)
project(dsim_depthcam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
# A frame is ~100k rays against every primitive; unoptimised it is several
# times slower. colcon passes no build type unless asked, so pick one here.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
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
find_package(tf2_ros REQUIRED)
find_package(dsim_time REQUIRED)
find_package(dsim_eval REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Header-only camera model and ray caster, so the maths is testable without a
# node. It reuses the referee's obstacle geometry (dsim_eval/obstacle.hpp)
# rather than restating it: one implementation of where an obstacle is.
add_library(dsim_depthcam_core INTERFACE)
target_include_directories(dsim_depthcam_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${dsim_eval_INCLUDE_DIRS})
target_link_libraries(dsim_depthcam_core INTERFACE Eigen3::Eigen)

add_executable(depthcam_node src/depthcam_node.cpp)
target_link_libraries(depthcam_node dsim_depthcam_core yaml-cpp)
ament_target_dependencies(depthcam_node
  rclcpp nav_msgs sensor_msgs geometry_msgs tf2_ros dsim_time)

install(TARGETS depthcam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # Pure geometry and camera model: also mutation-checked on the host by
  # scripts/mutation_check.sh, which compiles it with g++ and no ROS.
  ament_add_gtest(test_depthcam test/test_depthcam.cpp)
  target_link_libraries(test_depthcam dsim_depthcam_core)
  # The landmark loader needs yaml-cpp, which the host lacks: colcon only.
  ament_add_gtest(test_landmarks test/test_landmarks.cpp)
  target_link_libraries(test_landmarks dsim_depthcam_core yaml-cpp)
  target_compile_definitions(test_landmarks PRIVATE
    DSIM_DEPTHCAM_EXAMPLE_LANDMARKS="${CMAKE_CURRENT_SOURCE_DIR}/../../config/landmarks_example.yaml")
  # A measurement, not a test: built, never run by ctest.
  add_executable(bench_depthcam test/bench_depthcam.cpp)
  target_link_libraries(bench_depthcam dsim_depthcam_core)
endif()

ament_export_include_directories(include)
ament_package()
