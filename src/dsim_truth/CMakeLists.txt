cmake_minimum_required(VERSION 3.8)
project(dsim_truth)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)

# The arithmetic, header-only and Gazebo-free so it tests on the host.
add_library(dsim_truth_core INTERFACE)
target_include_directories(dsim_truth_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(dsim_truth_core INTERFACE Eigen3::Eigen)

# The Gazebo system plugin. Installed to lib/, which sim.launch.py puts on
# GZ_SIM_SYSTEM_PLUGIN_PATH; model.sdf names it by this library name.
add_library(dsim_truth_publisher SHARED src/truth_publisher.cpp)
target_link_libraries(dsim_truth_publisher
  dsim_truth_core
  gz-sim::gz-sim
  gz-plugin::register
  gz-transport::gz-transport)

install(TARGETS dsim_truth_publisher LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_truth test/test_truth.cpp)
  target_link_libraries(test_truth dsim_truth_core)
endif()

ament_package()
