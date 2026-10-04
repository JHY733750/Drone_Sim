// sensor_msgs/PointCloud2 -> points, the way realsense-ros 4.x publishes it.
//
// realsense-ros's pointcloud filter publishes x, y, z as float32 (plus rgb
// when textured), in the OPTICAL frame, unordered by default (zero-depth
// pixels dropped) or ordered with NaNs. Field offsets and the point step are
// read from the message, never assumed, so an rgb field, padding, or an
// ordered cloud all parse the same. Non-finite points are dropped here; the
// depth gate is the odometry's.
//
// Header-only and separate from the node so the conversion has its own test
// (test_cloud_conversion.cpp, ROS-dependent, colcon only).

#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace dsim_slam
{

/// False (and `why` set) if the cloud has no float32 x/y/z fields.
inline bool cloudToPoints(
  const sensor_msgs::msg::PointCloud2 & msg, std::vector<Eigen::Vector3d> * out,
  std::string * why)
{
  out->clear();
  bool has_x = false, has_y = false, has_z = false;
  for (const auto & f : msg.fields) {
    const bool f32 = f.datatype == sensor_msgs::msg::PointField::FLOAT32;
    if (f.name == "x") {has_x = f32;}
    if (f.name == "y") {has_y = f32;}
    if (f.name == "z") {has_z = f32;}
  }
  if (!(has_x && has_y && has_z)) {
    if (why != nullptr) {*why = "cloud has no float32 x/y/z fields";}
    return false;
  }
  out->reserve(static_cast<std::size_t>(msg.width) * msg.height);
  sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
  for (; x != x.end(); ++x, ++y, ++z) {
    const Eigen::Vector3d p(*x, *y, *z);
    if (p.allFinite()) {out->push_back(p);}
  }
  return true;
}

}  // namespace dsim_slam
