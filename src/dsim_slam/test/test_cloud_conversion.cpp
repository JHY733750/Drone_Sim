// PointCloud2 parsing, as realsense-ros publishes it. ROS-dependent, so this
// one runs under colcon only (the mutation harness builds test_slam.cpp).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "dsim_slam/cloud_conversion.hpp"

using dsim_slam::cloudToPoints;

namespace
{
/// x, y, z float32 plus an rgb field, the realsense-ros textured layout.
sensor_msgs::msg::PointCloud2 cloud(const std::vector<Eigen::Vector3f> & pts)
{
  sensor_msgs::msg::PointCloud2 m;
  m.header.frame_id = "camera_depth_optical_frame";
  sensor_msgs::PointCloud2Modifier mod(m);
  mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
  mod.resize(pts.size());
  sensor_msgs::PointCloud2Iterator<float> x(m, "x"), y(m, "y"), z(m, "z");
  for (const auto & p : pts) {
    *x = p.x();
    *y = p.y();
    *z = p.z();
    ++x; ++y; ++z;
  }
  return m;
}
}  // namespace

// FAILS IF: the parser assumes a 12-byte point (the rgb field shifts every
// point after the first), swaps axes, or passes NaN points through.
TEST(CloudConversion, ReadsXyzThroughExtraFieldsAndDropsNaN)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const auto m = cloud({{0.1f, -0.2f, 1.5f}, {nan, nan, nan}, {-0.3f, 0.4f, 2.5f}});
  std::vector<Eigen::Vector3d> out;
  ASSERT_TRUE(cloudToPoints(m, &out, nullptr));
  ASSERT_EQ(out.size(), 2u);
  EXPECT_NEAR(out[0].x(), 0.1, 1e-6);
  EXPECT_NEAR(out[0].y(), -0.2, 1e-6);
  EXPECT_NEAR(out[0].z(), 1.5, 1e-6);
  EXPECT_NEAR(out[1].x(), -0.3, 1e-6);
  EXPECT_NEAR(out[1].z(), 2.5, 1e-6);
}

// FAILS IF: a cloud without z is "parsed" into garbage instead of refused.
TEST(CloudConversion, RefusesACloudWithoutXyz)
{
  sensor_msgs::msg::PointCloud2 m;
  sensor_msgs::PointCloud2Modifier mod(m);
  mod.setPointCloud2FieldsByString(1, "rgb");
  mod.resize(3);
  std::vector<Eigen::Vector3d> out;
  std::string why;
  EXPECT_FALSE(cloudToPoints(m, &out, &why));
  EXPECT_FALSE(why.empty());
}
