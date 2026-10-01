// The landmark file loader. Needs yaml-cpp, so it runs under colcon only and
// is declared untestable on the host in scripts/mutation_check.sh; the ray
// casting it feeds is covered by test_depthcam.cpp.
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "dsim_depthcam/depthcam.hpp"
#include "dsim_depthcam/landmarks.hpp"

using dsim_depthcam::Primitive;
using V = Eigen::Vector3d;

#ifndef DSIM_DEPTHCAM_EXAMPLE_LANDMARKS
#error "the CMake file must point this at config/landmarks_example.yaml"
#endif

TEST(Landmarks, TheExampleFileLoadsAsWritten)
{
  // Catches: a field read into the wrong member (radius as height, size as
  // half size), a cylinder's centre taken as its base, yaw dropped.
  const auto l = dsim_depthcam::loadLandmarks(DSIM_DEPTHCAM_EXAMPLE_LANDMARKS);
  ASSERT_EQ(l.size(), 3u);

  EXPECT_EQ(l[0].name, "L0");
  EXPECT_EQ(l[0].type, Primitive::Type::Box);
  EXPECT_TRUE(l[0].center.isApprox(V(3.0, -1.5, 1.0)));
  EXPECT_TRUE(l[0].half_size.isApprox(V(0.5, 0.25, 1.0)));
  EXPECT_DOUBLE_EQ(l[0].yaw, 0.3);

  EXPECT_EQ(l[1].type, Primitive::Type::Cylinder);
  EXPECT_DOUBLE_EQ(l[1].radius, 0.4);
  EXPECT_DOUBLE_EQ(l[1].half_height, 1.0);
  EXPECT_DOUBLE_EQ(l[1].center.z(), 1.0);

  EXPECT_EQ(l[2].type, Primitive::Type::Sphere);
  EXPECT_DOUBLE_EQ(l[2].radius, 0.5);
}

TEST(Landmarks, ACylinderCentreIsItsMidHeight)
{
  // L1 is centred at z = 1 and 2 m tall, so it stands on the ground: its
  // top is at z = 2. Looking straight down from z = 5 above its axis, the
  // first surface is 3 m away. Centre-at-base would put it at 2 m (top at 3).
  const auto l = dsim_depthcam::loadLandmarks(DSIM_DEPTHCAM_EXAMPLE_LANDMARKS);
  EXPECT_NEAR(dsim_depthcam::intersect(l[1], V(4.0, 1.5, 5.0), V(0, 0, -1)), 3.0, 1e-12);
}

TEST(Landmarks, EmptyPathMeansNoneAndABadFileIsAnError)
{
  // Catches: a missing file silently treated as an empty world -- which
  // renders as free space exactly where the landmarks were meant to be.
  EXPECT_TRUE(dsim_depthcam::loadLandmarks("").empty());
  EXPECT_THROW(dsim_depthcam::loadLandmarks("/nonexistent/landmarks.yaml"), std::runtime_error);
  EXPECT_TRUE(dsim_depthcam::landmarksFromYaml(YAML::Load("landmarks: []")).empty());
}

TEST(Landmarks, MalformedEntriesAreRefusedNotSkipped)
{
  // Catches: an entry with a typo dropped quietly instead of stopping the
  // node.
  using dsim_depthcam::landmarksFromYaml;
  EXPECT_THROW(
    landmarksFromYaml(YAML::Load("landmarks: [{name: a, type: cone, center: [0, 0, 0]}]")),
    std::runtime_error);
  EXPECT_THROW(
    landmarksFromYaml(YAML::Load("landmarks: [{name: a, type: sphere, center: [0, 0, 0]}]")),
    std::runtime_error);
  EXPECT_THROW(
    landmarksFromYaml(YAML::Load(
      "landmarks: [{name: a, type: box, center: [0, 0, 0], size: [1, 1]}]")),
    std::runtime_error);
  EXPECT_THROW(
    landmarksFromYaml(YAML::Load(
      "landmarks: [{name: a, type: cylinder, center: [0, 0], radius: 1, height: 1}]")),
    std::runtime_error);
  EXPECT_THROW(
    landmarksFromYaml(YAML::Load(
      "landmarks: [{name: a, type: sphere, center: [0, 0, 0], radius: -1}]")),
    std::invalid_argument);
}

TEST(Landmarks, TheExampleRendersWhereItIsWritten)
{
  // End to end through the renderer: L2, a 0.5 m sphere at (6, 0, 1.5),
  // dead ahead of a level camera at (0.1, 0, 1.5): 6 - 0.5 - 0.1 = 5.4.
  dsim_depthcam::World w;
  w.ground = false;
  w.landmarks = dsim_depthcam::loadLandmarks(DSIM_DEPTHCAM_EXAMPLE_LANDMARKS);
  dsim_depthcam::CameraConfig c;
  c.width = 41;
  c.height = 31;
  c.noise = false;
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = V(0, 0, 1.5);
  const auto f = dsim_depthcam::render(
    dsim_depthcam::sceneAt(w, 0), dsim_depthcam::worldToOptical(T, dsim_depthcam::Mount{}), c, 1);
  EXPECT_NEAR(f.depth_m[15 * 41 + 20], 5.4, 1e-9);
}
