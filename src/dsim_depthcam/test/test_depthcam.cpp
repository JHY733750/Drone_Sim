// The depth camera's geometry, camera model and renderer.
//
// Every expected number is derived by hand in the comment beside it, never
// read back from the code under test. Each test names the bug that would
// make it fail; scripts/mutation_check.sh injects most of those bugs and
// requires this file to notice.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "dsim_depthcam/depthcam.hpp"

using dsim_depthcam::CameraConfig;
using dsim_depthcam::Frame;
using dsim_depthcam::Mount;
using dsim_depthcam::Primitive;
using dsim_depthcam::World;
using dsim_depthcam::kNoHit;
using dsim_depthcam::makeBox;
using dsim_depthcam::makeCylinder;
using dsim_depthcam::makePlane;
using dsim_depthcam::makeSphere;
using V = Eigen::Vector3d;

namespace
{
constexpr double kPi = 3.14159265358979323846;

/// Body pose: position and yaw, level.
Eigen::Isometry3d body(double x, double y, double z, double yaw = 0.0)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(yaw, V::UnitZ()).toRotationMatrix();
  T.translation() = V(x, y, z);
  return T;
}

/// An odd-sized, noiseless camera, so there IS a centre pixel whose ray is
/// exactly the optical axis: cx = (41-1)/2 = 20, cy = (31-1)/2 = 15.
CameraConfig small()
{
  CameraConfig c;
  c.width = 41;
  c.height = 31;
  c.noise = false;
  return c;
}

/// Depth at the centre pixel of a `small()` frame; 0 = no return.
double centre(const Frame & f)
{
  return f.depth_m[(f.height / 2) * f.width + f.width / 2];
}

/// Render the camera mounted on a body pose, with the default mount unless
/// one is given.
Frame shoot(
  const std::vector<Primitive> & scene, const Eigen::Isometry3d & T_world_base,
  const CameraConfig & cfg, const Mount & m = Mount{}, std::uint64_t seed = 1)
{
  return dsim_depthcam::render(scene, dsim_depthcam::worldToOptical(T_world_base, m), cfg, seed);
}

/// Distance from a point to a primitive's SURFACE, written independently of
/// the intersection code: what "the point is on that object" means.
double surfaceResidual(const Primitive & p, const V & x)
{
  constexpr double INF = std::numeric_limits<double>::infinity();
  switch (p.type) {
    case Primitive::Type::Plane:
      return std::abs(x.z() - p.center.z());
    case Primitive::Type::Sphere:
      return std::abs((x - p.center).norm() - p.radius);
    case Primitive::Type::Cylinder: {
        const double rho = std::hypot(x.x() - p.center.x(), x.y() - p.center.y());
        const double dz = std::abs(x.z() - p.center.z());
        const double side = dz <= p.half_height + 1e-9 ? std::abs(rho - p.radius) : INF;
        const double cap = (std::isfinite(p.half_height) && rho <= p.radius + 1e-9)
          ? std::abs(dz - p.half_height) : INF;
        return std::min(side, cap);
      }
    case Primitive::Type::Box: {
        const V w = x - p.center;
        const double c = std::cos(p.yaw), s = std::sin(p.yaw);
        const V l(c * w.x() + s * w.y(), -s * w.x() + c * w.y(), w.z());
        // On the surface exactly when the largest per-axis excess is zero.
        return std::abs((l.cwiseAbs() - p.half_size).maxCoeff());
      }
  }
  return INF;
}

/// A cluttered scene: ground, two landmarks and a scenario with a moving
/// sphere and a column, all in front of a camera at (0, 0, 1.5).
World clutter()
{
  World w;
  w.landmarks.push_back(makeSphere(V(4.0, 0.6, 1.4), 0.8, "S"));
  w.landmarks.push_back(makeCylinder(V(6.0, -2.0, 1.0), 0.4, 2.0, "C"));
  w.landmarks.push_back(makeBox(V(5.0, 2.2, 1.0), V(1.0, 2.0, 2.0), 0.5, "B"));
  w.obstacles.push_back(dsim_depthcam::obstacleFromFlat(
      "M", "sphere", 0.6, {7, -1, 1.2, 0, 7, 1, 2.0, 10}));
  w.obstacles.push_back(dsim_depthcam::obstacleFromFlat(
      "W", "column", 0.5, {8.5, 0.5, 1.5, 0, 8.5, 0.5, 1.5, 10}));
  w.scenario_start_s = 2.0;
  return w;
}
}  // namespace

// ============================================================================
// Analytic hit distances, one primitive at a time
// ============================================================================

TEST(Geometry, SphereHitDistances)
{
  // Catches: the |d| = 1 assumption (t not divided by d.d), the far root
  // taken first, a negative root accepted, the inside case returning nothing.
  const auto s = makeSphere(V(5, 0, 0), 1.0);
  // Along +x with |d| = 2: the surface is at x = 4, reached at t = 4/2 = 2.
  // (Without the division by a = d.d the half-b form gives 8.)
  EXPECT_NEAR(intersect(s, V(0, 0, 0), V(2, 0, 0)), 2.0, 1e-12);
  // Off axis, d = (1, 0.1, 0). The line y = 0.1 x passes the centre at
  // distance 5*0.1/sqrt(1.01); along the unit direction the entry is
  // 5/sqrt(1.01) - sqrt(1 - 0.25/1.01), and t is that over |d| = sqrt(1.01).
  const double expect = 5.0 / 1.01 - std::sqrt(1.0 - 0.25 / 1.01) / std::sqrt(1.01);
  EXPECT_NEAR(intersect(s, V(0, 0, 0), V(1, 0.1, 0)), expect, 1e-12);
  // From the centre, straight up with |d| = 0.5: the wall is 1 m away, t = 2.
  EXPECT_NEAR(intersect(s, V(5, 0, 0), V(0, 0, 0.5)), 2.0, 1e-12);
  // Behind the camera: nothing, not the mirror image at t = -4.
  EXPECT_EQ(intersect(s, V(0, 0, 0), V(-1, 0, 0)), kNoHit);
  // A clean miss: passes 2 m from the centre of a 1 m sphere.
  EXPECT_EQ(intersect(s, V(0, 2, 0), V(1, 0, 0)), kNoHit);
}

TEST(Geometry, CylinderHitDistances)
{
  // Catches: end caps dropped, the height ignored (an infinite tube), the
  // height tested at the ray origin rather than at the hit, a column given
  // ends.
  // Finite: radius 0.5, 2 m tall, centred at z = 1 -> spans z in [0, 2].
  const auto c = makeCylinder(V(5, 0, 1), 0.5, 2.0);
  // Level ray at z = 1: side wall at x = 4.5.
  EXPECT_NEAR(intersect(c, V(0, 0, 1), V(1, 0, 0)), 4.5, 1e-12);
  // Level ray at z = 2.5, above the top: misses. An infinite tube hits it.
  EXPECT_EQ(intersect(c, V(0, 0, 2.5), V(1, 0, 0)), kNoHit);
  // Down onto the top cap from z = 10 with |d| = 2: 8 m to z = 2, t = 4.
  EXPECT_NEAR(intersect(c, V(5, 0, 10), V(0, 0, -2)), 4.0, 1e-12);
  // Up into the bottom cap from z = -3: t = 3.
  EXPECT_NEAR(intersect(c, V(5.2, 0, -3), V(0, 0, 1)), 3.0, 1e-12);
  // Descending ray from (0, 0, 5) with d = (1, 0, -0.65): at the wall
  // (x = 4.5) it is at z = 5 - 0.65*4.5 = 2.075, above the top, so the wall
  // does not count; it then meets the top cap (z = 2) at t = 3/0.65, at
  // x = 4.615 -- inside the disc. A cylinder tested for height at the ray
  // origin, or with no caps, gets this wrong.
  EXPECT_NEAR(intersect(c, V(0, 0, 5), V(1, 0, -0.65)), 3.0 / 0.65, 1e-12);
  // Descending d = (1, 0, -0.4) from z = 3: wall at x = 4.5, z = 1.2 -> 4.5.
  EXPECT_NEAR(intersect(c, V(0, 0, 3), V(1, 0, -0.4)), 4.5, 1e-12);

  // A column (infinite height) at the same place: hit at any altitude, and
  // a vertical ray down its axis meets no cap -- there are none.
  const auto col = makeCylinder(V(5, 0, 1.5), 0.5, kNoHit);
  EXPECT_NEAR(intersect(col, V(0, 0, 50), V(1, 0, 0)), 4.5, 1e-12);
  EXPECT_NEAR(intersect(col, V(0, 0, -50), V(1, 0, 0)), 4.5, 1e-12);
  EXPECT_EQ(intersect(col, V(5, 0, 100), V(0, 0, -1)), kNoHit);
}

TEST(Geometry, BoxHitDistancesIncludingYawAndItsSign)
{
  // Catches: yaw ignored, yaw applied with the wrong sign, the |d| = 1
  // assumption, the inside case, a box behind the camera.
  const auto b = makeBox(V(5, 0, 0), V(2, 2, 2), 0.0);
  EXPECT_NEAR(intersect(b, V(0, 0, 0), V(1, 0, 0)), 4.0, 1e-12);
  EXPECT_NEAR(intersect(b, V(0, 0, 0), V(0.5, 0, 0)), 8.0, 1e-12);   // |d| = 0.5
  EXPECT_NEAR(intersect(b, V(5, 0, 0), V(1, 0, 0)), 1.0, 1e-12);     // from inside
  EXPECT_EQ(intersect(b, V(0, 0, 0), V(-1, 0, 0)), kNoHit);          // behind

  // Yawed 45 deg, the same cube presents a vertical EDGE: sqrt(2) from the
  // centre, so the hit is at 5 - sqrt(2). Ignoring yaw gives 4.
  const auto b45 = makeBox(V(5, 0, 0), V(2, 2, 2), kPi / 4);
  EXPECT_NEAR(intersect(b45, V(0, 0, 0), V(1, 0, 0)), 5.0 - std::sqrt(2.0), 1e-12);

  // The SIGN of yaw needs an asymmetric box: a 4 x 0.2 plank whose long axis
  // is turned +45 deg, i.e. along (1, 1)/sqrt(2) from (5, 0). A ray up the
  // line x = 6 from y = -10 meets the plank's centreline at y = +1, and its
  // near face 0.1/cos(45) = 0.1*sqrt(2) earlier. Turned -45 deg the plank
  // would be met near y = -1 instead: t = 8.86, not 10.86.
  const auto plank = makeBox(V(5, 0, 0), V(4, 0.2, 2), kPi / 4);
  EXPECT_NEAR(
    intersect(plank, V(6, -10, 0), V(0, 1, 0)), 10.0 + 1.0 - 0.1 * std::sqrt(2.0), 1e-12);
}

TEST(Geometry, PlaneHitDistances)
{
  // Catches: a negative t accepted (the ground seen through the sky), the
  // |d| = 1 assumption.
  const auto g = makePlane(0.0);
  // From 2 m up, d = (1, 0, -0.5): 2 / 0.5 = 4.
  EXPECT_NEAR(intersect(g, V(0, 0, 2), V(1, 0, -0.5)), 4.0, 1e-12);
  EXPECT_NEAR(intersect(g, V(0, 0, 2), V(0, 0, -4)), 0.5, 1e-12);
  EXPECT_EQ(intersect(g, V(0, 0, 2), V(0, 0, 1)), kNoHit);   // looking up
  EXPECT_EQ(intersect(g, V(0, 0, 2), V(1, 0, 0)), kNoHit);   // parallel
}

TEST(Geometry, ConstructorsRefuseDegenerateShapes)
{
  // Catches: a zero-size primitive accepted, which renders as nothing and
  // says nothing.
  EXPECT_THROW(makeSphere(V::Zero(), 0.0), std::invalid_argument);
  EXPECT_THROW(makeCylinder(V::Zero(), 0.5, -1.0), std::invalid_argument);
  EXPECT_THROW(makeBox(V::Zero(), V(1, 0, 1), 0.0), std::invalid_argument);
}

// ============================================================================
// The camera model
// ============================================================================

TEST(Camera, IntrinsicsComeFromTheFovAndResolution)
{
  // Catches: the full FOV used where the half-angle belongs (fx = W/2 /
  // tan(fov) gives 2.6x too wide a view), the principal point at W/2.
  const CameraConfig c;    // 424 x 240, 87 x 58 deg
  const auto k = dsim_depthcam::intrinsics(c);
  EXPECT_NEAR(k.fx, 212.0 / std::tan(43.5 * kPi / 180.0), 1e-9);   // 223.40
  EXPECT_NEAR(k.fy, 120.0 / std::tan(29.0 * kPi / 180.0), 1e-9);   // 216.48
  EXPECT_NEAR(k.fx, 223.40, 0.01);
  EXPECT_NEAR(k.cx, 211.5, 1e-12);
  EXPECT_NEAR(k.cy, 119.5, 1e-12);
}

TEST(Camera, EveryPointIsInsideTheFrustumAndRangeAndTheFrustumIsFilled)
{
  // Catches: a wider FOV than configured (points outside the frustum), a
  // narrower one (the extreme pixels fall short of its edge), a missing range
  // clip (the ground runs to the horizon well past 10 m).
  CameraConfig cfg;       // full resolution, noise ON: the clip applies to
  const auto w = clutter();   // what the camera reports
  const auto f = shoot(dsim_depthcam::sceneAt(w, 5.0), body(0, 0, 1.5, 0.1), cfg);
  ASSERT_GT(f.points.size(), 30000u) << "a frustum test on an empty frame tests nothing";
  const double tx = std::tan(0.5 * cfg.hfov_rad), ty = std::tan(0.5 * cfg.vfov_rad);
  for (const auto & p : f.points) {
    ASSERT_GE(p.z(), cfg.min_range_m);
    ASSERT_LE(p.z(), cfg.max_range_m);
    ASSERT_LE(std::abs(p.x()) / p.z(), tx + 1e-12);
    ASSERT_LE(std::abs(p.y()) / p.z(), ty + 1e-12);
  }
  // The level ground beyond 10 m fills the top of the ground half of the
  // image, so with the clip there MUST be pixels without a return below the
  // horizon row... and above it, sky.
  size_t holes = 0;
  for (const auto d : f.depth_m) {holes += (d == 0.0);}
  EXPECT_GT(holes, 1000u);

  // Two-sided: facing a wall that fills the view, the outermost pixel
  // centres sit half a pixel inside the edge: |x/z| = ((W-1)/2) / fx
  // = tan(hfov/2) * (W-1)/W exactly.
  World wall;
  wall.ground = false;
  wall.landmarks.push_back(makeBox(V(8.1, 0, 0), V(10, 200, 200), 0.0));
  const auto g = shoot(dsim_depthcam::sceneAt(wall, 0.0), body(0, 0, 0), cfg);
  ASSERT_EQ(g.points.size(), size_t(cfg.width * cfg.height));
  double mx = 0, my = 0;
  for (const auto & p : g.points) {
    mx = std::max(mx, std::abs(p.x()) / p.z());
    my = std::max(my, std::abs(p.y()) / p.z());
  }
  EXPECT_NEAR(mx, tx * (cfg.width - 1) / cfg.width, 1e-9);
  EXPECT_NEAR(my, ty * (cfg.height - 1) / cfg.height, 1e-9);
}

TEST(Camera, WithNoiseOffEveryPointLiesOnASurface)
{
  // Catches: depth reported as range along the ray (the |d| = 1 slip: points
  // off every surface by up to 38 % at the edges), points deprojected with
  // the wrong intrinsics, the optical frame mis-rotated back to the world.
  CameraConfig cfg;
  cfg.noise = false;
  const auto w = clutter();
  const double t = 5.0;
  const auto scene = dsim_depthcam::sceneAt(w, t);
  const Mount m{V(0.1, 0.0, -0.05), 0.2};
  const auto T = dsim_depthcam::worldToOptical(body(0.3, -0.2, 1.5, 0.15), m);
  const auto f = dsim_depthcam::render(scene, T, cfg, 1);
  ASSERT_GT(f.points.size(), 30000u);
  // Every primitive type must actually be in the picture, or the check is
  // silent about that type.
  std::map<Primitive::Type, size_t> seen;
  double worst = 0.0;
  for (const auto & p : f.points) {
    const V x = T * p;
    double best = std::numeric_limits<double>::infinity();
    Primitive::Type which = Primitive::Type::Plane;
    for (const auto & s : scene) {
      const double r = surfaceResidual(s, x);
      if (r < best) {best = r; which = s.type;}
    }
    worst = std::max(worst, best);
    ++seen[which];
  }
  EXPECT_LT(worst, 1e-6);
  EXPECT_GT(seen[Primitive::Type::Plane], 100u);
  EXPECT_GT(seen[Primitive::Type::Sphere], 100u);
  EXPECT_GT(seen[Primitive::Type::Cylinder], 100u);
  EXPECT_GT(seen[Primitive::Type::Box], 100u);
}

TEST(Camera, TheNearerObjectOccludesTheFartherWhateverTheOrder)
{
  // Catches: the last (or first) primitive in the list winning instead of
  // the nearest.
  const auto s = makeSphere(V(3.1, 0, 1.5), 0.5);
  const auto b = makeBox(V(6.1, 0, 1.5), V(1, 1, 1), 0.0);
  // Camera at x = 0.1 (the default mount) looking +x at z = 1.5: the
  // sphere's face is at 2.6, x = 2.6 - 0.1 = 2.5 from the camera; the box's
  // is at 5.6, i.e. 5.5.
  EXPECT_NEAR(centre(shoot({s, b}, body(0, 0, 1.5), small())), 2.5, 1e-9);
  EXPECT_NEAR(centre(shoot({b, s}, body(0, 0, 1.5), small())), 2.5, 1e-9);
  EXPECT_NEAR(centre(shoot({b}, body(0, 0, 1.5), small())), 5.5, 1e-9);
}

TEST(Camera, AnObjectInsideTheMinimumRangeLeavesAHoleNotTheBackground)
{
  // Catches: the minimum range applied per primitive before choosing the
  // nearest, which would let the camera see THROUGH a near object to the
  // wall behind. A real stereo camera returns no depth there.
  const auto near = makeSphere(V(0.35, 0, 1.5), 0.1);     // face at 0.25 - 0.1 = 0.15 m
  const auto wall = makeBox(V(4.1, 0, 1.5), V(2, 20, 20), 0.0);
  const auto f = shoot({near, wall}, body(0, 0, 1.5), small());
  EXPECT_EQ(centre(f), 0.0);
  // ...and the wall is there for the pixels the sphere does not cover (the
  // image corner): 4.1 - 1 - 0.1 = 3.0 m.
  EXPECT_NEAR(f.depth_m[0], 3.0, 1e-9);
}

TEST(Camera, NothingIsReturnedFromBehindTheCamera)
{
  // Catches: a negative ray parameter accepted, by any primitive.
  World w;
  w.ground = false;
  w.landmarks = {
    makeSphere(V(-5, 0, 1.5), 2.0), makeBox(V(-4, 0, 1.5), V(1, 6, 6), 0.3),
    makeCylinder(V(-3, 1, 1.5), 0.5, 3.0)};
  EXPECT_TRUE(shoot(dsim_depthcam::sceneAt(w, 0), body(0, 0, 1.5), CameraConfig{}).points.empty());
  // The ground, with the camera pitched 30 deg UP: the lowest ray is
  // 30 - 29 = 1 deg above the horizon, so no ray reaches the floor.
  World g;
  const auto f = shoot(
    dsim_depthcam::sceneAt(g, 0), body(0, 0, 1.5), CameraConfig{},
    Mount{V(0.1, 0, 0), -30.0 * kPi / 180.0});
  EXPECT_TRUE(f.points.empty());
  // Control: pitched 30 deg DOWN, the same ground fills the view's centre at
  // z = 1.5 / sin(30 deg) = 3.0. Without this, the line above passes for a
  // camera that sees nothing at all.
  const auto d = shoot(
    dsim_depthcam::sceneAt(g, 0), body(0, 0, 1.5), small(),
    Mount{V(0.1, 0, 0), 30.0 * kPi / 180.0});
  EXPECT_NEAR(centre(d), 3.0, 1e-9);
}

TEST(Camera, OpticalFrameAxesAreRightAndDown)
{
  // Catches: the optical rotation wrong in any axis, the body attitude not
  // applied (yawed vehicle looking along world x), the mount offset or pitch
  // ignored or reversed.
  World w;
  w.ground = false;
  auto only = [&](const Primitive & p) {World x = w; x.landmarks = {p}; return dsim_depthcam::sceneAt(x, 0);};
  const auto level = body(0, 0, 1.5);
  // To the camera's RIGHT is world -y for a vehicle facing +x.
  const auto right = shoot(only(makeBox(V(5, -2, 1.5), V(0.5, 0.5, 0.5), 0)), level, CameraConfig{});
  ASSERT_FALSE(right.points.empty());
  for (const auto & p : right.points) {EXPECT_GT(p.x(), 0.0); EXPECT_GT(p.z(), 0.0);}
  const auto left = shoot(only(makeBox(V(5, 2, 1.5), V(0.5, 0.5, 0.5), 0)), level, CameraConfig{});
  ASSERT_FALSE(left.points.empty());
  for (const auto & p : left.points) {EXPECT_LT(p.x(), 0.0);}
  // BELOW the camera: optical +y.
  const auto below = shoot(only(makeBox(V(5, 0, 0.5), V(0.5, 0.5, 0.5), 0)), level, CameraConfig{});
  ASSERT_FALSE(below.points.empty());
  for (const auto & p : below.points) {EXPECT_GT(p.y(), 0.0);}
  // Vehicle yawed +90 deg (facing world +y): world +x is now on its RIGHT.
  const auto yawed = shoot(
    only(makeBox(V(2, 5, 1.5), V(0.5, 0.5, 0.5), 0)), body(0, 0, 1.5, kPi / 2), CameraConfig{});
  ASSERT_FALSE(yawed.points.empty());
  for (const auto & p : yawed.points) {EXPECT_GT(p.x(), 0.0);}

  // The mount offset rides with the body: yawed +90 deg, a 0.10 m forward
  // offset puts the camera at (0, 0.1, 1.5), so a unit sphere at
  // (0, 5.1, 1.5) is exactly 4.0 away. Offset in world x instead: 4.10.
  EXPECT_NEAR(
    centre(shoot(only(makeSphere(V(0, 5.1, 1.5), 1.0)), body(0, 0, 1.5, kPi / 2), small())),
    4.0, 1e-9);
}

// ============================================================================
// Noise and dropout
// ============================================================================

namespace
{
/// Measured noise sigma and mean error on a wall facing the camera at depth z.
std::pair<double, double> wallNoise(double z)
{
  World w;
  w.ground = false;
  // Camera at x = 0.1 (default mount) facing +x; the wall's face at 0.1 + z.
  w.landmarks.push_back(makeBox(V(0.1 + z + 5.0, 0, 0), V(10, 400, 400), 0.0));
  const auto scene = dsim_depthcam::sceneAt(w, 0);
  CameraConfig cfg;
  double sum = 0, sum2 = 0;
  size_t n = 0;
  for (std::uint64_t seed : {11u, 12u, 13u}) {
    const auto f = shoot(scene, body(0, 0, 0), cfg, Mount{}, seed);
    for (const auto & p : f.points) {
      const double e = p.z() - z;
      sum += e; sum2 += e * e; ++n;
    }
  }
  // Nothing may be lost to the range clip, or the estimate is truncated.
  EXPECT_EQ(n, 3u * cfg.width * cfg.height);
  const double mean = sum / n;
  return {std::sqrt(sum2 / n - mean * mean), mean};
}
}  // namespace

TEST(Noise, MeasuredSigmaMatchesTheStereoFormulaAtTwoDepths)
{
  // sigma_z = z^2 * subpixel / (fx * baseline), with fx = 212 / tan(43.5 deg)
  // computed HERE, not by the library. At 1 m: 0.08 / (223.40 * 0.05)
  // = 7.16 mm; at 4 m, sixteen times that, 114.6 mm.
  //
  // Catches: noise off, noise linear in z (right at 1 m, a quarter of the
  // truth at 4 m -- why there are two depths), fy where fx belongs (3.2 %
  // off), the half-angle slip in f. 300k samples estimate sigma to 0.13 %,
  // so the 1.5 % tolerance is 11 standard errors wide and still tighter
  // than the fx/fy difference.
  const double fx = 212.0 / std::tan(43.5 * kPi / 180.0);
  for (const double z : {1.0, 4.0}) {
    const double expect = z * z * 0.08 / (fx * 0.05);
    const auto [sigma, mean] = wallNoise(z);
    EXPECT_NEAR(sigma / expect, 1.0, 0.015) << "at z = " << z;
    // Unbiased: the mean error is far inside 4 standard errors of zero.
    EXPECT_LT(std::abs(mean), 4.0 * expect / std::sqrt(3.0 * 424 * 240)) << "at z = " << z;
  }
}

TEST(Noise, DropoutRemovesTheConfiguredFractionAndOnlyThat)
{
  // Catches: dropout ignored, dropout applied twice, pixels lost when it is
  // configured off.
  World w;
  w.ground = false;
  w.landmarks.push_back(makeBox(V(8.1, 0, 0), V(10, 400, 400), 0.0));   // face at 3 m
  const auto scene = dsim_depthcam::sceneAt(w, 0);
  CameraConfig cfg;
  const double n = cfg.width * cfg.height;
  EXPECT_EQ(shoot(scene, body(0, 0, 0), cfg).points.size(), size_t(n));
  cfg.dropout = 0.25;
  // Binomial standard error sqrt(0.25 * 0.75 / 101760) = 0.0014: +/-0.01 is
  // seven of them.
  const double kept = shoot(scene, body(0, 0, 0), cfg).points.size() / n;
  EXPECT_NEAR(kept, 0.75, 0.01);
}

TEST(Noise, AFrameIsAPureFunctionOfItsSeed)
{
  // Catches: hidden global randomness (frames not reproducible), a seed that
  // is ignored (every frame identical), the stamp left out of the frame seed.
  const auto scene = dsim_depthcam::sceneAt(clutter(), 5.0);
  CameraConfig cfg;
  cfg.dropout = 0.1;
  const auto a = shoot(scene, body(0, 0, 1.5), cfg, Mount{}, 7);
  const auto b = shoot(scene, body(0, 0, 1.5), cfg, Mount{}, 7);
  const auto c = shoot(scene, body(0, 0, 1.5), cfg, Mount{}, 8);
  EXPECT_EQ(a.depth_raw, b.depth_raw);
  EXPECT_NE(a.depth_raw, c.depth_raw);
  EXPECT_NE(dsim_depthcam::frameSeed(1, 1000000000), dsim_depthcam::frameSeed(1, 1033333333));
  EXPECT_NE(dsim_depthcam::frameSeed(1, 1000000000), dsim_depthcam::frameSeed(2, 1000000000));
  EXPECT_EQ(dsim_depthcam::frameSeed(1, 1000000000), dsim_depthcam::frameSeed(1, 1000000000));
}

// ============================================================================
// The image and the cloud
// ============================================================================

TEST(Output, TheImageAndTheCloudAgreePointForPoint)
{
  // Catches: the image and the cloud indexed differently (a flipped row, a
  // transposed index), the image built from a different depth than the
  // cloud, a point kept where the image says no return or vice versa.
  CameraConfig cfg;
  cfg.dropout = 0.1;      // holes in the middle of surfaces, not only sky
  const auto f = shoot(dsim_depthcam::sceneAt(clutter(), 5.0), body(0, 0, 1.5, -0.2), cfg);
  // Intrinsics computed here, independently.
  const double fx = 212.0 / std::tan(43.5 * kPi / 180.0);
  const double fy = 120.0 / std::tan(29.0 * kPi / 180.0);
  const double cx = 211.5, cy = 119.5;

  size_t nonzero = 0;
  for (const auto r : f.depth_raw) {nonzero += (r != 0);}
  ASSERT_EQ(nonzero, f.points.size());
  ASSERT_EQ(f.pixel.size(), f.points.size());
  ASSERT_GT(f.points.size(), 30000u);
  for (size_t i = 0; i < f.points.size(); ++i) {
    const int idx = f.pixel[i];
    const int u = idx % cfg.width, v = idx / cfg.width;
    const auto & p = f.points[i];
    ASSERT_NE(f.depth_raw[idx], 0) << "point " << i << " has no pixel";
    // The image is the cloud's z rounded to the millimetre.
    ASSERT_LE(std::abs(p.z() * 1000.0 - f.depth_raw[idx]), 0.5 + 1e-9);
    ASSERT_EQ(p.z(), f.depth_m[idx]);
    // And the point IS that pixel deprojected at that depth.
    ASSERT_NEAR(p.x(), (u - cx) / fx * p.z(), 1e-9);
    ASSERT_NEAR(p.y(), (v - cy) / fy * p.z(), 1e-9);
    // Row-major order, no duplicates.
    if (i > 0) {ASSERT_GT(idx, f.pixel[i - 1]);}
  }
}

// ============================================================================
// The scenario's moving obstacles
// ============================================================================

TEST(Obstacles, AMovingObstacleIsWhereItsBezierSaysAtTheStampedScenarioTime)
{
  // A cubic in x over scenario t in [1, 5]: control x = 4, 9, 2, 7 at
  // y = 0, z = 1.5, radius 0.5. The scenario clock starts at sim 5 s, so a
  // frame stamped sim 7.5 s is scenario t = 2.5, s = (2.5 - 1)/4 = 0.375:
  //   x = 4(1-s)^3 + 27 s(1-s)^2 + 6 s^2(1-s) + 7 s^3 = 5.828125
  // (Bernstein, written out here rather than calling centreAt). The camera
  // is at x = 0.1, so the centre pixel reads 5.828125 - 0.5 - 0.1.
  //
  // Catches: the scenario start not subtracted (sim 7.5 is past the window,
  // so the obstacle vanishes), the motion ignored, a column read as a ball,
  // the active window ignored (parked at an endpoint instead of absent).
  World w;
  w.ground = false;
  w.scenario_start_s = 5.0;
  w.obstacles.push_back(dsim_depthcam::obstacleFromFlat(
      "B", "sphere", 0.5, {4, 0, 1.5, 1.0, 9, 0, 1.5, 7.0 / 3, 2, 0, 1.5, 11.0 / 3, 7, 0, 1.5, 5.0}));
  const auto cam = body(0, 0, 1.5);
  const double s = 0.375;
  const double x = 4 * std::pow(1 - s, 3) + 27 * s * (1 - s) * (1 - s) + 6 * s * s * (1 - s) +
    7 * s * s * s;
  ASSERT_NEAR(x, 5.828125, 1e-12);
  EXPECT_NEAR(centre(shoot(dsim_depthcam::sceneAt(w, 7.5), cam, small())), x - 0.6, 1e-9);
  // At the end of its window (scenario 5.0) it is at x = 7: present.
  EXPECT_NEAR(centre(shoot(dsim_depthcam::sceneAt(w, 10.0), cam, small())), 6.4, 1e-9);
  // Before and after the window: absent. Not parked at x = 4 or x = 7.
  EXPECT_EQ(centre(shoot(dsim_depthcam::sceneAt(w, 5.9), cam, small())), 0.0);
  EXPECT_EQ(centre(shoot(dsim_depthcam::sceneAt(w, 10.2), cam, small())), 0.0);

  // A column is a disc at EVERY altitude: drawn at z = 100, it still stands
  // in front of a camera at 1.5 m. Read as a ball at z = 100 it would not.
  World c;
  c.ground = false;
  c.obstacles.push_back(dsim_depthcam::obstacleFromFlat(
      "W", "column", 0.5, {5, 0, 100, 0, 5, 0, 100, 10}));
  EXPECT_NEAR(centre(shoot(dsim_depthcam::sceneAt(c, 3.0), cam, small())), 4.4, 1e-9);
}

TEST(Obstacles, TheFlatParameterFormIsValidatedLikeTheReferee)
{
  // Catches: half an obstacle accepted from a truncated parameter.
  using dsim_depthcam::obstacleFromFlat;
  EXPECT_THROW(obstacleFromFlat("a", "sphere", 1, {0, 0, 0, 0, 1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(obstacleFromFlat("a", "sphere", 1, {0, 0, 0, 0}), std::invalid_argument);
  EXPECT_THROW(obstacleFromFlat("a", "sphere", 1, {0, 0, 0, 5, 1, 1, 1, 4}), std::invalid_argument);
  EXPECT_THROW(obstacleFromFlat("a", "cube", 1, {0, 0, 0, 0, 1, 1, 1, 4}), std::invalid_argument);
  const auto o = obstacleFromFlat("a", "column", 0.5, {0, 0, 0, 0, 1, 1, 1, 4});
  EXPECT_EQ(o.control_points.size(), 2u);
  EXPECT_EQ(o.type, "column");
}

TEST(Obstacles, TheMountPitchTiltsTheCameraDown)
{
  // Catches: the pitch sign reversed (the camera looks at the sky), the
  // pitch ignored. Camera 1.5 m up, pitched 0.3 rad down, ground only: the
  // optical axis meets the floor at depth 1.5 / sin(0.3) = 5.075.
  World g;
  const auto f = shoot(
    dsim_depthcam::sceneAt(g, 0), body(0, 0, 1.5), small(), Mount{V(0.1, 0, 0), 0.3});
  EXPECT_NEAR(centre(f), 1.5 / std::sin(0.3), 1e-9);
}

TEST(Obstacles, ALoopingScenarioRepeatsAndANonLoopingOneDoesNot)
{
  // The drive runs loop the scenario so the obstacles keep moving for as
  // long as someone flies. Same cubic as above, window [1, 5] on a 10 s
  // scenario starting at sim 5: sim 17.5 is scenario 12.5 -> 2.5 looped,
  // which must put the obstacle exactly where sim 7.5 put it.
  // Catches: loop ignored (absent at sim 17.5), the wrap taken on sim time
  // instead of scenario time (off by start_s), looping before t = 0.
  World w;
  w.ground = false;
  w.scenario_start_s = 5.0;
  w.scenario_duration_s = 10.0;
  w.obstacles.push_back(dsim_depthcam::obstacleFromFlat(
      "B", "sphere", 0.5, {4, 0, 1.5, 1.0, 9, 0, 1.5, 7.0 / 3, 2, 0, 1.5, 11.0 / 3, 7, 0, 1.5, 5.0}));
  const auto cam = body(0, 0, 1.5);
  const double first = centre(shoot(dsim_depthcam::sceneAt(w, 7.5), cam, small()));
  ASSERT_GT(first, 0.0);
  EXPECT_EQ(centre(shoot(dsim_depthcam::sceneAt(w, 17.5), cam, small())), 0.0);  // no loop
  w.scenario_loop = true;
  EXPECT_NEAR(centre(shoot(dsim_depthcam::sceneAt(w, 17.5), cam, small())), first, 1e-9);
  EXPECT_NEAR(centre(shoot(dsim_depthcam::sceneAt(w, 27.5), cam, small())), first, 1e-9);
  // Before the scenario starts the clock is negative, looping or not.
  EXPECT_EQ(centre(shoot(dsim_depthcam::sceneAt(w, 2.0), cam, small())), 0.0);
}

TEST(Preview, ThinnedPointsAreTheCloudsOwnPointsInTheWorldFrame)
{
  // A wall at x = 4, camera at x = 0.1 yawed 90 deg so the wall is at y = 4
  // in its view... simpler: unyawed, the wall is a box face at world x = 4.
  // Every preview point must lie ON the wall in WORLD coordinates (x = 4)
  // and carry its optical depth (4 - 0.1). The count is capped and the
  // stride uniform.
  // Catches: points left in the optical frame (x would be ~0, z ~3.9), the
  // transform inverted, depth taken as range, the cap ignored.
  const auto wall = makeBox(V(4.5, 0, 1.5), V(1, 40, 40), 0.0);
  const auto cam = body(0, 0, 1.5);
  const auto T = dsim_depthcam::worldToOptical(cam, Mount{});
  const auto f = dsim_depthcam::render({wall}, T, small(), 1);
  ASSERT_EQ(f.points.size(), 41u * 31u);
  const auto pts = dsim_depthcam::previewPoints(f, T, 100);
  ASSERT_LE(pts.size(), 100u);
  ASSERT_GE(pts.size(), 90u);
  for (const auto & p : pts) {
    EXPECT_NEAR(p.world.x(), 4.0, 1e-9);
    EXPECT_NEAR(p.depth, 3.9, 1e-9);
  }
  // Spread over the image, not a crop: both ends of the y range appear.
  double ymin = 1e9, ymax = -1e9;
  for (const auto & p : pts) {ymin = std::min(ymin, p.world.y()); ymax = std::max(ymax, p.world.y());}
  EXPECT_LT(ymin, -2.0);
  EXPECT_GT(ymax, 2.0);
  EXPECT_TRUE(dsim_depthcam::previewPoints(f, T, 0).empty());
}

TEST(Preview, TheFrustumIsTheRendersFieldOfView)
{
  // The far corners at range d must sit at the outer edge of the rendered
  // image: a wall at optical depth d has its outermost pixels' points just
  // inside the drawn corners, and the corner rays are the FOV the config
  // says. Catches: half the FOV used, fx/fy swapped, corners in the optical
  // frame, the apex not at the camera.
  CameraConfig c = small();
  c.hfov_rad = 80.0 * kPi / 180.0;
  c.vfov_rad = 50.0 * kPi / 180.0;
  const auto cam = body(1, 2, 1.5, 0.3);
  const auto T = dsim_depthcam::worldToOptical(cam, Mount{});
  const auto fr = dsim_depthcam::frustumCorners(c, T, 3.0);
  EXPECT_LT((fr[0] - T.translation()).norm(), 1e-12);
  for (int i = 1; i <= 4; ++i) {
    const V local = T.inverse() * fr[i];          // back in the optical frame
    EXPECT_NEAR(local.z(), 3.0, 1e-9);
    EXPECT_NEAR(std::abs(std::atan2(local.x(), local.z())), 0.5 * c.hfov_rad, 1e-9);
    EXPECT_NEAR(std::abs(std::atan2(local.y(), local.z())), 0.5 * c.vfov_rad, 1e-9);
  }
  // Top-left is up-left as the camera sees it: optical -x, -y.
  const V tl = T.inverse() * fr[1];
  EXPECT_LT(tl.x(), 0.0);
  EXPECT_LT(tl.y(), 0.0);
}
