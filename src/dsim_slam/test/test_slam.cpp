// Invariants for the depth-camera scan-to-map odometry.
//
// Every test generates its own data: synthetic_world.hpp ray-casts a
// D435-like depth camera (424x240, 87x58 deg, 0.28-10 m, sigma_z ~ z^2) into
// a world of analytic shapes along a known trajectory, and synthesises the
// gyro from the same trajectory. The answer is therefore known exactly, and
// each test names the wrong implementation it fails on.
// scripts/mutation_check.sh injects those implementations to prove it.
//
// No ROS: this compiles on the host with g++ and gtest.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "dsim_slam/gyro_buffer.hpp"
#include "dsim_slam/lie.hpp"
#include "dsim_slam/registration.hpp"
#include "dsim_slam/scan_odometry.hpp"
#include "dsim_slam/voxel_map.hpp"
#include "synthetic_world.hpp"

using namespace dsim_slam;

namespace
{
constexpr double kDeg = M_PI / 180.0;
constexpr double kImuHz = 250.0;

/// The D435's noise at 424x240 (see OdometryConfig::depth_noise_quadratic).
synth::Camera noisyCamera()
{
  synth::Camera c;
  c.noise_floor = 0.003;
  c.noise_k = 0.0072;
  return c;
}

/// Camera on the body: 10 cm forward, 3 cm up, optical axes as built by the
/// test (independently of the library).
Eigen::Isometry3d mountTruth()
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = synth::opticalAxesInBody();
  T.translation() = Eigen::Vector3d(0.10, 0.0, 0.03);
  return T;
}

/// The same mount as the ODOMETRY is told it, through the library's helper.
OdometryConfig defaultConfig()
{
  OdometryConfig c;
  c.T_base_sensor = Eigen::Isometry3d::Identity();
  c.T_base_sensor.linear() = opticalToBodyRotation();
  c.T_base_sensor.translation() = Eigen::Vector3d(0.10, 0.0, 0.03);
  return c;
}

Eigen::Isometry3d poseFrom(const Eigen::Vector3d & p, double roll, double pitch, double yaw)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
  T.translation() = p;
  return T;
}

double rotationError(const Eigen::Isometry3d & a, const Eigen::Isometry3d & b)
{
  return Eigen::AngleAxisd(a.linear().transpose() * b.linear()).angle();
}

struct FlightSpec
{
  synth::Trajectory traj;
  double t0 {0.0};
  double t1 {1.0};
  double cloud_hz {30.0};
  synth::Camera cam {noisyCamera()};
  bool feed_gyro {true};
  double gyro_noise {1e-3};                                      // per sample, as config/drone.yaml
  Eigen::Vector3d gyro_bias {2e-4, -1.5e-4, 1e-4};
  unsigned seed {1};
  std::function<const synth::World &(double)> world;
};

struct FlightLog
{
  std::vector<double> t;
  std::vector<Eigen::Isometry3d> truth;
  std::vector<OdometryReport> rep;
  std::vector<double> ms;          // processScan wall time
  double path_length {0.0};
};

/// Fly `spec` with `odo`, which must have been constructed at the true
/// initial pose (the test's choice of map frame: the world).
FlightLog fly(const FlightSpec & spec, ScanOdometry & odo)
{
  FlightLog log;
  std::mt19937 rng(spec.seed);
  std::normal_distribution<double> unit(0.0, 1.0);
  const Eigen::Isometry3d mount = mountTruth();
  double ti = spec.t0 - 0.5;
  const int frames = static_cast<int>(std::floor((spec.t1 - spec.t0) * spec.cloud_hz)) + 1;
  Eigen::Vector3d prev = spec.traj(spec.t0).translation();
  for (int k = 0; k < frames; ++k) {
    const double tc = spec.t0 + k / spec.cloud_hz;
    // The IMU runs ahead of the camera, as on the vehicle: the cloud arrives
    // after its own exposure, so its whole interval is covered.
    while (spec.feed_gyro && ti <= tc + 0.008) {
      const Eigen::Vector3d w = synth::bodyRate(spec.traj, ti) + spec.gyro_bias +
        spec.gyro_noise * Eigen::Vector3d(unit(rng), unit(rng), unit(rng));
      odo.addGyro(ti, w);
      ti += 1.0 / kImuHz;
    }
    const Eigen::Isometry3d T = spec.traj(tc);
    const auto cloud = synth::render(spec.world(tc), T * mount, spec.cam, &rng);
    const auto a = std::chrono::steady_clock::now();
    const OdometryReport r = odo.processScan(tc, cloud);
    const auto b = std::chrono::steady_clock::now();
    log.ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    log.t.push_back(tc);
    log.truth.push_back(T);
    log.rep.push_back(r);
    log.path_length += (T.translation() - prev).norm();
    prev = T.translation();
  }
  return log;
}

const synth::World & rich()
{
  static const synth::World w = synth::richWorld();
  return w;
}

/// The 25 s survey: a 2.5 m circle at 0.9 m/s, looking along the path with
/// a slow +-17 deg look-around, bobbing 0.25 m, rolling and pitching a few
/// degrees -- all six degrees of freedom move.
Eigen::Isometry3d survey(double t)
{
  const double R = 2.5, v = 0.9, w = v / R;
  const Eigen::Vector3d p(R * std::cos(w * t), R * std::sin(w * t), 1.2 + 0.25 * std::sin(0.9 * t));
  const double yaw = w * t + M_PI / 2 + 0.3 * std::sin(0.7 * t);
  return poseFrom(p, 0.06 * std::sin(1.1 * t), 0.05 * std::sin(1.3 * t), yaw);
}
}  // namespace

// ===========================================================================
// Conventions
// ===========================================================================

// FAILS IF: the update and the error are not inverses (rotation composed on
// the wrong side in one of them, or a sign flipped).
TEST(Lie, UpdateAndErrorAreInverses)
{
  const auto A = poseFrom({1.0, -2.0, 0.5}, 0.3, -0.2, 2.0);
  const auto B = poseFrom({1.3, -1.7, 0.4}, 0.1, 0.25, 2.6);
  const auto C = applyUpdate(A, poseError(B, A));
  EXPECT_LT((C.translation() - B.translation()).norm(), 1e-12);
  EXPECT_LT(rotationError(C, B), 1e-12);
}

// FAILS IF: opticalToBodyRotation() is transposed (it maps body -> optical)
// or has an axis sign wrong. Three independent constructions must agree:
// the library's matrix, the axes by meaning, and the static transform
// realsense-ros publishes for camera_link -> camera_depth_optical_frame,
// rpy (-pi/2, 0, -pi/2) applied as Rz(yaw) Ry(pitch) Rx(roll).
TEST(Frames, OpticalToBodyMatchesRealsenseStaticTransform)
{
  const Eigen::Matrix3d rs = (Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX())).toRotationMatrix();
  EXPECT_LT((opticalToBodyRotation() - rs).norm(), 1e-12);
  EXPECT_LT((opticalToBodyRotation() - synth::opticalAxesInBody()).norm(), 1e-12);
  // And what it means: a point straight ahead of the lens is ahead of the body.
  EXPECT_LT((opticalToBodyRotation() * Eigen::Vector3d(0, 0, 2) - Eigen::Vector3d(2, 0, 0)).norm(),
    1e-12);
}

// FAILS IF: the body-frame mount is applied inverted (T^-1 for T), the
// optical rotation is transposed (forward motion then appears along -y), or
// the odometry reports map-to-body instead of body-to-map (-x). The camera
// moves 0.6 m along its own optical +z; the odometry must report it along
// base_link +x -- here the map frame IS the initial base_link, so that is
// map +x, whatever the world yaw (30 deg) is.
//
// The x bound is loose ON PURPOSE and the reason is a known error, not
// noise: measured 0.73 m, also with the depth noise off. Motion straight at
// surfaces is over-estimated ~20 % over the first second, while the map is
// one frame thick and the noise-scaled plane windows straddle junctions
// (README, failure modes). This test is about axes and signs; the survey
// test measures accuracy.
TEST(Frames, OpticalForwardMotionIsBodyPlusX)
{
  const Eigen::Isometry3d start = poseFrom({-1.0, -2.0, 1.2}, 0.0, 0.0, 30 * kDeg);
  FlightSpec s;
  s.traj = [&](double t) {
      Eigen::Isometry3d T = start;
      T.translation() += start.linear() * Eigen::Vector3d(0.6 * t, 0.0, 0.0);
      return T;
    };
  s.t1 = 1.0;
  s.world = [](double) -> const synth::World & {return rich();};
  ScanOdometry odo(defaultConfig());   // map = base_link at the first scan
  const FlightLog log = fly(s, odo);
  const Eigen::Vector3d p = log.rep.back().pose.translation();
  std::printf("  optical +z by 0.6 m -> map (%.4f, %.4f, %.4f)\n", p.x(), p.y(), p.z());
  EXPECT_GT(p.x(), 0.45);
  EXPECT_LT(p.x(), 0.85);
  EXPECT_LT(std::abs(p.y()), 0.06);
  EXPECT_LT(std::abs(p.z()), 0.06);
  // The velocity is in the BODY frame (REP-145): +x as well.
  EXPECT_GT(log.rep.back().velocity_body.x(), 0.3);
  EXPECT_LT(std::abs(log.rep.back().velocity_body.y()), 0.3);
}

// ===========================================================================
// Building blocks
// ===========================================================================

// FAILS IF: the voxel mean is unweighted (a far, noisy point counts as much
// as a near one) or the weights are 1/sigma instead of 1/sigma^2.
TEST(VoxelMap, MeanIsInverseVarianceWeighted)
{
  VoxelMapConfig c;
  c.voxel_size_m = 1.0;
  c.max_voxel_sigma_m = 1.0;
  c.min_neighbours = 1;
  VoxelMap m(c);
  m.insert({{0.2, 0.2, 0.2}, {0.8, 0.8, 0.8}}, {0.01, 0.04});
  const auto pts = m.points();
  ASSERT_EQ(pts.size(), 1u);
  const double expect = (0.2 / 0.01 + 0.8 / 0.04) / (1 / 0.01 + 1 / 0.04);   // 0.32
  EXPECT_NEAR(pts[0].x(), expect, 1e-12);
}

// FAILS IF: an unconfirmed voxel (one far, noisy observation) is used as a
// correspondence, or a confirmed one is not.
TEST(VoxelMap, VoxelIsRefusedUntilConfirmed)
{
  VoxelMapConfig c;
  c.voxel_size_m = 0.25;
  c.max_voxel_sigma_m = 0.05;    // needs weight >= 400
  VoxelMap m(c);
  std::vector<Eigen::Vector3d> plane;
  for (int i = -3; i <= 3; ++i) {
    for (int j = -3; j <= 3; ++j) {plane.push_back({0.25 * i + 0.1, 0.25 * j + 0.1, 0.0});}
  }
  m.insert(plane, std::vector<double>(plane.size(), 0.1 * 0.1));    // sigma 0.1: weight 100
  EXPECT_FALSE(m.query({0.1, 0.1, 0.02}).found);
  EXPECT_EQ(m.confirmedSize(), 0u);
  for (int k = 0; k < 3; ++k) {m.insert(plane, std::vector<double>(plane.size(), 0.01));}
  const PlaneMatch q = m.query({0.1, 0.1, 0.02});
  ASSERT_TRUE(q.found);
  EXPECT_NEAR(std::abs(q.normal.z()), 1.0, 1e-9);
}

// FAILS IF: the gyro is integrated in the world frame (dR * R instead of
// R * dR), or the interval ends are not clipped. Rotate about body x, then
// about body z: the order is observable.
TEST(GyroBuffer, IntegratesBodyRatesInOrderAndClipsTheInterval)
{
  GyroBuffer g(0.05);
  for (int i = 0; i < 100; ++i) {g.add(0.01 * i, Eigen::Vector3d(1.0, 0.0, 0.0));}
  for (int i = 100; i < 200; ++i) {g.add(0.01 * i, Eigen::Vector3d(0.0, 0.0, 1.0));}
  // [0.5, 1.5]: 0.5 rad about x, then 0.5 rad about z.
  const auto R = g.integrate(0.5, 1.5);
  ASSERT_TRUE(R.has_value());
  const Eigen::Matrix3d expect = expSO3({0.5, 0.0, 0.0}) * expSO3({0.0, 0.0, 0.5});
  EXPECT_LT((*R - expect).norm(), 1e-9);
  // Mid-sample ends: [0.505, 0.555] is exactly 0.05 rad about x.
  const auto R2 = g.integrate(0.505, 0.555);
  ASSERT_TRUE(R2.has_value());
  EXPECT_LT((*R2 - expSO3({0.05, 0.0, 0.0})).norm(), 1e-12);
}

// FAILS IF: a gap in the IMU stream is bridged by holding a stale rate, or
// an interval beyond the newest sample is "integrated" from nothing.
TEST(GyroBuffer, RefusesGapsAndUncoveredIntervals)
{
  GyroBuffer g(0.05);
  for (int i = 0; i < 50; ++i) {g.add(0.004 * i, Eigen::Vector3d::UnitZ());}
  for (int i = 0; i < 50; ++i) {g.add(0.5 + 0.004 * i, Eigen::Vector3d::UnitZ());}   // 0.3 s hole
  EXPECT_TRUE(g.integrate(0.02, 0.15).has_value());
  EXPECT_FALSE(g.integrate(0.1, 0.6).has_value());
  EXPECT_FALSE(g.integrate(0.6, 1.5).has_value());
}

// FAILS IF: roll and pitch are swapped or have a sign wrong, so the map
// would start tilted.
TEST(GravityAlignment, RecoversRollAndPitchFromAResting)
{
  const Eigen::Matrix3d R = poseFrom({0, 0, 0}, 12 * kDeg, -7 * kDeg, 40 * kDeg).linear();
  const Eigen::Vector3d f = R.transpose() * Eigen::Vector3d(0, 0, 9.81);
  const Eigen::Matrix3d Rg = gravityAlignedRotation(f, 40 * kDeg);
  EXPECT_LT(Eigen::AngleAxisd(R.transpose() * Rg).angle(), 1e-12);
}

// ===========================================================================
// Registration
// ===========================================================================

namespace
{
VoxelMap mapFromFrame(const Eigen::Isometry3d & T_world_body, unsigned seed, bool noise)
{
  VoxelMapConfig mc;
  VoxelMap m(mc);
  std::mt19937 rng(seed);
  const synth::Camera cam = noise ? noisyCamera() : synth::Camera{};
  const auto cloud = synth::render(rich(), T_world_body * mountTruth(), cam, &rng);
  ScanOdometry pre(defaultConfig());
  const auto src = pre.preprocess(cloud, nullptr);
  std::vector<Eigen::Vector3d> pts;
  std::vector<double> var;
  for (const auto & s : src) {pts.push_back(T_world_body * s.p); var.push_back(1e-4);}
  m.insert(pts, var);
  return m;
}
}  // namespace

// FAILS IF: the point-to-plane Jacobian is wrong (lever arm about the map
// origin instead of the body, (n x p) for (p x n)), the update is applied on
// the wrong side, or the solver stops early. The source points ARE the
// map's voxel means seen from the true pose, and the map only accepts
// patches whose means are coplanar to 1 um: every accepted plane then
// passes exactly through its source point at the true pose, the residual is
// zero there, and Gauss-Newton must land on it. Measured: 1.1e-6 m and
// 4.8e-7 rad from a 4 cm / 3 deg start (a residual floor of ~1e-7 m that is
// NOT eigen-solver precision -- an iterative solver gives the same -- and
// that I have not traced). Every bug named above errs by mm to cm.
TEST(Registration, RecoversAKnownTransformExactly)
{
  const Eigen::Isometry3d truth = poseFrom({0.0, -2.5, 1.2}, 0.05, -0.03, 0.4);
  VoxelMapConfig mc;
  mc.max_plane_distance_m = 1e-6;
  mc.min_voxel_sigma_m = 1e-5;
  // The patch window's depth band scales with the point's sigma (here
  // 0.1 mm); widen it to 10 cm so the 4 cm initial offset can pull in. The
  // coplanarity check that makes this exact does not depend on it.
  mc.window_band_sigmas = 1000.0;
  VoxelMap map(mc);
  {
    const auto cloud = synth::render(rich(), truth * mountTruth(), synth::Camera{}, nullptr);
    ScanOdometry pre(defaultConfig());
    std::vector<Eigen::Vector3d> pts;
    for (const auto & s : pre.preprocess(cloud, nullptr)) {pts.push_back(truth * s.p);}
    map.insert(pts, std::vector<double>(pts.size(), 1e-8));
  }
  std::vector<SourcePoint> src;
  for (const auto & m : map.points()) {src.push_back({truth.inverse() * m, 1e-8});}
  Vector6d off;
  off << 0.04, -0.03, 0.02, 1.5 * kDeg, -1.0 * kDeg, 2.5 * kDeg;
  RegistrationConfig rc;
  rc.max_iterations = 100;
  rc.convergence_eps = 1e-12;
  const auto res = registerScan(src, map, applyUpdate(truth, off), 0.5, nullptr, rc);
  const double dt = (res.pose.translation() - truth.translation()).norm();
  const double dr = rotationError(res.pose, truth);
  std::printf("  exact recovery: %.2e m, %.2e rad after %d iterations, %d correspondences\n",
    dt, dr, res.iterations, res.correspondences);
  EXPECT_GT(res.correspondences, 50);   // not "exact" because nothing matched
  EXPECT_LT(dt, 1e-5);
  EXPECT_LT(dr, 1e-5);
}

// FAILS IF: registration only works when points coincide. Two independent
// noise-free renders 12 cm and 3 deg apart sample the surfaces at different
// places, and point-to-plane must still recover the motion. Run as the
// odometry runs it: the rotation comes in with a gyro-grade prior (0.5
// mrad), because from one frame of a narrow-FOV camera a sideways shift and
// a yaw are nearly the same image change -- without the prior this case
// lands 2.3 cm and 0.42 deg off (measured), which is why the odometry
// trusts the gyro.
TEST(Registration, RecoversMotionBetweenTwoIndependentRenders)
{
  const Eigen::Isometry3d A = poseFrom({0.0, -2.5, 1.2}, 0.0, 0.0, 0.4);
  Vector6d motion;
  motion << 0.08, 0.06, -0.05, 1.0 * kDeg, -1.5 * kDeg, 2.0 * kDeg;
  const Eigen::Isometry3d B = applyUpdate(A, motion);
  const VoxelMap map = mapFromFrame(A, 1, false);
  const auto cloud = synth::render(rich(), B * mountTruth(), synth::Camera{}, nullptr);
  ScanOdometry pre(defaultConfig());
  const auto src = pre.preprocess(cloud, nullptr);
  PosePrior prior;
  prior.pose = A;
  prior.pose.linear() = B.linear();
  prior.information.diagonal() << 0, 0, 0, 4e6, 4e6, 4e6;
  const auto res = registerScan(src, map, prior.pose, 0.1, &prior, RegistrationConfig{});
  const double dt = (res.pose.translation() - B.translation()).norm();
  const double dr = rotationError(res.pose, B);
  std::printf("  two renders: %.2e m, %.3f deg, %d iterations, %d correspondences of %zu\n", dt, dr / kDeg,
    res.iterations, res.correspondences, src.size());
  EXPECT_LT(dt, 0.02);       // measured 1.3 cm
  EXPECT_LT(dr, 0.1 * kDeg);
}

namespace
{
/// Floor plus a wall `wall_x` metres along +x; camera at x = 0.5 facing it.
struct WallScan
{
  RegistrationResult res;
  Degeneracy d;
};
WallScan wallScan(double wall_x, bool noise, bool wide_judge)
{
  synth::World wall;
  wall.addGround();
  wall.addPlane(Eigen::Vector3d::UnitX(), wall_x);
  const Eigen::Isometry3d T = poseFrom({0.5, 0.0, 1.2}, 0.0, 0.0, 0.0);
  std::mt19937 rng(3);
  const synth::Camera cam = noise ? noisyCamera() : synth::Camera{};
  ScanOdometry pre(defaultConfig());
  VoxelMap map{VoxelMapConfig{}};
  for (int k = 0; k < 5; ++k) {       // five independent frames, as hovering
    std::vector<Eigen::Vector3d> p;
    std::vector<double> v;
    for (const auto & s : pre.preprocess(synth::render(wall, T * mountTruth(), cam, &rng), nullptr)) {
      p.push_back(T * s.p);
      v.push_back(s.variance);
    }
    map.insert(p, v);
  }
  const auto src = pre.preprocess(synth::render(wall, T * mountTruth(), cam, &rng), nullptr);
  RegistrationConfig rc;
  rc.judge_wide_normals = wide_judge;
  WallScan w;
  w.res = registerScan(src, map, T, 0.1, nullptr, rc);
  w.d = analyseDegeneracy(w.res.judge_hessian, w.res.judge_weight_sum);
  return w;
}
}  // namespace

// FAILS IF: degeneracy is judged on a matrix that cannot show it (point-to-
// point's sum(w) I would read ~1/3 here), the weak direction is reported in
// the wrong frame, or the threshold sits where a feature-rich scan also
// trips it. A camera facing a wall cannot see motion along it.
TEST(Registration, DegeneracyNamesTheDirectionAWallCannotSee)
{
  const OdometryConfig oc = defaultConfig();
  const WallScan w = wallScan(2.8, false, false);
  std::printf("  wall, noise-free: translation eig %.5f along (%.2f %.2f %.2f)\n",
    w.d.translation_min_eig, w.d.translation_weak_dir.x(), w.d.translation_weak_dir.y(),
    w.d.translation_weak_dir.z());
  EXPECT_LT(w.d.translation_min_eig, 0.1 * oc.degeneracy_translation_min);
  // Along the wall, horizontally or vertically (the floor may be out of
  // view): never across it.
  EXPECT_LT(std::abs(w.d.translation_weak_dir.x()), 0.1);

  const Eigen::Isometry3d T = poseFrom({0.0, 0.0, 1.2}, 0.0, 0.0, 0.0);
  const VoxelMap rmap = mapFromFrame(T, 1, true);
  std::mt19937 rng2(4);
  ScanOdometry pre(defaultConfig());
  const auto rsrc = pre.preprocess(synth::render(rich(), T * mountTruth(), noisyCamera(), &rng2), nullptr);
  const auto rres = registerScan(rsrc, rmap, T, 0.1, nullptr, RegistrationConfig{});
  const Degeneracy rd = analyseDegeneracy(rres.judge_hessian, rres.judge_weight_sum);
  std::printf("  rich world: translation eig %.5f, rotation eig %.4f\n",
    rd.translation_min_eig, rd.rotation_min_eig);
  EXPECT_GT(rd.translation_min_eig, 2.0 * oc.degeneracy_translation_min);
  EXPECT_GT(rd.rotation_min_eig, 2.0 * oc.degeneracy_rotation_min);
}

// FAILS IF: judgeNormal() is not used, or fits as narrow a patch as the
// solve. With the D435's noise, a wall 2.2 m away has narrow-patch normals
// scattered ~10 deg, which reads as several percent of information ALONG the
// wall; only the wide-support normals show it unobservable. This also pins
// the known limitation of the default (narrow) judge, so the README's
// statement of it stays true: if the narrow judge ever starts passing this,
// the default should be revisited.
TEST(Registration, WideNormalsSeeANoisyWallIsDegenerate)
{
  const OdometryConfig oc = defaultConfig();
  const WallScan wide = wallScan(2.8, true, true);
  const WallScan narrow = wallScan(2.8, true, false);
  std::printf("  noisy wall 2.2 m: judged eig %.5f wide, %.5f narrow\n",
    wide.d.translation_min_eig, narrow.d.translation_min_eig);
  EXPECT_LT(wide.d.translation_min_eig, oc.degeneracy_translation_min);
  EXPECT_GT(std::abs(wide.d.translation_weak_dir.y()), 0.9);
  EXPECT_GT(narrow.d.translation_min_eig, oc.degeneracy_translation_min);
}

// FAILS IF: the prior is dropped from the solve, or the covariance ignores
// it. Facing the wall, the along-wall coordinate is unobservable; started
// 10 cm off along the wall with a prior there, the answer must stay where
// the prior says (not wander), and its variance along the wall must be the
// prior's, while across the wall it is far smaller.
TEST(Registration, UnobservableDirectionFollowsThePriorAndSaysSo)
{
  synth::World wall;
  wall.addGround();
  wall.addPlane(Eigen::Vector3d::UnitX(), 4.0);
  const Eigen::Isometry3d T = poseFrom({0.0, 0.0, 1.2}, 0.0, 0.0, 0.0);
  ScanOdometry pre(defaultConfig());
  const auto src = pre.preprocess(synth::render(wall, T * mountTruth(), synth::Camera{}, nullptr), nullptr);
  VoxelMap map{VoxelMapConfig{}};
  {
    std::vector<Eigen::Vector3d> p;
    std::vector<double> v;
    const auto src0 = src;
    for (const auto & s : src0) {p.push_back(T * s.p); v.push_back(s.variance);}
    for (int k = 0; k < 4; ++k) {map.insert(p, v);}
  }
  Eigen::Isometry3d guess = T;
  guess.translation().y() += 0.10;
  guess.translation().x() += 0.05;
  PosePrior prior;
  prior.pose = guess;
  const double sp = 0.02, sr = 0.01;
  prior.information.diagonal() << 1 / (sp * sp), 1 / (sp * sp), 1 / (sp * sp),
    1 / (sr * sr), 1 / (sr * sr), 1 / (sr * sr);
  RegistrationConfig rc;
  rc.degenerate_translation_eig = defaultConfig().degeneracy_translation_min;
  rc.degenerate_rotation_eig = defaultConfig().degeneracy_rotation_min;
  const auto res = registerScan(src, map, guess, 0.1, &prior, rc);
  const Eigen::Vector3d p = res.pose.translation();
  std::printf("  wall with prior: x %.4f (truth 0), y %.4f (prior 0.10), sigma_x %.4f sigma_y %.4f\n",
    p.x(), p.y(), std::sqrt(res.covariance(0, 0)), std::sqrt(res.covariance(1, 1)));
  EXPECT_NEAR(p.x(), 0.0, 0.005);          // observable: pulled back to the truth
  EXPECT_NEAR(p.y(), 0.10, 0.002);         // unobservable: held at the prior
  EXPECT_NEAR(std::sqrt(res.covariance(1, 1)), sp, 0.05 * sp);
  EXPECT_LT(std::sqrt(res.covariance(0, 0)), 0.3 * sp);
}

// ===========================================================================
// The odometry, flying
// ===========================================================================

// FAILS IF: anything in the chain is off enough to drift: a wrong noise
// weighting, map blur, a biased Jacobian, the threshold collapsing. 25 s,
// ~23 m of 6-DOF flight through the rich world with the D435's noise and a
// biased, noisy gyro at 30 Hz. Also checks the covariance is honest: the
// true error lies inside the reported 3-sigma ellipsoid on (almost) every
// frame, and the ellipsoid is not uselessly large.
//
// Measured (host, -O2): final error 0.28 m = 1.22 % of 22.9 m, max 0.96 m,
// rotation max 0.99 deg, 100 % inside 3 sigma, final sigma 1.48 m, TRACKING
// on 72 % of frames (the rest DEGRADED, mostly a direction only far points
// constrain), 17 ms mean / 43 ms max per frame. Bounds have ~2x margin.
TEST(ScanOdometry, DriftOnA25SecondSurveyIsBounded)
{
  FlightSpec s;
  s.traj = survey;
  s.t1 = 25.0;
  s.world = [](double) -> const synth::World & {return rich();};
  ScanOdometry odo(defaultConfig(), survey(0.0));
  const FlightLog log = fly(s, odo);

  double max_err = 0.0, max_rot = 0.0, sum_ms = 0.0, max_ms = 0.0;
  int inside = 0, tracking = 0;
  for (std::size_t i = 0; i < log.rep.size(); ++i) {
    const Eigen::Vector3d e = log.rep[i].pose.translation() - log.truth[i].translation();
    max_err = std::max(max_err, e.norm());
    max_rot = std::max(max_rot, rotationError(log.rep[i].pose, log.truth[i]));
    const Eigen::Matrix3d P = log.rep[i].pose_covariance.topLeftCorner<3, 3>() +
      1e-12 * Eigen::Matrix3d::Identity();
    if (e.dot(P.ldlt().solve(e)) <= 14.16) {++inside;}   // chi^2_3 at 99.7 %
    if (log.rep[i].state == TrackingState::kTracking) {++tracking;}
    if (i > 0) {sum_ms += log.ms[i]; max_ms = std::max(max_ms, log.ms[i]);}
  }
  const auto & last = log.rep.back();
  const double final_err = (last.pose.translation() - log.truth.back().translation()).norm();
  const double sigma_final = std::sqrt(last.pose_covariance.topLeftCorner<3, 3>().trace());
  const double n = static_cast<double>(log.rep.size());
  std::printf("  survey: %.1f m flown, final error %.3f m (%.2f %% of distance), max %.3f m, "
    "max rotation error %.2f deg\n", log.path_length, final_err,
    100.0 * final_err / log.path_length, max_err, max_rot / kDeg);
  std::printf("  covariance: true error inside 3-sigma on %.1f %% of frames; final sigma %.3f m\n",
    100.0 * inside / n, sigma_final);
  std::printf("  tracking on %.1f %% of frames; compute %.1f ms mean, %.1f ms max per "
    "424x240 frame; final map %zu voxels, threshold %.3f m\n",
    100.0 * tracking / n, sum_ms / (n - 1), max_ms, last.map_voxels, last.threshold_m);

  EXPECT_LT(final_err / log.path_length, 0.025);
  EXPECT_LT(max_err, 1.5);
  EXPECT_LT(max_rot, 2.0 * kDeg);
  EXPECT_GT(tracking / n, 0.5);
  EXPECT_GT(inside / n, 0.95);
  EXPECT_LT(sigma_final, 0.15 * log.path_length);    // honest, but not "anywhere"
  EXPECT_LT(sum_ms / (n - 1), 100.0);
}

namespace
{
/// A pitched hover in the middle of the rich world, yawing +-86 deg at
/// 1 Hz: peak rate 9.4 rad/s. At 15 Hz, constant velocity mispredicts the
/// rotation by up to 15 deg per frame at the reversals; the gyro does not.
Eigen::Isometry3d yawShake(double t)
{
  return poseFrom({1.5, -1.8, 1.3}, 0.0, 12 * kDeg, 0.3 + 1.5 * std::sin(2 * M_PI * 1.0 * t));
}

struct ShakeResult { double max_err; double max_rot; int lost; int frames; };

ShakeResult shake(bool use_imu)
{
  FlightSpec s;
  s.traj = yawShake;
  s.t1 = 4.0;
  s.cloud_hz = 15.0;
  s.world = [](double) -> const synth::World & {return rich();};
  OdometryConfig c = defaultConfig();
  c.use_imu_prior = use_imu;
  ScanOdometry odo(c, yawShake(0.0));
  const FlightLog log = fly(s, odo);
  ShakeResult r {0.0, 0.0, 0, static_cast<int>(log.rep.size())};
  for (std::size_t i = 0; i < log.rep.size(); ++i) {
    r.max_err = std::max(r.max_err, (log.rep[i].pose.translation() - log.truth[i].translation()).norm());
    r.max_rot = std::max(r.max_rot, rotationError(log.rep[i].pose, log.truth[i]));
    if (log.rep[i].state == TrackingState::kLost) {++r.lost;}
  }
  return r;
}
}  // namespace

// FAILS IF: the gyro prior is ignored, integrated with the wrong sign, or
// composed in the world frame (the hover is pitched 12 deg, so body yaw and
// world yaw differ). The control run -- same code, use_imu_prior off --
// must FAIL on the same flight, which proves the flight is hard enough that
// the prior is what passes it.
TEST(ScanOdometry, GyroPriorCarriesAFastYaw)
{
  const ShakeResult with = shake(true);
  const ShakeResult without = shake(false);
  std::printf("  yaw shake, gyro prior:    max error %.3f m, %.2f deg, %d/%d frames lost\n",
    with.max_err, with.max_rot / kDeg, with.lost, with.frames);
  std::printf("  yaw shake, constant rate: max error %.3f m, %.2f deg, %d/%d frames lost\n",
    without.max_err, without.max_rot / kDeg, without.lost, without.frames);
  // Measured: 0.15 m / 2.2 deg / 0 lost with the gyro; 0.98 m / 179 deg /
  // 10 of 61 frames lost without.
  EXPECT_LT(with.max_err, 0.3);
  EXPECT_LT(with.max_rot, 4.0 * kDeg);
  EXPECT_EQ(with.lost, 0);
  // The control: the same flight without the gyro loses its heading.
  EXPECT_GT(without.max_rot, 20.0 * kDeg);
  EXPECT_LT(with.max_rot, 0.2 * without.max_rot);
}

namespace
{
/// A long wall at x = 4 (3.4 m in front of the lens) with nothing else but
/// the floor for y < 3; beyond
/// y = 3, a cluster of irregular objects in front of the wall.
const synth::World & corridor()
{
  static const synth::World w = [] {
      synth::World c;
      c.addGround();
      c.addPlane(Eigen::Vector3d::UnitX(), 4.0);
      c.addBox({3.0, 6.0, 0.5}, {0.4, 0.5, 0.5}, 0.5);
      c.addBox({2.2, 7.5, 0.9}, {0.3, 0.6, 0.9}, -0.4);
      c.addCylinder(3.2, 8.2, 0.3, 0.0, 2.0);
      c.addBox({2.8, 9.0, 0.3}, {0.6, 0.4, 0.3}, 1.0, 0.2);
      c.addSphere({2.5, 6.8, 1.6}, 0.4);
      return c;
    }();
  return w;
}

/// Facing the wall (+x), sliding along it (+y) at 0.5 m/s with a gentle
/// speed change the constant-velocity prior cannot know about.
Eigen::Isometry3d slide(double t)
{
  return poseFrom({0.5, -2.0 + 0.5 * t + 0.15 * std::sin(0.8 * t), 1.2}, 0.0, 0.0, 0.0);
}
}  // namespace

// FAILS IF: the blind stretch is reported TRACKING, the covariance does not
// grow in the blind direction, or the odometry never comes back to TRACKING
// once the features appear. Also: while degraded, the true along-wall error
// must lie inside the reported 3 sigma -- "not a confident wrong answer".
//
// At 3.4 m the D435's depth noise (~8 cm) leaves the wall barely matchable:
// the DEGRADED verdict mostly says "only 3 % of points matched", not
// "translation unobservable", and the along-wall error reaches ~7 m in
// 15 s, which the covariance covers. Measured: 188/188 blind frames
// DEGRADED and inside 3 sigma, var_y growing 80x faster than with features,
// 121/121 late frames TRACKING. A CLOSER wall (2.2 m) is the known failure
// of the default judge: see WideNormalsSeeANoisyWallIsDegenerate and the
// README.
TEST(ScanOdometry, AWallIsDegradedAndFeaturesRestoreTracking)
{
  FlightSpec s;
  s.traj = slide;
  s.t1 = 22.0;
  s.world = [](double) -> const synth::World & {return corridor();};
  ScanOdometry odo(defaultConfig(), slide(0.0));
  const FlightLog log = fly(s, odo);

  // The blind stretch: the features' near edge (y ~ 5.5) is still beyond the
  // depth gate's view until y ~ 2; take frames with y < 1 (t < ~6 s).
  int blind = 0, blind_degraded = 0, blind_inside = 0;
  double var_y0 = -1, var_y1 = 0, t_blind0 = 0, t_blind1 = 0;
  for (std::size_t i = 1; i < log.rep.size(); ++i) {
    if (log.truth[i].translation().y() > 1.0) {break;}
    const auto & r = log.rep[i];
    ++blind;
    if (r.state == TrackingState::kDegraded) {++blind_degraded;}
    const double ey = r.pose.translation().y() - log.truth[i].translation().y();
    if (std::abs(ey) <= 3.0 * std::sqrt(r.pose_covariance(1, 1))) {++blind_inside;}
    if (var_y0 < 0) {var_y0 = r.pose_covariance(1, 1); t_blind0 = log.t[i];}
    var_y1 = r.pose_covariance(1, 1);
    t_blind1 = log.t[i];
  }
  // Tracking again once the features are well in view (last 4 s).
  int late = 0, late_tracking = 0;
  double var_late0 = -1, var_late1 = 0, t_late0 = 0, t_late1 = 0;
  for (std::size_t i = 0; i < log.rep.size(); ++i) {
    if (log.t[i] < s.t1 - 4.0) {continue;}
    ++late;
    if (log.rep[i].state == TrackingState::kTracking) {++late_tracking;}
    if (var_late0 < 0) {var_late0 = log.rep[i].pose_covariance(1, 1); t_late0 = log.t[i];}
    var_late1 = log.rep[i].pose_covariance(1, 1);
    t_late1 = log.t[i];
  }
  const double growth_blind = (var_y1 - var_y0) / (t_blind1 - t_blind0);
  const double growth_late = (var_late1 - var_late0) / (t_late1 - t_late0);
  const auto & mid = log.rep[blind / 2];
  std::printf("  wall: %d/%d blind frames DEGRADED (e.g. \"%s\")\n", blind_degraded, blind,
    mid.reason.c_str());
  std::printf("  wall: along-wall error inside 3 sigma on %d/%d blind frames; final error %.3f m\n",
    blind_inside, blind,
    (log.rep.back().pose.translation() - log.truth.back().translation()).norm());
  std::printf("  wall: var_y growth %.2e m^2/s blind vs %.2e m^2/s with features; "
    "%d/%d late frames TRACKING\n", growth_blind, growth_late, late_tracking, late);

  EXPECT_GT(blind, 50);
  EXPECT_GE(blind_degraded, blind - 2);
  EXPECT_GE(blind_inside, static_cast<int>(0.95 * blind));
  EXPECT_GE(late_tracking, static_cast<int>(0.9 * late));
  EXPECT_GT(growth_blind, 10.0 * std::max(growth_late, 1e-9));
}

// FAILS IF: an empty scene is reported as anything but LOST, the reason is
// missing, the held pose's covariance does not grow, or the odometry cannot
// come back. Mid-hover, everything leaves the depth gate for 1.5 s (the
// lens covered, or nothing within 10 m), then returns.
TEST(ScanOdometry, NothingInRangeIsLostAndRecovers)
{
  static const synth::World empty;
  auto hover = [](double t) {
      return poseFrom({1.5, -1.8, 1.3 + 0.05 * std::sin(t)}, 0.0, 0.0, 0.3 + 0.1 * std::sin(0.5 * t));
    };
  FlightSpec s;
  s.traj = hover;
  s.t1 = 6.0;
  s.world = [](double t) -> const synth::World & {
      return (t > 2.0 && t < 3.5) ? empty : rich();
    };
  ScanOdometry odo(defaultConfig(), hover(0.0));
  const FlightLog log = fly(s, odo);
  int lost = 0, dark = 0;
  double sig_before = 0, sig_end_dark = 0;
  std::string reason;
  for (std::size_t i = 0; i < log.rep.size(); ++i) {
    const auto & r = log.rep[i];
    const double sig = std::sqrt(r.pose_covariance.topLeftCorner<3, 3>().trace());
    if (log.t[i] <= 2.0) {sig_before = sig;}
    if (log.t[i] > 2.0 && log.t[i] < 3.5) {
      ++dark;
      if (r.state == TrackingState::kLost) {++lost; reason = r.reason; sig_end_dark = sig;}
    }
  }
  const auto & last = log.rep.back();
  std::printf("  blackout: %d/%d dark frames LOST (\"%s\"), sigma %.3f m -> %.3f m; "
    "after: %s, error %.3f m\n", lost, dark, reason.c_str(), sig_before, sig_end_dark,
    toString(last.state), (last.pose.translation() - log.truth.back().translation()).norm());
  EXPECT_EQ(lost, dark);
  EXPECT_NE(reason.find("points in range"), std::string::npos);
  EXPECT_GT(sig_end_dark, 1.0);          // 1 m/s * 1.5 s: it could be anywhere nearby
  // Measured: sigma 0.08 m before, 2.54 m at the end of the dark, error
  // 0.14 m after re-registering from the held pose.
  EXPECT_LT(sig_before, 0.2);
  EXPECT_EQ(last.state, TrackingState::kTracking) << last.reason;
  EXPECT_LT((last.pose.translation() - log.truth.back().translation()).norm(), 0.3);
}

// FAILS IF: a camera that stops publishing is still reported as tracking.
TEST(ScanOdometry, WatchdogCallsASilentCameraLost)
{
  ScanOdometry odo(defaultConfig(), poseFrom({1.5, -1.8, 1.3}, 0, 0, 0.3));
  EXPECT_EQ(odo.watchdog(0.0).state, TrackingState::kLost);
  std::mt19937 rng(1);
  const auto cloud = synth::render(
    rich(), poseFrom({1.5, -1.8, 1.3}, 0, 0, 0.3) * mountTruth(), noisyCamera(), &rng);
  ASSERT_EQ(odo.processScan(10.0, cloud).state, TrackingState::kTracking);
  EXPECT_EQ(odo.watchdog(10.2).state, TrackingState::kTracking);
  const auto w = odo.watchdog(11.0);
  EXPECT_EQ(w.state, TrackingState::kLost);
  EXPECT_NE(w.reason.find("no point cloud"), std::string::npos);
}

// FAILS IF: a covariance-carrying odometry keeps its accumulated state
// across a clock reset (the simulator's RESET), so the new run starts with
// the old run's map and uncertainty.
TEST(ScanOdometry, ClockGoingBackwardsResets)
{
  const auto P = poseFrom({1.5, -1.8, 1.3}, 0, 0, 0.3);
  ScanOdometry odo(defaultConfig(), P);
  std::mt19937 rng(1);
  const auto cloud = synth::render(rich(), P * mountTruth(), noisyCamera(), &rng);
  odo.processScan(50.0, cloud);
  odo.processScan(50.1, cloud);
  const auto r = odo.processScan(0.5, cloud);
  EXPECT_EQ(r.reason, "map initialised from the first scan");
  EXPECT_LT(r.pose_covariance.norm(), 1e-12);
}
