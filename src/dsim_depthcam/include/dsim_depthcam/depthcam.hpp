#pragma once
// A simulated Intel RealSense D435 depth stream: camera model, the world it
// looks at, and the renderer. No ROS, so it is unit tested and mutation-checked
// on the host; depthcam_node.cpp only moves its output into messages.
//
// WHAT IS MODELLED
//
//   * A pinhole with the D435's depth field of view (87 x 58 deg nominal) at
//     a resolution it really streams (424 x 240), and the D435's minimum and
//     a usable maximum range.
//   * Stereo depth noise, which grows with the SQUARE of depth:
//
//         sigma_z = z^2 * subpixel / (f_px * baseline)
//
//     A stereo camera measures disparity, d = f*b/z, to about `subpixel`
//     pixels; differentiating, dz = z^2/(f*b) * dd. f_px is fx -- the focal
//     length along the BASELINE, which on a D435 is horizontal, because that
//     is the axis disparity is measured along. The noise moves a point along
//     its own pixel ray (z changes, the pixel does not), which is what a
//     disparity error does.
//   * Random dropout, a fraction of otherwise-valid pixels returning nothing,
//     standing in for the texture-less and specular patches a real stereo
//     matcher gives up on. Off by default.
//
// WHAT IS NOT
//
//   No IR projector pattern, no edge fattening at depth discontinuities, no
//   multipath, no rolling exposure, no lens distortion (the D435 depth stream
//   is rectified and reports zero distortion, so that one is free). Surfaces
//   reflect perfectly whatever their material. A planner tuned against this
//   sees cleaner edges than the real camera gives it.
//
// DEPTH IS z, NOT RANGE. A RealSense reports the distance along the optical
// axis, and so does this -- the minimum and maximum range clip z too. Rays are
// cast with direction (x_n, y_n, 1) in the optical frame so the ray parameter
// returned by geometry.hpp is z directly (see the note at the top of that
// file).
//
// RANDOMNESS is drawn from a generator seeded per FRAME, by the caller (the
// node derives it from the configured seed and the frame's stamp). The same
// pose at the same stamp therefore renders the same frame bit for bit,
// whatever order callbacks ran in -- a stronger promise than one generator
// shared across a node's timers can make.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "dsim_depthcam/geometry.hpp"
#include "dsim_eval/obstacle.hpp"

namespace dsim_depthcam
{

constexpr double kPi = 3.14159265358979323846;
inline double deg2rad(double d) {return d * kPi / 180.0;}

struct CameraConfig
{
  int width {424};
  int height {240};
  double hfov_rad {deg2rad(87.0)};
  double vfov_rad {deg2rad(58.0)};
  /// D435 minimum Z at this resolution class; nearer than this the stereo
  /// matcher has no disparity range left and returns nothing.
  double min_range_m {0.28};
  double max_range_m {10.0};
  bool noise {true};
  double baseline_m {0.050};
  /// Subpixel disparity accuracy, pixels RMS.
  double subpixel {0.08};
  /// Fraction of valid pixels dropped at random, [0, 1).
  double dropout {0.0};
  /// Depth unit of the 16-bit image. The D435's default depth scale.
  double depth_unit_m {0.001};

  void validate() const
  {
    if (width < 1 || height < 1) {throw std::invalid_argument("camera: resolution must be positive");}
    for (const double f : {hfov_rad, vfov_rad}) {
      if (!(f > 0.0 && f < kPi)) {throw std::invalid_argument("camera: FOV must be in (0, 180) deg");}
    }
    if (!(min_range_m > 0.0 && max_range_m > min_range_m)) {
      throw std::invalid_argument("camera: need 0 < min_range_m < max_range_m");
    }
    if (!(baseline_m > 0.0) || !(subpixel >= 0.0)) {
      throw std::invalid_argument("camera: baseline must be > 0 and subpixel >= 0");
    }
    if (!(dropout >= 0.0 && dropout < 1.0)) {throw std::invalid_argument("camera: dropout must be in [0, 1)");}
    if (!(depth_unit_m > 0.0) || max_range_m / depth_unit_m > 65535.0) {
      throw std::invalid_argument("camera: max range does not fit a 16-bit image at this depth unit");
    }
  }
};

/// Pinhole intrinsics, in the ROS / OpenCV convention: integer pixel
/// coordinates are pixel CENTRES, the image spans [-0.5, W-0.5], so a centred
/// principal point is (W-1)/2 and the FOV edges are W/2 pixels either side
/// of it.
struct Intrinsics
{
  double fx, fy, cx, cy;
};

/// fx and fy come from their own FOVs, so BOTH stated FOVs are exact. 87 x 58
/// deg at 424 x 240 is not quite square-pixel (fx 223.4, fy 216.5): the
/// nominal D435 FOV is quoted for its native 1280 x 800 sensor, and the
/// 16:9 modes crop it. Matching the numbers the data sheet gives was judged
/// more useful than square pixels; CameraInfo carries both, so nothing
/// downstream assumes fx == fy.
inline Intrinsics intrinsics(const CameraConfig & c)
{
  Intrinsics k;
  k.fx = 0.5 * c.width / std::tan(0.5 * c.hfov_rad);
  k.fy = 0.5 * c.height / std::tan(0.5 * c.vfov_rad);
  k.cx = 0.5 * (c.width - 1);
  k.cy = 0.5 * (c.height - 1);
  return k;
}

/// Stereo depth noise, one sigma, metres, at depth z. See the file header.
inline double depthSigma(double z, const CameraConfig & c, const Intrinsics & k)
{
  return z * z * c.subpixel / (k.fx * c.baseline_m);
}

// ---- frames ----------------------------------------------------------------
//
//   base_link                  body: x forward, y left, z up (REP 103)
//     -> camera_link           the mount: offset, then pitched about base y;
//                              x forward, y left, z up, like the body
//       -> camera_depth_frame  identity, as realsense-ros publishes it (the
//                              depth imager IS the D435's reference)
//         -> camera_depth_optical_frame
//                              z forward, x right, y down (REP 103 optical)

struct Mount
{
  /// camera_link origin in base_link, metres.
  Eigen::Vector3d offset {0.10, 0.0, 0.0};
  /// Rotation about base_link +y, right-handed: POSITIVE TILTS THE CAMERA
  /// DOWN (it turns +x toward -z).
  double pitch_rad {0.0};
};

inline Eigen::Isometry3d baseToCameraLink(const Mount & m)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = Eigen::AngleAxisd(m.pitch_rad, Eigen::Vector3d::UnitY()).toRotationMatrix();
  T.translation() = m.offset;
  return T;
}

/// camera_link -> optical. Columns are the optical axes expressed in
/// camera_link: x_opt = -y_link (right), y_opt = -z_link (down),
/// z_opt = +x_link (forward). The standard (-pi/2, 0, -pi/2) roll-pitch-yaw.
inline Eigen::Matrix3d linkToOpticalRotation()
{
  Eigen::Matrix3d R;
  R << 0.0, 0.0, 1.0,
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0;
  return R;
}

/// World pose of the optical frame, from the body's world pose.
inline Eigen::Isometry3d worldToOptical(const Eigen::Isometry3d & T_world_base, const Mount & m)
{
  Eigen::Isometry3d T_link_opt = Eigen::Isometry3d::Identity();
  T_link_opt.linear() = linkToOpticalRotation();
  return T_world_base * baseToCameraLink(m) * T_link_opt;
}

// ---- the world -------------------------------------------------------------

/// Everything the camera can see. Obstacles are the REFEREE's own type and
/// are moved by the referee's own code (dsim_eval::Obstacle::centreAt and
/// activeAt) -- there is exactly one implementation of where an obstacle is,
/// so the camera and the referee cannot disagree about it.
struct World
{
  bool ground {true};
  double ground_z {0.0};
  std::vector<Primitive> landmarks;
  std::vector<dsim_eval::Obstacle> obstacles;
  /// Scenario clock origin in sim seconds, `scenario.start_s` in the
  /// referee's obstacles file: scenario t = sim t - start_s, exactly as
  /// dsim_eval's eval_node computes it.
  double scenario_start_s {0.0};
  /// The referee's scenario.duration_s and scenario.loop: see
  /// dsim_eval::scenarioClock. The camera must loop exactly when the referee
  /// does, or it would draw obstacles the referee had already retired.
  double scenario_duration_s {0.0};
  bool scenario_loop {false};
};

/// The primitives present at a given SIM time. An obstacle outside its active
/// window is absent -- not parked at its endpoint -- for the reason in
/// dsim_eval/obstacle.hpp: the planner solved a problem without it.
inline std::vector<Primitive> sceneAt(const World & w, double sim_time_s)
{
  std::vector<Primitive> out;
  out.reserve(w.landmarks.size() + w.obstacles.size() + 1);
  if (w.ground) {out.push_back(makePlane(w.ground_z));}
  out.insert(out.end(), w.landmarks.begin(), w.landmarks.end());
  const double t = dsim_eval::scenarioClock(
    sim_time_s, w.scenario_start_s, w.scenario_duration_s, w.scenario_loop);
  for (const auto & o : w.obstacles) {
    if (!o.activeAt(t)) {continue;}
    const auto c = o.centreAt(t);
    const Eigen::Vector3d centre(c[0], c[1], c[2]);
    if (o.type == "column") {
      // A disc at every altitude: no ends. The z it carries is where the
      // viewer draws it and never enters the geometry.
      out.push_back(makeCylinder(centre, o.radius, kNoHit, o.name));
    } else {
      out.push_back(makeSphere(centre, o.radius, o.name));
    }
  }
  return out;
}

/// One obstacle from the referee's flat parameter form
/// (`obstacles.<name>.{type,radius,control_points}`), with the referee's
/// validation. The PARSING is repeated from dsim_eval/src/eval_node.cpp,
/// where it is inline in the node and not reachable from here; the GEOMETRY
/// is not repeated. It refuses one thing the referee does not: an unknown
/// type, which the referee would silently score as a sphere.
inline dsim_eval::Obstacle obstacleFromFlat(
  const std::string & name, const std::string & type, double radius,
  const std::vector<double> & flat)
{
  if (type != "sphere" && type != "column") {
    throw std::invalid_argument("obstacle " + name + ": unknown type '" + type + "'");
  }
  if (!(radius > 0.0)) {throw std::invalid_argument("obstacle " + name + ": radius must be > 0");}
  if (flat.size() < 8 || flat.size() % 4 != 0) {
    throw std::invalid_argument(
            "obstacle " + name + " has " + std::to_string(flat.size()) +
            " control-point numbers; need a multiple of 4, at least 8 (x,y,z,t per point)");
  }
  dsim_eval::Obstacle o;
  o.name = name;
  o.type = type;
  o.radius = radius;
  for (size_t i = 0; i < flat.size(); i += 4) {
    o.control_points.push_back({flat[i], flat[i + 1], flat[i + 2], flat[i + 3]});
  }
  if (o.tEnd() < o.tStart()) {
    throw std::invalid_argument("obstacle " + name + " has control point times running backwards");
  }
  return o;
}

// ---- rendering -------------------------------------------------------------

struct Frame
{
  int width {0};
  int height {0};
  /// Measured depth per pixel, row-major, metres; 0 = no return. Unquantised:
  /// this is what the cloud is built from.
  std::vector<double> depth_m;
  /// The same, in whole depth units (mm by default), 0 = no return: the
  /// 16UC1 image.
  std::vector<std::uint16_t> depth_raw;
  /// One point per valid pixel, optical frame, in row-major pixel order.
  std::vector<Eigen::Vector3d> points;
  /// The pixel index (v * width + u) each point came from.
  std::vector<int> pixel;
};

/// A per-frame seed from a base seed and the frame's stamp, so a frame is a
/// pure function of (scene, pose, stamp, seed).
inline std::uint64_t frameSeed(std::uint64_t base_seed, std::int64_t stamp_ns)
{
  std::seed_seq seq {
    static_cast<std::uint32_t>(base_seed), static_cast<std::uint32_t>(base_seed >> 32),
    static_cast<std::uint32_t>(stamp_ns), static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(stamp_ns) >> 32)};
  std::uint32_t out[2];
  seq.generate(out, out + 2);
  return (static_cast<std::uint64_t>(out[0]) << 32) | out[1];
}

/// Render one depth frame of `scene` from an optical frame at T_world_optical.
///
/// Per pixel: the nearest surface along the pixel's ray, then noise, then
/// dropout, then the range clip on the MEASURED depth -- so an object nearer
/// than the minimum range leaves a hole, exactly as the real camera does, and
/// does not reveal whatever is behind it.
inline Frame render(
  const std::vector<Primitive> & scene, const Eigen::Isometry3d & T_world_optical,
  const CameraConfig & cfg, std::uint64_t seed)
{
  const Intrinsics k = intrinsics(cfg);
  Frame f;
  f.width = cfg.width;
  f.height = cfg.height;
  const size_t n = static_cast<size_t>(cfg.width) * static_cast<size_t>(cfg.height);
  f.depth_m.assign(n, 0.0);
  f.depth_raw.assign(n, 0);
  f.points.reserve(n);
  f.pixel.reserve(n);

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  const Eigen::Matrix3d R = T_world_optical.linear();
  const Eigen::Vector3d o = T_world_optical.translation();

  for (int v = 0; v < cfg.height; ++v) {
    const double yn = (v - k.cy) / k.fy;
    for (int u = 0; u < cfg.width; ++u) {
      const double xn = (u - k.cx) / k.fx;
      const Eigen::Vector3d ray_opt(xn, yn, 1.0);
      const Eigen::Vector3d d = R * ray_opt;

      double z = kNoHit;
      for (const auto & p : scene) {
        z = std::min(z, intersect(p, o, d));
      }
      if (!std::isfinite(z)) {continue;}

      // A fixed number of draws per HIT pixel, in a fixed order: one
      // Gaussian if noise is on, then one uniform if dropout is on. So the
      // frame is reproducible from its seed, but changing either switch
      // reshuffles every later pixel's noise -- not a promise worth more code.
      if (cfg.noise) {z += depthSigma(z, cfg, k) * gauss(rng);}
      if (cfg.dropout > 0.0 && unit(rng) < cfg.dropout) {continue;}
      if (z < cfg.min_range_m || z > cfg.max_range_m) {continue;}

      const int idx = v * cfg.width + u;
      f.depth_m[idx] = z;
      const double raw = std::round(z / cfg.depth_unit_m);
      f.depth_raw[idx] = static_cast<std::uint16_t>(std::min(65535.0, std::max(1.0, raw)));
      f.points.push_back(ray_opt * z);
      f.pixel.push_back(idx);
    }
  }
  return f;
}

// ---- the viewer's preview ---------------------------------------------------
//
// NOT part of the camera's contract: realsense-ros publishes nothing like it,
// and nothing that would run on the real vehicle may read it. It exists so a
// person can watch the cloud in the browser, which cannot take 46,000 points
// thirty times a second over a tailnet. Built here, from the very points the
// cloud was built from, so the preview cannot show a different cloud.

/// A point of the preview: where it is in the WORLD, and its depth (optical z,
/// the camera's own measure) for colouring.
struct PreviewPoint
{
  Eigen::Vector3d world;
  double depth;
};

/// At most `max_points` of a frame's points, taken at an even stride through
/// the row-major pixel order so the thinning is uniform over the image (a
/// random pick would shimmer frame to frame; a crop would hide an edge), and
/// moved into the world frame with the pose the frame was rendered from.
inline std::vector<PreviewPoint> previewPoints(
  const Frame & f, const Eigen::Isometry3d & T_world_optical, std::size_t max_points)
{
  std::vector<PreviewPoint> out;
  if (max_points == 0 || f.points.empty()) {return out;}
  const std::size_t stride = (f.points.size() + max_points - 1) / max_points;
  out.reserve(f.points.size() / stride + 1);
  for (std::size_t i = 0; i < f.points.size(); i += stride) {
    out.push_back({T_world_optical * f.points[i], f.points[i].z()});
  }
  return out;
}

/// The camera's field of view as a pyramid: [apex, then the four far corners
/// at `range_m` of optical depth, clockwise from top-left as seen by the
/// camera]. In the world frame. From the same intrinsics the renderer casts
/// its rays with, so the frustum drawn is the one that produced the cloud.
inline std::array<Eigen::Vector3d, 5> frustumCorners(
  const CameraConfig & c, const Eigen::Isometry3d & T_world_optical, double range_m)
{
  const auto k = intrinsics(c);
  const double u[4] = {-0.5, c.width - 0.5, c.width - 0.5, -0.5};
  const double v[4] = {-0.5, -0.5, c.height - 0.5, c.height - 0.5};
  std::array<Eigen::Vector3d, 5> out;
  out[0] = T_world_optical.translation();
  for (int i = 0; i < 4; ++i) {
    const Eigen::Vector3d ray((u[i] - k.cx) / k.fx, (v[i] - k.cy) / k.fy, 1.0);
    out[i + 1] = T_world_optical * (ray * range_m);
  }
  return out;
}

}  // namespace dsim_depthcam
