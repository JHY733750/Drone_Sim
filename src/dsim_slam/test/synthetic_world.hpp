// A test-only synthetic world and depth camera, so the SLAM tests depend on
// nothing but this file and the library under test.
//
// The world is analytic primitives -- an infinite ground plane, oriented
// boxes, vertical cylinders, spheres, infinite walls -- and the camera casts
// one ray per pixel exactly like a pinhole depth camera: 424 x 240, 87 x 58
// degree field of view, depth gate 0.28 - 10 m, and stereo depth noise
// sigma_z = sqrt(floor^2 + (k z^2)^2) applied ALONG the depth axis (the
// point moves along its pixel's ray, as a stereo disparity error does).
// Points come out in the OPTICAL frame (x right, y down, z forward), the way
// realsense-ros publishes them.
//
// The camera rays are written with an unnormalised direction whose optical
// z component is 1, so the ray parameter at the hit IS the depth z. Every
// intersection routine below is written for non-unit directions for that
// reason.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace synth
{

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Plane { Eigen::Vector3d n; double d; };                 // n.x = d
struct Box { Eigen::Vector3d c; Eigen::Vector3d half; Eigen::Matrix3d R; };   // R: box -> world
struct Cylinder { double x, y, r, z0, z1; };                    // vertical
struct Sphere { Eigen::Vector3d c; double r; };

class World
{
public:
  void addPlane(const Eigen::Vector3d & n, double d) {planes_.push_back({n.normalized(), d});}
  void addGround() {addPlane(Eigen::Vector3d::UnitZ(), 0.0);}
  void addBox(const Eigen::Vector3d & c, const Eigen::Vector3d & half, double yaw,
    double tilt = 0.0)
  {
    const Eigen::Matrix3d R = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX())).toRotationMatrix();
    boxes_.push_back({c, half, R});
  }
  void addCylinder(double x, double y, double r, double z0, double z1)
  {
    cylinders_.push_back({x, y, r, z0, z1});
  }
  void addSphere(const Eigen::Vector3d & c, double r) {spheres_.push_back({c, r});}

  /// Smallest positive ray parameter t at which o + t d hits anything, or inf.
  double cast(const Eigen::Vector3d & o, const Eigen::Vector3d & d) const
  {
    double best = kInf;
    for (const auto & p : planes_) {
      const double den = p.n.dot(d);
      if (std::abs(den) < 1e-12) {continue;}
      const double t = (p.d - p.n.dot(o)) / den;
      if (t > 1e-9 && t < best) {best = t;}
    }
    for (const auto & b : boxes_) {
      // Slab test in the box frame.
      const Eigen::Vector3d ol = b.R.transpose() * (o - b.c);
      const Eigen::Vector3d dl = b.R.transpose() * d;
      double t0 = -kInf, t1 = kInf;
      bool miss = false;
      for (int i = 0; i < 3; ++i) {
        if (std::abs(dl(i)) < 1e-12) {
          if (std::abs(ol(i)) > b.half(i)) {miss = true; break;}
          continue;
        }
        double ta = (-b.half(i) - ol(i)) / dl(i);
        double tb = (b.half(i) - ol(i)) / dl(i);
        if (ta > tb) {std::swap(ta, tb);}
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        if (t0 > t1) {miss = true; break;}
      }
      if (!miss && t0 > 1e-9 && t0 < best) {best = t0;}
    }
    for (const auto & c : cylinders_) {
      const double ox = o.x() - c.x, oy = o.y() - c.y;
      const double a = d.x() * d.x() + d.y() * d.y();
      if (a > 1e-12) {
        const double bq = 2.0 * (ox * d.x() + oy * d.y());
        const double cq = ox * ox + oy * oy - c.r * c.r;
        const double disc = bq * bq - 4.0 * a * cq;
        if (disc >= 0.0) {
          const double t = (-bq - std::sqrt(disc)) / (2.0 * a);
          const double z = o.z() + t * d.z();
          if (t > 1e-9 && z >= c.z0 && z <= c.z1 && t < best) {best = t;}
        }
      }
      if (std::abs(d.z()) > 1e-12) {   // top cap (seen from above)
        const double t = (c.z1 - o.z()) / d.z();
        const double x = ox + t * d.x(), y = oy + t * d.y();
        if (t > 1e-9 && x * x + y * y <= c.r * c.r && t < best && o.z() > c.z1) {best = t;}
      }
    }
    for (const auto & s : spheres_) {
      const Eigen::Vector3d oc = o - s.c;
      const double a = d.squaredNorm();
      const double bq = 2.0 * oc.dot(d);
      const double cq = oc.squaredNorm() - s.r * s.r;
      const double disc = bq * bq - 4.0 * a * cq;
      if (disc < 0.0) {continue;}
      const double t = (-bq - std::sqrt(disc)) / (2.0 * a);
      if (t > 1e-9 && t < best) {best = t;}
    }
    return best;
  }

private:
  std::vector<Plane> planes_;
  std::vector<Box> boxes_;
  std::vector<Cylinder> cylinders_;
  std::vector<Sphere> spheres_;
};

struct Camera
{
  int width {424};
  int height {240};
  double hfov_rad {87.0 * M_PI / 180.0};
  double vfov_rad {58.0 * M_PI / 180.0};
  double min_depth {0.28};
  double max_depth {10.0};
  double noise_floor {0.0};
  double noise_k {0.0};     // sigma_z = sqrt(floor^2 + (k z^2)^2)
};

/// The camera's optical frame on the body, built from what the axes MEAN
/// (optical x = right = -body y, optical y = down = -body z, optical z =
/// forward = body x), independently of the library's opticalToBodyRotation().
inline Eigen::Matrix3d opticalAxesInBody()
{
  Eigen::Matrix3d R;
  R.col(0) = -Eigen::Vector3d::UnitY();
  R.col(1) = -Eigen::Vector3d::UnitZ();
  R.col(2) = Eigen::Vector3d::UnitX();
  return R;
}

/// Render one depth frame from T_world_optical. Points in the optical frame.
inline std::vector<Eigen::Vector3d> render(
  const World & w, const Eigen::Isometry3d & T_world_optical, const Camera & cam,
  std::mt19937 * rng)
{
  const double fx = 0.5 * cam.width / std::tan(0.5 * cam.hfov_rad);
  const double fy = 0.5 * cam.height / std::tan(0.5 * cam.vfov_rad);
  const double cx = 0.5 * (cam.width - 1), cy = 0.5 * (cam.height - 1);
  const Eigen::Matrix3d R = T_world_optical.linear();
  const Eigen::Vector3d o = T_world_optical.translation();
  std::normal_distribution<double> unit(0.0, 1.0);
  std::vector<Eigen::Vector3d> out;
  out.reserve(static_cast<std::size_t>(cam.width) * cam.height);
  for (int v = 0; v < cam.height; ++v) {
    for (int u = 0; u < cam.width; ++u) {
      const Eigen::Vector3d ray((u - cx) / fx, (v - cy) / fy, 1.0);
      double z = w.cast(o, R * ray);
      if (!(z < cam.max_depth * 1.5)) {continue;}
      if (rng != nullptr && (cam.noise_floor > 0.0 || cam.noise_k > 0.0)) {
        const double q = cam.noise_k * z * z;
        z += std::sqrt(cam.noise_floor * cam.noise_floor + q * q) * unit(*rng);
      }
      if (z < cam.min_depth || z > cam.max_depth) {continue;}
      out.push_back(ray * z);
    }
  }
  return out;
}

/// A trajectory: body pose in the world at time t.
using Trajectory = std::function<Eigen::Isometry3d(double)>;

/// Body angular rate at t, from the trajectory, by central difference of
/// the attitude: what a perfect strapdown gyro reads.
inline Eigen::Vector3d bodyRate(const Trajectory & traj, double t, double h = 1e-4)
{
  const Eigen::Matrix3d Ra = traj(t - h).linear();
  const Eigen::Matrix3d Rb = traj(t + h).linear();
  const Eigen::AngleAxisd aa(Ra.transpose() * Rb);
  return aa.angle() * aa.axis() / (2.0 * h);
}

/// The feature-rich test world: ground plus fourteen irregular objects --
/// boxes at odd yaws and tilts, cylinders, spheres, a sloped slab -- spread
/// over a 12 m disc so a camera anywhere on a 2.5 m circle sees several.
inline World richWorld()
{
  World w;
  w.addGround();
  w.addBox({4.0, 0.5, 0.6}, {0.5, 0.8, 0.6}, 0.4);
  w.addBox({1.0, 4.2, 0.9}, {0.6, 0.4, 0.9}, -0.7);
  w.addBox({-3.8, 2.0, 0.4}, {0.9, 0.5, 0.4}, 1.1, 0.15);
  w.addBox({-1.5, -4.0, 1.1}, {0.4, 0.4, 1.1}, 0.25);
  w.addBox({3.0, -3.2, 0.5}, {1.0, 0.3, 0.5}, 2.0);
  w.addBox({0.2, 0.3, 0.3}, {0.3, 0.6, 0.3}, 0.9, -0.2);   // centre: seen from everywhere
  w.addBox({-5.5, -1.0, 1.4}, {0.3, 1.5, 1.4}, -0.2);
  w.addBox({5.5, 3.5, 1.0}, {0.4, 1.2, 1.0}, 0.8);
  w.addBox({2.2, 2.4, 0.15}, {1.2, 0.8, 0.15}, 0.3, 0.25);  // a sloped slab
  w.addCylinder(-2.2, 3.6, 0.35, 0.0, 2.2);
  w.addCylinder(3.6, -0.9, 0.25, 0.0, 1.6);
  w.addCylinder(-3.2, -2.6, 0.5, 0.0, 1.0);
  w.addCylinder(0.8, -1.2, 0.2, 0.0, 2.5);
  w.addSphere({-1.2, 1.0, 0.7}, 0.5);
  w.addSphere({2.0, -1.0, 1.9}, 0.35);
  return w;
}

}  // namespace synth
