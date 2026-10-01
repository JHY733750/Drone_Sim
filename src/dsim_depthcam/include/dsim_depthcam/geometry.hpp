#pragma once
// Ray casting against analytic primitives, with no ROS in it so it can be unit
// tested and mutation-checked on the host.
//
// Four shapes, because that is everything the simulator's world is made of:
//
//   plane      horizontal, z = height. The ground. Two-sided: seen from below
//              too, which only matters for a camera that is in the floor.
//   sphere     the referee's 3D obstacles, and a landmark type.
//   cylinder   vertical axis, centre at MID-height, finite or infinite. A
//              referee "column" is the infinite case: a disc at every altitude
//              (see dsim_eval/obstacle.hpp for why a 2D obstacle must not be
//              given a top), and a landmark cylinder is the finite one.
//   box        axis-aligned in z, rotated by `yaw` about the vertical through
//              its centre (right-handed: positive yaw turns local +x toward
//              world +y).
//
// THE PARAMETER CONVENTION is the one thing worth reading before using this.
// Every intersection takes a ray  o + t*d  with d NOT necessarily unit length,
// and returns the smallest t > 0 at which the ray crosses the surface, or
// kNoHit. The renderer exploits that: it casts each pixel's ray with a
// direction whose optical-z component is exactly 1, so the t that comes back
// IS the depth z the camera reports, with no square root and no division. A
// function that silently assumed |d| = 1 would return range-along-the-ray
// instead, which is wrong by up to 1/cos(43.5 deg) = 38 % at the image edge
// and correct at the centre -- the most invisible kind of wrong. The tests
// cast deliberately non-unit directions for exactly that reason.
//
// "Smallest t > 0" over every boundary crossing is the first surface the ray
// meets, for a convex solid, wherever the ray starts. A camera that has ended
// up inside a body therefore sees its inside wall rather than nothing; a real
// stereo camera pressed against something returns no depth at all, but the
// renderer's minimum range already removes anything that close.
//
// Nothing at t <= 0 is ever returned: that is behind the camera.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

namespace dsim_depthcam
{

constexpr double kNoHit = std::numeric_limits<double>::infinity();

/// Below this, a direction component is treated as zero: the ray is parallel
/// to that face. 1e-12 on a direction whose largest component is O(1).
constexpr double kParallel = 1e-12;

struct Primitive
{
  enum class Type { Plane, Sphere, Cylinder, Box };

  Type type {Type::Sphere};
  std::string name;
  /// Plane: only z is used (the plane's height). Cylinder: centre at
  /// mid-height. Sphere, box: centre.
  Eigen::Vector3d center {Eigen::Vector3d::Zero()};
  /// Sphere and cylinder.
  double radius {0.0};
  /// Cylinder: half the height; +infinity for a column with no ends.
  double half_height {0.0};
  /// Box: half the edge lengths, in the box's own (yawed) frame.
  Eigen::Vector3d half_size {Eigen::Vector3d::Zero()};
  /// Box: rotation about world z, radians. Set through makeBox(), which
  /// caches its cosine and sine below: they are needed once per RAY, and a
  /// frame is a hundred thousand rays.
  double yaw {0.0};
  double cos_yaw {1.0};
  double sin_yaw {0.0};
};

// ---- constructors ---------------------------------------------------------
// They validate, because a zero-radius sphere or a negative box still
// produces a picture -- of nothing -- and nothing in the image says why.

inline Primitive makePlane(double height, const std::string & name = "ground")
{
  Primitive p;
  p.type = Primitive::Type::Plane;
  p.name = name;
  p.center = Eigen::Vector3d(0.0, 0.0, height);
  return p;
}

inline Primitive makeSphere(
  const Eigen::Vector3d & center, double radius, const std::string & name = "")
{
  if (!(radius > 0.0)) {throw std::invalid_argument("sphere '" + name + "': radius must be > 0");}
  Primitive p;
  p.type = Primitive::Type::Sphere;
  p.name = name;
  p.center = center;
  p.radius = radius;
  return p;
}

/// `height` = +infinity gives a column with no ends.
inline Primitive makeCylinder(
  const Eigen::Vector3d & center, double radius, double height, const std::string & name = "")
{
  if (!(radius > 0.0)) {throw std::invalid_argument("cylinder '" + name + "': radius must be > 0");}
  if (!(height > 0.0)) {throw std::invalid_argument("cylinder '" + name + "': height must be > 0");}
  Primitive p;
  p.type = Primitive::Type::Cylinder;
  p.name = name;
  p.center = center;
  p.radius = radius;
  p.half_height = 0.5 * height;      // inf * 0.5 stays inf
  return p;
}

inline Primitive makeBox(
  const Eigen::Vector3d & center, const Eigen::Vector3d & size, double yaw,
  const std::string & name = "")
{
  if (!(size.minCoeff() > 0.0)) {
    throw std::invalid_argument("box '" + name + "': every size must be > 0");
  }
  Primitive p;
  p.type = Primitive::Type::Box;
  p.name = name;
  p.center = center;
  p.half_size = 0.5 * size;
  p.yaw = yaw;
  p.cos_yaw = std::cos(yaw);
  p.sin_yaw = std::sin(yaw);
  return p;
}

// ---- intersections --------------------------------------------------------

/// Horizontal plane z = height.
inline double intersectPlane(const Primitive & p, const Eigen::Vector3d & o, const Eigen::Vector3d & d)
{
  if (std::abs(d.z()) < kParallel) {return kNoHit;}
  const double t = (p.center.z() - o.z()) / d.z();
  return t > 0.0 ? t : kNoHit;
}

inline double intersectSphere(const Primitive & p, const Eigen::Vector3d & o, const Eigen::Vector3d & d)
{
  // |o + t d - c|^2 = r^2  ->  a t^2 + 2 b t + cc = 0, with the half-b form
  // so the discriminant needs no factor of 4.
  const Eigen::Vector3d oc = o - p.center;
  const double a = d.squaredNorm();
  const double b = oc.dot(d);
  const double cc = oc.squaredNorm() - p.radius * p.radius;
  const double disc = b * b - a * cc;
  if (disc < 0.0 || a <= 0.0) {return kNoHit;}
  const double sq = std::sqrt(disc);
  // Near root first: from outside it is the entry point. Only if that is
  // behind the camera (we are inside, or the sphere is behind us) does the
  // far root count, and then only if IT is in front.
  const double t0 = (-b - sq) / a;
  if (t0 > 0.0) {return t0;}
  const double t1 = (-b + sq) / a;
  return t1 > 0.0 ? t1 : kNoHit;
}

/// Vertical cylinder, centre at mid-height, possibly infinite.
inline double intersectCylinder(const Primitive & p, const Eigen::Vector3d & o, const Eigen::Vector3d & d)
{
  double best = kNoHit;
  const double ox = o.x() - p.center.x();
  const double oy = o.y() - p.center.y();
  const double r2 = p.radius * p.radius;

  // Side wall: the circle in xy, accepted only where the hit's z is within
  // the cylinder's height. A vertical ray (a == 0) never meets the wall.
  const double a = d.x() * d.x() + d.y() * d.y();
  if (a > kParallel * kParallel) {
    const double b = ox * d.x() + oy * d.y();
    const double cc = ox * ox + oy * oy - r2;
    const double disc = b * b - a * cc;
    if (disc >= 0.0) {
      const double sq = std::sqrt(disc);
      for (const double t : {(-b - sq) / a, (-b + sq) / a}) {
        if (t > 0.0 && t < best &&
          std::abs(o.z() + t * d.z() - p.center.z()) <= p.half_height)
        {
          best = t;
        }
      }
    }
  }

  // End caps: discs at centre z +/- half_height. A column has none -- its
  // half_height is infinite and the test below is skipped.
  if (std::isfinite(p.half_height) && std::abs(d.z()) >= kParallel) {
    for (const double zc : {p.center.z() - p.half_height, p.center.z() + p.half_height}) {
      const double t = (zc - o.z()) / d.z();
      if (t > 0.0 && t < best) {
        const double hx = ox + t * d.x();
        const double hy = oy + t * d.y();
        if (hx * hx + hy * hy <= r2) {best = t;}
      }
    }
  }
  return best;
}

/// Box rotated by yaw about the vertical through its centre: the slab method
/// in the box's own frame.
inline double intersectBox(const Primitive & p, const Eigen::Vector3d & o, const Eigen::Vector3d & d)
{
  // World -> box frame is a rotation by -yaw about z.
  const double c = p.cos_yaw, s = p.sin_yaw;
  const Eigen::Vector3d w = o - p.center;
  const Eigen::Vector3d lo(c * w.x() + s * w.y(), -s * w.x() + c * w.y(), w.z());
  const Eigen::Vector3d ld(c * d.x() + s * d.y(), -s * d.x() + c * d.y(), d.z());

  double t_near = -std::numeric_limits<double>::infinity();
  double t_far = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i) {
    const double h = p.half_size[i];
    if (std::abs(ld[i]) < kParallel) {
      // Parallel to this pair of faces: inside the slab forever, or never.
      if (std::abs(lo[i]) > h) {return kNoHit;}
      continue;
    }
    double t1 = (-h - lo[i]) / ld[i];
    double t2 = (h - lo[i]) / ld[i];
    if (t1 > t2) {std::swap(t1, t2);}
    t_near = std::max(t_near, t1);
    t_far = std::min(t_far, t2);
    if (t_near > t_far) {return kNoHit;}
  }
  if (t_near > 0.0) {return t_near;}     // entering from outside
  if (t_far > 0.0) {return t_far;}       // starting inside: the exit wall
  return kNoHit;                         // entirely behind
}

inline double intersect(const Primitive & p, const Eigen::Vector3d & o, const Eigen::Vector3d & d)
{
  switch (p.type) {
    case Primitive::Type::Plane: return intersectPlane(p, o, d);
    case Primitive::Type::Sphere: return intersectSphere(p, o, d);
    case Primitive::Type::Cylinder: return intersectCylinder(p, o, d);
    case Primitive::Type::Box: return intersectBox(p, o, d);
  }
  return kNoHit;
}

}  // namespace dsim_depthcam
