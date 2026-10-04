// SO(3) helpers and the one pose parameterisation the whole package uses.
//
// A pose error / update is a 6-vector xi = [rho; phi], ORDERED THE WAY
// nav_msgs/Odometry orders its covariance (x, y, z, rot x, rot y, rot z):
//
//   position:  t  <-  t + rho                 (map frame, metres)
//   rotation:  R  <-  Exp(phi) * R            (map-frame axes, through the
//                                              body origin, radians)
//
// That is the "rotation about fixed axes, position in the parent frame"
// convention nav_msgs documents, so the 6x6 matrices the registration
// produces go into the message unchanged. It also makes the registration
// Jacobian simple: a map point p moved by xi about the current body position
// c goes to  c + Exp(phi) (p - c) + rho, whose derivative at zero is
// [ I , -[p - c]x ].

#pragma once

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace dsim_slam
{

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

inline Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(),
    v.z(), 0.0, -v.x(),
    -v.y(), v.x(), 0.0;
  return s;
}

/// Rodrigues. Exact for any angle; the small-angle branch avoids 0/0.
inline Eigen::Matrix3d expSO3(const Eigen::Vector3d & phi)
{
  const double theta = phi.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity() + skew(phi);
  }
  return Eigen::AngleAxisd(theta, phi / theta).toRotationMatrix();
}

/// Inverse of expSO3, angle in [0, pi].
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d & R)
{
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}

/// Apply a [rho; phi] update in the convention above.
inline Eigen::Isometry3d applyUpdate(const Eigen::Isometry3d & T, const Vector6d & xi)
{
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.linear() = expSO3(xi.tail<3>()) * T.linear();
  out.translation() = T.translation() + xi.head<3>();
  return out;
}

/// The [rho; phi] that takes `from` to `to`:  applyUpdate(from, poseError(to, from)) == to.
inline Vector6d poseError(const Eigen::Isometry3d & to, const Eigen::Isometry3d & from)
{
  Vector6d e;
  e.head<3>() = to.translation() - from.translation();
  e.tail<3>() = logSO3(to.linear() * from.linear().transpose());
  return e;
}

/// The rotation from the camera's OPTICAL frame (x right, y down, z forward)
/// to a body / link frame (x forward, y left, z up), i.e. p_body = R * p_optical.
///
/// realsense-ros publishes exactly this as the static transform
/// camera_link -> camera_depth_optical_frame (rpy = -pi/2, 0, -pi/2). The node
/// takes the real extrinsic from TF and does not use this; it exists so the
/// tests can state the convention once and the frame test can catch it
/// transposed.
inline Eigen::Matrix3d opticalToBodyRotation()
{
  Eigen::Matrix3d R;
  R << 0.0, 0.0, 1.0,     // body x  =  optical z  (forward)
    -1.0, 0.0, 0.0,       // body y  = -optical x  (left = -right)
    0.0, -1.0, 0.0;       // body z  = -optical y  (up = -down)
  return R;
}

}  // namespace dsim_slam
