// Invariants for the truth plugin's arithmetic. Each comment says what bug
// would make it fail.

#include <gtest/gtest.h>
#include <cmath>

#include "dsim_truth/body_twist.hpp"

using dsim_truth::bodyTwistFromWorld;

namespace
{
Eigen::Quaterniond yawPitchRoll(double yaw, double pitch, double roll)
{
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
}
}  // namespace

// FAILS IF: the velocity is rotated by R instead of R^T (world -> body is the
// inverse rotation). Yawed 90 deg left and flying world +x, the vehicle is
// moving towards its own right: body -y. R instead of R^T gives body +y.
TEST(BodyTwist, WorldVelocityIsExpressedInTheBodyFrame)
{
  const auto q = yawPitchRoll(M_PI / 2.0, 0.0, 0.0);
  const auto t = bodyTwistFromWorld(q, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  EXPECT_NEAR(t.linear.x(), 0.0, 1e-12);
  EXPECT_NEAR(t.linear.y(), -2.0, 1e-12);
  EXPECT_NEAR(t.linear.z(), 0.0, 1e-12);
}

// FAILS IF: the angular velocity is left in the world frame, or rotated the
// wrong way. Rolled 90 deg, a world yaw rate is a body -y rate.
TEST(BodyTwist, WorldAngularVelocityIsExpressedInTheBodyFrame)
{
  const auto q = yawPitchRoll(0.0, 0.0, M_PI / 2.0);
  const auto t = bodyTwistFromWorld(q, {0.0, 0.0, 0.0}, {0.0, 0.0, 1.5});
  EXPECT_NEAR(t.angular.x(), 0.0, 1e-12);
  EXPECT_NEAR(t.angular.y(), 1.5, 1e-12);
  EXPECT_NEAR(t.angular.z(), 0.0, 1e-12);
}

// FAILS IF: anything but a pure rotation is applied (a rotation cannot change a
// length), and the consumer's inverse (R * v_body, which is what the
// controller does with twist_in_body_frame) must give back the world vector
// to rounding -- the round trip the truth path actually takes.
TEST(BodyTwist, TheControllersInverseRecoversTheWorldVectorExactly)
{
  const auto q = yawPitchRoll(2.7, -0.4, 0.9);
  const Eigen::Vector3d v(1.3, -0.7, 0.25), w(0.3, -2.1, 0.8);
  const auto t = bodyTwistFromWorld(q, v, w);
  EXPECT_NEAR(t.linear.norm(), v.norm(), 1e-12);
  EXPECT_LT((q * t.linear - v).norm(), 1e-12);
  EXPECT_LT((q * t.angular - w).norm(), 1e-12);
}

// FAILS IF: the result depends on the quaternion's sign. q and -q are one
// rotation; this is the property /drone/truth's differentiated twist lacked.
TEST(BodyTwist, QuaternionSignDoesNotMatter)
{
  const auto q = yawPitchRoll(-1.1, 0.3, 0.2);
  const Eigen::Quaterniond nq(-q.w(), -q.x(), -q.y(), -q.z());
  const Eigen::Vector3d v(0.4, 1.9, -0.3), w(1.2, 0.1, -0.6);
  const auto a = bodyTwistFromWorld(q, v, w);
  const auto b = bodyTwistFromWorld(nq, v, w);
  EXPECT_LT((a.linear - b.linear).norm(), 1e-15);
  EXPECT_LT((a.angular - b.angular).norm(), 1e-15);
}
