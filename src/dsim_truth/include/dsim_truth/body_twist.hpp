#pragma once
// The one piece of arithmetic between the physics engine's state and
// /drone/truth_exact: express the link's world-frame velocities in its own
// body frame, as nav_msgs/Odometry requires (REP-145: the twist is in the
// child frame).
//
// Header-only and Eigen-only so it is testable on the host with no Gazebo
// (scripts/mutation_check.sh builds test/test_truth.cpp that way). The plugin
// converts gz::math types to these and back; nothing here knows about Gazebo.

#include <Eigen/Dense>

namespace dsim_truth
{

struct BodyTwist
{
  Eigen::Vector3d linear;   // body frame, m/s
  Eigen::Vector3d angular;  // body frame, rad/s
};

/// q_world_body rotates body vectors into the world; its inverse takes world
/// vectors into the body. Both velocities come from the same physics step as
/// the pose -- no differencing, so no lag and no quaternion double cover.
inline BodyTwist bodyTwistFromWorld(
  const Eigen::Quaterniond & q_world_body,
  const Eigen::Vector3d & v_world,
  const Eigen::Vector3d & w_world)
{
  const Eigen::Quaterniond q = q_world_body.normalized();
  return BodyTwist{q.conjugate() * v_world, q.conjugate() * w_world};
}

}  // namespace dsim_truth
