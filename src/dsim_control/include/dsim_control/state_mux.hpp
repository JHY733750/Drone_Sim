#pragma once
// Where each part of the controller's state comes from, field by field.
//
// The controller consumes four things: position, velocity, attitude and body
// rate. Each can be flown on the onboard estimate or on ground truth, chosen
// independently, so one sensor's noise can be ruled in or out without
// replacing everything else at once:
//
//   field       est                                   truth
//   position    /drone/state_est pose.position        /drone/truth_exact
//   velocity    /drone/state_est twist.linear         /drone/truth_exact
//   attitude    /drone/state_est pose.orientation     /drone/truth_exact
//   body_rate   /drone/imu gyro, gated + low-passed   /drone/truth_exact twist.angular
//
// "truth" means the physics engine's value, unprocessed: /drone/truth_exact is
// published by the dsim_truth Gazebo plugin every physics step, read straight
// from the link's pose and velocity components (no differencing, no rolling
// mean -- see src/dsim_truth/src/truth_publisher.cpp for why /drone/truth's own
// twist does not qualify). On this side a truth field is copied, not
// filtered: the body rate goes through the plausibility gate with the
// low-pass disabled, and the gate returns its input unchanged for anything a
// quadrotor can physically do.
//
// position has a third source, slam: /slam/odom from dsim_slam, the depth
// camera's scan-matching odometry (map frame = world, anchored at the spawn
// pose by the launch). Only position may take it. SLAM going LOST or quiet is
// NOT a reason to cut the motors the way a stale truth/est source is: the
// controller latches a position HOLD instead (controller_node.cpp), so here a
// slam source only gates Waiting -- its age is reported by slamAge() for the
// controller to judge.
//
// Nothing here knows about ROS, so it is unit tested on the host.

#include <cstddef>
#include <string>

#include <Eigen/Dense>

#include "dsim_control/body_rate_source.hpp"
#include "dsim_control/se3_controller.hpp"

namespace dsim_control
{

enum class Source { Est, Truth, Slam };

const char * toString(Source s);

struct StateSources
{
  Source position {Source::Est};
  Source velocity {Source::Est};
  Source attitude {Source::Est};
  Source body_rate {Source::Est};

  /// Parse and validate the four names. THE place a selection is accepted or
  /// refused: unknown names and slam on a field other than position throw
  /// std::invalid_argument with a message meant for a person reading the
  /// launch log.
  static StateSources fromNames(
    const std::string & position, const std::string & velocity,
    const std::string & attitude, const std::string & body_rate);

  /// True if any field reads from s.
  bool uses(Source s) const;
  /// True if any of position / velocity / attitude reads the estimate's
  /// Odometry. body_rate=est reads the gyro, not that message.
  bool usesEstOdometry() const;
  bool usesGyro() const {return body_rate == Source::Est;}

  /// "position=truth velocity=est attitude=truth body_rate=truth"
  std::string describe() const;
};

/// One Odometry message, unpacked. twist is as received (body frame when the
/// publisher follows REP-145, which both /drone/state_est and
/// /drone/truth_exact do).
struct OdomSample
{
  Eigen::Vector3d position {Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation {Eigen::Quaterniond::Identity()};
  Eigen::Vector3d twist_linear {Eigen::Vector3d::Zero()};
  Eigen::Vector3d twist_angular {Eigen::Vector3d::Zero()};
};

class StateMux
{
public:
  /// max_body_rate / gyro_tau_s configure the est body-rate path exactly as
  /// before. The truth body-rate path gets the same plausibility gate and NO
  /// low-pass, whatever gyro_tau_s says.
  StateMux(
    const StateSources & sources, bool twist_in_body_frame,
    double max_body_rate_rad_s, double gyro_tau_s);

  void onTruth(const OdomSample & m, double now);
  void onEst(const OdomSample & m, double now);
  /// One /slam/odom pose (position is all that is read).
  void onSlam(const OdomSample & m, double now);
  /// Seconds since the last SLAM pose, or +inf if none has arrived.
  double slamAge(double now) const;
  /// One gyro sample. Feeds the est body-rate path; state() reads that path
  /// only when body_rate=est.
  void onGyro(const Eigen::Vector3d & measured, double now);

  /// Simulated time went backwards: forget every state sample (it belongs to
  /// the old run) and restart the staleness clocks at now. The gyro filter is
  /// kept, as the controller always did -- a rate is not a place.
  void reset(double now);

  enum class Status { Ready, Waiting, Stale };
  struct Health
  {
    Status status;
    std::string why;   // for Stale: which source, how long
  };
  /// Ready only if every field's own source has delivered and is fresh.
  /// Waiting: some selected source has never delivered (idle quietly, as the
  /// controller always has before its first state). Stale: one has gone
  /// quiet past its timeout (cut the motors and say which).
  Health health(double now, double state_timeout_s, double gyro_timeout_s) const;

  /// The composed state. Only meaningful when health() is Ready.
  State state() const;

  const StateSources & sources() const {return sources_;}

  /// Rates rejected by the gate on the selected body-rate path, and the last
  /// raw sample offered to it -- for the controller's warning.
  std::size_t rejectedRates() const;
  double maxRate() const {return gyro_.maxRate();}

private:
  /// World-frame velocity of a sample, rotated with THAT sample's own
  /// orientation. Mixing attitude=truth with velocity=est must not rotate the
  /// estimate's body velocity with the true attitude: the estimator expressed
  /// it in its own estimated body frame.
  Eigen::Vector3d worldVelocity(const OdomSample & m) const;

  StateSources sources_;
  bool twist_in_body_frame_;

  OdomSample truth_;
  OdomSample est_;
  bool have_truth_ {false};
  bool have_est_ {false};
  double last_truth_s_ {0.0};
  double last_est_s_ {0.0};
  OdomSample slam_;
  bool have_slam_ {false};
  double last_slam_s_ {0.0};

  BodyRateSource gyro_;        // body_rate=est
  BodyRateSource truth_rate_;  // body_rate=truth: gate only, tau = 0
  bool have_gyro_ {false};
  double last_gyro_s_ {0.0};
};

}  // namespace dsim_control
