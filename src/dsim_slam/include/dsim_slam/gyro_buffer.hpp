// Gyro samples between point clouds, integrated into the rotation prior.
//
// The camera runs at 15-30 Hz and the IMU at 250 Hz. Between two clouds the
// gyro has measured the body's rotation directly; integrating it gives the
// registration a starting rotation that is right to the gyro's noise and
// bias times the frame interval (~1e-4 rad), where a constant-velocity guess
// is wrong by the change in angular rate times dt -- 10 degrees per frame in
// a yaw reversal at 15 Hz, which is more than ICP's basin of convergence.
//
// Integration is in the BODY frame, sample by sample: R(t1) = R(t0) *
// prod Exp(w_i dt_i), because a strapdown gyro measures body rates. Composing
// on the other side would be right only while the vehicle is level.
//
// Samples are treated as zero-order holds starting at their stamp (the rate
// measured at t_i holds until t_{i+1}); the interval is clipped at both ends
// so a cloud stamped between two samples gets exactly its share.

#pragma once

#include <deque>
#include <optional>

#include <Eigen/Core>

namespace dsim_slam
{

class GyroBuffer
{
public:
  /// `max_gap_s`: the largest hole in the samples an integration may span;
  /// a longer one (a dropped IMU stream) makes integrate() refuse rather
  /// than hold a stale rate across it.
  explicit GyroBuffer(double max_gap_s = 0.05, double keep_s = 2.0)
  : max_gap_s_(max_gap_s), keep_s_(keep_s) {}

  /// Add one body-frame angular rate sample (rad/s). Out-of-order samples
  /// are dropped; a sample far in the past of the newest (clock reset) clears
  /// the buffer.
  void add(double t, const Eigen::Vector3d & gyro);

  /// The body rotation from t0 to t1 (R_t0^T R_t1), or nothing if the
  /// samples do not cover [t0, t1] without a gap.
  std::optional<Eigen::Matrix3d> integrate(double t0, double t1) const;

  /// Mean angular rate over [t0, t1], or nothing (same coverage rule).
  std::optional<Eigen::Vector3d> meanRate(double t0, double t1) const;

  void clear() {samples_.clear();}
  bool empty() const {return samples_.empty();}
  double newest() const {return samples_.empty() ? -1e300 : samples_.back().t;}

private:
  struct Sample
  {
    double t;
    Eigen::Vector3d w;
  };
  bool covers(double t0, double t1) const;

  double max_gap_s_;
  double keep_s_;
  std::deque<Sample> samples_;
};

}  // namespace dsim_slam
