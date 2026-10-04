#include "dsim_slam/gyro_buffer.hpp"

#include <algorithm>

#include "dsim_slam/lie.hpp"

namespace dsim_slam
{

void GyroBuffer::add(double t, const Eigen::Vector3d & gyro)
{
  if (!gyro.allFinite()) {return;}
  if (!samples_.empty()) {
    if (t < samples_.back().t - 1.0) {
      samples_.clear();            // the clock went backwards: a reset
    } else if (t <= samples_.back().t) {
      return;                      // duplicate or out of order
    }
  }
  samples_.push_back({t, gyro});
  while (!samples_.empty() && samples_.front().t < t - keep_s_) {
    samples_.pop_front();
  }
}

bool GyroBuffer::covers(double t0, double t1) const
{
  if (samples_.empty() || !(t1 >= t0)) {return false;}
  // The sample in force at t0 must exist, and the rate it holds must not be
  // stale by t1: the newest sample has to be within one gap of t1.
  if (samples_.front().t > t0 || samples_.back().t < t1 - max_gap_s_) {return false;}
  for (std::size_t i = 1; i < samples_.size(); ++i) {
    if (samples_[i].t <= t0) {continue;}
    if (samples_[i - 1].t >= t1) {break;}
    if (samples_[i].t - samples_[i - 1].t > max_gap_s_) {return false;}
  }
  return true;
}

std::optional<Eigen::Matrix3d> GyroBuffer::integrate(double t0, double t1) const
{
  if (!covers(t0, t1)) {return std::nullopt;}
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  for (std::size_t i = 0; i < samples_.size(); ++i) {
    const double a = std::max(samples_[i].t, t0);
    const double b = std::min(i + 1 < samples_.size() ? samples_[i + 1].t : t1, t1);
    if (b <= a) {continue;}
    R = R * expSO3(samples_[i].w * (b - a));
  }
  return R;
}

std::optional<Eigen::Vector3d> GyroBuffer::meanRate(double t0, double t1) const
{
  if (!covers(t0, t1) || !(t1 > t0)) {return std::nullopt;}
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < samples_.size(); ++i) {
    const double a = std::max(samples_[i].t, t0);
    const double b = std::min(i + 1 < samples_.size() ? samples_[i + 1].t : t1, t1);
    if (b <= a) {continue;}
    sum += samples_[i].w * (b - a);
  }
  return sum / (t1 - t0);
}

}  // namespace dsim_slam
