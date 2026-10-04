#include "dsim_slam/scan_odometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace dsim_slam
{

const char * toString(TrackingState s)
{
  switch (s) {
    case TrackingState::kTracking: return "TRACKING";
    case TrackingState::kDegraded: return "DEGRADED";
    case TrackingState::kLost: return "LOST";
  }
  return "?";
}

namespace
{
std::string fmt(const char * f, double a, double b = 0.0, double c = 0.0, double d = 0.0)
{
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a, b, c, d);
  return buf;
}
}  // namespace

ScanOdometry::ScanOdometry(const OdometryConfig & cfg, const Eigen::Isometry3d & initial_pose)
: cfg_(cfg), map_(cfg.map), initial_pose_(initial_pose), pose_(initial_pose)
{
  resetVelocity();
  threshold_ = cfg_.initial_threshold_m;
  last_.pose = pose_;
}

void ScanOdometry::addGyro(double t, const Eigen::Vector3d & rate)
{
  gyro_.add(t, rate);
}

void ScanOdometry::reset(const Eigen::Isometry3d & pose)
{
  map_.clear();
  gyro_.clear();
  have_scan_ = false;
  map_initialised_ = false;
  pose_ = pose;
  cov_.setZero();
  distance_ = 0.0;
  resetVelocity();
  omega_body_cv_.setZero();
  threshold_sq_sum_ = 0.0;
  threshold_n_ = 0;
  threshold_ = cfg_.initial_threshold_m;
  last_ = OdometryReport{};
  last_.pose = pose_;
}

std::vector<SourcePoint> ScanOdometry::preprocess(
  const std::vector<Eigen::Vector3d> & points_sensor, int * in_gate) const
{
  // One REAL measurement per voxel -- the first to land in it, as KISS-ICP
  // does -- not the voxel's centroid. A centroid of points that straddle an
  // edge, or lie on a curved surface, is on no surface at all, and those
  // offsets all point the same way (into convex corners, inside cylinders):
  // on two noise-free renders of the test world that was a 5.9 cm, 1.1 deg
  // registration bias. A raw point carries its full depth noise, but that
  // noise is zero-mean, and its variance below is that point's own.
  std::unordered_map<int64_t, int> cells;
  cells.reserve(4096);
  const double inv = 1.0 / cfg_.source_voxel_m;
  const double f2 = cfg_.depth_noise_floor_m * cfg_.depth_noise_floor_m;
  int gate = 0;
  std::vector<SourcePoint> out;
  out.reserve(4096);
  for (const auto & ps : points_sensor) {
    if (!ps.allFinite()) {continue;}
    const double z = ps.z();
    if (z < cfg_.min_depth_m || z > cfg_.max_depth_m) {continue;}
    ++gate;
    const Eigen::Vector3d pb = cfg_.T_base_sensor * ps;
    // 21 bits per axis (+-1M voxels) packed into one key.
    const int64_t ix = static_cast<int64_t>(std::floor(pb.x() * inv)) & 0x1FFFFF;
    const int64_t iy = static_cast<int64_t>(std::floor(pb.y() * inv)) & 0x1FFFFF;
    const int64_t iz = static_cast<int64_t>(std::floor(pb.z() * inv)) & 0x1FFFFF;
    if (!cells.emplace((ix << 42) | (iy << 21) | iz, 0).second) {continue;}
    const double sq = cfg_.depth_noise_quadratic * z * z;
    out.push_back({pb, f2 + sq * sq});
  }
  if (in_gate != nullptr) {*in_gate = gate;}
  return out;
}

void ScanOdometry::insertScan(const std::vector<SourcePoint> & src, const Eigen::Isometry3d & pose)
{
  std::vector<Eigen::Vector3d> pts;
  std::vector<double> var;
  pts.reserve(src.size());
  var.reserve(src.size());
  for (const auto & s : src) {
    pts.push_back(pose * s.p);
    var.push_back(s.variance);
  }
  map_.insert(pts, var);
  map_.removeFar(pose.translation());
}

void ScanOdometry::updateThreshold(
  const Eigen::Isometry3d & predicted, const Eigen::Isometry3d & estimated,
  const Eigen::Isometry3d & previous)
{
  // KISS-ICP's adaptive threshold: how far did registration have to move
  // the prediction, expressed as the largest displacement it causes at the
  // edge of the depth gate. Only frames that actually moved count, so a
  // hover does not shrink the threshold to nothing.
  const Vector6d dev = poseError(estimated, predicted);
  const double model_error = dev.head<3>().norm() +
    2.0 * cfg_.max_depth_m * std::sin(0.5 * std::min(M_PI, dev.tail<3>().norm()));
  const Vector6d motion = poseError(estimated, previous);
  const double moved = motion.head<3>().norm() +
    2.0 * cfg_.max_depth_m * std::sin(0.5 * std::min(M_PI, motion.tail<3>().norm()));
  if (moved > cfg_.min_motion_m) {
    threshold_sq_sum_ += model_error * model_error;
    threshold_n_ += 1;
  }
  if (threshold_n_ > 0) {
    threshold_ = std::max(cfg_.min_threshold_m, 3.0 * std::sqrt(threshold_sq_sum_ / threshold_n_));
  }
}

OdometryReport ScanOdometry::lostFrame(
  double t, double dt, const Eigen::Matrix3d & rotation, const std::string & reason,
  OdometryReport rep)
{
  // Hold the position: with nothing registered there is no translation
  // measurement, and extrapolating a velocity for an unknown time is how a
  // lost estimate flies into a wall. The covariance says how far the
  // vehicle could have gone: sigma = lost_velocity_sigma * time lost, so the
  // variance increment over [t - dt, t] is s^2 (T^2 - (T - dt)^2).
  const double T = std::max(0.0, t - last_accepted_t_);
  const double T0 = std::max(0.0, T - dt);
  const double s2 = cfg_.lost_velocity_sigma * cfg_.lost_velocity_sigma;
  const double dvar_p = s2 * (T * T - T0 * T0);
  const double dvar_r = rep.imu_prior_used ?
    cfg_.lost_rate_sigma * cfg_.lost_rate_sigma * dt :
    std::pow(cfg_.prior_rate_sigma_cv * dt, 2);
  for (int i = 0; i < 3; ++i) {
    cov_(i, i) += dvar_p;
    cov_(3 + i, 3 + i) += dvar_r;
  }
  pose_.linear() = rotation;
  resetVelocity();
  omega_body_cv_.setZero();
  last_t_ = t;

  rep.state = TrackingState::kLost;
  rep.reason = reason;
  rep.pose = pose_;
  rep.pose_covariance = reportedCovariance();
  rep.velocity_valid = false;
  rep.threshold_m = threshold_;
  rep.map_voxels = map_.size();
  rep.time_since_good_s = map_initialised_ ? t - last_good_t_ : 0.0;
  last_ = rep;
  return rep;
}

OdometryReport ScanOdometry::processScan(
  double t, const std::vector<Eigen::Vector3d> & points_sensor)
{
  OdometryReport rep;
  rep.stamp = t;
  rep.pose_valid = true;

  if (have_scan_ && t <= last_t_) {
    if (t < last_t_ - 1.0) {
      // Time went backwards: a simulator reset (or a replayed bag). Nothing
      // accumulated so far belongs to the new timeline.
      reset(initial_pose_);
    } else {
      OdometryReport r = last_;
      r.reason = "cloud stamp not after the previous one: ignored";
      return r;
    }
  }

  const std::vector<SourcePoint> src = preprocess(points_sensor, &rep.raw_points);
  rep.source_points = static_cast<int>(src.size());
  rep.threshold_m = threshold_;

  // ---- first scan: it IS the map --------------------------------------
  if (!map_initialised_) {
    have_scan_ = true;
    if (rep.source_points < cfg_.min_source_points) {
      last_t_ = t;
      rep.state = TrackingState::kLost;
      rep.reason = fmt("only %.0f points in range (need %.0f): no map yet",
          rep.source_points, cfg_.min_source_points);
      rep.pose = pose_;
      rep.pose_covariance = reportedCovariance();
      last_ = rep;
      return rep;
    }
    insertScan(src, pose_);
    map_initialised_ = true;
    last_t_ = last_good_t_ = last_accepted_t_ = t;
    rep.state = TrackingState::kTracking;
    rep.reason = "map initialised from the first scan";
    rep.pose = pose_;
    rep.pose_covariance = reportedCovariance();
    rep.map_voxels = map_.size();
    last_ = rep;
    return rep;
  }

  const double dt = t - last_t_;

  // ---- prediction ---------------------------------------------------------
  Eigen::Matrix3d dR = expSO3(omega_body_cv_ * dt);
  if (cfg_.use_imu_prior) {
    if (const auto r = gyro_.integrate(last_t_, t)) {
      dR = *r;
      rep.imu_prior_used = true;
    }
  }
  const Eigen::Isometry3d previous = pose_;
  Eigen::Isometry3d predicted = Eigen::Isometry3d::Identity();
  // The gyro measures BODY rates, so its rotation composes on the right.
  predicted.linear() = pose_.linear() * dR;
  predicted.translation() = pose_.translation() +
    velocity_map_ * cfg_.velocity_correlation_time_s * (1.0 - std::exp(-dt / cfg_.velocity_correlation_time_s));

  // Translation: a Kalman filter on [p; v] (map frame) whose velocity is a
  // first-order Gauss-Markov process -- it decays towards zero with time
  // constant tau and has a bounded stationary sigma. Its predicted position
  // covariance IS the registration's translation prior, so the prior is as
  // loose as the velocity is unknown, and in a direction the scans do not
  // observe the position diffuses at a bounded rate. Two simpler models
  // failed, both facing a wall: a fixed velocity sigma carried the first
  // frame's 0.9 m/s error for ever at a few-cm/s sigma (6 m, far outside 3
  // sigma); constant velocity with growing variance let any leaked kick run
  // the velocity away (to 5 m/s, 30-80 m). A drone that nothing pushes slows
  // down; this says so.
  const double tau = cfg_.velocity_correlation_time_s;
  const double a = std::exp(-dt / tau);
  const double sv2 = cfg_.velocity_sigma * cfg_.velocity_sigma;
  Eigen::Matrix<double, 6, 6> Fk = Eigen::Matrix<double, 6, 6>::Identity();
  Fk.topRightCorner<3, 3>() = tau * (1.0 - a) * Eigen::Matrix3d::Identity();
  Fk.bottomRightCorner<3, 3>() = a * Eigen::Matrix3d::Identity();
  const double qv = sv2 * (1.0 - a * a);
  Eigen::Matrix<double, 6, 6> Qk = Eigen::Matrix<double, 6, 6>::Zero();
  Qk.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * qv * dt * dt / 3.0;
  Qk.topRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * qv * dt / 2.0;
  Qk.bottomLeftCorner<3, 3>() = Qk.topRightCorner<3, 3>();
  Qk.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * qv;
  const Eigen::Matrix<double, 6, 6> P_pred = Fk * kf_P_ * Fk.transpose() + Qk;
  const Eigen::Matrix3d P_pp = 0.5 * (P_pred.topLeftCorner<3, 3>() +
    P_pred.topLeftCorner<3, 3>().transpose());
  const double sig_r = rep.imu_prior_used ?
    std::max(cfg_.prior_rotation_floor_imu, cfg_.prior_rate_sigma_imu * dt) :
    std::max(cfg_.prior_rotation_floor_imu, cfg_.prior_rate_sigma_cv * dt);
  PosePrior prior;
  prior.pose = predicted;
  prior.information.setZero();
  prior.information.topLeftCorner<3, 3>() = P_pp.inverse();
  for (int i = 0; i < 3; ++i) {
    prior.information(3 + i, 3 + i) = 1.0 / (sig_r * sig_r);
  }

  if (rep.source_points < cfg_.min_source_points) {
    return lostFrame(
      t, dt, predicted.linear(),
      fmt("only %.0f points in range (need %.0f)", rep.source_points, cfg_.min_source_points),
      rep);
  }

  // ---- registration -------------------------------------------------------
  std::vector<SourcePoint> reg_src;
  reg_src.reserve(src.size());
  const double max_var = cfg_.max_registration_sigma_m * cfg_.max_registration_sigma_m;
  for (const auto & s : src) {
    if (s.variance <= max_var) {reg_src.push_back(s);}
  }
  RegistrationConfig rc = cfg_.registration;
  rc.degenerate_translation_eig = cfg_.degeneracy_translation_min;
  rc.degenerate_rotation_eig = cfg_.degeneracy_rotation_min;
  rc.sticky_translation_dirs = sticky_dirs_;
  const RegistrationResult reg =
    registerScan(reg_src, map_, predicted, threshold_, &prior, rc);
  rep.correspondences = reg.correspondences;
  rep.inlier_fraction = reg_src.empty() ? 0.0 :
    static_cast<double>(reg.correspondences) / static_cast<double>(reg_src.size());
  rep.rms_residual_m = reg.rms_residual_m;
  rep.chi2_per_dof = reg.chi2_per_dof;
  rep.iterations = reg.iterations;
  rep.converged = reg.converged;
  rep.degeneracy = analyseDegeneracy(reg.judge_hessian, reg.judge_weight_sum);
  sticky_dirs_ = reg.projected_translation_dirs;

  if (reg.correspondences < cfg_.min_correspondences) {
    const double lost_for = t - last_good_t_;
    if (lost_for > cfg_.reset_map_after_lost_s) {
      // Long enough without a match that the old map is no use from here:
      // start a new one at the held pose. The position may be off from the
      // old map by however far the vehicle moved while lost, and the
      // covariance already says so -- it is kept, not reset.
      map_.clear();
      pose_.linear() = predicted.linear();
      insertScan(src, pose_);
      resetVelocity();
      last_t_ = last_accepted_t_ = t;
      rep.state = TrackingState::kDegraded;
      rep.map_reset = true;
      rep.reason = fmt("map re-initialised after %.1f s without a match: position "
          "continuity with the old map is not guaranteed", lost_for);
      rep.pose = pose_;
      rep.pose_covariance = reportedCovariance();
      rep.map_voxels = map_.size();
      rep.time_since_good_s = lost_for;
      last_ = rep;
      return rep;
    }
    return lostFrame(
      t, dt, predicted.linear(),
      fmt("only %.0f of %.0f points matched the map (need %.0f)",
          reg.correspondences, rep.source_points, cfg_.min_correspondences),
      rep);
  }

  // ---- accepted: covariance ------------------------------------------------
  // A rotation error phi at the previous pose swings this frame's
  // displacement d by phi x d, so the previous covariance is transported by
  // F = [I -[d]x; 0 I] before this frame's registration covariance adds to
  // it. Random-walk accumulation: conservative while hovering in a mapped
  // area (the real error does not grow there), right while exploring.
  const Eigen::Vector3d d = reg.pose.translation() - previous.translation();
  distance_ += d.norm();
  Matrix6d F = Matrix6d::Identity();
  F.topRightCorner<3, 3>() = -skew(d);
  cov_ = F * cov_ * F.transpose() + reg.covariance;
  cov_ = 0.5 * (cov_ + cov_.transpose());

  // ---- velocity: Kalman update by conditioning on the registered position --
  // The registration already fused the prediction with the scan; the
  // velocity follows by Gaussian conditioning on that posterior position
  // (mean reg.pose.t, covariance S): x+ = x- + G (p+ - p-), P+ = P- - G
  // (P_pp - S) G^T, G = P_xp P_pp^-1. Where the scan said nothing, S = P_pp
  // and the velocity is left alone.
  {
    const Eigen::Matrix3d S = 0.5 * (reg.covariance.topLeftCorner<3, 3>() +
      reg.covariance.topLeftCorner<3, 3>().transpose());
    const Eigen::Matrix<double, 6, 3> G = P_pred.leftCols<3>() * P_pp.inverse();
    Eigen::Matrix<double, 6, 1> x;
    x.head<3>() = predicted.translation();
    x.tail<3>() = velocity_map_ * std::exp(-dt / cfg_.velocity_correlation_time_s);
    // The innovation along a direction judged unobserved is not evidence
    // (the prior decided it); feeding it to the velocity would let the mean
    // follow whatever moved the pose there while the covariance says nothing
    // was learned.
    Eigen::Vector3d innov = reg.pose.translation() - predicted.translation();
    for (const auto & u : reg.projected_translation_dirs) {innov -= u * u.dot(innov);}
    x += G * innov;
    kf_P_ = P_pred - G * (P_pp - S) * G.transpose();
    kf_P_ = 0.5 * (kf_P_ + kf_P_.transpose());
    velocity_map_ = x.tail<3>();
  }
  if (dt > 0.0) {
    rep.velocity_valid = velocity_known_;    // needs two consecutive accepted frames
    rep.velocity_body = reg.pose.linear().transpose() * velocity_map_;
    rep.velocity_covariance = reg.pose.linear().transpose() *
      kf_P_.bottomRightCorner<3, 3>() * reg.pose.linear();
    velocity_known_ = true;
    omega_body_cv_ = logSO3(previous.linear().transpose() * reg.pose.linear()) / dt;
    if (const auto w = gyro_.meanRate(last_t_, t)) {
      rep.angular_velocity_body = *w;
    } else {
      rep.angular_velocity_body = omega_body_cv_;
    }
    rep.angular_velocity_valid = true;
  }

  // ---- verdict ---------------------------------------------------------------
  rep.state = TrackingState::kTracking;
  std::string why;
  const auto & dg = rep.degeneracy;
  if (dg.translation_min_eig < cfg_.degeneracy_translation_min) {
    rep.state = TrackingState::kDegraded;
    why += fmt("translation along map (%.2f, %.2f, %.2f) unobservable (eig %.4f); ",
        dg.translation_weak_dir.x(), dg.translation_weak_dir.y(), dg.translation_weak_dir.z(),
        dg.translation_min_eig);
  }
  if (dg.rotation_min_eig < cfg_.degeneracy_rotation_min) {
    rep.state = TrackingState::kDegraded;
    why += fmt("rotation about map (%.2f, %.2f, %.2f) unobservable (eig %.4f m^2); ",
        dg.rotation_weak_dir.x(), dg.rotation_weak_dir.y(), dg.rotation_weak_dir.z(),
        dg.rotation_min_eig);
  }
  if (rep.inlier_fraction < cfg_.degraded_inlier_fraction) {
    rep.state = TrackingState::kDegraded;
    why += fmt("only %.0f%% of points matched; ", 100.0 * rep.inlier_fraction);
  }
  if (rep.state == TrackingState::kTracking) {
    rep.reason = "tracking";
    last_good_t_ = t;
    updateThreshold(predicted, reg.pose, previous);
  } else {
    rep.reason = why.substr(0, why.size() - 2);
  }

  pose_ = reg.pose;
  insertScan(src, pose_);
  last_t_ = last_accepted_t_ = t;

  rep.pose = pose_;
  rep.pose_covariance = reportedCovariance();
  rep.threshold_m = threshold_;
  rep.map_voxels = map_.size();
  rep.time_since_good_s = t - last_good_t_;
  last_ = rep;
  return rep;
}

void ScanOdometry::resetVelocity()
{
  sticky_dirs_.clear();
  velocity_map_.setZero();
  velocity_known_ = false;
  kf_P_.setZero();
  const double u = cfg_.velocity_sigma;
  for (int i = 0; i < 3; ++i) {
    kf_P_(i, i) = 1e-4;          // position relative to the map: as registered
    kf_P_(3 + i, 3 + i) = u * u;
  }
}

Matrix6d ScanOdometry::reportedCovariance() const
{
  // The random walk in cov_ treats each frame's registration error as
  // independent. It is not: map blur and the lateral/yaw ambiguity of a
  // narrow field of view give an error that grows with distance flown, and
  // the walk alone held the true error inside 3 sigma on only ~70 % of the
  // survey. The drift term is (drift_per_metre * distance)^2 -- an
  // EMPIRICAL calibration, measured by the survey test, not derived.
  Matrix6d P = cov_;
  const double sp = cfg_.drift_per_metre * distance_;
  const double sr = cfg_.rotation_drift_per_metre * distance_;
  for (int i = 0; i < 3; ++i) {
    P(i, i) += sp * sp;
    P(3 + i, 3 + i) += sr * sr;
  }
  return P;
}

OdometryReport ScanOdometry::watchdog(double now) const
{
  OdometryReport r = last_;
  if (!have_scan_) {
    r.state = TrackingState::kLost;
    r.reason = "no point cloud received yet";
    return r;
  }
  const double silent = now - last_t_;
  if (silent > cfg_.scan_timeout_s) {
    r.state = TrackingState::kLost;
    r.reason = fmt("no point cloud for %.2f s", silent);
    r.time_since_good_s = now - last_good_t_;
  }
  return r;
}

Eigen::Matrix3d gravityAlignedRotation(const Eigen::Vector3d & f, double yaw)
{
  const double roll = std::atan2(f.y(), f.z());
  const double pitch = std::atan2(-f.x(), std::hypot(f.y(), f.z()));
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
}

}  // namespace dsim_slam
