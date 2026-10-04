// Depth-camera scan-to-local-map odometry with a gyro rotation prior.
//
//   point cloud (sensor frame, z = depth)          gyro (body frame, 250 Hz)
//        |  range gate, depth-noise variance           |
//        |  per point, to body frame, voxel             |  integrated between
//        |  downsample                                  |  cloud stamps
//        v                                              v
//   source points --------> registerScan(map, prediction, prior) <-- constant velocity
//                                |                                    translation
//                                v
//            verdict: TRACKING / DEGRADED / LOST, with the reason
//                                |
//              pose, covariance, velocity, and the scan into the map
//
// KISS-ICP's shape (voxel local map, adaptive correspondence threshold from
// the motion model's own error, constant-velocity prediction), with three
// changes for a narrow, noisy stereo camera instead of a 360 degree lidar:
// point-to-plane so degeneracy is visible (registration.hpp), noise-weighted
// voxel means (voxel_map.hpp), and the gyro instead of constant velocity for
// the rotation guess (gyro_buffer.hpp). The map frame is wherever the body
// was at the first scan (or the pose given to the constructor / reset()).
// Nothing here knows the true pose of anything.
//
// The verdict is the contract with the consumer:
//   TRACKING  the scan constrained all six directions and most of it matched.
//   DEGRADED  a pose was computed but some direction rests on the prior
//             (degenerate geometry: one wall, a floor), or few points
//             matched, or the map was just re-initialised. The pose is the
//             best available, its covariance says how bad, `reason` says why.
//   LOST      no registration this frame (too few points in range, too few
//             matches). The position is HELD, the rotation follows the gyro,
//             and the covariance grows as an unknown velocity would carry it.
//             A consumer must not close a position loop on it.
//
// No ROS.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "dsim_slam/gyro_buffer.hpp"
#include "dsim_slam/lie.hpp"
#include "dsim_slam/registration.hpp"
#include "dsim_slam/voxel_map.hpp"

namespace dsim_slam
{

enum class TrackingState : uint8_t {kTracking = 0, kDegraded = 1, kLost = 2};
const char * toString(TrackingState s);

struct OdometryConfig
{
  /// The camera's pose on the body: p_body = T_base_sensor * p_sensor. The
  /// node fills it from TF (base_link -> camera_depth_optical_frame).
  Eigen::Isometry3d T_base_sensor {Eigen::Isometry3d::Identity()};

  // ---- depth model. Points must be in an OPTICAL frame: z is depth. ------
  double min_depth_m {0.28};
  /// The D435 reports to 10 m, but at 424x240 its depth sigma there is ~0.7 m;
  /// beyond ~6 m a point costs more compute than it adds information.
  double max_depth_m {6.0};
  /// sigma_z = sqrt(floor^2 + (k z^2)^2). Stereo: k = subpixel / (f_px * B).
  /// D435 at 424x240: f = 212 / tan(43.5 deg) = 223 px, B = 0.050 m,
  /// subpixel ~0.08 px -> k = 0.08 / (223 * 0.050) = 0.0072 /m.
  double depth_noise_floor_m {0.005};
  double depth_noise_quadratic {0.0072};
  /// Body-frame voxel the scan is averaged into before registration.
  double source_voxel_m {0.20};
  /// Points noisier than this are put in the MAP (repeated observations
  /// confirm far surfaces) but not REGISTERED: at sigma 0.15 m (z ~ 4.5 m)
  /// a point weighs 1/50 of one at 2 m, rarely finds a confirmed plane, and
  /// noise scatters such points over so many voxels that they were the bulk
  /// of the per-frame cost.
  double max_registration_sigma_m {0.15};

  VoxelMapConfig map {};
  RegistrationConfig registration {};

  // ---- adaptive correspondence threshold (KISS-ICP) ----------------------
  /// Correspondences further than tau (+3 sigma of the point) off the plane
  /// are rejected. tau = 3 * RMS of how far registration moved the pose
  /// from the prediction, over frames that moved at least min_motion_m.
  double initial_threshold_m {0.10};
  double min_threshold_m {0.05};
  double min_motion_m {0.02};

  // ---- motion prior --------------------------------------------------------
  bool use_imu_prior {true};
  /// Translation prior: a Kalman filter on position and velocity, the
  /// velocity a Gauss-Markov process (see processScan). velocity_sigma is
  /// its stationary sigma -- also what an unknown velocity (first frame,
  /// after LOST) starts at -- and the correlation time is how long a
  /// velocity persists with nothing measuring it.
  double velocity_sigma {1.0};                 // m/s
  double velocity_correlation_time_s {5.0};
  /// Rotation prior sigma per frame: with the gyro, rate_sigma * dt but at
  /// least the floor; without it, the constant-angular-velocity guess's
  /// uncertainty. The gyro is TRUSTED: an 87 deg field of view barely tells
  /// a sideways shift from a yaw, and with a loose floor (3 mrad) the scan's
  /// poorly conditioned yaw won and the survey drifted 17-35 deg. The
  /// integrated gyro is good to ~1e-5 rad per frame; the 0.5 mrad floor is
  /// for what it does not see -- camera-IMU timing offset (rate x offset:
  /// 3 rad/s x 0.2 ms) and extrinsic error. On hardware with an unsynchronised
  /// camera, RAISE it.
  double prior_rate_sigma_imu {0.01};      // rad/s
  double prior_rotation_floor_imu {0.0005}; // rad
  double prior_rate_sigma_cv {2.0};        // rad/s

  // ---- covariance ----------------------------------------------------------
  /// Added to the reported pose covariance as (rate * distance flown)^2:
  /// the part of the drift that does not average out frame to frame. An
  /// empirical calibration -- the survey test measures ~1 % of distance on
  /// its synthetic world; re-measure on real data.
  double drift_per_metre {0.01};
  double rotation_drift_per_metre {0.001};   // rad/m

  // ---- verdicts -------------------------------------------------------------
  int min_source_points {60};              // after downsampling
  int min_correspondences {40};
  /// A weak signal: far points (sigma ~ z^2) only become usable after
  /// several frames and patches must pass a planarity test, so on the
  /// survey a frame that registers to a few cm matches 10-50 %.
  double degraded_inlier_fraction {0.10};
  /// Below these (see Degeneracy) the scan does not observe some direction.
  double degeneracy_translation_min {0.025};
  double degeneracy_rotation_min {0.02};   // m^2
  /// While LOST the held position drifts like an unknown velocity of this
  /// sigma: position sigma grows as lost_velocity_sigma * time lost.
  double lost_velocity_sigma {1.0};        // m/s
  double lost_rate_sigma {0.02};           // rad/s, gyro-only rotation, random walk
  /// Lost this long, the next scan that cannot be registered re-initialises
  /// the map at the held pose instead of staying lost for ever.
  double reset_map_after_lost_s {1.0};
  /// The watchdog calls it LOST when no cloud has arrived for this long.
  double scan_timeout_s {0.5};
};

struct OdometryReport
{
  double stamp {0.0};
  bool pose_valid {false};
  TrackingState state {TrackingState::kLost};
  std::string reason {"no point cloud received yet"};

  /// body -> map.
  Eigen::Isometry3d pose {Eigen::Isometry3d::Identity()};
  /// [x y z rx ry rz]: position in map, rotation about map axes through the
  /// body origin -- the nav_msgs/Odometry pose covariance convention.
  Matrix6d pose_covariance {Matrix6d::Zero()};

  bool velocity_valid {false};
  Eigen::Vector3d velocity_body {Eigen::Vector3d::Zero()};       // m/s, body frame (REP-145)
  Eigen::Matrix3d velocity_covariance {Eigen::Matrix3d::Identity()};
  bool angular_velocity_valid {false};
  Eigen::Vector3d angular_velocity_body {Eigen::Vector3d::Zero()};

  int raw_points {0};          // in the depth gate
  int source_points {0};       // after downsampling
  int correspondences {0};
  double inlier_fraction {0.0};
  double rms_residual_m {0.0};
  double chi2_per_dof {0.0};
  int iterations {0};
  bool converged {false};
  Degeneracy degeneracy {};
  double threshold_m {0.0};
  std::size_t map_voxels {0};
  bool imu_prior_used {false};
  bool map_reset {false};
  double time_since_good_s {0.0};   // since the last TRACKING frame
};

class ScanOdometry
{
public:
  explicit ScanOdometry(
    const OdometryConfig & cfg,
    const Eigen::Isometry3d & initial_pose = Eigen::Isometry3d::Identity());

  /// One gyro sample, body frame, rad/s, on the same clock as the clouds.
  void addGyro(double t, const Eigen::Vector3d & rate);

  /// One point cloud in the SENSOR frame, stamped t.
  OdometryReport processScan(double t, const std::vector<Eigen::Vector3d> & points_sensor);

  /// The last report, re-judged at `now`: LOST if no cloud for scan_timeout_s.
  OdometryReport watchdog(double now) const;

  /// Forget the map and start again at `pose` (zero covariance).
  void reset(const Eigen::Isometry3d & pose);

  const OdometryReport & last() const {return last_;}
  const VoxelMap & map() const {return map_;}
  const OdometryConfig & config() const {return cfg_;}

  /// Depth gate, noise model, body frame, voxel average. Public for tests.
  std::vector<SourcePoint> preprocess(
    const std::vector<Eigen::Vector3d> & points_sensor, int * in_gate) const;

private:
  OdometryReport lostFrame(
    double t, double dt, const Eigen::Matrix3d & rotation, const std::string & reason,
    OdometryReport rep);
  Matrix6d reportedCovariance() const;
  void resetVelocity();
  void insertScan(const std::vector<SourcePoint> & src, const Eigen::Isometry3d & pose);
  void updateThreshold(const Eigen::Isometry3d & predicted, const Eigen::Isometry3d & estimated,
    const Eigen::Isometry3d & previous);

  OdometryConfig cfg_;
  VoxelMap map_;
  GyroBuffer gyro_;

  bool have_scan_ {false};
  bool map_initialised_ {false};
  double last_t_ {0.0};
  double last_good_t_ {0.0};
  double last_accepted_t_ {0.0};
  Eigen::Isometry3d initial_pose_;
  Eigen::Isometry3d pose_;
  Matrix6d cov_ {Matrix6d::Zero()};
  double distance_ {0.0};   // flown since start/reset, accepted frames
  Eigen::Vector3d velocity_map_ {Eigen::Vector3d::Zero()};
  bool velocity_known_ {false};
  Eigen::Vector3d omega_body_cv_ {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> sticky_dirs_;   // degenerate last frame
  Eigen::Matrix<double, 6, 6> kf_P_ {Eigen::Matrix<double, 6, 6>::Zero()};   // [p; v] filter

  double threshold_sq_sum_ {0.0};
  int threshold_n_ {0};
  double threshold_ {0.0};

  OdometryReport last_;
};

/// The attitude whose roll and pitch put a resting accelerometer's reading
/// (specific force, body frame: +g up when still) on the map's +z, with the
/// given yaw. Lets the map frame start gravity-aligned from the IMU alone.
Eigen::Matrix3d gravityAlignedRotation(const Eigen::Vector3d & specific_force_body, double yaw);

}  // namespace dsim_slam
