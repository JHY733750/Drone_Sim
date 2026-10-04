// Depth-camera SLAM odometry: where the vehicle is, from what a real drone
// carries and nothing else.
//
//   /camera/camera/depth/color/points  (sensor_msgs/PointCloud2)  15-30 Hz, optical frame
//   /drone/imu                         (sensor_msgs/Imu)          250 Hz, base_link
//   TF base_link -> <cloud frame>       static, from the URDF / realsense-ros
//        |
//        v  ScanOdometry (scan_odometry.hpp): scan-to-local-map point-to-plane
//        |  registration, gyro rotation prior, degeneracy-aware
//        v
//   /slam/odom    (nav_msgs/Odometry)                map -> base_link, per cloud
//   /slam/status  (diagnostic_msgs/DiagnosticStatus) per cloud AND on a timer
//   TF map -> base_link                              only if publish_tf (default off)
//
// What this node must never do, because it has to run unchanged on a real
// drone with a real D435: read /drone/truth, a scenario, an obstacle list,
// or anything else a real vehicle does not have. It starts at the origin of
// its own map frame (or at initial_* / ~/set_pose), roll and pitch levelled
// by the accelerometer if gravity_align is on, yaw as given.
//
// The status topic is the contract with the consumer: level OK = TRACKING,
// WARN = DEGRADED, ERROR = LOST, and the message says why. It is also
// published from a timer, so a camera that stops publishing turns into
// LOST "no point cloud for X s" instead of a stale OK. A controller must not
// close a position loop on anything but OK; on WARN the covariance says
// which direction rests on the motion prior.
//
// Time: every stamp is the message's header stamp, so the node runs on
// whatever clock the sensors stamp with (use_sim_time in the sim). A cloud
// stamp that jumps back by more than a second (a simulator reset) restarts
// the map.

#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include "dsim_slam/cloud_conversion.hpp"
#include "dsim_slam/scan_odometry.hpp"

namespace dsim_slam
{

class SlamNode : public rclcpp::Node
{
public:
  SlamNode()
  : Node("dsim_slam")
  {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/camera/camera/depth/color/points");
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/drone/imu");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/slam/odom");
    status_topic_ = declare_parameter<std::string>("status_topic", "/slam/status");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    publish_tf_ = declare_parameter<bool>("publish_tf", false);
    gravity_align_ = declare_parameter<bool>("gravity_align", true);
    initial_xyz_ = declare_parameter<std::vector<double>>("initial_xyz", {0.0, 0.0, 0.0});
    initial_yaw_ = declare_parameter<double>("initial_yaw", 0.0);
    const double status_hz = declare_parameter<double>("status_rate_hz", 5.0);
    gyro_noise_ = declare_parameter<double>("gyro_noise_rad_s", 1e-3);

    // ---- algorithm (defaults are the library's; see scan_odometry.hpp) ----
    auto d = [this](const std::string & n, double v) {return declare_parameter<double>(n, v);};
    auto i = [this](const std::string & n, int v) {return static_cast<int>(declare_parameter<int>(n, v));};
    OdometryConfig c;
    c.min_depth_m = d("min_depth_m", c.min_depth_m);
    c.max_depth_m = d("max_depth_m", c.max_depth_m);
    c.depth_noise_floor_m = d("depth_noise_floor_m", c.depth_noise_floor_m);
    c.depth_noise_quadratic = d("depth_noise_quadratic", c.depth_noise_quadratic);
    c.source_voxel_m = d("source_voxel_m", c.source_voxel_m);
    c.max_registration_sigma_m = d("max_registration_sigma_m", c.max_registration_sigma_m);
    c.map.voxel_size_m = d("map_voxel_m", c.map.voxel_size_m);
    c.map.max_distance_m = d("map_radius_m", c.map.max_distance_m);
    c.map.max_voxel_sigma_m = d("max_voxel_sigma_m", c.map.max_voxel_sigma_m);
    c.registration.max_iterations = i("max_iterations", c.registration.max_iterations);
    c.registration.judge_wide_normals =
      declare_parameter<bool>("judge_wide_normals", c.registration.judge_wide_normals);
    c.initial_threshold_m = d("initial_threshold_m", c.initial_threshold_m);
    c.min_threshold_m = d("min_threshold_m", c.min_threshold_m);
    c.use_imu_prior = declare_parameter<bool>("use_imu_prior", c.use_imu_prior);
    c.velocity_sigma = d("velocity_sigma", c.velocity_sigma);
    c.velocity_correlation_time_s = d("velocity_correlation_time_s", c.velocity_correlation_time_s);
    c.prior_rate_sigma_imu = d("prior_rate_sigma_imu", c.prior_rate_sigma_imu);
    c.prior_rotation_floor_imu = d("prior_rotation_floor_imu", c.prior_rotation_floor_imu);
    c.drift_per_metre = d("drift_per_metre", c.drift_per_metre);
    c.min_source_points = i("min_source_points", c.min_source_points);
    c.min_correspondences = i("min_correspondences", c.min_correspondences);
    c.degraded_inlier_fraction = d("degraded_inlier_fraction", c.degraded_inlier_fraction);
    c.degeneracy_translation_min = d("degeneracy_translation_min", c.degeneracy_translation_min);
    c.degeneracy_rotation_min = d("degeneracy_rotation_min", c.degeneracy_rotation_min);
    c.lost_velocity_sigma = d("lost_velocity_sigma", c.lost_velocity_sigma);
    c.reset_map_after_lost_s = d("reset_map_after_lost_s", c.reset_map_after_lost_s);
    c.scan_timeout_s = d("scan_timeout_s", c.scan_timeout_s);
    cfg_ = c;

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    if (publish_tf_) {tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);}

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    status_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticStatus>(status_topic_, 10);
    // KeepLast(500): two seconds of IMU, so a slow registration (the
    // single-threaded executor runs it in the cloud callback) cannot make
    // the gyro samples for the next interval fall out of the queue.
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS().keep_last(500),
      [this](sensor_msgs::msg::Imu::ConstSharedPtr m) {onImu(*m);});
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS().keep_last(2),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) {onCloud(*m);});
    set_pose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "~/set_pose", 1,
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr m) {onSetPose(*m);});
    status_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(0.5, status_hz)), [this] {onStatusTimer();});

    RCLCPP_INFO(get_logger(), "dsim_slam: %s + %s -> %s, %s (tf %s)", cloud_topic_.c_str(),
      imu_topic_.c_str(), odom_topic_.c_str(), status_topic_.c_str(), publish_tf_ ? "on" : "off");
  }

private:
  static double stampSeconds(const builtin_interfaces::msg::Time & t)
  {
    return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
  }

  void onImu(const sensor_msgs::msg::Imu & m)
  {
    const double t = stampSeconds(m.header.stamp);
    const Eigen::Vector3d w(m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z);
    const Eigen::Vector3d f(m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z);
    std::lock_guard<std::mutex> lock(mutex_);
    imu_.push_back({t, w, f});
    while (imu_.size() > 1000) {imu_.pop_front();}
    if (odo_) {odo_->addGyro(t, w);}
  }

  /// The body -> cloud-frame extrinsic, from TF. Static, so looked up once.
  bool extrinsic(const std::string & cloud_frame, Eigen::Isometry3d * T)
  {
    if (have_extrinsic_ && cloud_frame == extrinsic_frame_) {*T = T_base_sensor_; return true;}
    try {
      const auto tf = tf_buffer_->lookupTransform(base_frame_, cloud_frame, tf2::TimePointZero);
      T_base_sensor_ = tf2::transformToEigen(tf.transform);
      extrinsic_frame_ = cloud_frame;
      have_extrinsic_ = true;
      *T = T_base_sensor_;
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "no TF %s -> %s yet (%s): clouds are dropped until it exists",
        base_frame_.c_str(), cloud_frame.c_str(), e.what());
      return false;
    }
  }

  /// The pose the map starts at: initial_xyz / initial_yaw, with roll and
  /// pitch from the mean accelerometer reading of the last half second if
  /// gravity_align is on and the vehicle was still enough for that to mean
  /// gravity (|f| within 5 % of g, gyro below 0.1 rad/s).
  Eigen::Isometry3d initialPose(double t)
  {
    Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
    P.translation() = Eigen::Vector3d(initial_xyz_.at(0), initial_xyz_.at(1), initial_xyz_.at(2));
    P.linear() = Eigen::AngleAxisd(initial_yaw_, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    if (!gravity_align_) {return P;}
    Eigen::Vector3d fsum = Eigen::Vector3d::Zero();
    int n = 0;
    double wmax = 0.0;
    for (const auto & s : imu_) {
      if (s.t < t - 0.5 || s.t > t) {continue;}
      fsum += s.f;
      wmax = std::max(wmax, s.w.norm());
      ++n;
    }
    if (n < 10) {
      RCLCPP_WARN(get_logger(), "gravity_align: no IMU in the last 0.5 s; map starts unlevelled");
      return P;
    }
    const Eigen::Vector3d f = fsum / n;
    if (std::abs(f.norm() - 9.80665) > 0.5 || wmax > 0.1) {
      RCLCPP_WARN(get_logger(), "gravity_align: vehicle not still (|f| %.2f, |w| %.2f); "
        "map starts unlevelled", f.norm(), wmax);
      return P;
    }
    P.linear() = gravityAlignedRotation(f, initial_yaw_);
    return P;
  }

  void onCloud(const sensor_msgs::msg::PointCloud2 & msg)
  {
    const auto wall0 = std::chrono::steady_clock::now();
    const double t = stampSeconds(msg.header.stamp);
    std::vector<Eigen::Vector3d> pts;
    std::string why;
    if (!cloudToPoints(msg, &pts, &why)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "%s", why.c_str());
      return;
    }
    Eigen::Isometry3d T_bs;
    if (!extrinsic(msg.header.frame_id, &T_bs)) {return;}
    if (msg.header.frame_id.find("optical") == std::string::npos && !warned_frame_) {
      warned_frame_ = true;
      RCLCPP_WARN(get_logger(), "cloud frame '%s' is not an optical frame; the depth noise "
        "model takes z as depth", msg.header.frame_id.c_str());
    }

    OdometryReport r;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!odo_ || T_bs.matrix() != cfg_.T_base_sensor.matrix()) {
        cfg_.T_base_sensor = T_bs;
        odo_ = std::make_unique<ScanOdometry>(cfg_, pending_pose_ ? *pending_pose_ : initialPose(t));
        pending_pose_.reset();
        for (const auto & s : imu_) {odo_->addGyro(s.t, s.w);}
      }
      r = odo_->processScan(t, pts);
    }
    const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall0).count();
    publishOdometry(msg.header.stamp, r);
    publishStatus(r, ms);
  }

  void onSetPose(const geometry_msgs::msg::PoseWithCovarianceStamped & m)
  {
    Eigen::Isometry3d P = Eigen::Isometry3d::Identity();
    const auto & p = m.pose.pose;
    P.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
    P.linear() = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y,
      p.orientation.z).normalized().toRotationMatrix();
    std::lock_guard<std::mutex> lock(mutex_);
    if (odo_) {odo_->reset(P);} else {pending_pose_ = P;}
    RCLCPP_INFO(get_logger(), "map reset to the pose given on ~/set_pose");
  }

  void onStatusTimer()
  {
    OdometryReport r;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!odo_) {
        r.reason = have_extrinsic_ ? "no point cloud received yet" :
          "no point cloud received yet (or no TF to its frame)";
      } else {
        r = odo_->watchdog(now().seconds());
      }
    }
    publishStatus(r, -1.0);
  }

  void publishOdometry(const builtin_interfaces::msg::Time & stamp, const OdometryReport & r)
  {
    nav_msgs::msg::Odometry o;
    o.header.stamp = stamp;
    o.header.frame_id = map_frame_;
    o.child_frame_id = base_frame_;
    const Eigen::Quaterniond q(r.pose.linear());
    o.pose.pose.position.x = r.pose.translation().x();
    o.pose.pose.position.y = r.pose.translation().y();
    o.pose.pose.position.z = r.pose.translation().z();
    o.pose.pose.orientation.w = q.w();
    o.pose.pose.orientation.x = q.x();
    o.pose.pose.orientation.y = q.y();
    o.pose.pose.orientation.z = q.z();
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {o.pose.covariance[6 * a + b] = r.pose_covariance(a, b);}
    }
    // Twist in base_link (REP-145). An unknown velocity is published as zero
    // with a huge variance, never as a confident zero.
    o.twist.twist.linear.x = r.velocity_body.x();
    o.twist.twist.linear.y = r.velocity_body.y();
    o.twist.twist.linear.z = r.velocity_body.z();
    o.twist.twist.angular.x = r.angular_velocity_body.x();
    o.twist.twist.angular.y = r.angular_velocity_body.y();
    o.twist.twist.angular.z = r.angular_velocity_body.z();
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        o.twist.covariance[6 * a + b] = r.velocity_valid ? r.velocity_covariance(a, b) :
          (a == b ? 1e6 : 0.0);
      }
      o.twist.covariance[6 * (3 + a) + 3 + a] =
        r.angular_velocity_valid ? gyro_noise_ * gyro_noise_ : 1e6;
    }
    odom_pub_->publish(o);

    if (tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header = o.header;
      tf.child_frame_id = base_frame_;
      tf.transform.translation.x = o.pose.pose.position.x;
      tf.transform.translation.y = o.pose.pose.position.y;
      tf.transform.translation.z = o.pose.pose.position.z;
      tf.transform.rotation = o.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  void publishStatus(const OdometryReport & r, double compute_ms)
  {
    diagnostic_msgs::msg::DiagnosticStatus s;
    s.name = "dsim_slam";
    s.hardware_id = extrinsic_frame_;
    switch (r.state) {
      case TrackingState::kTracking: s.level = diagnostic_msgs::msg::DiagnosticStatus::OK; break;
      case TrackingState::kDegraded: s.level = diagnostic_msgs::msg::DiagnosticStatus::WARN; break;
      case TrackingState::kLost: s.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR; break;
    }
    s.message = std::string(toString(r.state)) + ": " + r.reason;
    auto kv = [&s](const std::string & k, const std::string & v) {
        diagnostic_msgs::msg::KeyValue e;
        e.key = k;
        e.value = v;
        s.values.push_back(e);
      };
    auto num = [](double v) {return std::to_string(v);};
    const auto & dg = r.degeneracy;
    kv("state", toString(r.state));
    kv("reason", r.reason);
    kv("stamp", num(r.stamp));
    kv("time_since_good_s", num(r.time_since_good_s));
    kv("source_points", std::to_string(r.source_points));
    kv("correspondences", std::to_string(r.correspondences));
    kv("inlier_fraction", num(r.inlier_fraction));
    kv("rms_residual_m", num(r.rms_residual_m));
    kv("chi2_per_dof", num(r.chi2_per_dof));
    kv("translation_min_eig", num(dg.translation_min_eig));
    kv("translation_weak_dir", num(dg.translation_weak_dir.x()) + " " +
      num(dg.translation_weak_dir.y()) + " " + num(dg.translation_weak_dir.z()));
    kv("rotation_min_eig_m2", num(dg.rotation_min_eig));
    kv("position_sigma_m", num(std::sqrt(std::max(0.0, r.pose_covariance.topLeftCorner<3, 3>().trace()))));
    kv("correspondence_threshold_m", num(r.threshold_m));
    kv("iterations", std::to_string(r.iterations));
    kv("map_voxels", std::to_string(r.map_voxels));
    kv("imu_prior_used", r.imu_prior_used ? "true" : "false");
    kv("map_reset", r.map_reset ? "true" : "false");
    if (compute_ms >= 0.0) {kv("compute_ms", num(compute_ms));}
    status_pub_->publish(s);
  }

  struct ImuSample
  {
    double t;
    Eigen::Vector3d w, f;
  };

  std::string cloud_topic_, imu_topic_, odom_topic_, status_topic_, map_frame_, base_frame_;
  bool publish_tf_ {false};
  bool gravity_align_ {true};
  std::vector<double> initial_xyz_;
  double initial_yaw_ {0.0};
  double gyro_noise_ {1e-3};
  OdometryConfig cfg_;

  std::mutex mutex_;
  std::unique_ptr<ScanOdometry> odo_;
  std::deque<ImuSample> imu_;
  std::optional<Eigen::Isometry3d> pending_pose_;
  bool have_extrinsic_ {false};
  bool warned_frame_ {false};
  std::string extrinsic_frame_;
  Eigen::Isometry3d T_base_sensor_ {Eigen::Isometry3d::Identity()};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr set_pose_sub_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};

}  // namespace dsim_slam

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<dsim_slam::SlamNode>());
  rclcpp::shutdown();
  return 0;
}
