// A simulated Intel RealSense D435 depth camera.
//
//   /drone/truth  (nav_msgs/Odometry, world -> base_link)
//   referee obstacles file (ROS params), landmarks file (YAML), ground plane
//        |
//        v
//   /camera/camera/depth/image_rect_raw  (sensor_msgs/Image, 16UC1, mm)
//   /camera/camera/depth/camera_info     (sensor_msgs/CameraInfo)
//   /camera/camera/depth/color/points    (sensor_msgs/PointCloud2, x y z)
//   /tf_static: base_link -> camera_link -> camera_depth_frame
//                 -> camera_depth_optical_frame
//
// THE SPLIT. Everything on the output side is shaped like what realsense-ros
// 4.x publishes for a D435 with its default `camera` namespace and name, so a
// consumer written against this node runs unchanged against the real camera.
// Everything on the input side -- the truth pose, the obstacle file, the
// landmarks -- is simulation only and has no counterpart on hardware. Swapping
// in a real D435 means deleting this node and keeping its consumers; the one
// thing that has to come from somewhere else then is base_link -> camera_link,
// which on hardware belongs to the vehicle's URDF (`publish_mount_tf:=false`
// hands it over while this node is still running).
//
// THE CLOCK. A frame is rendered from ONE truth message and stamped with that
// message's stamp, so the pose, the obstacle positions (scenario t = stamp -
// scenario.start_s, the referee's arithmetic) and the stamp all refer to the
// same instant. Frames fall on the camera's rate grid in sim time: the first
// truth message at or past each 1/rate boundary renders one. Truth arrives at
// 250 Hz, so a 30 Hz frame is stamped within 4 ms of its grid point -- and
// that 4 ms is not an error, because the stamp says exactly which pose was
// used. Rendering happens in the subscription callback; while a frame renders
// (milliseconds), truth messages queue in the sensor-data QoS depth and the
// stale ones are dropped, which is what we want: only the newest matters.
//
// A sim reset sends the clock backwards; the rate grid then restarts from the
// new time (dsim_time::SimEpoch), or the camera would go silent until sim
// time caught up with the old run.
//
// NOT HERE: the colour stream, IR images, IMU, depth-to-colour alignment.
// Only the depth topics realsense-ros publishes are simulated.

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <dsim_time/sim_epoch.hpp>

#include "dsim_depthcam/depthcam.hpp"
#include "dsim_depthcam/landmarks.hpp"

namespace dsim_depthcam
{

class DepthCamNode : public rclcpp::Node
{
public:
  DepthCamNode()
  : Node("dsim_depthcam")
  {
    // ---- topics and frames: realsense-ros 4.x names by default -----------
    const auto truth_topic = declare_parameter("truth_topic", std::string("/drone/truth"));
    const auto points_topic = declare_parameter(
      "points_topic", std::string("/camera/camera/depth/color/points"));
    const auto image_topic = declare_parameter(
      "image_topic", std::string("/camera/camera/depth/image_rect_raw"));
    const auto info_topic = declare_parameter(
      "info_topic", std::string("/camera/camera/depth/camera_info"));
    base_frame_ = declare_parameter("base_frame", std::string("base_link"));
    link_frame_ = declare_parameter("camera_link_frame", std::string("camera_link"));
    depth_frame_ = declare_parameter("depth_frame", std::string("camera_depth_frame"));
    optical_frame_ = declare_parameter(
      "optical_frame", std::string("camera_depth_optical_frame"));
    world_frame_ = declare_parameter("world_frame", std::string("world"));
    publish_mount_tf_ = declare_parameter("publish_mount_tf", true);

    // ---- the mount --------------------------------------------------------
    mount_.offset = Eigen::Vector3d(
      declare_parameter("mount.x", 0.10), declare_parameter("mount.y", 0.0),
      declare_parameter("mount.z", 0.0));
    mount_.pitch_rad = declare_parameter("mount.pitch_rad", 0.0);

    // ---- the sensor -------------------------------------------------------
    // Defaults here ARE the model (a D435's numbers), not copies of a
    // generated value, so unlike dsim_sensors this node runs standalone.
    cfg_.width = declare_parameter("width", 424);
    cfg_.height = declare_parameter("height", 240);
    cfg_.hfov_rad = deg2rad(declare_parameter("hfov_deg", 87.0));
    cfg_.vfov_rad = deg2rad(declare_parameter("vfov_deg", 58.0));
    cfg_.min_range_m = declare_parameter("min_range_m", 0.28);
    cfg_.max_range_m = declare_parameter("max_range_m", 10.0);
    cfg_.noise = declare_parameter("noise.enable", true);
    cfg_.baseline_m = declare_parameter("noise.baseline_m", 0.050);
    cfg_.subpixel = declare_parameter("noise.subpixel", 0.08);
    cfg_.dropout = declare_parameter("noise.dropout", 0.0);
    cfg_.validate();
    rate_hz_ = declare_parameter("rate_hz", 30.0);
    if (!(rate_hz_ > 0.0)) {throw std::invalid_argument("rate_hz must be > 0");}
    // 0 = a fresh seed per run. Otherwise each frame's noise is a pure
    // function of (seed, stamp): see frameSeed().
    const int seed = declare_parameter("noise.seed", 1);
    seed_ = seed != 0 ? static_cast<std::uint64_t>(seed) :
      (static_cast<std::uint64_t>(std::random_device{}()) << 32) | std::random_device{}();

    // ---- the world --------------------------------------------------------
    world_.ground = declare_parameter("ground.enable", true);
    world_.ground_z = declare_parameter("ground.z", 0.0);
    const auto landmarks_file = declare_parameter("landmarks_file", std::string(""));
    world_.landmarks = loadLandmarks(landmarks_file);
    world_.scenario_start_s = declare_parameter("scenario.start_s", 0.0);
    world_.scenario_duration_s = declare_parameter("scenario.duration_s", 0.0);
    world_.scenario_loop = declare_parameter("scenario.loop", false);
    loadObstacles();

    // ---- I/O --------------------------------------------------------------
    // Reliable, depth 10 on the outputs: a reliable publisher serves both
    // reliable and best-effort subscribers, a best-effort one only the
    // latter, so this is the choice that connects to everything.
    image_pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic, rclcpp::QoS(10));
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(info_topic, rclcpp::QoS(10));
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(points_topic, rclcpp::QoS(10));
    // The viewer's preview (see previewPoints in depthcam.hpp): a thinned
    // copy of the cloud in the WORLD frame, plus the field-of-view pyramid.
    // Not part of the realsense-ros contract; preview.rate_hz 0 turns it off.
    preview_rate_hz_ = declare_parameter("preview.rate_hz", 10.0);
    preview_max_points_ = static_cast<std::size_t>(
      std::max<int64_t>(0, declare_parameter<int64_t>("preview.max_points", 3000)));
    preview_range_m_ = declare_parameter("preview.frustum_range_m", 4.0);
    if (preview_rate_hz_ > 0.0) {
      preview_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/camera/viz/points_world", rclcpp::QoS(2));
      frustum_pub_ = create_publisher<geometry_msgs::msg::PolygonStamped>(
        "/camera/viz/frustum", rclcpp::QoS(2));
    }
    truth_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      truth_topic, rclcpp::SensorDataQoS(),
      [this](nav_msgs::msg::Odometry::SharedPtr m) {onTruth(*m);});

    publishStaticTf();
    info_template_ = makeCameraInfo();

    const auto k = intrinsics(cfg_);
    RCLCPP_INFO(
      get_logger(),
      "depth camera up: %dx%d @ %.0f Hz, fx %.1f fy %.1f, range %.2f-%.1f m, noise %s "
      "(sigma %.1f mm at 1 m), dropout %.2f; %zu landmarks, %zu obstacles, "
      "scenario t=0 at sim %.1f s, ground %s",
      cfg_.width, cfg_.height, rate_hz_, k.fx, k.fy, cfg_.min_range_m, cfg_.max_range_m,
      cfg_.noise ? "on" : "off", 1000.0 * depthSigma(1.0, cfg_, k), cfg_.dropout,
      world_.landmarks.size(), world_.obstacles.size(), world_.scenario_start_s,
      world_.ground ? "on" : "off");
  }

private:
  /// The referee's obstacles, from the same params file the referee is given
  /// (config/obstacles_<scenario>.yaml, whose `/**:` key reaches any node).
  void loadObstacles()
  {
    const auto names = declare_parameter<std::vector<std::string>>(
      "obstacles.names", std::vector<std::string>{});
    for (const auto & n : names) {
      const auto type = declare_parameter("obstacles." + n + ".type", std::string("sphere"));
      const double radius = declare_parameter("obstacles." + n + ".radius", 1.0);
      const auto flat = declare_parameter<std::vector<double>>(
        "obstacles." + n + ".control_points", std::vector<double>{});
      world_.obstacles.push_back(obstacleFromFlat(n, type, radius, flat));
    }
  }

  void onTruth(const nav_msgs::msg::Odometry & m)
  {
    const double t = rclcpp::Time(m.header.stamp).seconds();
    if (epoch_.restarted(t)) {
      RCLCPP_INFO(get_logger(), "sim time went backwards (%.3f s): restarting the frame grid", t);
      next_frame_s_ = -1.0;
      next_preview_s_ = -1.0;
    }
    if (!m.header.frame_id.empty() && m.header.frame_id != world_frame_ && !warned_frame_) {
      warned_frame_ = true;
      RCLCPP_WARN(
        get_logger(), "truth is in frame '%s', expected '%s'; rendering as if it were",
        m.header.frame_id.c_str(), world_frame_.c_str());
    }
    // First frame: render now and put the next one on the rate grid.
    if (next_frame_s_ >= 0.0 && t < next_frame_s_) {return;}
    next_frame_s_ = (std::floor(t * rate_hz_ + 1e-9) + 1.0) / rate_hz_;

    const auto & p = m.pose.pose.position;
    const auto & q = m.pose.pose.orientation;
    Eigen::Isometry3d T_world_base = Eigen::Isometry3d::Identity();
    T_world_base.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
    T_world_base.translation() = Eigen::Vector3d(p.x, p.y, p.z);

    const rclcpp::Time stamp(m.header.stamp);
    const Eigen::Isometry3d T_world_opt = worldToOptical(T_world_base, mount_);
    const auto frame = render(
      sceneAt(world_, t), T_world_opt, cfg_, frameSeed(seed_, stamp.nanoseconds()));
    publish(frame, m.header.stamp);
    if (preview_pub_ && (next_preview_s_ < 0.0 || t >= next_preview_s_)) {
      next_preview_s_ = t + 1.0 / preview_rate_hz_;
      publishPreview(frame, T_world_opt, m.header.stamp);
    }
  }

  void publishPreview(
    const Frame & f, const Eigen::Isometry3d & T_world_opt,
    const builtin_interfaces::msg::Time & stamp)
  {
    const auto pts = previewPoints(f, T_world_opt, preview_max_points_);
    sensor_msgs::msg::PointCloud2 pc;
    pc.header.stamp = stamp;
    pc.header.frame_id = world_frame_;
    pc.height = 1;
    pc.width = static_cast<std::uint32_t>(pts.size());
    pc.is_bigendian = false;
    pc.is_dense = true;
    pc.point_step = 16;
    pc.row_step = pc.point_step * pc.width;
    for (const auto & [name, offset] : {std::pair<const char *, std::uint32_t>{"x", 0},
        {"y", 4}, {"z", 8}, {"depth", 12}})
    {
      sensor_msgs::msg::PointField fld;
      fld.name = name;
      fld.offset = offset;
      fld.datatype = sensor_msgs::msg::PointField::FLOAT32;
      fld.count = 1;
      pc.fields.push_back(fld);
    }
    pc.data.resize(pc.row_step);
    auto * out = reinterpret_cast<float *>(pc.data.data());
    for (size_t i = 0; i < pts.size(); ++i) {
      out[4 * i + 0] = static_cast<float>(pts[i].world.x());
      out[4 * i + 1] = static_cast<float>(pts[i].world.y());
      out[4 * i + 2] = static_cast<float>(pts[i].world.z());
      out[4 * i + 3] = static_cast<float>(pts[i].depth);
    }
    preview_pub_->publish(pc);

    geometry_msgs::msg::PolygonStamped fr;
    fr.header = pc.header;
    for (const auto & c : frustumCorners(cfg_, T_world_opt, preview_range_m_)) {
      geometry_msgs::msg::Point32 q;
      q.x = static_cast<float>(c.x());
      q.y = static_cast<float>(c.y());
      q.z = static_cast<float>(c.z());
      fr.polygon.points.push_back(q);
    }
    frustum_pub_->publish(fr);
  }

  void publish(const Frame & f, const builtin_interfaces::msg::Time & stamp)
  {
    // Depth image: 16UC1 in depth units (mm), 0 = no return -- the
    // realsense-ros convention for image_rect_raw.
    sensor_msgs::msg::Image img;
    img.header.stamp = stamp;
    img.header.frame_id = optical_frame_;
    img.height = f.height;
    img.width = f.width;
    img.encoding = "16UC1";
    img.is_bigendian = 0;
    img.step = f.width * sizeof(std::uint16_t);
    img.data.resize(img.step * img.height);
    std::memcpy(img.data.data(), f.depth_raw.data(), img.data.size());

    auto info = info_template_;
    info.header = img.header;

    // Point cloud: UNORGANISED (height 1), valid points only, x y z float32
    // with a 12-byte point step. That is realsense-ros 4.x's default
    // (`pointcloud.ordered_pc: false`): an organised cloud would have to carry
    // NaN for every missing return, and consumers written against the real
    // camera do not expect them. No rgb field: the real cloud carries one
    // only when a colour stream textures it, which this does not simulate.
    sensor_msgs::msg::PointCloud2 pc;
    pc.header = img.header;
    pc.height = 1;
    pc.width = static_cast<std::uint32_t>(f.points.size());
    pc.is_bigendian = false;
    pc.is_dense = true;
    pc.point_step = 12;
    pc.row_step = pc.point_step * pc.width;
    for (const auto & [name, offset] : {std::pair<const char *, std::uint32_t>{"x", 0}, {"y", 4}, {"z", 8}}) {
      sensor_msgs::msg::PointField fld;
      fld.name = name;
      fld.offset = offset;
      fld.datatype = sensor_msgs::msg::PointField::FLOAT32;
      fld.count = 1;
      pc.fields.push_back(fld);
    }
    pc.data.resize(pc.row_step);
    auto * out = reinterpret_cast<float *>(pc.data.data());
    for (size_t i = 0; i < f.points.size(); ++i) {
      out[3 * i + 0] = static_cast<float>(f.points[i].x());
      out[3 * i + 1] = static_cast<float>(f.points[i].y());
      out[3 * i + 2] = static_cast<float>(f.points[i].z());
    }

    image_pub_->publish(img);
    info_pub_->publish(info);
    points_pub_->publish(pc);
  }

  sensor_msgs::msg::CameraInfo makeCameraInfo() const
  {
    // Rectified, undistorted: realsense-ros reports plumb_bob with five
    // zeros for the depth stream, and so does this.
    const auto k = intrinsics(cfg_);
    sensor_msgs::msg::CameraInfo ci;
    ci.width = cfg_.width;
    ci.height = cfg_.height;
    ci.distortion_model = "plumb_bob";
    ci.d = {0.0, 0.0, 0.0, 0.0, 0.0};
    ci.k = {k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0};
    ci.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    ci.p = {k.fx, 0.0, k.cx, 0.0, 0.0, k.fy, k.cy, 0.0, 0.0, 0.0, 1.0, 0.0};
    return ci;
  }

  void publishStaticTf()
  {
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    auto tf = [this](const std::string & parent, const std::string & child,
        const Eigen::Vector3d & t, const Eigen::Matrix3d & R) {
        geometry_msgs::msg::TransformStamped m;
        m.header.stamp = now();
        m.header.frame_id = parent;
        m.child_frame_id = child;
        m.transform.translation.x = t.x();
        m.transform.translation.y = t.y();
        m.transform.translation.z = t.z();
        const Eigen::Quaterniond q(R);
        m.transform.rotation.x = q.x();
        m.transform.rotation.y = q.y();
        m.transform.rotation.z = q.z();
        m.transform.rotation.w = q.w();
        return m;
      };
    std::vector<geometry_msgs::msg::TransformStamped> all;
    if (publish_mount_tf_) {
      const auto T = baseToCameraLink(mount_);
      all.push_back(tf(base_frame_, link_frame_, T.translation(), T.linear()));
    }
    // camera_link -> camera_depth_frame is the identity on a D435: the depth
    // imager is the device's reference, as realsense-ros publishes it.
    all.push_back(tf(link_frame_, depth_frame_, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()));
    all.push_back(tf(depth_frame_, optical_frame_, Eigen::Vector3d::Zero(), linkToOpticalRotation()));
    tf_->sendTransform(all);
  }

  CameraConfig cfg_;
  Mount mount_;
  World world_;
  double rate_hz_ {30.0};
  std::uint64_t seed_ {1};
  double next_frame_s_ {-1.0};
  double next_preview_s_ {-1.0};
  double preview_rate_hz_ {10.0};
  std::size_t preview_max_points_ {3000};
  double preview_range_m_ {4.0};
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr preview_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr frustum_pub_;
  bool warned_frame_ {false};
  bool publish_mount_tf_ {true};
  std::string base_frame_, link_frame_, depth_frame_, optical_frame_, world_frame_;
  dsim_time::SimEpoch epoch_;
  sensor_msgs::msg::CameraInfo info_template_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr truth_sub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
};

}  // namespace dsim_depthcam

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<dsim_depthcam::DepthCamNode>());
  rclcpp::shutdown();
  return 0;
}
