// Controller node: turns a planner's trajectory into rotor speeds.
//
//   /drone/state_est   (nav_msgs/Odometry)      estimated pose + velocity  } which of these
//   /drone/truth_exact (nav_msgs/Odometry)      exact physics state        } feeds each field
//   /drone/imu         (sensor_msgs/Imu)        gyro, for body_rate=est    } is per field
//   /drone/trajectory  (dsim_msgs/Trajectory) planner input  <-- YOUR PLANNER
//        |
//        v  SE(3) geometric control + mixer
//   /drone/command/motor_speed (actuator_msgs/Actuators) -> bridged to Gazebo
//   /drone/control_debug       (dsim_msgs/ControlDebug)   telemetry
//
// Also republishes the currently-tracked setpoint on /drone/setpoint so that
// tracking error is measured against what the controller actually chased, not
// against a re-interpolation done elsewhere.
//
// Each of position, velocity, attitude and body rate is taken from the
// estimate or from ground truth independently -- the state.source.* parameters,
// resolved and validated in ONE place (StateSources::fromNames, state_mux.hpp),
// logged once at startup and stamped on every /drone/control_debug message, so
// a run's own telemetry says what it flew on.
//
// The angular rate never comes from a differentiated pose. Gazebo's
// OdometryPublisher is a wheeled-robot plugin that reconstructs velocity by
// differencing successive poses; a quaternion and its negation are the same
// rotation, so once per revolution its reported angular velocity jumped from
// 1.8 to 626 rad/s while the pose stayed perfectly smooth. See
// body_rate_source.hpp for what that did to the torque demand. body_rate=est
// is the gyro; body_rate=truth is the physics engine's own angular velocity
// from the dsim_truth plugin, which does not differentiate anything.
//
// /drone/control_debug carries what every loop of the cascade demanded and what
// the rotors could deliver. It is what lets the viewer draw the control action
// instead of only the resulting motion -- those look identical from position
// alone, whether the stack is working hard or coasting.

#include <chrono>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <actuator_msgs/msg/actuators.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <dsim_msgs/msg/control_debug.hpp>
#include <dsim_msgs/msg/trajectory.hpp>
#include <dsim_msgs/msg/trajectory_setpoint.hpp>

#include <dsim_time/sim_epoch.hpp>
#include "dsim_control/mixer.hpp"
#include "dsim_control/se3_controller.hpp"
#include "dsim_control/state_mux.hpp"
#include "dsim_control/trajectory_buffer.hpp"

using namespace std::chrono_literals;

namespace dsim_control
{

class ControllerNode : public rclcpp::Node
{
public:
  ControllerNode()
  : Node("dsim_controller")
  {
    // ---- parameters -------------------------------------------------------
    const double mass        = declare_parameter("vehicle.mass_kg", 1.5);
    const double gravity     = declare_parameter("vehicle.gravity_m_s2", 9.80665);
    const double arm_length  = declare_parameter("vehicle.arm_length_m", 0.18);
    const double motor_const = declare_parameter("vehicle.motor_constant", 3.29e-06);
    const double moment_const = declare_parameter("vehicle.moment_constant", 0.016);
    const double max_rot_vel = declare_parameter("vehicle.max_rot_velocity", 1500.0);
    const auto inertia = declare_parameter<std::vector<double>>(
      "vehicle.inertia", {0.012, 0.012, 0.023});

    Gains g;
    g.mass = mass;
    g.gravity = gravity;
    g.inertia = Eigen::Vector3d(inertia.at(0), inertia.at(1), inertia.at(2)).asDiagonal();
    auto vec3 = [this](const std::string & name, std::vector<double> def) {
        const auto v = declare_parameter<std::vector<double>>(name, def);
        return Eigen::Vector3d(v.at(0), v.at(1), v.at(2));
      };
    g.kp     = vec3("gains.kp",     {6.0, 6.0, 8.0});
    g.kv     = vec3("gains.kv",     {4.0, 4.0, 5.0});
    g.ki     = vec3("gains.ki",     {1.5, 1.5, 2.0});
    g.max_integral_n = declare_parameter("gains.max_integral_n", 6.0);
    g.kR     = vec3("gains.kR",     {3.0, 3.0, 0.5});
    g.komega = vec3("gains.komega", {0.5, 0.5, 0.1});
    g.max_tilt_rad = declare_parameter("gains.max_tilt_rad", 0.7);
    g.min_thrust_n = declare_parameter("gains.min_thrust_n", 0.5);

    control_rate_hz_ = declare_parameter("control_rate_hz", 250.0);
    odom_timeout_s_  = declare_parameter("odom_timeout_s", 0.5);

    // A quadrotor of this size cannot exceed a few rad/s under its own power:
    // full roll authority is 2.55 N.m against Ixx = 0.015, so about 170 rad/s^2
    // -- it would need most of a second at full deflection to reach 30 rad/s,
    // and the tilt clamp prevents that. Anything past this is the state source
    // malfunctioning, not the vehicle manoeuvring.
    const double max_body_rate = declare_parameter("state.max_body_rate_rad_s", 30.0);
    // Off by default: with a noiseless gyro there is nothing to filter, and a
    // filter that is not needed only adds phase lag to the fastest loop.
    // config/drone.yaml turns it on when the IMU has noise configured.
    const double gyro_tau = declare_parameter("state.gyro_lowpass_tau_s", 0.0);
    imu_timeout_s_ = declare_parameter("imu_timeout_s", 0.5);
    start_armed_     = declare_parameter("start_armed", true);
    armed_ = start_armed_;

    // nav_msgs/Odometry states twist in the child frame (REP-145). Gazebo's
    // OdometryPublisher follows that. Kept as a parameter because getting this
    // wrong is silent at hover and only shows up in banked flight — see
    // scripts/check_twist_frame.py, which measures it instead of assuming.
    const bool twist_in_body_frame = declare_parameter("odom.twist_in_body_frame", true);

    // ---- where each field of the state comes from -------------------------
    // The launch resolves state:= and the src_* overrides into these four; the
    // defaults reproduce state:=est. Validation -- including the refusal of
    // position=slam until it is wired -- is StateSources::fromNames, and a
    // refusal here stops the node rather than flying on a fallback.
    const std::string src_pos = declare_parameter("state.source.position", std::string("est"));
    const std::string src_vel = declare_parameter("state.source.velocity", std::string("est"));
    const std::string src_att = declare_parameter("state.source.attitude", std::string("est"));
    const std::string src_rate = declare_parameter("state.source.body_rate", std::string("est"));
    const std::string truth_topic =
      declare_parameter("state.truth_topic", std::string("/drone/truth_exact"));
    const std::string est_topic =
      declare_parameter("state.est_topic", std::string("/drone/state_est"));
    const std::string slam_topic =
      declare_parameter("state.slam_topic", std::string("/slam/odom"));
    const std::string slam_status_topic =
      declare_parameter("state.slam_status_topic", std::string("/slam/status"));
    // No SLAM pose for this long is treated exactly like SLAM saying LOST.
    slam_timeout_s_ = declare_parameter("state.slam_timeout_s", 0.5);
    StateSources sources;
    try {
      sources = StateSources::fromNames(src_pos, src_vel, src_att, src_rate);
    } catch (const std::invalid_argument & e) {
      RCLCPP_FATAL(get_logger(), "%s", e.what());
      throw;
    }
    mux_ = std::make_unique<StateMux>(sources, twist_in_body_frame, max_body_rate, gyro_tau);

    controller_ = std::make_unique<SE3Controller>(g);
    mixer_ = std::make_unique<Mixer>(arm_length, moment_const, motor_const, max_rot_vel);

    runSelfTest(g);

    // ---- interfaces -------------------------------------------------------
    // Only the sources actually selected are subscribed, so each staleness
    // check guards something the loop is really flying on.
    const auto qos = rclcpp::SensorDataQoS();
    if (sources.uses(Source::Truth)) {
      truth_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        truth_topic, qos,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {onTruth(*msg);});
    }
    if (sources.usesEstOdometry()) {
      est_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        est_topic, qos,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {onEst(*msg);});
    }
    if (sources.uses(Source::Slam)) {
      slam_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        slam_topic, qos,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {onSlam(*msg);});
      slam_status_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticStatus>(
        slam_status_topic, rclcpp::QoS(10),
        [this](diagnostic_msgs::msg::DiagnosticStatus::SharedPtr msg) {onSlamStatus(*msg);});
    }
    if (sources.usesGyro()) {
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/drone/imu", qos,
        [this](sensor_msgs::msg::Imu::SharedPtr msg) {onImu(*msg);});
    }
    source_names_[0] = toString(sources.position);
    source_names_[1] = toString(sources.velocity);
    source_names_[2] = toString(sources.attitude);
    source_names_[3] = toString(sources.body_rate);
    RCLCPP_INFO(
      get_logger(), "state sources: %s  (est = %s%s, truth = %s)",
      sources.describe().c_str(), est_topic.c_str(),
      sources.usesGyro() ? " + /drone/imu gyro" : "", truth_topic.c_str());

    traj_sub_ = create_subscription<dsim_msgs::msg::Trajectory>(
      "/drone/trajectory", rclcpp::QoS(4),
      [this](dsim_msgs::msg::Trajectory::SharedPtr msg) {
        if (msg->points.empty()) {
          RCLCPP_WARN(get_logger(), "ignoring empty trajectory");
          return;
        }
        buffer_.set(*msg);
        // Throttled, and on the STEADY clock. At 5 Hz this line was 99.55% of
        // an eleven-hour log -- 200,741 of 201,647 lines -- and the 903
        // warnings that mattered were the other 0.45%. A log that has to be
        // filtered before it can be read is not a log.
        RCLCPP_INFO_THROTTLE(
          get_logger(), steady_, 10000, "accepting trajectories: %zu points, %.2f s",
          msg->points.size(), buffer_.duration());
      });

    cmd_pub_ = create_publisher<actuator_msgs::msg::Actuators>(
      "/drone/command/motor_speed", rclcpp::QoS(1));
    setpoint_pub_ = create_publisher<dsim_msgs::msg::TrajectorySetpoint>(
      "/drone/setpoint", rclcpp::QoS(10));
    // Best-effort, shallow queue: telemetry must never be able to back-pressure
    // the control loop, and a viewer that misses a frame at 250 Hz loses
    // nothing it could have drawn anyway.
    debug_pub_ = create_publisher<dsim_msgs::msg::ControlDebug>(
      "/drone/control_debug", rclcpp::SensorDataQoS());

    arm_srv_ = create_service<std_srvs::srv::SetBool>(
      "/drone/arm",
      [this](
        const std_srvs::srv::SetBool::Request::SharedPtr req,
        std_srvs::srv::SetBool::Response::SharedPtr res) {
        armed_ = req->data;
        hold_valid_ = false;
        res->success = true;
        res->message = armed_ ? "armed" : "disarmed";
        RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
      });

    // Sim-clock timer, not a wall timer: if Gazebo runs slower than real time
    // a wall timer would tick faster than simulated time and silently change
    // the effective control rate.
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / control_rate_hz_),
      [this] {step();});

    RCLCPP_INFO(
      get_logger(),
      "controller up: %.0f Hz, hover thrust %.2f N, max thrust %.2f N (TWR %.2f)",
      control_rate_hz_, controller_->hoverThrust(), 4.0 * mixer_->maxThrustPerRotor(),
      4.0 * mixer_->maxThrustPerRotor() / controller_->hoverThrust());
  }

private:
  /// Startup checks that CAN fail. Each asserts something that must hold if
  /// the geometry and conventions are right, and is cheap to evaluate.
  void runSelfTest(const Gains & g)
  {
    // 1. The mixer inverse must actually invert, for a spread of wrenches.
    double worst = 0.0;
    const double hover = g.mass * g.gravity;
    const std::vector<Eigen::Vector4d> probes = {
      {hover, 0.0, 0.0, 0.0},
      {hover, 0.5, 0.0, 0.0},
      {hover, 0.0, 0.5, 0.0},
      {hover, 0.0, 0.0, 0.2},
      {1.4 * hover, -0.3, 0.4, -0.1},
    };
    for (const auto & w : probes) {
      worst = std::max(worst, mixer_->roundTripError(w));
    }
    if (worst > 1e-9) {
      RCLCPP_FATAL(get_logger(), "mixer round-trip error %.3e — allocation is wrong", worst);
      throw std::runtime_error("mixer self-test failed");
    }

    // 2. Hovering must be achievable well inside saturation. If hover already
    //    needs >90% of max rotor speed the vehicle cannot manoeuvre, and every
    //    tracking result afterwards would be actuator-limited rather than
    //    control-limited.
    const double hover_per_rotor = g.mass * g.gravity / 4.0;
    const double hover_omega = mixer_->thrustToSpeed(hover_per_rotor);
    const double util = hover_omega / mixer_->maxRotVelocity();
    if (util > 0.9) {
      RCLCPP_FATAL(
        get_logger(), "hover needs %.0f%% of max rotor speed — vehicle cannot fly", util * 100.0);
      throw std::runtime_error("thrust budget self-test failed");
    }

    // 3. A pure hover wrench must produce four equal rotor speeds. Catches
    //    sign or index errors in the allocation columns.
    const Eigen::Vector4d f = mixer_->rotorThrusts({g.mass * g.gravity, 0, 0, 0});
    if ((f.array() - f(0)).abs().maxCoeff() > 1e-9) {
      RCLCPP_FATAL(get_logger(), "hover produced asymmetric rotor thrusts — mixer columns are wrong");
      throw std::runtime_error("hover symmetry self-test failed");
    }

    RCLCPP_INFO(
      get_logger(),
      "self-test passed: mixer residual %.2e, condition %.2f, hover at %.0f%% rotor speed",
      worst, mixer_->conditionNumber(), util * 100.0);
  }

  static OdomSample unpack(const nav_msgs::msg::Odometry & msg)
  {
    OdomSample s;
    s.position = Eigen::Vector3d(
      msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z);
    s.orientation = Eigen::Quaterniond(
      msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
      msg.pose.pose.orientation.y, msg.pose.pose.orientation.z);
    s.twist_linear = Eigen::Vector3d(
      msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z);
    s.twist_angular = Eigen::Vector3d(
      msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z);
    return s;
  }

  /// The exact physics state. Its twist.angular is the engine's own body
  /// angular velocity -- NOT /drone/truth's, which is differentiated from the
  /// pose and blind to the quaternion double cover (body_rate_source.hpp).
  void onTruth(const nav_msgs::msg::Odometry & msg)
  {
    const OdomSample s = unpack(msg);
    const double now = nowSeconds();
    std::lock_guard<std::mutex> lock(state_mutex_);
    const std::size_t before = mux_->rejectedRates();
    mux_->onTruth(s, now);
    warnIfRejected(before, s.twist_angular);
  }

  /// The estimate. Its twist.angular is not read: body_rate=est means the
  /// gyro, gated and filtered here, as it always has.
  void onEst(const nav_msgs::msg::Odometry & msg)
  {
    const OdomSample s = unpack(msg);
    const double now = nowSeconds();
    std::lock_guard<std::mutex> lock(state_mutex_);
    mux_->onEst(s, now);
  }

  /// position=slam: dsim_slam's pose, map frame = world (the launch anchors
  /// the map at the spawn pose).
  void onSlam(const nav_msgs::msg::Odometry & msg)
  {
    const OdomSample s = unpack(msg);
    const double now = nowSeconds();
    std::lock_guard<std::mutex> lock(state_mutex_);
    mux_->onSlam(s, now);
  }

  /// SLAM's own verdict. ERROR = LOST: latch the hold. WARN (degraded) keeps
  /// flying -- SLAM's covariance already says how much to trust it -- but is
  /// logged, so a run that crept through a degenerate stretch says so.
  void onSlamStatus(const diagnostic_msgs::msg::DiagnosticStatus & msg)
  {
    // LOST before SLAM has ever tracked is start-up ("no point cloud received
    // yet"), not a loss: measured, it latched the hold 0.2 s into every run,
    // before the camera's first frame. Only a SLAM that was tracking can lose.
    if (msg.level == diagnostic_msgs::msg::DiagnosticStatus::OK) {
      slam_ever_tracked_ = true;
    }
    if (!slam_ever_tracked_) {return;}
    if (msg.level == diagnostic_msgs::msg::DiagnosticStatus::ERROR) {
      latchSlamHold("SLAM reports LOST: " + msg.message);
    } else if (msg.level == diagnostic_msgs::msg::DiagnosticStatus::WARN) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_, 2000, "SLAM degraded: %s", msg.message.c_str());
    }
  }

  /// The user's rule for losing SLAM: HOLD. Stop following the plan and hover,
  /// for the rest of the run (a reset clears it). Latched rather than resumed
  /// on recovery, because a pose that came back after a gap may have jumped,
  /// and silently picking the plan up again from there would hide that. Not a
  /// motor cut: velocity and attitude come from other sources and are still
  /// good, which is what holding needs.
  ///
  /// The hold is on VELOCITY, not on a point: the position error is zeroed
  /// every step, so the loop brakes to a stop and keeps velocity at zero. A
  /// point hold steers by the position source, and the position source is
  /// the thing that just failed -- measured: holding a point on a drifting
  /// SLAM pose flew the vehicle 84 m away while it was "holding".
  void latchSlamHold(const std::string & why)
  {
    if (slam_lost_) {return;}
    slam_lost_ = true;
    slam_hold_pending_ = true;
    RCLCPP_ERROR(
      get_logger(), "%s -- abandoning the plan and HOLDING position until reset",
      why.c_str());
  }

  /// Body angular rate, from the gyro, for body_rate=est. Everything about why
  /// is in body_rate_source.hpp; the short version is that a gyro measures
  /// rate while a differentiated attitude can jump without the attitude moving.
  void onImu(const sensor_msgs::msg::Imu & msg)
  {
    const Eigen::Vector3d measured(
      msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
    const double now = nowSeconds();
    std::lock_guard<std::mutex> lock(state_mutex_);
    const std::size_t before = mux_->rejectedRates();
    mux_->onGyro(measured, now);
    warnIfRejected(before, measured);
  }

  /// Caller holds state_mutex_.
  void warnIfRejected(std::size_t before, const Eigen::Vector3d & offered)
  {
    if (mux_->rejectedRates() == before) {return;}
    RCLCPP_WARN_THROTTLE(
      get_logger(), steady_, 2000,
      "discarded an implausible body rate of %.1f rad/s (limit %.0f); "
      "%zu rejected so far -- the state source is malfunctioning",
      offered.norm(), mux_->maxRate(), mux_->rejectedRates());
  }

  double nowSeconds() const
  {
    return get_clock()->now().seconds();
  }

  void step()
  {
    const double now = nowSeconds();

    // A reset sends simulated time back to zero, which leaves a trajectory
    // stamped in the old run sitting in the buffer, dated minutes into the
    // future. Sampling it would hold the last commanded point until the
    // planner published again; clearing it drops the controller into its
    // position hold, which is the correct behaviour with no plan.
    if (epoch_.restarted(now)) {
      RCLCPP_INFO(get_logger(), "simulated time went backwards — dropping the stale plan");
      buffer_.clear();
      hold_valid_ = false;
      // The integral is a claim about a force acting NOW. Carrying one across
      // a reset makes the new run open by correcting for a gust that belongs
      // to the old one, and the vehicle leans into nothing for several
      // seconds while it unwinds.
      controller_->resetIntegral();
      have_step_ = false;
      std::lock_guard<std::mutex> lock(state_mutex_);
      // The state in hand is from the OLD run. With the plan gone the next
      // step latches a position hold on whatever state it has, and if no
      // post-reset odometry has arrived yet that is where the vehicle was
      // before the reset -- measured: nine metres away, a 40 degree tilt
      // commanded from the pad for as long as it took the planner to
      // republish, and two collisions before the plan had even started.
      // Forgetting the state idles the motors until a fresh one arrives --
      // from every selected source, not just the first to speak.
      mux_->reset(now);
      slam_lost_ = false;
      slam_hold_pending_ = false;
      slam_ever_tracked_ = false;
    }

    State state;
    StateMux::Health health;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      health = mux_->health(now, odom_timeout_s_, imu_timeout_s_);
      state = mux_->state();
    }

    // No rate feedback means no rotational loop. Commanding anyway would fly
    // on a fabricated zero rate, which looks fine at hover and diverges the
    // moment the vehicle is disturbed.
    if (!armed_ || health.status == StateMux::Status::Waiting) {
      // Same reasoning as a reset: a disarmed vehicle is not being pushed by
      // anything the loop should remember, and the gap while it is disarmed is
      // not time the integrator may accumulate over.
      controller_->resetIntegral();
      have_step_ = false;
      publishIdle();
      return;
    }
    if (health.status == StateMux::Status::Stale) {
      // Any selected source going quiet cuts the motors, and the warning
      // names it: with mixed sources, "no state" alone would not say which.
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_, 2000, "%s — cutting motors", health.why.c_str());
      publishIdle();
      return;
    }

    double slam_age = 0.0;
    if (mux_->sources().position == Source::Slam) {
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        slam_age = mux_->slamAge(now);
      }
      if (slam_ever_tracked_ && slam_age > slam_timeout_s_) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "no SLAM pose for %.2f s", slam_age);
        latchSlamHold(buf);
      }
    }

    Reference ref;
    if (slam_lost_) {
      if (slam_hold_pending_) {
        slam_hold_yaw_ = yawFromQuat(state.orientation);
        slam_hold_pending_ = false;
        controller_->resetIntegral();   // it integrated a position error that no longer exists
      }
      ref.position = state.position;    // zero position error: a velocity hold
      ref.yaw = slam_hold_yaw_;
      RCLCPP_ERROR_THROTTLE(
        get_logger(), steady_, 5000, "HOLDING (zero velocity): SLAM was lost");
    } else if (auto sampled = buffer_.sample(now)) {
      ref = *sampled;
      hold_valid_ = false;
    } else {
      // No trajectory yet: hold the position we were at when we first noticed.
      // Latching the hold point matters — holding "current position" every tick
      // would let the vehicle drift away with no error signal to correct it.
      if (!hold_valid_) {
        hold_position_ = state.position;
        hold_yaw_ = yawFromQuat(state.orientation);
        hold_valid_ = true;
      }
      ref.position = hold_position_;
      ref.yaw = hold_yaw_;
    }

    // Measured, not assumed to be 1/rate: the timer can be late, and a step
    // that took twice as long must integrate twice as much or the integral
    // silently depends on how busy the machine is. A first step, or one after
    // a gap large enough to be a stall rather than jitter, integrates nothing
    // -- there is no evidence about what happened during a gap.
    double dt = 0.0;
    if (have_step_) {
      const double measured = now - last_step_s_;
      if (measured > 0.0 && measured < max_step_dt_s_) {dt = measured;}
    }
    last_step_s_ = now;
    have_step_ = true;

    ControlDebug dbg;
    const Eigen::Vector4d wrench = controller_->compute(state, ref, dt, &dbg);
    // Per-rotor thrusts first, then the speeds derived from them. Calling
    // rotorSpeeds() as well would compute the same clamp a second time and let
    // the commanded speeds and the reported thrusts drift apart under a future
    // edit; this way they are the same numbers by construction.
    const Eigen::Vector4d thrusts = mixer_->rotorThrusts(wrench);
    Eigen::Vector4d speeds;
    for (int i = 0; i < 4; ++i) {
      speeds(i) = mixer_->thrustToSpeed(thrusts(i));
    }

    if (dbg.tilt_clamped) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_, 1000,
        "tilt clamped at %.0f deg — trajectory is more aggressive than the vehicle allows",
        controller_->gains().max_tilt_rad * 180.0 / M_PI);
    }

    actuator_msgs::msg::Actuators cmd;
    cmd.header.stamp = get_clock()->now();
    cmd.velocity = {speeds(0), speeds(1), speeds(2), speeds(3)};
    cmd_pub_->publish(cmd);

    publishDebug(true, wrench, thrusts, dbg, state.velocity);

    dsim_msgs::msg::TrajectorySetpoint sp;
    sp.position.x = ref.position.x();
    sp.position.y = ref.position.y();
    sp.position.z = ref.position.z();
    sp.velocity.x = ref.velocity.x();
    sp.velocity.y = ref.velocity.y();
    sp.velocity.z = ref.velocity.z();
    sp.acceleration.x = ref.acceleration.x();
    sp.acceleration.y = ref.acceleration.y();
    sp.acceleration.z = ref.acceleration.z();
    sp.yaw = ref.yaw;
    sp.yaw_rate = ref.yaw_rate;
    setpoint_pub_->publish(sp);
  }

  void publishIdle()
  {
    actuator_msgs::msg::Actuators cmd;
    cmd.header.stamp = get_clock()->now();
    cmd.velocity = {0.0, 0.0, 0.0, 0.0};
    cmd_pub_->publish(cmd);
    // Publish the debug frame here too, with armed=false. A viewer that kept
    // showing the last commanded thrust after the motors were cut would be
    // drawing a vehicle that no longer exists.
    publishDebug(
      false, Eigen::Vector4d::Zero(), Eigen::Vector4d::Zero(), ControlDebug{},
      Eigen::Vector3d::Zero());
  }

  static geometry_msgs::msg::Vector3 toVec3(const Eigen::Vector3d & v)
  {
    geometry_msgs::msg::Vector3 m;
    m.x = v.x();
    m.y = v.y();
    m.z = v.z();
    return m;
  }

  /// Telemetry for one control step: the demand, what the rotors could actually
  /// deliver, and what each loop was reacting to.
  ///
  /// The realised wrench is recomputed here from the per-rotor thrusts rather
  /// than tracked alongside them, so the two cannot disagree. Consumers may
  /// assert sum(rotor_thrust_n) == realised_thrust_n; that identity is exactly
  /// what breaks if the allocation matrix ever stops inverting.
  void publishDebug(
    bool armed, const Eigen::Vector4d & wrench,
    const Eigen::Vector4d & thrusts, const ControlDebug & dbg,
    const Eigen::Vector3d & state_velocity)
  {
    const Eigen::Vector4d realised = mixer_->wrenchFromThrusts(thrusts);

    dsim_msgs::msg::ControlDebug m;
    m.header.stamp = get_clock()->now();
    m.header.frame_id = "drone/base_link";
    m.armed = armed;

    // Sent on the wire so no consumer needs its own copy of config/drone.yaml.
    // A viewer that hardcoded the mass would draw wrong forces the moment the
    // vehicle changed, and would do it silently.
    m.mass_kg = controller_->gains().mass;
    m.gravity_m_s2 = controller_->gains().gravity;
    m.max_rotor_thrust_n = mixer_->maxThrustPerRotor();

    m.thrust_n = wrench(0);
    m.torque_nm = toVec3(Eigen::Vector3d(wrench.tail<3>()));
    m.realised_thrust_n = realised(0);
    m.realised_torque_nm = toVec3(Eigen::Vector3d(realised.tail<3>()));
    // Saturation is defined by its observable consequence -- the rotors could
    // not deliver what was asked -- not by bookkeeping inside the clamp. That
    // keeps the flag true even for a saturation route added later.
    m.saturated = (realised - wrench).cwiseAbs().maxCoeff() > kWrenchTol;

    for (int i = 0; i < 4; ++i) {
      m.rotor_position[i] = toVec3(mixer_->rotorPosition(i));
      m.rotor_thrust_n[i] = thrusts(i);
      m.rotor_speed_rad_s[i] = mixer_->thrustToSpeed(thrusts(i));
    }

    m.velocity_world = toVec3(state_velocity);
    m.position_error = toVec3(dbg.position_error);
    m.velocity_error = toVec3(dbg.velocity_error);
    m.attitude_error = toVec3(dbg.attitude_error);
    m.body_rate_error = toVec3(dbg.body_rate_error);
    m.desired_force = toVec3(dbg.desired_force);
    m.commanded_tilt_rad = dbg.commanded_tilt_rad;
    m.tilt_clamped = dbg.tilt_clamped;
    m.integral_force_n = toVec3(dbg.integral_force);
    m.integral_held = dbg.integral_held;

    m.source_position = source_names_[0];
    m.source_velocity = source_names_[1];
    m.source_attitude = source_names_[2];
    m.source_body_rate = source_names_[3];

    debug_pub_->publish(m);
  }

  static double yawFromQuat(const Eigen::Quaterniond & q)
  {
    const Eigen::Vector3d x_body = q * Eigen::Vector3d::UnitX();
    return std::atan2(x_body.y(), x_body.x());
  }

  std::unique_ptr<SE3Controller> controller_;
  std::unique_ptr<Mixer> mixer_;
  std::unique_ptr<StateMux> mux_;
  std::string source_names_[4];
  dsim_time::SimEpoch epoch_;
  // Throttling on the SIMULATED clock goes permanently mute after a reset:
  // rcutils compares `now >= last_logged + period`, and a clock that jumped
  // back has to re-reach the old value before another line is allowed through.
  // Every warning in this file is about right now, so all of them are timed by
  // a clock that only moves forwards.
  rclcpp::Clock steady_ {RCL_STEADY_TIME};
  TrajectoryBuffer buffer_;

  mutable std::mutex state_mutex_;

  bool armed_ {true};
  bool have_step_ {false};
  double last_step_s_ {0.0};
  /// Longer than this between control steps and the gap is a stall, not
  /// jitter: 250 Hz is a 4 ms step, so 50 ms is twelve missed steps.
  double max_step_dt_s_ {0.05};
  bool start_armed_ {true};
  double control_rate_hz_ {250.0};
  double odom_timeout_s_ {0.5};
  double imu_timeout_s_ {0.5};

  /// Below this, a demanded-vs-realised difference is numerical noise from the
  /// mixer inverse, not saturation. The self-test asserts the round-trip
  /// residual is under 1e-9, so this sits an order of magnitude above it.
  static constexpr double kWrenchTol = 1e-8;

  double slam_timeout_s_ {0.5};
  bool slam_lost_ {false};
  bool slam_hold_pending_ {false};
  bool slam_ever_tracked_ {false};
  double slam_hold_yaw_ {0.0};

  bool hold_valid_ {false};
  Eigen::Vector3d hold_position_ {Eigen::Vector3d::Zero()};
  double hold_yaw_ {0.0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr truth_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr est_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr slam_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr slam_status_sub_;
  rclcpp::Subscription<dsim_msgs::msg::Trajectory>::SharedPtr traj_sub_;
  rclcpp::Publisher<actuator_msgs::msg::Actuators>::SharedPtr cmd_pub_;
  rclcpp::Publisher<dsim_msgs::msg::TrajectorySetpoint>::SharedPtr setpoint_pub_;
  rclcpp::Publisher<dsim_msgs::msg::ControlDebug>::SharedPtr debug_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr arm_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace dsim_control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<dsim_control::ControllerNode>());
  rclcpp::shutdown();
  return 0;
}
