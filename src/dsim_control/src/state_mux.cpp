#include "dsim_control/state_mux.hpp"

#include <cstdio>
#include <stdexcept>

namespace dsim_control
{

const char * toString(Source s)
{
  switch (s) {
    case Source::Est: return "est";
    case Source::Truth: return "truth";
    case Source::Slam: return "slam";
  }
  return "?";
}

namespace
{
Source parseOne(const std::string & field, const std::string & name)
{
  if (name == "est") {return Source::Est;}
  if (name == "truth") {return Source::Truth;}
  if (name == "slam") {
    if (field != "position") {
      throw std::invalid_argument(
              "state source '" + field + "=slam' is not a thing: slam provides "
              "position only. " + field + " takes est | truth.");
    }
    return Source::Slam;
  }
  throw std::invalid_argument(
          "state source '" + field + "=" + name + "' is not one of est | truth" +
          (field == "position" ? " | slam" : "") + ".");
}
}  // namespace

StateSources StateSources::fromNames(
  const std::string & position, const std::string & velocity,
  const std::string & attitude, const std::string & body_rate)
{
  StateSources s;
  s.position = parseOne("position", position);
  s.velocity = parseOne("velocity", velocity);
  s.attitude = parseOne("attitude", attitude);
  s.body_rate = parseOne("body_rate", body_rate);
  if (s.position == Source::Slam) {
    // Refused, not degraded. A controller asked to fly on SLAM that quietly
    // flew on something else would make every comparison drawn from that run
    // wrong, and nothing in the output would say so.
    throw std::invalid_argument(
            "position=slam is not wired yet: nothing feeds /slam/odom into the "
            "controller. Fly on position=est or position=truth until it is.");
  }
  return s;
}

bool StateSources::uses(Source s) const
{
  return position == s || velocity == s || attitude == s || body_rate == s;
}

bool StateSources::usesEstOdometry() const
{
  return position == Source::Est || velocity == Source::Est || attitude == Source::Est;
}

std::string StateSources::describe() const
{
  return std::string("position=") + toString(position) +
         " velocity=" + toString(velocity) +
         " attitude=" + toString(attitude) +
         " body_rate=" + toString(body_rate);
}

StateMux::StateMux(
  const StateSources & sources, bool twist_in_body_frame,
  double max_body_rate_rad_s, double gyro_tau_s)
: sources_(sources),
  twist_in_body_frame_(twist_in_body_frame),
  gyro_(max_body_rate_rad_s, gyro_tau_s),
  truth_rate_(max_body_rate_rad_s, 0.0)
{
  if (sources_.uses(Source::Slam)) {
    throw std::invalid_argument("StateMux: slam is not a wired source");
  }
}

void StateMux::onTruth(const OdomSample & m, double now)
{
  const double dt = have_truth_ ? (now - last_truth_s_) : 0.0;
  truth_ = m;
  truth_.orientation = m.orientation.normalized();
  have_truth_ = true;
  last_truth_s_ = now;
  // The sample's own angular velocity -- never one derived from consecutive
  // orientations. With tau = 0 the gate is the only thing applied, and it
  // returns its input unchanged for any rate this airframe can reach.
  truth_rate_.update(m.twist_angular, dt);
}

void StateMux::onEst(const OdomSample & m, double now)
{
  est_ = m;
  est_.orientation = m.orientation.normalized();
  have_est_ = true;
  last_est_s_ = now;
}

void StateMux::onGyro(const Eigen::Vector3d & measured, double now)
{
  const double dt = have_gyro_ ? (now - last_gyro_s_) : 0.0;
  gyro_.update(measured, dt);
  last_gyro_s_ = now;
  have_gyro_ = true;
}

void StateMux::reset(double now)
{
  have_truth_ = false;
  have_est_ = false;
  last_truth_s_ = now;
  last_est_s_ = now;
  last_gyro_s_ = now;
}

StateMux::Health StateMux::health(
  double now, double state_timeout_s, double gyro_timeout_s) const
{
  const bool need_truth = sources_.uses(Source::Truth);
  const bool need_est = sources_.usesEstOdometry();
  const bool need_gyro = sources_.usesGyro();

  if ((need_truth && !have_truth_) || (need_est && !have_est_) ||
    (need_gyro && !(have_gyro_ && gyro_.valid())) ||
    (sources_.body_rate == Source::Truth && !truth_rate_.valid()))
  {
    return {Status::Waiting, ""};
  }

  char buf[160];
  if (need_truth && (now - last_truth_s_) > state_timeout_s) {
    std::snprintf(
      buf, sizeof(buf), "no truth state (/drone/truth_exact) for %.2f s", now - last_truth_s_);
    return {Status::Stale, buf};
  }
  if (need_est && (now - last_est_s_) > state_timeout_s) {
    std::snprintf(
      buf, sizeof(buf), "no estimated state (/drone/state_est) for %.2f s", now - last_est_s_);
    return {Status::Stale, buf};
  }
  if (need_gyro && (now - last_gyro_s_) > gyro_timeout_s) {
    std::snprintf(buf, sizeof(buf), "no gyro for %.2f s", now - last_gyro_s_);
    return {Status::Stale, buf};
  }
  return {Status::Ready, ""};
}

Eigen::Vector3d StateMux::worldVelocity(const OdomSample & m) const
{
  return twist_in_body_frame_ ? Eigen::Vector3d(m.orientation * m.twist_linear) : m.twist_linear;
}

State StateMux::state() const
{
  auto pick = [this](Source s) -> const OdomSample & {
      return s == Source::Truth ? truth_ : est_;
    };
  State s;
  s.position = pick(sources_.position).position;
  s.orientation = pick(sources_.attitude).orientation;
  s.velocity = worldVelocity(pick(sources_.velocity));
  s.angular_rate = sources_.body_rate == Source::Truth ? truth_rate_.value() : gyro_.value();
  return s;
}

std::size_t StateMux::rejectedRates() const
{
  return sources_.body_rate == Source::Truth ? truth_rate_.rejected() : gyro_.rejected();
}

}  // namespace dsim_control
