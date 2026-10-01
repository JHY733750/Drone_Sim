// Gazebo system plugin: the vehicle's exact state, straight out of the physics
// engine, on /drone/truth_exact (gz.msgs.Odometry -> nav_msgs/Odometry via the
// bridge).
//
// Why this exists alongside /drone/truth. /drone/truth comes from Gazebo's
// OdometryPublisher, and its pose is exact but its TWIST is not:
//
//   * twist.linear and twist.angular are finite differences of the pose,
//     taken every physics step and passed through a rolling mean whose window
//     is hard-coded to 10 samples (libgz-sim8-odometry-publisher-system
//     8.15.0: four `mov $0xa,%esi` before the RollingMean::SetWindowSize
//     calls in its constructor; there is no SDF parameter for it). At the
//     1 ms step every world here uses, that is a 10 ms boxcar: the published
//     velocity is the average over the last 10 ms, about 5 ms old.
//   * twist.angular is differenced from quaternions, and once per revolution
//     jumped to 626 rad/s on a sign flip (README, "A bug worth knowing about").
//
// "Truth means truth" -- the controller's truth sources must carry no sensor
// rate, no delay and no filtering. Nothing differentiated can promise that, so
// this plugin does not differentiate. It reads the link's world pose and its
// world linear and angular velocity components, which the physics system
// writes on the same step, and rotates the velocities into the body frame.
//
// SDF parameters (all optional):
//   <link_name>          link to report, default base_link
//   <topic>              gz topic, default /drone/truth_exact
//   <publish_frequency>  Hz; 0 (the default) publishes every physics step, so a
//                        consumer's most recent sample is never older than one
//                        step. Anything slower is a sample-and-hold, i.e. a
//                        delay of up to one period.
//   <frame_id> / <child_frame_id>  default world / base_link

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/odometry.pb.h>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/Conversions.hh>
#include <gz/transport/Node.hh>

#include "dsim_truth/body_twist.hpp"

namespace dsim_truth
{

class TruthPublisher
  : public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPostUpdate
{
public:
  void Configure(
    const gz::sim::Entity & entity,
    const std::shared_ptr<const sdf::Element> & sdf,
    gz::sim::EntityComponentManager & ecm,
    gz::sim::EventManager &) override
  {
    const gz::sim::Model model(entity);
    if (!model.Valid(ecm)) {
      gzerr << "[dsim_truth] must be attached to a model; publishing nothing\n";
      return;
    }
    const std::string link_name = sdf->Get<std::string>("link_name", "base_link").first;
    topic_ = sdf->Get<std::string>("topic", "/drone/truth_exact").first;
    frame_id_ = sdf->Get<std::string>("frame_id", "world").first;
    child_frame_id_ = sdf->Get<std::string>("child_frame_id", "base_link").first;
    const double hz = sdf->Get<double>("publish_frequency", 0.0).first;
    period_ = hz > 0.0 ?
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / hz)) :
      std::chrono::steady_clock::duration::zero();

    link_ = gz::sim::Link(model.LinkByName(ecm, link_name));
    if (!link_.Valid(ecm)) {
      // Loud, not silent: a truth source that never publishes leaves the
      // controller idling with "no truth for N s", and this line is the cause.
      gzerr << "[dsim_truth] model has no link named '" << link_name
            << "'; publishing nothing\n";
      return;
    }
    // Gazebo does not compute link velocities unless asked; without this
    // WorldLinearVelocity() returns nullopt forever.
    link_.EnableVelocityChecks(ecm, true);
    pub_ = node_.Advertise<gz::msgs::Odometry>(topic_);
    configured_ = true;
    gzmsg << "[dsim_truth] publishing link '" << link_name << "' on " << topic_
          << (hz > 0.0 ? "" : " every physics step") << "\n";
  }

  void PostUpdate(
    const gz::sim::UpdateInfo & info,
    const gz::sim::EntityComponentManager & ecm) override
  {
    if (!configured_ || info.paused) {return;}
    // A reset sends simTime backwards; restart the throttle with it.
    if (info.simTime < last_pub_) {have_pub_ = false;}
    if (period_.count() > 0 && have_pub_ && info.simTime - last_pub_ < period_) {return;}

    // gz::sim::worldPose composes the Pose components up the tree -- the same
    // call OdometryPublisher makes for /drone/truth, so the two poses agree.
    // Link::WorldPose() would return nullopt unless something had created a
    // WorldPose component on the link, which nothing here guarantees.
    const std::optional<gz::math::Pose3d> pose = gz::sim::worldPose(link_.Entity(), ecm);
    const auto v = link_.WorldLinearVelocity(ecm);
    const auto w = link_.WorldAngularVelocity(ecm);
    // The first step after EnableVelocityChecks has no velocity yet. Publish
    // nothing rather than a fabricated zero.
    if (!v || !w) {return;}

    const Eigen::Quaterniond q(pose->Rot().W(), pose->Rot().X(), pose->Rot().Y(), pose->Rot().Z());
    const BodyTwist t = bodyTwistFromWorld(
      q, Eigen::Vector3d(v->X(), v->Y(), v->Z()), Eigen::Vector3d(w->X(), w->Y(), w->Z()));

    gz::msgs::Odometry msg;
    msg.mutable_header()->mutable_stamp()->CopyFrom(
      gz::sim::convert<gz::msgs::Time>(info.simTime));
    auto * f = msg.mutable_header()->add_data();
    f->set_key("frame_id");
    f->add_value(frame_id_);
    auto * c = msg.mutable_header()->add_data();
    c->set_key("child_frame_id");
    c->add_value(child_frame_id_);
    gz::msgs::Set(msg.mutable_pose(), *pose);
    gz::msgs::Set(msg.mutable_twist()->mutable_linear(),
      gz::math::Vector3d(t.linear.x(), t.linear.y(), t.linear.z()));
    gz::msgs::Set(msg.mutable_twist()->mutable_angular(),
      gz::math::Vector3d(t.angular.x(), t.angular.y(), t.angular.z()));
    pub_.Publish(msg);
    last_pub_ = info.simTime;
    have_pub_ = true;
  }

private:
  gz::sim::Link link_;
  gz::transport::Node node_;
  gz::transport::Node::Publisher pub_;
  std::string topic_;
  std::string frame_id_;
  std::string child_frame_id_;
  std::chrono::steady_clock::duration period_ {std::chrono::steady_clock::duration::zero()};
  std::chrono::steady_clock::duration last_pub_ {std::chrono::steady_clock::duration::zero()};
  bool have_pub_ {false};
  bool configured_ {false};
};

}  // namespace dsim_truth

GZ_ADD_PLUGIN(
  dsim_truth::TruthPublisher,
  gz::sim::System,
  dsim_truth::TruthPublisher::ISystemConfigure,
  dsim_truth::TruthPublisher::ISystemPostUpdate)
