// These tests are written as invariants: statements that cannot be false if
// the implementation is correct. Each comment says what would make it fail.

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "dsim_control/body_rate_source.hpp"
#include "dsim_control/mixer.hpp"
#include "dsim_control/se3_controller.hpp"
#include "dsim_control/angles.hpp"
#include "dsim_control/state_mux.hpp"

using namespace dsim_control;

namespace
{
// Mirrors PARAMS in scripts/gen_assets.py. These tests check the CODE, not the
// shipped config -- whether the config itself describes a flyable vehicle is
// asserted by check_flyable() in that generator, which runs on every
// regeneration and refuses to emit an airframe that cannot hover.
constexpr double kArm = 0.20;
constexpr double kMoment = 0.016;
constexpr double kMotor = 9.0e-06;
constexpr double kMaxRot = 1000.0;
constexpr double kMass = 1.5;
constexpr double kG = 9.80665;

Mixer makeMixer() {return Mixer(kArm, kMoment, kMotor, kMaxRot);}

State hoverState()
{
  State s;
  s.position = Eigen::Vector3d(0, 0, 1.5);
  return s;
}
}  // namespace

// FAILS IF: the allocation matrix and its inverse disagree — i.e. the geometry
// or spin signs were edited in one place and not the other.
TEST(Mixer, InverseActuallyInverts)
{
  const auto m = makeMixer();
  const std::vector<Eigen::Vector4d> probes = {
    {14.7, 0, 0, 0}, {14.7, 0.5, 0, 0}, {14.7, 0, 0.5, 0},
    {14.7, 0, 0, 0.2}, {20.0, -0.3, 0.4, -0.1}, {5.0, 1.0, -1.0, 0.05},
  };
  for (const auto & w : probes) {
    EXPECT_LT(m.roundTripError(w), 1e-9) << "wrench " << w.transpose();
  }
}

// FAILS IF: a rotor column is duplicated or mis-signed. Hover is the one
// wrench whose answer is known without any derivation: all four equal.
TEST(Mixer, HoverIsSymmetric)
{
  const auto m = makeMixer();
  const Eigen::Vector4d f = m.rotorThrusts({kMass * kG, 0, 0, 0});
  const double expected = kMass * kG / 4.0;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(f(i), expected, 1e-9) << "rotor " << i;
  }
}

// FAILS IF: the roll axis is swapped with pitch, or the y-offsets have the
// wrong sign. Body frame is FLU, so +tau_x must lift the +y (left) side:
// rotors 0 and 3 sit at y = +d.
TEST(Mixer, PositiveRollLoadsTheLeftRotors)
{
  const auto m = makeMixer();
  const Eigen::Vector4d f = m.rotorThrusts({kMass * kG, 0.4, 0, 0});
  EXPECT_GT(f(0), f(2));   // front-left  > front-right
  EXPECT_GT(f(3), f(1));   // back-left   > back-right
}

// FAILS IF: pitch sign is inverted. +tau_y lifts the -x (rear) side, so the
// rear rotors (1, 3) must work harder.
TEST(Mixer, PositivePitchLoadsTheRearRotors)
{
  const auto m = makeMixer();
  const Eigen::Vector4d f = m.rotorThrusts({kMass * kG, 0, 0.4, 0});
  EXPECT_GT(f(1), f(0));
  EXPECT_GT(f(3), f(2));
}

// FAILS IF: the reaction-torque sign disagrees with Gazebo's motor plugin.
// A rotor's reaction torque opposes its own spin, so +tau_z (CCW about +z)
// comes from spinning the CW rotors (2, 3) harder.
TEST(Mixer, PositiveYawLoadsTheClockwiseRotors)
{
  const auto m = makeMixer();
  const Eigen::Vector4d f = m.rotorThrusts({kMass * kG, 0, 0, 0.15});
  EXPECT_GT(f(2), f(0));
  EXPECT_GT(f(3), f(1));
}

// FAILS IF: thrust/speed conversion is not a true inverse pair.
TEST(Mixer, ThrustSpeedRoundTrip)
{
  const auto m = makeMixer();
  for (double thrust : {0.5, 1.0, 3.678, 7.0}) {
    EXPECT_NEAR(m.speedToThrust(m.thrustToSpeed(thrust)), thrust, 1e-9);
  }
}

// FAILS IF: the vehicle cannot hover with margin. This is a modelling check,
// not a code check: it catches a parameter edit that makes the drone unflyable.
TEST(Mixer, ThrustBudgetAllowsHoverWithMargin)
{
  const auto m = makeMixer();
  const double twr = 4.0 * m.maxThrustPerRotor() / (kMass * kG);
  EXPECT_GT(twr, 1.5) << "thrust-to-weight too low to manoeuvre";
}

// FAILS IF: gravity compensation is missing or scaled wrong. At rest on the
// reference, the only correct answer is exactly hover thrust and zero torque.
TEST(SE3Controller, PerfectHoverCommandsExactlyWeight)
{
  SE3Controller c{Gains{}};
  const State s = hoverState();
  Reference r;
  r.position = s.position;

  const Eigen::Vector4d w = c.compute(s, r, 0.0);
  EXPECT_NEAR(w(0), kMass * kG, 1e-6);
  EXPECT_NEAR(w(1), 0.0, 1e-9);
  EXPECT_NEAR(w(2), 0.0, 1e-9);
  EXPECT_NEAR(w(3), 0.0, 1e-9);
}

// FAILS IF: the position error sign is flipped — the classic bug that makes a
// drone accelerate away from its setpoint. Below the reference must mean
// more thrust than weight.
TEST(SE3Controller, BelowSetpointCommandsMoreThanWeight)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  s.position.z() -= 0.5;
  Reference r;
  r.position = Eigen::Vector3d(0, 0, 1.5);

  EXPECT_GT(c.compute(s, r, 0.0)(0), kMass * kG);
}

// FAILS IF: the horizontal error sign is flipped. A setpoint ahead in +x must
// produce a desired force with a +x component (nose-down pitch).
TEST(SE3Controller, SetpointAheadTiltsForward)
{
  SE3Controller c{Gains{}};
  const State s = hoverState();
  Reference r;
  r.position = s.position + Eigen::Vector3d(1.0, 0.0, 0.0);

  ControlDebug dbg;
  c.compute(s, r, 0.0, &dbg);
  EXPECT_GT(dbg.desired_force.x(), 0.0);
  EXPECT_NEAR(dbg.desired_force.y(), 0.0, 1e-9);
}

// FAILS IF: the tilt clamp does not engage. A huge error must not produce an
// arbitrarily large bank angle.
TEST(SE3Controller, TiltIsClamped)
{
  Gains g;
  g.max_tilt_rad = 0.5;
  SE3Controller c{g};
  const State s = hoverState();
  Reference r;
  r.position = s.position + Eigen::Vector3d(100.0, 0.0, 0.0);

  ControlDebug dbg;
  c.compute(s, r, 0.0, &dbg);
  EXPECT_TRUE(dbg.tilt_clamped);
  EXPECT_LE(dbg.commanded_tilt_rad, 0.5 + 1e-9);
}

// FAILS IF: acceleration feedforward is dropped. Two states identical except
// for reference acceleration must produce different thrust.
TEST(SE3Controller, AccelerationFeedforwardIsUsed)
{
  SE3Controller c{Gains{}};
  const State s = hoverState();
  Reference r0;
  r0.position = s.position;
  Reference r1 = r0;
  r1.acceleration = Eigen::Vector3d(0, 0, 2.0);

  EXPECT_NEAR(c.compute(s, r1, 0.0)(0) - c.compute(s, r0, 0.0)(0), kMass * 2.0, 1e-6);
}

// FAILS IF: yaw interpolation goes the long way around the circle. Crossing
// the +/-pi seam must take the 20-degree path, not the 340-degree one.
TEST(TrajectoryBuffer, YawInterpolationTakesTheShortArc)
{
  const double a = 3.0;            // ~172 deg
  const double b = -3.0;           // ~-172 deg
  const double mid = interpolateAngle(a, b, 0.5);
  // Shortest arc passes through +/-pi, not through zero.
  EXPECT_GT(std::abs(mid), 3.0);
}

// FAILS IF: the arm length is wrong, or the allocation does not actually scale
// torque with geometry. The previous tests could not catch this: they compare
// rotors against each other, and the round-trip check uses the same matrix on
// both sides, so a wrong arm length is self-consistent and invisible.
//
// This asserts against physics derived independently of the implementation:
//   tau_x = d * ((f0 + f3) - (f1 + f2))   with d = L / sqrt(2)
// so the left/right thrust differential MUST equal tau_x / d exactly.
TEST(Mixer, RollTorqueMatchesArmLengthAnalytically)
{
  const auto m = makeMixer();
  const double d = kArm / std::sqrt(2.0);
  const double tau_x = 0.4;

  const Eigen::Vector4d f = m.rotorThrusts({kMass * kG, tau_x, 0, 0});
  const double differential = (f(0) + f(3)) - (f(1) + f(2));
  EXPECT_NEAR(differential, tau_x / d, 1e-9);
}

// Same idea for yaw, against the moment constant instead of the arm length:
//   tau_z = c * ((f2 + f3) - (f0 + f1))
TEST(Mixer, YawTorqueMatchesMomentConstantAnalytically)
{
  const auto m = makeMixer();
  const double tau_z = 0.15;

  const Eigen::Vector4d f = m.rotorThrusts({kMass * kG, 0, 0, tau_z});
  const double differential = (f(2) + f(3)) - (f(0) + f(1));
  EXPECT_NEAR(differential, tau_z / kMoment, 1e-9);
}

// FAILS IF: geometry is hard-coded rather than read from the constructor.
// Physical impossibility check: a longer arm gives more leverage, so the SAME
// torque cannot require MORE thrust differential. This cannot be false for any
// correct implementation, whatever the actual numbers are.
TEST(Mixer, LongerArmNeedsLessThrustDifferential)
{
  const Mixer shortArm(0.15, kMoment, kMotor, kMaxRot);
  const Mixer longArm(0.30, kMoment, kMotor, kMaxRot);
  const Eigen::Vector4d w{kMass * kG, 0.4, 0, 0};

  auto differential = [&w](const Mixer & m) {
      const Eigen::Vector4d f = m.rotorThrusts(w);
      return (f(0) + f(3)) - (f(1) + f(2));
    };
  EXPECT_LT(differential(longArm), differential(shortArm));
}

// FAILS IF: total thrust leaks into the torque channels. Commanding pure torque
// on top of hover must not change the total thrust produced.
TEST(Mixer, TorqueDemandDoesNotChangeTotalThrust)
{
  const auto m = makeMixer();
  const double total = kMass * kG;
  for (const auto & w : std::vector<Eigen::Vector4d>{
      {total, 0.3, 0, 0}, {total, 0, 0.3, 0}, {total, 0, 0, 0.1}})
  {
    const Eigen::Vector4d f = m.rotorThrusts(w);
    EXPECT_NEAR(f.sum(), total, 1e-9);
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// ---------------------------------------------------------------------------
// Invariants behind /drone/control_debug.
//
// The viewer draws force arrows straight from that message. A field wired to
// the wrong quantity, or a sign flip, would draw a confident picture of
// something that is not happening -- worse than drawing nothing, because it
// sends you looking for the bug in the wrong place. These are the checks that
// would catch it.
// ---------------------------------------------------------------------------

// FAILS IF: the per-rotor thrusts stop summing to the thrust that was asked
// for. ControlDebug publishes both numbers; this is the identity a consumer is
// invited to assert, so it had better hold.
TEST(Mixer, RotorThrustsSumToDemandedThrust)
{
  const auto m = makeMixer();
  const std::vector<Eigen::Vector4d> probes = {
    {kMass * kG, 0.0, 0.0, 0.0},
    {kMass * kG, 0.4, -0.3, 0.05},
    {1.6 * kMass * kG, -0.2, 0.2, -0.04},
  };
  for (const auto & w : probes) {
    const Eigen::Vector4d f = m.rotorThrusts(w);
    EXPECT_NEAR(f.sum(), w(0), 1e-9) << "wrench " << w.transpose();
  }
}

// FAILS IF: an unsaturated demand comes back changed. This is what makes the
// `saturated` flag meaningful -- it is defined as "realised differs from
// demanded", so if the round trip were lossy the flag would be stuck true and
// the viewer would permanently claim the vehicle is out of authority.
TEST(Mixer, RealisedWrenchEqualsDemandWhenUnsaturated)
{
  const auto m = makeMixer();
  const Eigen::Vector4d w(kMass * kG, 0.3, -0.2, 0.03);
  const Eigen::Vector4d realised = m.wrenchFromThrusts(m.rotorThrusts(w));
  EXPECT_LT((realised - w).cwiseAbs().maxCoeff(), 1e-9);
}

// FAILS IF: an impossible demand is reported as delivered. Asking for more
// thrust than four rotors can produce MUST show up as a shortfall, otherwise
// the saturation flag can never fire and the HUD would show a vehicle happily
// tracking a trajectory it cannot fly.
TEST(Mixer, SaturationIsVisibleInTheRealisedWrench)
{
  const auto m = makeMixer();
  const double beyond_max = 8.0 * m.maxThrustPerRotor();
  const Eigen::Vector4d w(beyond_max, 0.0, 0.0, 0.0);
  const Eigen::Vector4d realised = m.wrenchFromThrusts(m.rotorThrusts(w));
  EXPECT_LT(realised(0), w(0) - 1.0);
  EXPECT_NEAR(realised(0), 4.0 * m.maxThrustPerRotor(), 1e-9);
}

// FAILS IF: the debug errors are wired backwards (reference - state instead of
// state - reference), or to the wrong axis. A sign flip here would draw every
// force arrow pointing the wrong way while the numbers still looked plausible.
TEST(SE3Controller, DebugErrorsHaveTheDocumentedSign)
{
  Gains g;
  g.mass = kMass;
  g.gravity = kG;
  SE3Controller c(g);

  State s = hoverState();
  s.position = Eigen::Vector3d(0.0, 0.0, 2.0);      // one metre ABOVE the ref
  s.velocity = Eigen::Vector3d(0.5, 0.0, 0.0);      // moving +x, ref is still

  Reference ref;
  ref.position = Eigen::Vector3d(0.0, 0.0, 1.0);

  ControlDebug dbg;
  c.compute(s, ref, 0.0, &dbg);

  EXPECT_NEAR(dbg.position_error.z(), +1.0, 1e-12);   // state - reference
  EXPECT_NEAR(dbg.velocity_error.x(), +0.5, 1e-12);
}

// FAILS IF: perfect hover reports a non-zero error anywhere, or a desired force
// that is not straight up at exactly one weight. This is the anchor for every
// overlay: at rest on the setpoint the four rotor arrows must be equal, the
// thrust arrow must exactly cancel the gravity arrow, and the attitude ghost
// must sit on top of the actual axis. If this test passes and the picture still
// looks wrong, the bug is in the viewer, not the controller.
TEST(SE3Controller, PerfectHoverProducesZeroErrorsAndVerticalForce)
{
  Gains g;
  g.mass = kMass;
  g.gravity = kG;
  SE3Controller c(g);

  const State s = hoverState();
  Reference ref;
  ref.position = s.position;

  ControlDebug dbg;
  const Eigen::Vector4d wrench = c.compute(s, ref, 0.0, &dbg);

  EXPECT_LT(dbg.position_error.norm(), 1e-12);
  EXPECT_LT(dbg.velocity_error.norm(), 1e-12);
  EXPECT_LT(dbg.attitude_error.norm(), 1e-12);
  EXPECT_LT(dbg.body_rate_error.norm(), 1e-12);
  EXPECT_NEAR(dbg.desired_force.x(), 0.0, 1e-12);
  EXPECT_NEAR(dbg.desired_force.y(), 0.0, 1e-12);
  EXPECT_NEAR(dbg.desired_force.z(), kMass * kG, 1e-9);
  EXPECT_NEAR(dbg.commanded_tilt_rad, 0.0, 1e-12);
  EXPECT_FALSE(dbg.tilt_clamped);
  EXPECT_NEAR(wrench(0), kMass * kG, 1e-9);
}

// FAILS IF: the yaw-rate feedforward is dropped or double-counted. A vehicle
// already yawing at exactly the commanded yaw rate has NO rate error to
// correct; if it reported one, the torque arrow would show the controller
// fighting a manoeuvre it asked for.
TEST(SE3Controller, MatchedYawRateLeavesNoBodyRateError)
{
  Gains g;
  g.mass = kMass;
  g.gravity = kG;
  SE3Controller c(g);

  State s = hoverState();
  s.angular_rate = Eigen::Vector3d(0.0, 0.0, 0.7);   // yawing, level

  Reference ref;
  ref.position = s.position;
  ref.yaw_rate = 0.7;                                // ...exactly as commanded

  ControlDebug dbg;
  c.compute(s, ref, 0.0, &dbg);
  EXPECT_NEAR(dbg.body_rate_error.z(), 0.0, 1e-12);

  // ...and the control must notice when they DISagree, or the test above would
  // pass for a controller that always reports zero.
  ref.yaw_rate = 0.0;
  c.compute(s, ref, 0.0, &dbg);
  EXPECT_NEAR(dbg.body_rate_error.z(), 0.7, 1e-12);
}

// FAILS IF: the rotor positions the mixer reports stop matching the allocation
// matrix it actually uses. Telemetry says "rotor i sits here and is producing
// this much thrust", and a viewer draws an arrow from that position; if the two
// were derived separately, the arrow could land on the wrong arm and make a
// correct controller look broken.
TEST(Mixer, RotorPositionsGenerateTheAllocationMatrix)
{
  const auto m = makeMixer();
  const Eigen::Matrix4d & A = m.allocation();
  for (int i = 0; i < Mixer::rotorCount(); ++i) {
    const Eigen::Vector3d r = m.rotorPosition(i);
    // Roll torque is +y * f, pitch torque is -x * f. Those ARE the columns.
    EXPECT_NEAR(A(1, i), r.y(), 1e-12) << "rotor " << i;
    EXPECT_NEAR(A(2, i), -r.x(), 1e-12) << "rotor " << i;
    // ...and every hub is one arm length from the centre.
    EXPECT_NEAR(r.norm(), kArm, 1e-6) << "rotor " << i;
  }
  // The four hubs must be distinct, or two arrows would stack on one arm.
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      EXPECT_GT((m.rotorPosition(i) - m.rotorPosition(j)).norm(), 1e-6)
        << "rotors " << i << " and " << j << " are in the same place";
    }
  }
}

// ---------------------------------------------------------------------------
// BodyRateSource: the gate that would have caught the 626 rad/s.
//
// The simulator's OdometryPublisher reported a body rate of 626 rad/s once per
// revolution -- a quaternion sign flip differentiated blind to the double
// cover. The pose stayed smooth, the flight looked fine, and the controller
// quietly demanded 111 N.m against an airframe good for 2.55. These are the
// checks that make that impossible to repeat.
// ---------------------------------------------------------------------------

// FAILS IF: an impossible body rate is accepted. This is the exact number the
// simulator produced, and the exact bug: 626 rad/s is 100 revolutions per
// second, which this airframe cannot approach at 170 rad/s^2 of angular
// authority.
TEST(BodyRateSource, RejectsThePhysicallyImpossibleRateTheSimActuallyProduced)
{
  BodyRateSource src(30.0, 0.0);
  const Eigen::Vector3d sane(0.1, -0.2, 1.795);
  EXPECT_EQ(src.update(sane, 0.004), sane);
  EXPECT_TRUE(src.valid());

  // The real sample: (+9.9, +186.7, -598.0), magnitude 626.5.
  const Eigen::Vector3d garbage(9.897, 186.709, -597.973);
  ASSERT_GT(garbage.norm(), 600.0);
  const Eigen::Vector3d out = src.update(garbage, 0.004);

  EXPECT_EQ(src.rejected(), 1u);
  // Holds the last good value: zeroing would claim the vehicle had stopped
  // rotating, which is a different lie and provokes its own wrong correction.
  EXPECT_EQ(out, sane);
}

// FAILS IF: a NaN slips through. NaN fails every comparison, so a bare
// magnitude test would pass it straight into the rotational loop, and one NaN
// in the torque poisons all four rotor commands with no error anywhere.
TEST(BodyRateSource, RejectsNonFiniteSamples)
{
  BodyRateSource src(30.0, 0.0);
  const Eigen::Vector3d sane(0.0, 0.0, 1.0);
  src.update(sane, 0.004);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(src.update(Eigen::Vector3d(nan, 0.0, 0.0), 0.004), sane);
  EXPECT_EQ(
    src.update(Eigen::Vector3d(std::numeric_limits<double>::infinity(), 0, 0), 0.004),
    sane);
  EXPECT_EQ(src.rejected(), 2u);
}

// FAILS IF: the gate is so tight it rejects real flight. A quadrotor tracking
// an aggressive trajectory genuinely reaches several rad/s; a limit that
// clipped those would silently degrade the rotational loop, which is a worse
// bug than the one being fixed because it looks like poor tuning.
TEST(BodyRateSource, AcceptsRatesARealVehicleActuallyReaches)
{
  BodyRateSource src(30.0, 0.0);
  for (const double w : {0.5, 2.0, 5.0, 10.0, 25.0}) {
    const Eigen::Vector3d r(0.0, 0.0, w);
    EXPECT_EQ(src.update(r, 0.004), r) << "rejected a plausible " << w << " rad/s";
  }
  EXPECT_EQ(src.rejected(), 0u);
}

// FAILS IF: the very first sample is filtered instead of adopted. Ramping from
// zero towards the true rate would fabricate a spin-up that never happened,
// and on a vehicle already rotating at arming time that is a real transient.
TEST(BodyRateSource, AdoptsTheFirstGoodSampleOutright)
{
  BodyRateSource src(30.0, 0.05);
  const Eigen::Vector3d first(0.0, 0.0, 2.0);
  EXPECT_EQ(src.update(first, 0.004), first);
}

// FAILS IF: the low-pass is not the time constant it claims. After one tau of
// a unit step a first-order filter must have covered 1 - 1/e = 63.2%; a filter
// whose alpha ignored dt would move at a rate that changed with the control
// rate, so the same gains would behave differently at 250 Hz and 500 Hz.
TEST(BodyRateSource, LowPassHonoursItsTimeConstant)
{
  const double tau = 0.05;
  const double dt = 0.001;
  BodyRateSource src(30.0, tau);
  src.update(Eigen::Vector3d::Zero(), dt);              // establish the origin
  const Eigen::Vector3d step(0.0, 0.0, 1.0);
  for (int i = 0; i < static_cast<int>(tau / dt); ++i) {
    src.update(step, dt);
  }
  EXPECT_NEAR(src.value().z(), 1.0 - std::exp(-1.0), 0.02);

  // ...and with filtering disabled it must pass the sample through untouched,
  // or the "off" setting would still add lag to the fastest loop.
  BodyRateSource unfiltered(30.0, 0.0);
  unfiltered.update(Eigen::Vector3d::Zero(), dt);
  EXPECT_EQ(unfiltered.update(step, dt), step);
}

// FAILS IF: a run of bad samples ever produces a usable-looking output. If the
// state source breaks permanently, the controller must keep seeing the same
// stale value and the rejection count must keep climbing, so the log says what
// happened instead of the vehicle silently flying on a frozen rate.
TEST(BodyRateSource, KeepsCountingWhileTheSourceStaysBroken)
{
  BodyRateSource src(30.0, 0.0);
  const Eigen::Vector3d sane(0.0, 0.0, 1.5);
  src.update(sane, 0.004);
  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(src.update(Eigen::Vector3d(0.0, 0.0, 626.5), 0.004), sane);
  }
  EXPECT_EQ(src.rejected(), 50u);
}

// ---------------------------------------------------------------------------
// The integral term.
//
// This stack was deliberately PD: with an exact model and exact state there is
// no steady-state error for an integrator to remove. External forces are the
// case that argument excluded, and the tests below are about the three ways an
// integrator goes wrong -- wrong sign, unbounded growth, and accumulating
// while the actuator is already saturated.
// ---------------------------------------------------------------------------
namespace
{
/// Closed-loop point-mass simulation of the translational loop.
///
/// The controller returns a body wrench, so the honest way to close the loop
/// would be to simulate attitude too. This instead drives a point mass with
/// `desired_force`, which is what the attitude loop exists to realise, and is
/// a good approximation while the tilt stays small -- the disturbances below
/// are a few newtons against a 14.7 N weight, so the commanded tilt stays
/// under 20 degrees. The approximation is stated because it is the one thing
/// that could make this test agree with a controller that would not fly.
///
/// Returns the position error at the end of the run.
double settleUnderDisturbance(
  SE3Controller & c, const Eigen::Vector3d & disturbance_n,
  double seconds, double dt = 0.004)
{
  State s = hoverState();
  Reference r;
  r.position = s.position;
  ControlDebug dbg;
  for (int i = 0; i < static_cast<int>(seconds / dt); ++i) {
    c.compute(s, r, dt, &dbg);
    // The vehicle feels what the loop asked for, its own weight, and the
    // disturbance. desired_force already contains the gravity compensation.
    const Eigen::Vector3d accel =
      (dbg.desired_force - Eigen::Vector3d(0, 0, kMass * kG) + disturbance_n) / kMass;
    s.velocity += accel * dt;
    s.position += s.velocity * dt;
  }
  return (s.position - r.position).norm();
}
}  // namespace

// FAILS IF: there is no integrator, or it is too weak to matter. This is the
// whole reason the term exists: a PD loop answers a constant force with a
// standing offset of F/kp -- 0.83 m for 5 N against kp = 6 -- and holds it for
// as long as the force lasts. Both halves are asserted, because "the error is
// small" proves nothing without the number it would otherwise have been.
TEST(SE3Controller, AConstantDisturbanceIsDrivenOutByTheIntegrator)
{
  const Eigen::Vector3d gust(5.0, 0.0, 0.0);

  Gains pd{};
  pd.ki.setZero();
  SE3Controller no_integral{pd};
  const double pd_error = settleUnderDisturbance(no_integral, gust, 20.0);
  EXPECT_NEAR(pd_error, 5.0 / pd.kp.x(), 0.05)
    << "a PD loop must sit at exactly F/kp; it is not the controller under "
       "test here, it is the baseline the integrator has to beat";

  SE3Controller with_integral{Gains{}};
  const double pid_error = settleUnderDisturbance(with_integral, gust, 20.0);
  EXPECT_LT(pid_error, 0.05)
    << "the integrator left " << pid_error << " m of standing error";
  EXPECT_LT(pid_error, pd_error / 10.0);
}

// FAILS IF: someone expects the integrator to fix a tracking error that turns
// with the vehicle. It cannot, and the reason is worth pinning down rather
// than rediscovering: the integral accumulates in the WORLD frame, so a
// disturbance that rotates through a full circle integrates to nothing. This
// is not a defect, it is the boundary of what the term can do -- measured on
// the real thing, adding this integrator removed a held gust completely (83 cm
// of standing error to 0.4 cm) and left the circle's constant radial offset
// where it was.
TEST(SE3Controller, ARotatingDisturbanceIsNotSomethingTheIntegratorCanCancel)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  Reference r;
  r.position = s.position;
  ControlDebug dbg;

  // A 5 N force turning once every 6 seconds -- the lap period of the demo
  // circle -- applied as an error the loop sees.
  const double dt = 0.004, period = 6.0;
  double worst = 0.0;
  for (int i = 0; i < static_cast<int>(60.0 / dt); ++i) {
    const double phase = 2.0 * M_PI * (i * dt) / period;
    s.position = r.position +
      Eigen::Vector3d(0.05 * std::cos(phase), 0.05 * std::sin(phase), 0.0);
    c.compute(s, r, dt, &dbg);
    if (i * dt > 12.0) {worst = std::max(worst, dbg.integral_force.norm());}
  }
  // ki * e / omega is the most a rotating error of this size can ever build:
  // 1.5 * 0.05 / (2*pi/6) = 0.072 N, against the 6 N clamp. It never
  // accumulates, whatever the loop is doing.
  EXPECT_LT(worst, 0.15)
    << "the integral reached " << worst << " N against a rotating error it "
       "cannot help with";
}

// FAILS IF: the integral has the wrong sign -- which does not look like a bug
// at first, it looks like a slow instability. Above the reference on +x, the
// integral force must push back along -x, the same direction the proportional
// term already pushes.
TEST(SE3Controller, TheIntegralPushesBackTowardsTheReference)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  s.position.x() += 0.2;                    // 20 cm past the reference
  Reference r;
  r.position = hoverState().position;

  ControlDebug dbg;
  for (int i = 0; i < 250; ++i) {c.compute(s, r, 0.004, &dbg);}   // 1 s held

  EXPECT_LT(dbg.integral_force.x(), 0.0);
  // ki * e * t = 1.5 * 0.2 * 1.0 = 0.3 N, and it opposes the error.
  EXPECT_NEAR(dbg.integral_force.x(), -0.3, 1e-3);
  EXPECT_NEAR(dbg.integral_force.y(), 0.0, 1e-12);
}

// FAILS IF: the integral can grow without bound. An integrator asking for more
// lateral force than the tilt clamp will ever deliver is not merely useless --
// every newton of it has to be unwound before the vehicle can come back, which
// turns a brief excursion into a long one.
TEST(SE3Controller, TheIntegralIsClampedInNewtons)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  // Half a metre, held. Chosen so the tilt clamp does NOT engage: 0.5 m
  // against kp = 6 is 3 N of lateral demand, 11.5 degrees against a 40 degree
  // clamp. The first version of this test used a 20 m error and measured
  // nothing -- the clamp engaged on the first step, the anti-windup freeze
  // below stopped the integrator at 0.12 N, and the newton clamp was never
  // reached. The two mechanisms have to be tested apart or each one hides
  // whether the other works.
  s.position.x() += 0.5;
  Reference r;
  r.position = hoverState().position;

  ControlDebug dbg;
  for (int i = 0; i < 15000; ++i) {c.compute(s, r, 0.004, &dbg);}  // 60 s

  const Gains g{};
  ASSERT_FALSE(dbg.tilt_clamped) << "this test must exercise the clamp, "
                                    "not the anti-windup freeze";
  // Unclamped this would reach ki * e * t = 1.5 * 0.5 * 60 = 45 N.
  EXPECT_NEAR(dbg.integral_force.x(), -g.max_integral_n, 1e-9);
  EXPECT_LE(c.integral().cwiseAbs().maxCoeff(), g.max_integral_n + 1e-9);
}

// FAILS IF: the integrator keeps accumulating while the demand is already
// clamped. That is the textbook windup failure, and it is invisible in normal
// flight: it only shows up as a vehicle that overshoots badly coming out of a
// manoeuvre it was never going to make.
TEST(SE3Controller, TheIntegratorHoldsWhileTheTiltIsClamped)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  s.position.x() += 5.0;         // far enough that the tilt clamp engages
  Reference r;
  r.position = hoverState().position;

  ControlDebug dbg;
  c.compute(s, r, 0.004, &dbg);
  ASSERT_TRUE(dbg.tilt_clamped) << "this test needs the clamp to engage";

  // One more step to pick up the clamp from the previous one, then measure.
  c.compute(s, r, 0.004, &dbg);
  const Eigen::Vector3d held = dbg.integral_force;
  for (int i = 0; i < 500; ++i) {c.compute(s, r, 0.004, &dbg);}
  EXPECT_TRUE(dbg.integral_held);
  EXPECT_NEAR((dbg.integral_force - held).norm(), 0.0, 1e-12)
    << "the integral grew by " << (dbg.integral_force - held).norm()
    << " N while the demand was clamped";
}

// FAILS IF: a reset leaves the disturbance estimate behind. An integral is a
// claim about a force acting NOW; carried across a run boundary it makes the
// new run open by leaning into a gust that is no longer there.
TEST(SE3Controller, ResettingForgetsTheDisturbance)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  s.position.x() += 0.5;
  Reference r;
  r.position = hoverState().position;

  ControlDebug dbg;
  for (int i = 0; i < 500; ++i) {c.compute(s, r, 0.004, &dbg);}
  ASSERT_GT(dbg.integral_force.norm(), 0.1);

  c.resetIntegral();
  c.compute(s, r, 0.0, &dbg);
  EXPECT_NEAR(dbg.integral_force.norm(), 0.0, 1e-12);
}

// FAILS IF: the integral follows the number of STEPS rather than the time they
// covered. Hardcoding the nominal 4 ms passes every test whose loop happens to
// run at 250 Hz -- which was all of them, and the mutation harness said so:
// "integral accumulates without regard to the step length" SURVIVED. The node
// measures dt because the timer can be late, and an integrator that ignores
// that is one whose behaviour depends on how busy the machine is.
TEST(SE3Controller, TheIntegralFollowsElapsedTimeNotStepCount)
{
  State s = hoverState();
  s.position.x() += 0.2;
  Reference r;
  r.position = hoverState().position;

  SE3Controller fast{Gains{}};
  SE3Controller slow{Gains{}};
  ControlDebug fast_dbg, slow_dbg;
  for (int i = 0; i < 250; ++i) {fast.compute(s, r, 0.004, &fast_dbg);}   // 1 s
  for (int i = 0; i < 100; ++i) {slow.compute(s, r, 0.010, &slow_dbg);}   // 1 s

  // Same second of the same error: the same accumulated force, whatever rate
  // the loop happened to run at.
  EXPECT_NEAR(fast_dbg.integral_force.x(), slow_dbg.integral_force.x(), 1e-9);
  EXPECT_NEAR(fast_dbg.integral_force.x(), -0.3, 1e-3);
}

// FAILS IF: a zero or missing dt still integrates. The node passes dt = 0 for
// the first step and after a stall, where there is no evidence about what
// happened during the gap; integrating over an unmeasured interval would put a
// step into the demand at exactly the moment the stack is least healthy.
TEST(SE3Controller, NoTimeMeansNoIntegration)
{
  SE3Controller c{Gains{}};
  State s = hoverState();
  s.position.x() += 0.5;
  Reference r;
  r.position = hoverState().position;

  ControlDebug dbg;
  for (int i = 0; i < 1000; ++i) {c.compute(s, r, 0.0, &dbg);}
  EXPECT_NEAR(dbg.integral_force.norm(), 0.0, 1e-12);
}

// FAILS IF: the shipped gains are outside the stability bound. With an
// integrator the translational loop is m*s^3 + kv*s^2 + kp*s + ki, and
// Routh-Hurwitz requires ki < kp*kv/m on every axis. This is a check on the
// DEFAULTS rather than on the code: raising ki is the obvious thing to try
// when a disturbance is rejected too slowly, and the loop goes unstable
// without anything in the maths ever looking wrong.
TEST(SE3Controller, TheShippedIntegralGainsAreInsideTheStabilityBound)
{
  const Gains g{};
  for (int i = 0; i < 3; ++i) {
    const double bound = g.kp(i) * g.kv(i) / g.mass;
    EXPECT_LT(g.ki(i), bound)
      << "axis " << i << ": ki " << g.ki(i) << " against the Routh-Hurwitz "
      << "bound kp*kv/m = " << bound;
    // ...and comfortably inside it, not scraping the edge, because the bound
    // itself assumes a point mass with no actuator lag.
    EXPECT_LT(g.ki(i), 0.5 * bound);
  }
}

// ---------------------------------------------------------------------------
// State sources: each of position, velocity, attitude and body rate is taken
// from the estimate or from ground truth independently. The user's rule is
// "truth means truth": a field on truth is the truth input, unprocessed, even
// while every sensor is noisy; a field on est is the estimate; and choosing
// one never leaks into another.
// ---------------------------------------------------------------------------
namespace
{
constexpr double kMaxRate = 30.0;
constexpr double kGyroTau = 0.05;   // the est path's low-pass, ON, so a test
                                    // can tell whether truth went through it

Eigen::Quaterniond ypr(double yaw, double pitch, double roll)
{
  return Eigen::Quaterniond(
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
}

/// A truth sample and an estimate of it with every field visibly wrong --
/// noise "injected" large enough that no field of one could pass for the
/// other. Attitudes differ by 35 deg of yaw, so a velocity rotated with the
/// wrong attitude is off by metres per second, not by rounding.
OdomSample truthSample()
{
  OdomSample t;
  t.position = Eigen::Vector3d(1.25, -0.5, 1.5);
  t.orientation = ypr(0.7, 0.12, -0.08);
  t.twist_linear = Eigen::Vector3d(1.8, 0.3, -0.1);    // body frame
  t.twist_angular = Eigen::Vector3d(0.2, -0.1, 1.8);   // body frame
  return t;
}

OdomSample estSample()
{
  OdomSample e;
  e.position = Eigen::Vector3d(1.31, -0.46, 1.43);
  e.orientation = ypr(0.7 + 0.61, 0.10, -0.05);
  e.twist_linear = Eigen::Vector3d(1.7, 0.42, -0.02);
  e.twist_angular = Eigen::Vector3d(9.0, 9.0, 9.0);    // never read
  return e;
}

const Eigen::Vector3d kGyro(0.23, -0.14, 1.77);   // what the noisy gyro says

StateSources pick(Source p, Source v, Source a, Source r)
{
  StateSources s;
  s.position = p;
  s.velocity = v;
  s.attitude = a;
  s.body_rate = r;
  return s;
}

/// Feed one of everything, as the node would, and return the mux.
StateMux fed(const StateSources & s, const OdomSample & truth, const OdomSample & est)
{
  StateMux m(s, true, kMaxRate, kGyroTau);
  m.onGyro(kGyro, 0.0);
  m.onTruth(truth, 0.0);
  m.onEst(est, 0.0);
  return m;
}

double quatDistance(const Eigen::Quaterniond & a, const Eigen::Quaterniond & b)
{
  return std::min((a.coeffs() - b.coeffs()).norm(), (a.coeffs() + b.coeffs()).norm());
}
}  // namespace

// FAILS IF: any truth field is processed on its way through -- filtered,
// gated into a held value, rotated with the wrong attitude, or simply taken
// from the estimate. Every sensor here is wrong (estSample, kGyro) and the
// gyro low-pass is ON; with all four on truth the output must still be the
// truth input to rounding. The body rate is fed twice with a step between, so
// a truth rate passed through the 50 ms low-pass would land 7% of the way to
// the second value instead of on it.
TEST(StateMux, AllTruthIsTheTruthInputExactlyWhileTheSensorsAreNoisy)
{
  const auto all = pick(Source::Truth, Source::Truth, Source::Truth, Source::Truth);
  const OdomSample t = truthSample();
  StateMux m = fed(all, t, estSample());
  OdomSample t2 = t;
  t2.twist_angular = Eigen::Vector3d(-0.4, 0.3, 2.6);
  m.onTruth(t2, 0.004);
  ASSERT_EQ(m.health(0.004, 0.5, 0.5).status, StateMux::Status::Ready);

  const State s = m.state();
  EXPECT_LT((s.position - t.position).norm(), 1e-12);
  EXPECT_LT(quatDistance(s.orientation, t.orientation), 1e-12);
  EXPECT_LT((s.velocity - t.orientation * t.twist_linear).norm(), 1e-12);
  EXPECT_LT((s.angular_rate - t2.twist_angular).norm(), 1e-12);
}

// FAILS IF: an est field is quietly served from truth (which would make every
// "est vs truth" comparison compare truth with itself), or the est body rate
// stops being the gyro through its configured low-pass.
TEST(StateMux, AllEstIsTheEstimateAndNotTheTruth)
{
  const auto all = pick(Source::Est, Source::Est, Source::Est, Source::Est);
  const OdomSample t = truthSample(), e = estSample();
  StateMux m = fed(all, t, e);
  ASSERT_EQ(m.health(0.0, 0.5, 0.5).status, StateMux::Status::Ready);

  const State s = m.state();
  EXPECT_LT((s.position - e.position).norm(), 1e-12);
  EXPECT_LT(quatDistance(s.orientation, e.orientation), 1e-12);
  EXPECT_LT((s.velocity - e.orientation * e.twist_linear).norm(), 1e-12);
  EXPECT_LT((s.angular_rate - kGyro).norm(), 1e-12);   // first sample adopted

  EXPECT_GT((s.position - t.position).norm(), 0.05);
  EXPECT_GT((s.velocity - t.orientation * t.twist_linear).norm(), 0.1);
  EXPECT_GT(quatDistance(s.orientation, t.orientation), 0.1);

  // Second gyro sample 4 ms later: the est path is low-passed (tau 50 ms), so
  // it moves 7.4% of the way, not all of it.
  const Eigen::Vector3d g2 = kGyro + Eigen::Vector3d(0.0, 0.0, 1.0);
  m.onGyro(g2, 0.004);
  EXPECT_NEAR(m.state().angular_rate.z(), kGyro.z() + 0.004 / (kGyroTau + 0.004), 1e-12);
}

// FAILS IF: one field's choice leaks into another, in any of the 16
// combinations. The classic leak is the velocity: both Odometry messages carry
// it in their OWN body frame, so attitude=truth + velocity=est must rotate the
// estimate's body velocity with the ESTIMATED attitude (giving the estimator's
// world velocity), not with the true one -- that would be 35 deg off here,
// about 1.1 m/s, and a velocity that is neither the estimate nor the truth.
TEST(StateMux, EachFieldComesFromItsOwnSourceInEveryCombination)
{
  const OdomSample t = truthSample(), e = estSample();
  const Source both[2] = {Source::Est, Source::Truth};
  for (Source p : both) {
    for (Source v : both) {
      for (Source a : both) {
        for (Source r : both) {
          const auto sel = pick(p, v, a, r);
          StateMux m = fed(sel, t, e);
          ASSERT_EQ(m.health(0.0, 0.5, 0.5).status, StateMux::Status::Ready) << sel.describe();
          const State s = m.state();
          const OdomSample & ps = p == Source::Truth ? t : e;
          const OdomSample & vs = v == Source::Truth ? t : e;
          const OdomSample & as = a == Source::Truth ? t : e;
          const Eigen::Vector3d rate = r == Source::Truth ? t.twist_angular : kGyro;
          EXPECT_LT((s.position - ps.position).norm(), 1e-12) << sel.describe();
          EXPECT_LT((s.velocity - vs.orientation * vs.twist_linear).norm(), 1e-12)
            << sel.describe();
          EXPECT_LT(quatDistance(s.orientation, as.orientation), 1e-12) << sel.describe();
          EXPECT_LT((s.angular_rate - rate).norm(), 1e-12) << sel.describe();
        }
      }
    }
  }
  // And the leak above is not a rounding-sized distinction: the two candidate
  // world velocities for attitude=truth + velocity=est really differ.
  EXPECT_GT((e.orientation * e.twist_linear - t.orientation * e.twist_linear).norm(), 0.5);
}

// FAILS IF: the truth body rate is derived from the pose. Gazebo's
// OdometryPublisher did exactly that and reported 626 rad/s once per
// revolution when the quaternion changed sheet. Here the vehicle yaws steadily
// at 1.8 rad/s through yaw = pi, and the published quaternion is negated half
// way -- the same rotation, a different sign. The rate handed to the loop must
// be the sample's own angular velocity at every step: no spike, no hold.
TEST(StateMux, TruthBodyRateIsUntouchedAcrossAQuaternionSignFlip)
{
  const auto sel = pick(Source::Truth, Source::Truth, Source::Truth, Source::Truth);
  StateMux m(sel, true, kMaxRate, kGyroTau);
  const double dt = 0.004, rate = 1.8;
  double worst = 0.0;
  for (int k = 0; k < 40; ++k) {
    OdomSample s = truthSample();
    const double yaw = M_PI - 20 * dt * rate + k * dt * rate;
    s.orientation = ypr(yaw, 0.0, 0.0);
    if (k >= 20) {s.orientation.coeffs() = -s.orientation.coeffs();}
    s.twist_angular = Eigen::Vector3d(0.0, 0.0, rate);
    m.onTruth(s, k * dt);
    worst = std::max(worst, (m.state().angular_rate - s.twist_angular).norm());
  }
  EXPECT_EQ(worst, 0.0);
  EXPECT_EQ(m.rejectedRates(), 0u);
}

// FAILS IF: the gyro can reach the loop while body_rate=truth. A noisy gyro
// sample arriving after the truth sample must not overwrite it.
TEST(StateMux, TheGyroIsIgnoredWhenBodyRateIsTruth)
{
  const auto sel = pick(Source::Est, Source::Est, Source::Est, Source::Truth);
  const OdomSample t = truthSample();
  StateMux m = fed(sel, t, estSample());
  m.onGyro(Eigen::Vector3d(5.0, -5.0, 5.0), 0.002);
  EXPECT_LT((m.state().angular_rate - t.twist_angular).norm(), 1e-12);
}

// FAILS IF: slam is accepted for a field it cannot provide, or refused for
// the one it can.
TEST(StateSources, SlamIsAPositionSourceOnly)
{
  EXPECT_EQ(StateSources::fromNames("slam", "est", "est", "est").position, Source::Slam);
  EXPECT_THROW(StateSources::fromNames("est", "slam", "est", "est"), std::invalid_argument);
  EXPECT_THROW(StateSources::fromNames("est", "est", "slam", "est"), std::invalid_argument);
  EXPECT_THROW(StateSources::fromNames("est", "est", "est", "slam"), std::invalid_argument);
  // Typos are refused, not defaulted.
  EXPECT_THROW(StateSources::fromNames("truht", "est", "est", "est"), std::invalid_argument);
  EXPECT_THROW(StateSources::fromNames("est", "est", "", "est"), std::invalid_argument);
  // The two wired names parse to what they say, per field.
  const auto s = StateSources::fromNames("truth", "est", "truth", "est");
  EXPECT_EQ(s.position, Source::Truth);
  EXPECT_EQ(s.velocity, Source::Est);
  EXPECT_EQ(s.attitude, Source::Truth);
  EXPECT_EQ(s.body_rate, Source::Est);
  EXPECT_EQ(s.describe(), "position=truth velocity=est attitude=truth body_rate=est");
}

// FAILS IF: a selected source can go quiet without the loop noticing. With
// mixed sources there are up to three streams, and each one the selection
// actually uses must hold the vehicle on the ground when it stops -- while a
// stream the selection does NOT use must not (an all-truth run has no gyro
// subscription at all, and must still fly).
TEST(StateMux, EverySelectedSourceIsWatchedAndNoOtherIs)
{
  // velocity on est, the rest on truth: needs truth and the estimate, no gyro.
  const auto sel = pick(Source::Truth, Source::Est, Source::Truth, Source::Truth);
  StateMux m(sel, true, kMaxRate, kGyroTau);
  EXPECT_EQ(m.health(0.0, 0.5, 0.5).status, StateMux::Status::Waiting);
  m.onTruth(truthSample(), 0.0);
  EXPECT_EQ(m.health(0.0, 0.5, 0.5).status, StateMux::Status::Waiting)
    << "flew with no estimate in hand while velocity=est";
  m.onEst(estSample(), 0.0);
  EXPECT_EQ(m.health(0.1, 0.5, 0.5).status, StateMux::Status::Ready)
    << "a gyro that is not selected was required";

  // The estimate stops; truth keeps coming.
  for (int k = 1; k <= 200; ++k) {m.onTruth(truthSample(), k * 0.004);}
  const auto h = m.health(0.8, 0.5, 0.5);
  EXPECT_EQ(h.status, StateMux::Status::Stale);
  EXPECT_NE(h.why.find("state_est"), std::string::npos) << h.why;

  // And the other way round: truth stops while the estimate keeps coming.
  StateMux n(sel, true, kMaxRate, kGyroTau);
  n.onTruth(truthSample(), 0.0);
  for (int k = 0; k <= 200; ++k) {n.onEst(estSample(), k * 0.004);}
  const auto h2 = n.health(0.8, 0.5, 0.5);
  EXPECT_EQ(h2.status, StateMux::Status::Stale);
  EXPECT_NE(h2.why.find("truth"), std::string::npos) << h2.why;

  // body_rate=est: the gyro is watched with its own timeout.
  const auto est = pick(Source::Est, Source::Est, Source::Est, Source::Est);
  StateMux g(est, true, kMaxRate, kGyroTau);
  g.onGyro(kGyro, 0.0);
  for (int k = 0; k <= 200; ++k) {g.onEst(estSample(), k * 0.004);}
  const auto h3 = g.health(0.8, 0.5, 0.5);
  EXPECT_EQ(h3.status, StateMux::Status::Stale);
  EXPECT_NE(h3.why.find("gyro"), std::string::npos) << h3.why;
}

// FAILS IF: a reset forgets only one of the sources. After a reset the state
// in hand belongs to the old run (nine metres away, measured), so the loop
// must wait for a fresh sample from EVERY selected source, not just the first
// that speaks.
TEST(StateMux, AResetWaitsForEverySelectedSourceAgain)
{
  const auto sel = pick(Source::Truth, Source::Est, Source::Truth, Source::Est);
  StateMux m = fed(sel, truthSample(), estSample());
  ASSERT_EQ(m.health(0.0, 0.5, 0.5).status, StateMux::Status::Ready);
  m.reset(0.0);
  EXPECT_EQ(m.health(0.0, 0.5, 0.5).status, StateMux::Status::Waiting);
  m.onTruth(truthSample(), 0.004);
  EXPECT_EQ(m.health(0.004, 0.5, 0.5).status, StateMux::Status::Waiting);
  m.onEst(estSample(), 0.008);
  EXPECT_EQ(m.health(0.008, 0.5, 0.5).status, StateMux::Status::Ready);
}


// FAILS IF: position=slam reads position from anything but /slam/odom, or
// lets SLAM's (stale, drifting) orientation or velocity leak into the other
// fields, or flies before the first SLAM pose.
TEST(StateMux, SlamFeedsPositionAndNothingElse)
{
  const auto sel = pick(Source::Slam, Source::Est, Source::Est, Source::Est);
  StateMux m(sel, true, kMaxRate, kGyroTau);
  const OdomSample t = truthSample();
  const OdomSample e = estSample();
  m.onGyro(kGyro, 0.0);
  m.onTruth(t, 0.0);
  m.onEst(e, 0.0);
  EXPECT_EQ(m.health(0.0, 0.1, 0.1).status, StateMux::Status::Waiting);
  EXPECT_TRUE(std::isinf(m.slamAge(0.0)));

  OdomSample slam;
  slam.position = Eigen::Vector3d(7.0, -3.0, 1.25);
  slam.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(2.0, Eigen::Vector3d::UnitZ()));
  slam.twist_linear = Eigen::Vector3d(9.0, 9.0, 9.0);
  m.onSlam(slam, 0.01);
  EXPECT_EQ(m.health(0.01, 0.1, 0.1).status, StateMux::Status::Ready);
  const State s = m.state();
  EXPECT_LT((s.position - slam.position).norm(), 1e-12);
  EXPECT_LT(quatDistance(s.orientation, e.orientation.normalized()), 1e-12);
  EXPECT_LT((s.velocity - e.orientation.normalized() * e.twist_linear).norm(), 1e-12);
  EXPECT_NEAR(m.slamAge(0.31), 0.30, 1e-12);
}

// FAILS IF: a quiet SLAM cuts the motors like a stale truth/est source. The
// controller must still have a state to HOLD on (its hold is the whole point
// of noticing LOST), so SLAM's age is reported, not turned into Stale.
TEST(StateMux, AQuietSlamIsNotAMotorCut)
{
  const auto sel = pick(Source::Slam, Source::Est, Source::Est, Source::Est);
  StateMux m(sel, true, kMaxRate, kGyroTau);
  m.onGyro(kGyro, 0.0);
  m.onEst(estSample(), 0.0);
  m.onSlam(truthSample(), 0.0);
  m.onEst(estSample(), 5.0);
  m.onGyro(kGyro, 5.0);
  EXPECT_EQ(m.health(5.0, 0.1, 0.1).status, StateMux::Status::Ready);
  EXPECT_NEAR(m.slamAge(5.0), 5.0, 1e-12);
  m.reset(6.0);
  EXPECT_TRUE(std::isinf(m.slamAge(6.0)));   // a reset forgets the old run's pose
}
