"""The message the bridge builds. Needs dsim_msgs, so this runs in the
container (colcon test) and is skipped on the host."""
import math

import pytest

pytest.importorskip("dsim_msgs")

from dsim_planner.bridge_node import build_trajectory  # noqa: E402

SEED = [[0.5 + 1.125 * i, 5.0, 0.5, 1.25 * i] for i in range(9)]


def to_s(d):
    return d.sec + 1e-9 * d.nanosec


def test_stamp_is_the_absolute_start_time():
    msg, _ = build_trajectory(SEED, 10.25, 0.02, 0.0)
    assert to_s(msg.header.stamp) == pytest.approx(10.25, abs=1e-9)


def test_time_from_start_counts_from_the_plans_first_point():
    # A plan whose own clock starts at t=3 still has points[0] at offset 0:
    # the header stamp is where scenario time is anchored, not the plan's t0.
    shifted = [[x, y, z, t + 3.0] for x, y, z, t in SEED]
    msg, _ = build_trajectory(shifted, 10.0, 0.5, 0.0)
    offsets = [to_s(p.time_from_start) for p in msg.points]
    assert offsets[0] == 0.0
    assert offsets[-1] == pytest.approx(10.0)
    assert all(b - a == pytest.approx(0.5, abs=1e-9) for a, b in zip(offsets, offsets[1:]))


def test_nanoseconds_never_overflow_a_second():
    # 0.02 s steps accumulate float error; a rounding that produced
    # nanosec == 1e9 is a message the controller's binary search cannot sort.
    msg, _ = build_trajectory(SEED, 10.0, 0.02, 0.0)
    assert all(0 <= p.time_from_start.nanosec < 1_000_000_000 for p in msg.points)
    assert 0 <= msg.header.stamp.nanosec < 1_000_000_000


def test_points_carry_the_feedforward():
    msg, samples = build_trajectory(SEED, 10.0, 0.02, 0.3)
    assert len(msg.points) == len(samples) == 501
    assert msg.points[100].velocity.x == pytest.approx(0.9)
    assert msg.points[100].acceleration.x == pytest.approx(0.0, abs=1e-9)
    assert msg.points[100].yaw == 0.3
    assert msg.points[-1].position.x == pytest.approx(9.5)


# ---- the frame check ------------------------------------------------------
# FAILS IF: the bridge flies a plan that is not in its scenario's frame --
# above all one with no frame record, which is every plan solved before
# frames.py and is in the planner's coordinates, beside its obstacles.
import json  # noqa: E402
import pathlib  # noqa: E402

from dsim_planner import spacetime  # noqa: E402
from dsim_planner.bridge_node import load_plan  # noqa: E402

PLANS = pathlib.Path(__file__).resolve().parents[3] / "plans"


def _write(tmp_path, doc):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_the_committed_seed_loads():
    plan = load_plan(str(PLANS / "wall_seed.json"), "wall")
    assert plan["frame"]["scenario"] == "wall"


def test_a_plan_with_no_frame_is_refused(tmp_path):
    doc = json.loads((PLANS / "wall_seed.json").read_text())
    del doc["frame"]
    with pytest.raises(spacetime.PlanError, match="no 'frame'"):
        load_plan(_write(tmp_path, doc), "wall")


def test_a_plan_in_a_stale_frame_is_refused(tmp_path):
    # The frame the planner's loiter used to be flown in: a plan solved then
    # is in the wrong place now, and has to be refused rather than flown.
    doc = json.loads((PLANS / "loiter_sim_seed.json").read_text())
    doc["frame"]["scale"] = 0.04
    with pytest.raises(spacetime.PlanError, match="re-solve"):
        load_plan(_write(tmp_path, doc), "loiter_sim")


def test_a_plan_for_another_world_is_refused(tmp_path):
    doc = json.loads((PLANS / "fence3d_seed.json").read_text())
    with pytest.raises(spacetime.PlanError, match="solves scenario"):
        load_plan(_write(tmp_path, doc), "wall")


def test_with_no_scenario_given_the_plan_is_held_to_its_own(tmp_path):
    doc = json.loads((PLANS / "wall_seed.json").read_text())
    del doc["frame"]
    with pytest.raises(spacetime.PlanError):
        load_plan(_write(tmp_path, doc))


def test_travel_mode_puts_the_nose_along_the_plan():
    # FAILS IF: the bridge ignores yaw_mode and flies the fixed yaw.
    diag = [[0.0, 0.0, 1.0, 0.0], [3.0, 3.0, 1.0, 4.0]]
    msg, _ = build_trajectory(diag, 10.0, 0.05, 0.0, "travel")
    assert msg.points[10].yaw == pytest.approx(math.pi / 4)
    msg, _ = build_trajectory(diag, 10.0, 0.05, 0.0, "fixed")
    assert msg.points[10].yaw == 0.0
    with pytest.raises(Exception):
        build_trajectory(diag, 10.0, 0.05, 0.0, "sideways")
