"""The scenario frames: one table, one similarity map, and plans that say
which frame they are in.

The load-bearing test is the commutation: mapping a Bezier's control points
and then evaluating must equal evaluating and then mapping, at every
parameter, and the sampled velocity and acceleration must pick up exactly the
factor s. That is the property that makes "transform the plan" exact rather
than approximately right, and it fails for any map that is not affine --
including a correct-looking one that forgets to scale z, or scales about a
different centre for positions than for control points.
"""
import json
import math
import pathlib
import random

import pytest

from dsim_planner import frames, spacetime as st

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCENARIOS = sorted((ROOT / "scenarios").glob("*.json"))

CLUSTERED = [
    [0.0, 0.0, 1.0, 0.0],
    [0.5, 0.2, 1.0, 0.4],
    [1.0, 0.9, 1.4, 0.9],
    [3.0, 1.5, 2.2, 1.6],
    [6.0, 1.2, 1.5, 4.0],
    [8.0, 0.4, 1.0, 7.5],
    [9.0, 0.0, 1.0, 10.0],
]


def _frame(s, cx, cy):
    return {"scenario": "x", "version": frames.FRAME_VERSION, "scale": s,
            "centre": [cx, cy, 0.0]}


@pytest.mark.parametrize("s,cx,cy", [(1.0, 4.75, 5.0), (0.04, 0.0, 1.88), (2.5, -3.0, 7.0)])
def test_mapping_control_points_maps_the_curve_exactly(s, cx, cy):
    # FAILS IF: the map is not affine, or positions and control points are
    # mapped differently (e.g. a centre applied to one and not the other).
    f = _frame(s, cx, cy)
    Q = frames.points_to_sim(CLUSTERED, f)
    for k in range(0, 101):
        tau = k / 100
        a = frames.to_sim(st.de_casteljau(CLUSTERED, tau), f)
        b = st.de_casteljau(Q, tau)
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-12 * max(1.0, s * 10)


@pytest.mark.parametrize("s", [1.0, 0.04, 3.0])
def test_speed_and_acceleration_scale_by_exactly_s(s):
    # FAILS IF: time is touched by the map (then the factor is not s), or z
    # is left unscaled (then the 3D speed ratio is not s).
    f = _frame(s, 1.0, -2.0)
    Q = frames.points_to_sim(CLUSTERED, f)
    for t in (0.0, 0.3, 1.7, 5.0, 9.9):
        p0, v0, a0 = st.sample_at_time(CLUSTERED, t)
        p1, v1, a1 = st.sample_at_time(Q, t)
        assert p1 == pytest.approx(frames.to_sim(p0, f), abs=1e-9)
        assert v1 == pytest.approx([s * c for c in v0], rel=1e-9, abs=1e-12)
        assert a1 == pytest.approx([s * c for c in a0], rel=1e-9, abs=1e-12)


def test_round_trip_is_the_identity():
    rng = random.Random(1)
    for name in frames.TABLE:
        f = frames.frame_for(name)
        for _ in range(50):
            p = [rng.uniform(-200, 200) for _ in range(3)] + [rng.uniform(0, 80)]
            assert frames.to_planner(frames.to_sim(p, f), f) == pytest.approx(p, abs=1e-9)


def test_a_similarity_scales_every_distance_by_s():
    # The referee's clearance and sight-line margins are distances; they carry
    # over only because this holds for every pair of points.
    rng = random.Random(2)
    f = frames.frame_for("loiter")
    for _ in range(100):
        a = [rng.uniform(-100, 100) for _ in range(3)]
        b = [rng.uniform(-100, 100) for _ in range(3)]
        assert math.dist(frames.to_sim(a, f), frames.to_sim(b, f)) == pytest.approx(
            f["scale"] * math.dist(a, b), rel=1e-12)


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_every_scenario_has_a_table_entry_whose_centre_is_the_definition(path):
    # FAILS IF: the planner moves a scenario and the table is not updated.
    sc = json.loads(path.read_text())
    frames.check_table_centre(sc)


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_the_scenario_is_centred_on_the_origin_in_the_sim(path):
    # What the transform is FOR: the bounding-box centre lands within a
    # centimetre (times s) of the origin.
    sc = frames.scenario_to_sim(json.loads(path.read_text()))
    xs = [sc["start"][0], sc["end"][0]] + [c[0] for o in sc["obstacles"] for c in o["control_points"]]
    ys = [sc["start"][1], sc["end"][1]] + [c[1] for o in sc["obstacles"] for c in o["control_points"]]
    assert abs(0.5 * (min(xs) + max(xs))) <= 0.006
    assert abs(0.5 * (min(ys) + max(ys))) <= 0.006


def test_loiter_fits_a_depth_camera_world_and_the_flow_band():
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / "loiter.json").read_text()))
    assert abs(sc["end"][0] - sc["start"][0]) <= 10.0
    assert 0.1 < sc["start"][2] < 3.0          # optical flow's band (config/drone.yaml)
    assert sc["stations"][0][2] == 0.0         # still on the ground
    assert all(o["radius"] == pytest.approx(6.0 * 0.04) for o in sc["obstacles"])


@pytest.mark.parametrize("path", sorted((ROOT / "plans").glob("*_seed.json")),
                         ids=lambda p: p.stem)
def test_every_committed_seed_is_in_its_scenarios_frame(path):
    plan = json.loads(path.read_text())
    frames.check_plan_frame(plan, plan["scenario"])


def _seed(name="wall"):
    return json.loads((ROOT / "plans" / f"{name}_seed.json").read_text())


def test_a_plan_with_no_frame_is_refused():
    # Every plan written before frames.py: in the planner's coordinates.
    plan = _seed()
    del plan["frame"]
    with pytest.raises(frames.FrameError, match="no 'frame'"):
        frames.check_plan_frame(plan, "wall")


@pytest.mark.parametrize("field,value", [("scale", 0.5), ("centre", [0.0, 0.0, 0.0]),
                                         ("version", 0), ("scenario", "fence3d")])
def test_a_plan_in_another_frame_is_refused(field, value):
    plan = _seed()
    plan["frame"] = dict(plan["frame"], **{field: value})
    with pytest.raises(frames.FrameError):
        frames.check_plan_frame(plan, "wall")


def test_a_plan_for_another_scenario_is_refused():
    with pytest.raises(frames.FrameError, match="solves scenario"):
        frames.check_plan_frame(_seed("fence3d"), "wall")


def test_a_scenario_with_no_table_entry_is_refused():
    with pytest.raises(frames.FrameError, match="no entry"):
        frames.frame_for("not_a_scenario")
