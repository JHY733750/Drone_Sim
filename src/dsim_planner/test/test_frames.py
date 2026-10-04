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
    # The s = 0.04 the planner's loiter was once flown at: a non-unit scale is
    # the case worth testing, and no scenario in the table has one any more.
    f = _frame(0.04, 0.0, 1.88)
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


IMPORTED = [p for p in SCENARIOS if not json.loads(p.read_text()).get("native")]
NATIVE = [p for p in SCENARIOS if json.loads(p.read_text()).get("native")]


@pytest.mark.parametrize("path", IMPORTED, ids=lambda p: p.stem)
def test_the_scenario_is_centred_on_the_origin_in_the_sim(path):
    # What the transform is FOR: the bounding-box centre lands within a
    # centimetre (times s) of the origin. Imported scenarios only: a
    # sim-native one is not re-centred (its origin is its ground station).
    sc = frames.scenario_to_sim(json.loads(path.read_text()))
    xs = [sc["start"][0], sc["end"][0]] + [c[0] for o in sc["obstacles"] for c in o["control_points"]]
    ys = [sc["start"][1], sc["end"][1]] + [c[1] for o in sc["obstacles"] for c in o["control_points"]]
    assert abs(0.5 * (min(xs) + max(xs))) <= 0.006
    assert abs(0.5 * (min(ys) + max(ys))) <= 0.006


def test_there_is_a_sim_native_scenario_and_no_imported_loiter():
    # FAILS IF: the planner's loiter is re-imported (import_scenarios.py would
    # also refuse it as an orphan) or loiter_sim stops being sim-native.
    assert not (ROOT / "scenarios" / "loiter.json").exists()
    assert [p.stem for p in NATIVE] == ["loiter_sim"]
    assert set(frames.NATIVE) == {"loiter_sim"}


@pytest.mark.parametrize("path", NATIVE, ids=lambda p: p.stem)
def test_a_sim_native_scenario_is_already_in_the_sim_frame(path):
    # FAILS IF: the identity frame is not the identity -- the file's numbers
    # would then not be where the sim flies them.
    raw = json.loads(path.read_text())
    sc = frames.scenario_to_sim(raw)
    assert sc["frame"]["scale"] == 1.0 and sc["frame"]["centre"] == [0.0, 0.0, 0.0]
    assert sc["start"] == raw["start"] and sc["end"] == raw["end"]
    for a, b in zip(sc["obstacles"], raw["obstacles"]):
        assert a["control_points"] == b["control_points"] and a["radius"] == b["radius"]
    assert "native" not in sc and "solve" not in sc      # the solver's, not the sim's


def test_check_table_centre_holds_both_halves_of_the_native_claim(monkeypatch):
    raw = json.loads((ROOT / "scenarios" / "loiter_sim.json").read_text())
    frames.check_table_centre(raw)
    # FAILS IF: a file can claim the identity frame without the table.
    with pytest.raises(frames.FrameError, match="does not list it"):
        frames.check_table_centre(dict(raw, name="wall"))
    # FAILS IF: the table can claim it for a file that does not.
    with pytest.raises(frames.FrameError, match="does not declare"):
        frames.check_table_centre({k: v for k, v in raw.items() if k != "native"})
    # FAILS IF: a native scenario may sit in a non-identity frame.
    monkeypatch.setitem(frames.TABLE, "loiter_sim", (1.0, (0.5, 0.0)))
    with pytest.raises(frames.FrameError, match="identity"):
        frames.check_table_centre(raw)


def _drone_yaml():
    import yaml
    return yaml.safe_load((ROOT / "config" / "drone.yaml").read_text())["drone"]


def test_loiter_sim_is_the_design_it_claims():
    # The geometry scripts/sim_scenarios.py promises, read back from the file.
    # FAILS IF: a number is changed and the design quietly stops holding --
    # the corridor leaves the flow band or the camera's world, the grown body
    # reaches the band, the corridor leaves the shadow ring, or it grows wide
    # enough for a sidestep to clear the shadow.
    sc = json.loads((ROOT / "scenarios" / "loiter_sim.json").read_text())
    drone = _drone_yaml()
    flow = drone["sensors"]["optical_flow"]
    grown = drone["radius_m"]
    zc = sc["start"][2]
    lo, hi = sc["coord_bounds"][2]
    assert flow["min_height_m"] < lo < zc < hi < flow["max_height_m"]
    assert 10.0 <= abs(sc["end"][0] - sc["start"][0]) <= 14.0
    assert sc["stations"] == [[0.0, 0.0, 0.0]]
    body_z = sc["obstacles"][0]["control_points"][0][2]
    body_r = sc["obstacles"][0]["radius"]
    orbit_r = math.hypot(*sc["obstacles"][0]["control_points"][0][:2])
    for o in sc["obstacles"]:
        for cp in (o["control_points"][0], o["control_points"][-1]):
            assert math.hypot(cp[0], cp[1]) == pytest.approx(orbit_r, abs=1e-8)
            assert cp[2] == body_z
    assert body_z + body_r + grown < lo                     # keep-out under the band
    assert sc["start"][1] == pytest.approx(orbit_r * zc / body_z, abs=1e-9)   # tangent
    half = 0.5 * (sc["coord_bounds"][1][1] - sc["coord_bounds"][1][0])
    assert half < body_r * zc / body_z                      # narrower than the spot
    # One lap, on contiguous windows, over exactly the horizon.
    windows = [(o["control_points"][0][3], o["control_points"][-1][3]) for o in sc["obstacles"]]
    assert windows[0][0] == 0.0 and windows[-1][1] == sc["T"]
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))
    assert sc["solve"]["free_arrival_time"] and sc["solve"]["v_max"] < 2.0


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
