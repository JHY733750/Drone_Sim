"""SLAM landmarks: the distance arithmetic, the file format, and the gate.

scripts/landmarks.py lives beside gen_assets.py, which generates from it;
these run with the planner tests because the landmarks are part of the
scenario, and the gate reads the same plans and frames.

The distance tests check the signed-distance functions against a SECOND,
independent computation -- the distance to explicitly sampled surface points
-- because the generator and the gate share the SDFs, and a shared bug would
make them agree with each other and both be wrong.
"""
import json
import math
import pathlib
import random
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import landmarks as L  # noqa: E402
import check_landmarks as G  # noqa: E402
from dsim_planner import frames  # noqa: E402

NAMES = sorted(p.stem for p in (ROOT / "scenarios").glob("*.json"))

SHAPES = [
    {"name": "b", "type": "box", "center": [1.0, -2.0, 0.6], "size": [1.4, 0.5, 1.2], "yaw": 0.7},
    {"name": "c", "type": "cylinder", "center": [-0.5, 1.0, 1.1], "radius": 0.4, "height": 2.2},
    {"name": "s", "type": "sphere", "center": [0.3, 0.2, 2.0], "radius": 0.55},
]


def surface_points(lm, n=40):
    """Points ON the surface, parametrically -- no SDF involved."""
    c = lm["center"]
    out = []
    if lm["type"] == "sphere":
        for i in range(n):
            for j in range(n):
                th, ph = math.pi * (i + 0.5) / n, 2 * math.pi * j / n
                out.append([c[0] + lm["radius"] * math.sin(th) * math.cos(ph),
                            c[1] + lm["radius"] * math.sin(th) * math.sin(ph),
                            c[2] + lm["radius"] * math.cos(th)])
    elif lm["type"] == "cylinder":
        r, h = lm["radius"], lm["height"]
        for i in range(n + 1):
            for j in range(n):
                a = 2 * math.pi * j / n
                out.append([c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2] - h / 2 + h * i / n])
                for z in (c[2] - h / 2, c[2] + h / 2):     # the caps
                    out.append([c[0] + r * i / n * math.cos(a), c[1] + r * i / n * math.sin(a), z])
    else:
        hx, hy, hz = (0.5 * v for v in lm["size"])
        cy_, sy_ = math.cos(lm["yaw"]), math.sin(lm["yaw"])
        g = [k / n * 2 - 1 for k in range(n + 1)]
        for u in g:
            for v in g:
                for q in ([hx, u * hy, v * hz], [-hx, u * hy, v * hz], [u * hx, hy, v * hz],
                          [u * hx, -hy, v * hz], [u * hx, v * hy, hz], [u * hx, v * hy, -hz]):
                    out.append([c[0] + cy_ * q[0] - sy_ * q[1], c[1] + sy_ * q[0] + cy_ * q[1],
                                c[2] + q[2]])
    return out


@pytest.mark.parametrize("lm", SHAPES, ids=lambda s: s["type"])
def test_sdf_outside_equals_distance_to_the_sampled_surface(lm):
    # FAILS IF: a box is rotated the wrong way, a cylinder's cap or a box's
    # half-extent is off by a factor of two, or an inside/outside sign flips.
    rng = random.Random(3)
    surf = surface_points(lm)
    for _ in range(60):
        p = [lm["center"][i] + rng.uniform(-3, 3) for i in range(3)]
        d = L.sdf(lm, p)
        if d <= 0.05:
            continue
        brute = min(math.dist(p, q) for q in surf)
        # brute over-estimates by at most the surface sampling gap.
        assert d <= brute + 1e-9
        assert brute - d < 0.12


@pytest.mark.parametrize("lm", SHAPES, ids=lambda s: s["type"])
def test_sdf_is_negative_exactly_inside(lm):
    rng = random.Random(4)
    for _ in range(200):
        p = [lm["center"][i] + rng.uniform(-2, 2) for i in range(3)]
        c = lm["center"]
        if lm["type"] == "sphere":
            inside = math.dist(p, c) < lm["radius"]
        elif lm["type"] == "cylinder":
            inside = (math.hypot(p[0] - c[0], p[1] - c[1]) < lm["radius"]
                      and abs(p[2] - c[2]) < lm["height"] / 2)
        else:
            cy_, sy_ = math.cos(lm["yaw"]), math.sin(lm["yaw"])
            dx, dy = p[0] - c[0], p[1] - c[1]
            q = (cy_ * dx + sy_ * dy, -sy_ * dx + cy_ * dy, p[2] - c[2])
            inside = all(abs(q[i]) < lm["size"][i] / 2 for i in range(3))
        assert (L.sdf(lm, p) < 0) == inside


@pytest.mark.parametrize("lm", SHAPES, ids=lambda s: s["type"])
def test_segment_distance_is_a_lower_bound(lm):
    # The gate's whole claim: it may call a clear landmark close, never the
    # reverse. Checked against a much finer sampling of the same segment.
    rng = random.Random(5)
    for _ in range(40):
        a = [lm["center"][i] + rng.uniform(-3, 3) for i in range(3)]
        b = [lm["center"][i] + rng.uniform(-3, 3) for i in range(3)]
        fine = min(L.sdf(lm, [a[i] + k / 4000 * (b[i] - a[i]) for i in range(3)])
                   for k in range(4001))
        assert L.seg_distance(lm, a, b) <= fine + 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_the_file_is_exactly_the_agreed_format(name):
    doc = yaml.safe_load((ROOT / "config" / f"landmarks_{name}.yaml").read_text())
    assert list(doc) == ["landmarks"]
    keys = {"box": ["name", "type", "center", "size", "yaw"],
            "cylinder": ["name", "type", "center", "radius", "height"],
            "sphere": ["name", "type", "center", "radius"]}
    for i, e in enumerate(doc["landmarks"]):
        assert list(e) == keys[e["type"]]
        assert e["name"] == f"L{i}"
        assert len(e["center"]) == 3 and all(isinstance(v, (int, float)) for v in e["center"])


@pytest.mark.parametrize("name", NAMES)
def test_landmarks_are_informative_not_a_row_of_parallel_boxes(name):
    lms = L.parse_yaml((ROOT / "config" / f"landmarks_{name}.yaml").read_text())
    types = {lm["type"] for lm in lms}
    assert len(types) == 3
    yaws = sorted(lm["yaw"] % (math.pi / 2) for lm in lms if lm["type"] == "box")
    # Box faces not all parallel: yaws modulo a right angle span > 20 deg.
    assert len(yaws) >= 2 and yaws[-1] - yaws[0] > math.radians(20)
    heights = [lm["center"][2] for lm in lms]
    assert max(heights) - min(heights) > 0.5
    # In view, on both sides of the CAMERA's axis, from every point of the
    # seed. (Not both sides of the path: flying yaw 0 on a 45 deg diagonal,
    # the path's left is outside a +/-43.5 deg view. See landmarks.py.)
    seed = json.loads((ROOT / "plans" / f"{name}_seed.json").read_text())["control_points"]
    cams, _ = L.curve_polyline(seed, step=0.25)
    assert all(L.covered(c) for c in L.coverage(cams, lms))


@pytest.mark.parametrize("name", NAMES)
def test_the_committed_landmarks_pass_the_gate(name):
    assert G.check(name, quiet=True) == []


@pytest.mark.parametrize("name", NAMES)
def test_a_landmark_planted_on_the_seed_fails_the_gate(name):
    # FAILS IF: the gate stops reading the plans, or reads them in the wrong
    # frame (the seed's midpoint would then be somewhere else entirely).
    seed = json.loads((ROOT / "plans" / f"{name}_seed.json").read_text())["control_points"]
    mid = L.bezier_points(seed, 2)[1]
    fails = G.check(name, extra={"name": "PLANT", "type": "box", "center": mid,
                                 "size": [0.2, 0.2, 0.2], "yaw": 0.0}, quiet=True)
    assert any(f.startswith("PLANT") and "corridor" in f for f in fails), fails


def test_a_landmark_in_an_obstacle_sweep_fails_the_gate():
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / "wall.json").read_text()))
    w = sc["obstacles"][0]["control_points"][0]
    fails = G.check("wall", extra={"name": "PLANT", "type": "sphere",
                                   "center": [w[0], w[1], 4.0], "radius": 0.2}, quiet=True)
    assert any(f.startswith("PLANT") and "obstacle W0" in f for f in fails), fails


def test_a_landmark_across_a_sight_line_fails_the_gate():
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / "loiter_sim.json").read_text()))
    st = sc["stations"][0]
    seed = json.loads((ROOT / "plans" / "loiter_sim_seed.json").read_text())["control_points"]
    m = L.bezier_points(seed, 2)[1]
    p = [st[i] + 0.4 * (m[i] - st[i]) for i in range(3)]
    fails = G.check("loiter_sim", extra={"name": "PLANT", "type": "sphere", "center": p,
                                     "radius": 0.1}, quiet=True)
    assert any(f.startswith("PLANT") and "sight line" in f for f in fails), fails


def test_generation_is_deterministic():
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / "wall.json").read_text()))
    seed = json.loads((ROOT / "plans" / "wall_seed.json").read_text())["control_points"]
    assert L.generate(sc, seed)[0] == L.generate(sc, seed)[0]
