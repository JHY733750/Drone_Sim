"""Where each scenario sits in the simulator's world. One table, one map.

The planner's six imported scenarios all live in roughly 0..10 m, a quarter
of the world around the origin. The sim places each one with an explicit
SIMILARITY transform,

    p_sim = s * (p_planner - c),        c = (cx, cy, 0),    t_sim = t_planner

applied AFTER import: scenarios/*.json stay the planner's numbers verbatim, so
`import_scenarios.py --check` still compares against the planner, and the
solver is still handed exactly what the planner defines. Everything the sim
flies, scores or draws -- worlds, the referee's obstacle list, the viewer's
scene, the seed, a solved plan -- is in the SIM frame, and records which
transform put it there.

Why a similarity and nothing looser: a Bezier curve is an affine combination
of its control points, so an affine map of the control points IS the same map
of the curve -- exactly, at every parameter, not approximately. Time is left
alone, so dt/dtau is unchanged and the bridge's v = p'/t', a = (p''t' - p't'')/t'^3
both pick up exactly the factor s. A UNIFORM scale keeps a sphere a sphere
(radius times s) and keeps a sight line a sight line, so the referee's
clearance and line-of-sight arithmetic carry over unchanged. Scaling x and y
but not z would turn every sphere into an ellipsoid, which nothing here can
score -- so there is no separate z scale; z scales with s about the ground
(cz = 0), and for s = 1 it is simply untouched.

For an IMPORTED scenario, c is the centre of its horizontal bounding box:
start, end and every obstacle control point (the control polygon contains the
curve, so this box contains every obstacle path). Rounded to the centimetre
and written into the table below as a literal, because the table is the claim;
gen_assets.py --check recomputes the definition and fails if a planner change
has moved it.

A SIM-NATIVE scenario (NATIVE below) is authored in this repo, in sim metres,
by scripts/sim_scenarios.py. Its frame is the identity, s = 1 and c = 0, BY
CONSTRUCTION rather than by measurement: its numbers are already where the sim
flies them, and its origin is meaningful (loiter_sim's ground station stands
there), so re-centring it on a bounding box would move the station off the
origin for nothing. check_table_centre() holds both directions: a native
scenario must declare itself native in its file AND be listed here with
exactly (1, (0, 0)); an imported one may do neither.

  loiter_sim   replaces the planner's 200 m `loiter`, which this table used to
               scale by s = 0.04. That put the corridor in the optical flow's
               band, but the 0.3 m vehicle became 7.5 planner metres, so the
               bodies had to be grown by that much before solving; grown, they
               reached into the altitude band and the run lost its occlusion
               certificate at N8_seg16. Redesigned at the sim's own size
               instead: see scripts/sim_scenarios.py.

Everything that maps a coordinate between the frames calls this module. There
is no other copy of s or c anywhere.
"""
from __future__ import annotations

import math

#: Bumped if the MEANING of the transform changes (not its numbers), so a plan
#: written under an older convention is refused rather than misread.
FRAME_VERSION = 1

#: name -> (s, (cx, cy)). cx, cy in PLANNER metres. See the module docstring.
TABLE = {
    "curve":    (1.0,  (5.00, 5.00)),
    "diverse":  (1.0,  (4.75, 4.75)),
    "door3d":   (1.0,  (4.25, 5.00)),
    "fence3d":  (1.0,  (5.00, 5.00)),
    # Sim-native (NATIVE below): the identity, by construction.
    "loiter_sim": (1.0, (0.00, 0.00)),
    "original": (1.0,  (4.50, 4.75)),
    "wall":     (1.0,  (4.75, 5.00)),
}

#: Scenarios AUTHORED in the sim (scripts/sim_scenarios.py), in sim metres.
#: Their frame is the identity; see the module docstring. Their files carry
#: "native": true, and check_table_centre() holds the two together.
NATIVE = frozenset({"loiter_sim"})

#: How far a table centre may sit from the bounding-box centre it claims to
#: be, in planner metres: half a centimetre, i.e. "rounded to the cm".
CENTRE_ROUNDING_M = 0.005 + 1e-9


class FrameError(ValueError):
    """A coordinate, plan or scenario is not in the frame it has to be in."""


def frame_for(name: str) -> dict:
    """The transform record for scenario `name`, as written into every file."""
    if name not in TABLE:
        raise FrameError(
            f"scenario {name!r} has no entry in dsim_planner/frames.py TABLE; "
            f"add one (s and the bounding-box centre) before generating it")
    s, (cx, cy) = TABLE[name]
    return {"scenario": name, "version": FRAME_VERSION,
            "scale": float(s), "centre": [float(cx), float(cy), 0.0]}


def same_frame(a, b, tol: float = 1e-12) -> bool:
    """True when two transform records describe the same map."""
    try:
        return (a["scenario"] == b["scenario"]
                and int(a["version"]) == int(b["version"])
                and abs(float(a["scale"]) - float(b["scale"])) <= tol
                and len(a["centre"]) == 3 and len(b["centre"]) == 3
                and all(abs(float(x) - float(y)) <= tol
                        for x, y in zip(a["centre"], b["centre"])))
    except (KeyError, TypeError, ValueError):
        return False


def describe(frame) -> str:
    c = frame["centre"]
    return (f"{frame['scenario']}: p_sim = {frame['scale']:g} * (p - "
            f"({c[0]:g}, {c[1]:g}, {c[2]:g})), t unchanged")


# --------------------------------------------------------------------------
# The map. Points are (x, y, z) or (x, y, z, t); a 4th coordinate is time and
# passes through.
# --------------------------------------------------------------------------
def to_sim(point, frame) -> list:
    s, c = float(frame["scale"]), frame["centre"]
    p = [float(v) for v in point]
    if len(p) not in (3, 4):
        raise FrameError(f"expected (x, y, z) or (x, y, z, t), got {len(p)} values")
    out = [s * (p[i] - float(c[i])) for i in range(3)]
    return out + p[3:]


def to_planner(point, frame) -> list:
    s, c = float(frame["scale"]), frame["centre"]
    p = [float(v) for v in point]
    if len(p) not in (3, 4):
        raise FrameError(f"expected (x, y, z) or (x, y, z, t), got {len(p)} values")
    out = [p[i] / s + float(c[i]) for i in range(3)]
    return out + p[3:]


def length_to_sim(d: float, frame) -> float:
    """A length (radius, clearance, margin) in sim metres."""
    return float(frame["scale"]) * float(d)


def length_to_planner(d: float, frame) -> float:
    return float(d) / float(frame["scale"])


def points_to_sim(points, frame) -> list:
    """Control points of a Bezier, mapped. Exact for the curve: see top."""
    return [to_sim(p, frame) for p in points]


# --------------------------------------------------------------------------
# Scenarios and plans
# --------------------------------------------------------------------------
def bbox_centre(scenario) -> tuple:
    """The DEFINITION of c, from the planner-frame scenario: the centre of the
    horizontal bounding box of start, end and every obstacle control point."""
    xs = [float(scenario["start"][0]), float(scenario["end"][0])]
    ys = [float(scenario["start"][1]), float(scenario["end"][1])]
    for o in scenario["obstacles"]:
        for cp in o["control_points"]:
            xs.append(float(cp[0]))
            ys.append(float(cp[1]))
    return (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))


def check_table_centre(scenario) -> None:
    """Raise if the table's c is not the bounding-box centre, rounded to cm.

    For a sim-native scenario there is no bounding-box definition to hold it
    to: raise unless the file declares itself native, the name is in NATIVE
    and the table entry is exactly the identity. Each half alone is refused,
    so a planner scenario cannot be waved through by listing it in NATIVE and
    a hand-edited file cannot be waved through by writing "native" into it.
    """
    name = scenario["name"]
    frame = frame_for(name)
    declared = bool(scenario.get("native", False))
    if declared or name in NATIVE:
        if not declared:
            raise FrameError(
                f"{name}: listed in frames.NATIVE but its file does not declare "
                f'"native": true -- it is not the sim-authored scenario NATIVE means')
        if name not in NATIVE:
            raise FrameError(
                f'{name}: file declares "native": true but frames.NATIVE does not '
                f"list it; a sim-native frame is the identity and must be claimed here")
        if frame["scale"] != 1.0 or frame["centre"] != [0.0, 0.0, 0.0]:
            raise FrameError(
                f"{name}: a sim-native scenario is authored in sim metres, so its "
                f"frame must be the identity (s = 1, c = 0); the table says "
                f"{describe(frame)}")
        return
    want = bbox_centre(scenario)
    got = frame["centre"][:2]
    if any(abs(a - b) > CENTRE_ROUNDING_M for a, b in zip(got, want)):
        raise FrameError(
            f"{name}: frames.TABLE centre {got} is not the bounding-box "
            f"centre ({want[0]:.4f}, {want[1]:.4f}) rounded to the cm -- the planner's "
            f"scenario has moved. Update TABLE, regenerate, and re-solve its plans.")


def scenario_to_sim(scenario) -> dict:
    """The planner-frame scenario (scenarios/<name>.json) in the sim frame.

    Only what the SIM reads is carried: start, end, spawn, the flight altitude
    of a 2D scenario, the obstacles and the stations. The planner-only keys
    (coord_bounds, trust_radius, init_curve, configs) are in planner metres and
    mean something only to the solver, which is handed the planner-frame file;
    they are dropped here so nothing in the sim can misread them. So are a
    sim-native scenario's "native" marker and "solve" flags, which are the
    frame table's and the solver's business respectively.
    """
    frame = frame_for(scenario["name"])
    out = {k: v for k, v in scenario.items()
           if k not in ("coord_bounds", "trust_radius", "init_curve", "configs",
                        "_generated_by", "native", "solve")}
    out["frame"] = frame
    out["start"] = to_sim(scenario["start"], frame)
    out["end"] = to_sim(scenario["end"], frame)
    sim = dict(scenario["sim"])
    sim["spawn"] = to_sim(scenario["sim"]["spawn"], frame)
    out["sim"] = sim
    if "lift_z" in scenario:
        # z' = s * z about the ground; c has no z component.
        out["lift_z"] = to_sim([0.0, 0.0, scenario["lift_z"]], frame)[2] + 0.0
    out["obstacles"] = [
        dict(o, control_points=points_to_sim(o["control_points"], frame),
             radius=length_to_sim(o["radius"], frame))
        for o in scenario["obstacles"]]
    if scenario.get("stations"):
        out["stations"] = [to_sim((list(st) + [0.0, 0.0, 0.0])[:3], frame)
                           for st in scenario["stations"]]
    return out


def check_plan_frame(plan, scenario_name: str) -> dict:
    """Return the scenario's frame if `plan` is in it; raise FrameError if not.

    A plan with no frame record is REFUSED, not assumed to be in the planner's
    frame: every plan before this module existed was in the planner's frame,
    and flying one in a recentred world is flying it beside its obstacles.
    """
    want = frame_for(scenario_name)
    if plan.get("scenario") != scenario_name:
        raise FrameError(f"plan solves scenario {plan.get('scenario')!r}, "
                         f"not {scenario_name!r}")
    got = plan.get("frame")
    if got is None:
        raise FrameError(
            f"plan has no 'frame' record, so it is in the planner's coordinates "
            f"(it predates frames.py); the world is in {describe(want)}. "
            f"Re-solve it, or map it with scripts/reframe_plan.py")
    if not same_frame(got, want):
        raise FrameError(
            f"plan is in frame {got!r} but scenario {scenario_name!r} is now in "
            f"{describe(want)}; re-solve it")
    return want
