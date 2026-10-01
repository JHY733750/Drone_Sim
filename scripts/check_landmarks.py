#!/usr/bin/env python3
"""Gate: no SLAM landmark interferes with any scenario's flight.

    python3 scripts/check_landmarks.py                 # every scenario
    python3 scripts/check_landmarks.py wall loiter     # some
    python3 scripts/check_landmarks.py --self-test     # prove it can fail

For each config/landmarks_<scenario>.yaml, against EVERY plan for that
scenario in plans/ (the committed seed and every solved plan on this machine),
in the sim frame:

  corridor   landmark surface >= CORRIDOR_MARGIN_M (1.0 m) from every point of
             every plan, start and goal included;
  obstacles  >= OBSTACLE_MARGIN_M (0.5 m) from every moving obstacle's swept
             volume (its centre path over its whole window, at its radius; a
             column at every height);
  sight      >= SIGHT_MARGIN_M (0.5 m) from every segment between a station and
             any point of any plan;
  ground     nothing below z = 0;
  file       the format the SLAM side reads, names unique, and NOT one of the
             referee's obstacles.

The distances are lower bounds (see scripts/landmarks.py), so this can call a
clear landmark too close but cannot call a close one clear. A plan whose frame
is not the scenario's fails the gate: it is not the plan that would be flown,
and checking landmarks against it would check nothing.

FAILS IF: a landmark is moved into a plan's corridor, an obstacle's sweep or a
sight line; a re-solved plan wanders into a landmark; a plan is left in the
wrong frame. --self-test plants one landmark in each of those places and
exits non-zero unless every one of them is caught and the real set passes.

Also reports coverage -- how many landmark centres the camera (yaw 0, body
+x, 87 x 58 deg, 0.28..10 m) has in view along each plan -- which is not
gated: it is the generator's target, not an interference property.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "dsim_planner"))
sys.path.insert(0, str(ROOT / "scripts"))
from dsim_planner import frames  # noqa: E402
import landmarks as L  # noqa: E402


def scenario_names():
    return sorted(p.stem for p in (ROOT / "scenarios").glob("*.json"))


def plans_for(name):
    """[(label, control points)] for every plan file that declares `name`,
    and [(file, reason)] for those that cannot be checked."""
    good, bad = [], []
    for path in sorted((ROOT / "plans").glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            bad.append((path.name, f"unreadable: {exc}"))
            continue
        if doc.get("scenario") != name:
            continue
        try:
            frames.check_plan_frame(doc, name)
        except frames.FrameError as exc:
            bad.append((path.name, str(exc)))
            continue
        good.append((path.stem, doc["control_points"]))
    return good, bad


def check(name, extra=None, quiet=False):
    """Return the list of failures for one scenario. `extra` is a landmark to
    plant on top of the file's set (the self-test)."""
    fails = []
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / f"{name}.json").read_text()))
    path = ROOT / "config" / f"landmarks_{name}.yaml"
    try:
        lms = L.parse_yaml(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"{path.name}: {exc}"]
    if extra is not None:
        lms = lms + [extra]
    if not lms:
        fails.append(f"{path.name}: no landmarks")
    names = [lm["name"] for lm in lms]
    if len(set(names)) != len(names):
        fails.append(f"{path.name}: duplicate names")
    clash = set(names) & {o["name"] for o in sc["obstacles"]}
    if clash:
        fails.append(f"{path.name}: {sorted(clash)} share a name with the referee's obstacles")
    plans, bad = plans_for(name)
    for f, why in bad:
        fails.append(f"plans/{f}: {why}")
    if not plans:
        fails.append(f"no plan for {name} in plans/ to check against")
        return fails
    field = L.Field(sc, plans, reserve=0.0)
    worst = {}
    for lm in lms:
        for kind, against, d, need in field.violations(lm):
            fails.append(f"{lm['name']} ({lm['type']}): {kind} {against}: "
                         f"{d:+.3f} m, needs >= {need:.2f}")
        # Closest approach of each kind, for the report.
        for label, pts, slack in field.paths:
            d = L.polyline_distance(lm, pts) - slack
            worst["corridor"] = min(worst.get("corridor", 1e9), d)
    if not quiet:
        cov = []
        for label, cps in plans:
            cams, _ = L.curve_polyline(cps, step=0.25)
            c = L.coverage(cams, [x for x in lms if x is not extra])
            cov.append(f"{label} min {min(v[0] for v in c)} visible, "
                       f"{100 * sum(L.covered(v) for v in c) / len(c):.0f}% covered")
        print(f"  {name:9} {len(lms)} landmarks vs {len(plans)} plan(s); closest to any "
              f"plan {worst.get('corridor', float('nan')):.2f} m; " + "; ".join(cov))
    return fails


def self_test():
    """Plant a landmark where each rule forbids one; every plant must fail."""
    import math
    cases = []
    # In the corridor: a 0.3 m sphere at the midpoint of a SOLVED plan if
    # there is one (so the gate is shown reading solved plans), else the seed.
    for name in scenario_names():
        plans, _ = plans_for(name)
        label, cps = next(((lab, c) for lab, c in plans if not lab.endswith("_seed")), plans[0])
        mid = L.bezier_points(cps, 2)[1]
        cases.append((name, f"sphere at the middle of {label}",
                      {"name": "PLANT", "type": "sphere", "center": mid, "radius": 0.3},
                      "corridor"))
    # In an obstacle's sweep, away from every plan: a column obstacle's track
    # in wall (the 21-column wall, which no plan crosses while it stands).
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / "wall.json").read_text()))
    w0 = sc["obstacles"][0]["control_points"][0]
    cases.append(("wall", "box on the wall's bottom column, far from the path",
                  {"name": "PLANT", "type": "box", "center": [w0[0], w0[1], 3.5],
                   "size": [0.3, 0.3, 0.3], "yaw": 0.4}, "obstacle"))
    # Across loiter's sight lines and nowhere near the path: halfway up the
    # segment from the station to the seed's midpoint.
    sc = frames.scenario_to_sim(json.loads((ROOT / "scenarios" / "loiter.json").read_text()))
    st = sc["stations"][0]
    seed = json.loads((ROOT / "plans" / "loiter_seed.json").read_text())["control_points"]
    m = L.bezier_points(seed, 2)[1]
    p = [st[i] + 0.4 * (m[i] - st[i]) for i in range(3)]
    cases.append(("loiter", "cylinder 40% of the way up a sight line",
                  {"name": "PLANT", "type": "cylinder", "center": p, "radius": 0.1,
                   "height": 0.2}, "sight line"))
    # Below ground.
    cases.append(("fence3d", "sphere half buried, far from everything",
                  {"name": "PLANT", "type": "sphere", "center": [0.0, -9.0, 0.0],
                   "radius": 0.4}, "ground"))
    # A plan left in the planner's frame (the pre-frames.py kind).
    bad = 0
    print("self-test: each planted landmark must FAIL the gate")
    for name, what, lm, kind in cases:
        fails = check(name, extra=lm, quiet=True)
        hit = [f for f in fails if f.startswith("PLANT") and kind in f]
        ok = bool(hit)
        bad += not ok
        print(f"  {'caught' if ok else 'MISSED'}  {name:8} {what}: "
              f"{hit[0] if hit else fails}")
    clean = sum(len(check(n, quiet=True)) for n in scenario_names())
    print(f"  {'pass  ' if clean == 0 else 'FAIL  '}  the unplanted sets: {clean} failure(s)")
    # The plant must only be caught for the RIGHT reason: a gate that fails
    # everything would also "catch" every plant.
    return 1 if bad or clean else 0


def main(argv):
    if "--self-test" in argv:
        return self_test()
    names = [a for a in argv if not a.startswith("-")] or scenario_names()
    total = 0
    print(f"landmark gate: corridor {L.CORRIDOR_MARGIN_M} m, obstacle sweep "
          f"{L.OBSTACLE_MARGIN_M} m, sight line {L.SIGHT_MARGIN_M} m (lower bounds)")
    for name in names:
        fails = check(name)
        for f in fails:
            print(f"    FAIL {f}")
        total += len(fails)
    print("PASS: no landmark interferes with any plan, obstacle or sight line"
          if total == 0 else f"FAILED: {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
