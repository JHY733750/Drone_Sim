#!/usr/bin/env python3
"""Import the planner's scenarios -- and render the sim's own -- into scenarios/*.json.

Run INSIDE the container, where spacetime_bezier imports:

    python3 scripts/import_scenarios.py            # regenerate
    python3 scripts/import_scenarios.py --check    # fail if any drifted

These files were hand-copied once and that was a standing drift risk: a
scenario that means one thing to the solver and another to the referee is the
whole failure the generator chain exists to prevent. Now they are GENERATED
from SCENARIO_MAP, in the planner's own canonical form -- obstacles as lifted
control points in (x, y, [z], t), with the active window intrinsic to the
first and last of them -- so nothing is transcribed and `--check` catches a
planner change that the sim has not been told about.

Two things this adds on top of the planner's dict, both under keys the planner
ignores:

  spatial_dim / lift_z
      Four of the seven scenarios are planned in TWO spatial dimensions plus
      time. They are NOT lifted into 3D by making their obstacles spheres:
      that would hand the solver an escape route over the top that the 2D
      problem never had, and the scenario would stop being the scenario. They
      stay 2D for the solver, and the sim flies the resulting plan at a fixed
      altitude, with each obstacle treated as the vertical COLUMN it is -- a
      disc extruded through every height, which is what a 2D obstacle means.

  sim.start_s / sim.spawn
      When scenario t=0 happens in simulated seconds, and where the vehicle
      starts. It spawns AT the plan's first point and settles there; the
      transit from a pad is not part of any plan and nothing certifies it, so
      there is no reason to fly one.

And two things that are not the planner's at all, both printed by `--check`:

  REPLACED (scripts/sim_scenarios.py)
      Planner scenarios the sim deliberately does NOT import, each with what
      replaced it and why. Today: `loiter`, whose 200 m scene does not survive
      being shrunk to the sim's size (see sim_scenarios.py). `--check` fails if
      the planner stops defining a replaced name (the record is stale) or if a
      file for one turns up in scenarios/ (it would be flown as if current).

  SIM_SCENARIOS (scripts/sim_scenarios.py)
      Scenarios AUTHORED in the sim, in sim metres -- frame s = 1, c = 0,
      listed in dsim_planner/frames.py NATIVE. Rendered here by the same code
      into the same schema, plus "native": true and the solve flags they are
      meant to be solved with; `--check` holds the file to its factory exactly
      as it holds an imported one to the planner. A sim-native name may not
      also be a planner name.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "scenarios"
sys.path.insert(0, str(ROOT / "src" / "dsim_planner"))
sys.path.insert(0, str(ROOT / "scripts"))
from dsim_planner import frames  # noqa: E402
import sim_scenarios  # noqa: E402  (scripts/sim_scenarios.py, beside this file)

#: Altitude a 2-spatial-dimension scenario is flown at, in metres.
LIFT_Z = 1.5
#: Simulated seconds before scenario t=0: the vehicle spawns at the plan's
#: first point and holds there. Long enough for the attitude estimator to
#: settle (it initialises from the first mag and accel samples) and for the
#: viewer to be connected before anything moves.
START_S = 5.0


def scenario_doc(name, factory, configs, native=None):
    """The file for one scenario. `native` is None for one imported from the
    planner, or the solve flags of one authored in the sim (sim_scenarios.py)."""
    sc = factory()
    spatial = len(sc["start"]) - 1
    if spatial not in (2, 3):
        raise SystemExit(f"{name}: {spatial} spatial dimensions; this sim flies in 3")

    def lift(point):
        """(x, y, t) -> (x, y, LIFT_Z, t), or pass a 3D point through."""
        p = [float(v) for v in point]
        return p if spatial == 3 else [p[0], p[1], LIFT_Z, p[-1]]

    obstacles = []
    for o in sc["obstacles"]:
        cps = [lift(row) for row in o["control_points"]]
        entry = {
            "name": o.get("name") or f"O{len(obstacles)}",
            # A 2D obstacle is a disc at every altitude, not a ball at one.
            "type": "sphere" if spatial == 3 else "column",
            "radius": float(o.get("radius", o.get("r"))),
            "control_points": cps,
        }
        if o.get("color"):
            entry["color"] = o["color"]
        obstacles.append(entry)

    if native is None:
        made_by = ("scripts/import_scenarios.py from the planner's SCENARIO_MAP "
                   "-- do not edit; edit the planner and re-import")
    else:
        made_by = ("scripts/import_scenarios.py from scripts/sim_scenarios.py -- "
                   "SIM-NATIVE: authored in sim metres, frame s = 1, c = 0 "
                   "(frames.NATIVE); do not edit, edit the factory and re-import")
    doc = {
        "_generated_by": made_by,
        "name": name,
        "title": sc.get("title", name),
        "spatial_dim": spatial,
        "start": lift(sc["start"]),
        "end": lift(sc["end"]),
        "T": float(sc["T"]),
        # The (degree, segments) pairs the planner records as worth solving for
        # this scene. `make solve SOLVE_ARGS=--sweep` walks them.
        "configs": [[int(a), int(b)] for a, b in configs],
        "obstacles": obstacles,
        "sim": {"start_s": START_S, "spawn": lift(sc["start"])[:3]},
    }
    if spatial == 2:
        doc["lift_z"] = LIFT_Z
    if native is not None:
        # Read by frames.check_table_centre (the identity frame is claimed in
        # BOTH places or neither) and by solve_plan.py (the flags).
        doc["native"] = True
        doc["solve"] = dict(native)
    for key in ("init_curve", "stations", "coord_bounds", "trust_radius"):
        if sc.get(key) is not None:
            doc[key] = sc[key]
    return doc


def render(doc):
    """Stable text: control points one per line, everything else indented."""
    obstacles = doc.pop("obstacles")
    body = json.dumps(doc, indent=1)
    doc["obstacles"] = obstacles
    rows = []
    for o in obstacles:
        cps = ",\n".join("    " + json.dumps([round(c, 9) for c in row])
                         for row in o["control_points"])
        head = {k: v for k, v in o.items() if k != "control_points"}
        rows.append("  " + json.dumps(head)[:-1] + ', "control_points": [\n'
                    + cps + "\n  ]}")
    return body[:-2] + ',\n "obstacles": [\n' + ",\n".join(rows) + "\n ]\n}\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    try:
        from spacetime_bezier import SCENARIO_MAP
    except ImportError as exc:
        raise SystemExit(
            f"the space-time planner is not importable ({exc}).\n"
            f"Run this INSIDE the container after `make planner`. See docs/PLANNER.md.")

    OUT_DIR.mkdir(exist_ok=True)
    problems = []
    # The records that say what the sim does instead of importing, checked
    # before anything is written: a stale one is a claim about a planner
    # scenario that no longer exists.
    for name, (by, why) in sorted(sim_scenarios.REPLACED.items()):
        if name not in SCENARIO_MAP:
            problems.append(f"REPLACED names {name!r}, which the planner no longer "
                            f"defines; drop the record from scripts/sim_scenarios.py")
        if by not in sim_scenarios.SIM_SCENARIOS:
            problems.append(f"REPLACED says {name!r} is replaced by {by!r}, which "
                            f"scripts/sim_scenarios.py does not define")
    for name in sorted(sim_scenarios.SIM_SCENARIOS):
        if name in SCENARIO_MAP:
            problems.append(f"sim-native {name!r} is also a planner scenario; one "
                            f"name, two problems -- rename it")
        if name not in frames.NATIVE:
            problems.append(f"sim-native {name!r} is not in dsim_planner/frames.py "
                            f"NATIVE, so its identity frame is not claimed")
    for name in sorted(frames.NATIVE):
        if name not in sim_scenarios.SIM_SCENARIOS:
            problems.append(f"frames.NATIVE lists {name!r} but scripts/sim_scenarios.py "
                            f"does not define it")
    if problems:
        for p in problems:
            print("ERROR  " + p)
        return 1

    jobs = [(name, factory, configs, None)
            for name, (factory, configs) in sorted(SCENARIO_MAP.items())
            if name not in sim_scenarios.REPLACED]
    jobs += [(name, factory, configs, flags)
             for name, (factory, configs, flags) in sorted(sim_scenarios.SIM_SCENARIOS.items())]

    stale, seen = [], set()
    for name, factory, configs, flags in jobs:
        path = OUT_DIR / f"{name}.json"
        seen.add(path.name)
        new = render(scenario_doc(name, factory, configs, native=flags))
        old = path.read_text() if path.exists() else None
        if args.check:
            if old != new:
                stale.append(path.name)
        else:
            path.write_text(new)
            doc = json.loads(new)
            print(f"{'unchanged' if old == new else 'wrote    '}  scenarios/{path.name}"
                  f"  ({doc['spatial_dim']}D+t, {len(doc['obstacles'])} obstacles, "
                  f"T={doc['T']:g}s{', SIM-NATIVE' if flags is not None else ''})")

    print("\nnot imported, deliberately (scripts/sim_scenarios.py REPLACED):")
    for name, (by, why) in sorted(sim_scenarios.REPLACED.items()):
        print(f"  {name} -> {by}: {why}")

    orphans = sorted(p.name for p in OUT_DIR.glob("*.json") if p.name not in seen)
    if orphans:
        # A scenario file with no scenario behind it would still generate a
        # world and a seed, and would be flyable and ungradeable. A file for a
        # REPLACED name is one of these, and the worst kind: it looks current.
        print(f"\nscenarios/ has files nothing defines: {orphans}")
        print("  delete them, or add them to the planner's SCENARIO_MAP "
              "(or to scripts/sim_scenarios.py)")
        return 1

    if args.check:
        if stale:
            print("STALE (run scripts/import_scenarios.py):")
            for s in stale:
                print("  scenarios/" + s)
            return 1
        print("every scenario matches the planner, or its sim_scenarios.py factory")
        return 0
    print("\nnow run: python3 scripts/gen_assets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
