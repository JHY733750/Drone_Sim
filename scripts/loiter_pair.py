#!/usr/bin/env python3
"""Measure loiter_sim as a PAIR: does the line-of-sight constraint do anything?

Run INSIDE the container (the solver is a Rust extension):

    python3 scripts/loiter_pair.py                      # the registered configs
    python3 scripts/loiter_pair.py -N 8 --n-seg 8
    python3 scripts/loiter_pair.py --phase 0 15 30      # a phase scan
    python3 scripts/loiter_pair.py --phase 0 --trust 0.5   # trust-radius sensitivity

Two solves per config, identical in everything but one thing: the CONSTRAINED
run is the scenario as solve_plan.py solves it (bodies grown by the vehicle
radius, the scenario's own solve flags), and the BASELINE is the same call
with the stations removed, so it builds no shadow rows at all. Then three
conditions, the planner's own (`scenario_loiter` docstring), each measured
with the planner's los_margin_at against the REAL bodies on 4001 samples
uniform in the curve parameter -- never with a certificate, because the
baseline's occlusion certificate is 0.0 by construction for a trajectory that
loses the link outright:

  (a) the constrained run certifies (converged, keep-out and occlusion
      certificates zero, no occlusion plane dropped) AND its sampled margin
      stays positive -- two computations of one fact, which must agree;
  (b) the baseline's sampled margin goes negative: without the rows the link
      is lost, or the constraint is decoration;
  (c) the constrained PATH on the baseline's SCHEDULE (same curve parameter,
      the baseline's time coordinate) goes negative too: the path alone does
      not keep the link, the retiming does.

It also reports the reverse graft, how far the constrained path strays from
the corridor's axis, and the margin while the vehicle HOLDS the goal from its
arrival to the end of the horizon, which no certificate covers.

Exit status 0 when every config measured satisfies all three, 1 otherwise.
FAILS IF: any of (a), (b), (c) fails for a config -- including the case the
design exists to exclude, a baseline that never loses the link.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "dsim_planner"))
import solve_plan  # noqa: E402
import sim_scenarios  # noqa: E402
from dsim_planner import frames  # noqa: E402

NAME = "loiter_sim"
SAMPLES = 4001


def scenario(phase):
    """The scenario file, or the factory at another phase (a scan)."""
    path = ROOT / "scenarios" / f"{NAME}.json"
    sc = json.loads(path.read_text())
    if frames.frame_for(NAME)["scale"] != 1.0 or not sc.get("native"):
        raise SystemExit(f"{NAME} is not sim-native; these lengths would be in the wrong frame")
    if phase is None:
        return sc
    factory, _, _ = sim_scenarios.SIM_SCENARIOS[NAME]
    fresh = factory(phase_deg=phase)
    out = dict(sc)
    # Rounded exactly as import_scenarios.render() writes the file, so that
    # --phase P with P the file's phase solves the FILE's problem. It matters:
    # the 3e-10 s by which an unrounded window time differs from the written
    # one has been measured to flip N8_seg16's certificate at 15 deg.
    out["obstacles"] = [dict(o, type="sphere",
                             control_points=[[round(c, 9) for c in row]
                                             for row in o["control_points"]])
                        for o in fresh["obstacles"]]
    return out


def solve_one(sc, cfg, kw, grow, stations):
    from spacetime_bezier import optimize_scenario
    problem = {k: v for k, v in sc.items() if k != "stations"}
    if stations:
        problem["stations"] = sc["stations"]
    problem = solve_plan.inflate(problem, grow)
    out = optimize_scenario(problem, [cfg], verbose=False, **kw)
    return out["results"][f"N{cfg[0]}_seg{cfg[1]}"]


def measure(sc, cfg, kw, grow):
    import numpy as np
    from spacetime_bezier.geometry import bezier_curve, los_margin_at

    con = solve_one(sc, cfg, kw, grow, stations=True)
    base = solve_one(sc, cfg, kw, grow, stations=False)
    station = sc["stations"][0]
    real = sc["obstacles"]                      # NOT grown: line of sight is a segment
    fc = bezier_curve(np.asarray(con["control_points"], float), num_pts=SAMPLES)
    fb = bezier_curve(np.asarray(base["control_points"], float), num_pts=SAMPLES)
    m_con = los_margin_at(fc[:, :3], fc[:, 3], station, real)
    m_base = los_margin_at(fb[:, :3], fb[:, 3], station, real)
    graft = los_margin_at(fc[:, :3], fb[:, 3], station, real)
    rgraft = los_margin_at(fb[:, :3], fc[:, 3], station, real)
    blocked = m_base < 0.0
    # The hold: the vehicle sits at the goal from arrival to T.
    t_hold = np.linspace(float(fc[-1, 3]), float(sc["T"]), 400)
    hold = los_margin_at(np.repeat(fc[-1:, :3], len(t_hold), axis=0), t_hold, station, real)
    certified = bool(con["converged"] and con["certified"] and con.get("occlusion_certified")
                     and float(con.get("occlusion_planes_dropped", 1)) == 0.0)
    r = {
        "config": f"N{cfg[0]}_seg{cfg[1]}",
        "a_certified": certified,
        "a_min_los": float(np.min(m_con)),
        "b_min_los": float(np.min(m_base)),
        "b_blocked_s": ([round(float(fb[blocked, 3].min()), 3), round(float(fb[blocked, 3].max()), 3)]
                        if blocked.any() else None),
        "c_graft_min_los": float(np.min(graft)),
        "reverse_graft_min_los": float(np.min(rgraft)),
        "arrival_con_s": float(con["arrival_time"]),
        "arrival_base_s": float(base["arrival_time"]),
        "lateral_m": float(np.max(np.abs(fc[:, 1] - sc["start"][1]))),
        "hold_min_los": float(np.min(hold)) if len(hold) else float("inf"),
        "iterations": int(con["iterations"]),
        "stop": con.get("stop_label"),
    }
    r["all3"] = bool(certified and r["a_min_los"] > 0.0 and r["b_min_los"] < 0.0
                     and r["c_graft_min_los"] < 0.0)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-N", type=int, default=None)
    ap.add_argument("--n-seg", type=int, default=None)
    ap.add_argument("--phase", type=float, nargs="*", default=None,
                    help="orbit phase(s) in degrees; default: the scenario file's")
    ap.add_argument("--trust", type=float, default=None,
                    help="override the scenario's trust_radius (a sensitivity scan)")
    args = ap.parse_args(argv)

    base_sc = json.loads((ROOT / "scenarios" / f"{NAME}.json").read_text())
    configs = ([(args.N, args.n_seg)] if args.N is not None and args.n_seg is not None
               else [tuple(c) for c in base_sc["configs"]])
    flags = base_sc.get("solve") or {}
    kw = dict(min_dt=0.1, max_iter=200,
              time_weight=float(flags.get("time_weight", 0.0)),
              free_arrival_time=bool(flags.get("free_arrival_time", False)))
    if flags.get("v_max") is not None:
        kw["v_max"] = float(flags["v_max"])
    if flags.get("elastic_weight") is not None:
        kw["elastic_weight"] = float(flags["elastic_weight"])
    grow = solve_plan.vehicle()["radius_m"]
    print(f"{NAME}: bodies grown {grow} m for the solve, measured against the real "
          f"ones; flags {flags}")
    ok = True
    for phase in (args.phase or [None]):
        sc = scenario(phase)
        if args.trust is not None:
            sc = dict(sc, trust_radius=args.trust)
        for cfg in configs:
            r = measure(sc, cfg, kw, grow)
            ok &= r["all3"]
            tag = "file" if phase is None else f"{phase:g} deg"
            print(f"  phase {tag:>8} {r['config']:9} (a) cert={r['a_certified']} "
                  f"los={r['a_min_los']:+.3f}  (b) {r['b_min_los']:+.3f} blocked {r['b_blocked_s']}  "
                  f"(c) {r['c_graft_min_los']:+.3f}  rev {r['reverse_graft_min_los']:+.3f}  "
                  f"arrive {r['arrival_con_s']:.2f}/{r['arrival_base_s']:.2f} s  "
                  f"lateral {r['lateral_m']:.3f} m  hold {r['hold_min_los']:+.3f}  "
                  f"{'ALL THREE' if r['all3'] else 'no'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
