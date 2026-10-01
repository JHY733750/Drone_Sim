#!/usr/bin/env python3
"""Move a plan solved before frames.py existed into its scenario's sim frame.

    python3 scripts/reframe_plan.py plans/fence3d_N8_seg2.json [more.json ...]

Such a plan has no 'frame' record: its control points are in the planner's
coordinates, and the launch and the bridge refuse it. For a scenario whose
transform is a pure TRANSLATION (s = 1) the map is exact and changes nothing
else: velocities, accelerations, the inflation, the speed cap and the
predicted clearance are all invariant under a translation, so the plan's
record stays true and only its control points and frame change.

For s != 1 (loiter) it REFUSES. Scaling space at fixed time scales every speed
and acceleration by s, which a plan's record would no longer describe, and the
obstacles were grown by the vehicle radius in planner metres -- 0.3 m there is
0.3 * s here, so the certificate would be about a much smaller vehicle. Those
plans have to be re-solved (scripts/solve_plan.py), which grows by 0.3 / s.

The mapping is frames.points_to_sim; nothing here does arithmetic on a
coordinate.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "dsim_planner"))
from dsim_planner import frames  # noqa: E402


def reframe(path):
    path = pathlib.Path(path)
    plan = json.loads(path.read_text())
    name = plan.get("scenario")
    frame = frames.frame_for(name)
    if plan.get("frame") is not None:
        if frames.same_frame(plan["frame"], frame):
            return f"unchanged  {path} (already in {frames.describe(frame)})"
        raise SystemExit(f"{path}: in frame {plan['frame']}, not the planner's and not "
                         f"the scenario's; re-solve it")
    if frame["scale"] != 1.0:
        raise SystemExit(f"{path}: scenario {name!r} is scaled (s = {frame['scale']:g}); "
                         f"a scaled plan's speeds and inflation no longer match its record. "
                         f"Re-solve it with scripts/solve_plan.py --scenario {name}")
    plan["control_points"] = frames.points_to_sim(plan["control_points"], frame)
    plan["frame"] = frame
    plan["_reframed_by"] = ("scripts/reframe_plan.py: translated from the planner's "
                            "coordinates; nothing else changed (s = 1)")
    head = {k: v for k, v in plan.items() if k != "control_points"}
    text = json.dumps(head, indent=1)
    rows = ",\n".join("  " + json.dumps([round(c, 9) for c in row])
                      for row in plan["control_points"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text[:-2] + ',\n "control_points": [\n' + rows + "\n ]\n}\n")
    tmp.replace(path)
    return f"reframed   {path} -> {frames.describe(frame)}"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    for p in argv:
        print(reframe(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
