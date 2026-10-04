"""Static SLAM landmarks: what they are, where they may go, and the check.

A depth-camera SLAM needs something to lock onto, and a scenario's obstacles
are no use for that: they move, they switch off, and they are not in the
physics. So each scenario gets a set of static landmarks -- boxes, vertical
cylinders and spheres -- placed by a SEEDED generator, written to
config/landmarks_<scenario>.yaml, drawn by the viewer as scenery, and kept OUT
of the referee's obstacle list and out of the physics. They must not change
what a run means, so they keep out of three regions:

  the flight corridor   every point of every plan for the scenario (the seed
                        and each solved plan), with the vehicle holding at
                        start and goal: the landmark's SURFACE stays at least
                        CORRIDOR_MARGIN_M from it. 1.0 m = the 0.3 m vehicle
                        envelope plus 0.7 m.
  obstacle sweeps       every moving obstacle's swept volume -- its centre
                        path over the whole window, at its radius; a column
                        at every height -- plus OBSTACLE_MARGIN_M.
  sight lines           every segment from a station to any point of any plan,
                        plus SIGHT_MARGIN_M. Only loiter_sim has a station.

The generator cannot see solved plans -- they are gitignored outputs, and a
generated file has to come out the same on every machine for `gen_assets.py
--check` to mean anything. So it keeps out of the SEED's corridor widened by a
per-scenario RESERVE_M: if every solved plan stays within RESERVE_M of the
seed, the margins above hold for it too (distance is 1-Lipschitz, so a plan
point within R of a seed point moves every distance by at most R, and a sight
segment's far end moving by R moves any point of it by at most R). The
reserve is sized from the plans solved on 2026-10-01; scripts/check_landmarks.py
checks every plan actually present, directly, so a re-solve that wanders
further fails the gate instead of silently clipping a landmark.

Visible on purpose: the bridge flies yaw 0, so the camera (body +x, 87 x 58
deg, 0.28..10 m) always looks along world +x. The generator anchors each
candidate at the least-covered point of the seed and drops it ahead of that
point, within the field of view, and stops only when every point of the seed
sees at least VISIBLE_MIN landmark centres with some on each side OF THE
CAMERA'S AXIS. Not of the path: on the diagonal scenarios (curve, diverse,
original head 42-45 deg off +x) every ray of a +/-43.5 deg field of view
pointed along +x lands on the path's right, so a landmark on its left is
never seen. That is a consequence of flying yaw 0, not of this generator; a
bridge that yawed along the path would need the generator to know the yaw.
Sizes, yaws and types are mixed so scan matching has corners and curvature
in every direction -- a row of identical parallel boxes is degenerate along
the row -- and box yaws are stratified (see _PHI) rather than left to chance.

Distances are computed by signed-distance functions sampled along segments at
a known step h, minus h/2: an SDF is 1-Lipschitz, so the sampled minimum
minus half the step is a LOWER bound on the true minimum. The gate can say
"too close" when it is not; it cannot say "clear" when it is not.
"""
from __future__ import annotations

import math
import random
import zlib

#: Landmark surface to any point of any plan (sim metres).
CORRIDOR_MARGIN_M = 1.0
#: Landmark surface to any moving obstacle's swept surface.
OBSTACLE_MARGIN_M = 0.5
#: Landmark surface to any station -> path sight segment.
SIGHT_MARGIN_M = 0.5
#: Landmark surface to landmark surface, so they read as separate objects.
SEPARATION_M = 0.6

#: How far (sim metres) a solved plan may stray from the seed before the
#: generator's guarantee stops covering it. Measured max deviation of the
#: plans solved 2026-10-01, rounded up with room: curve 0.86 (uncertified),
#: diverse 2.38, door3d 0.05, fence3d 1.70 (the climb), original 1.72,
#: wall 0.39; loiter_sim (2026-10-05) 0.19 at N8_seg8, 0.23 at N8_seg16 --
#: the corridor is only 0.2 m wide and the band 0.3 m deep, so a plan that
#: certifies cannot stray much further than that.
RESERVE_M = {
    "curve": 1.5, "diverse": 3.0, "door3d": 1.0, "fence3d": 2.0,
    "loiter_sim": 0.5, "original": 2.0, "wall": 1.0,
}

#: The camera, from the task's D435 figures. Body +x, yaw 0 in flight.
CAM_HFOV_DEG = 87.0
CAM_VFOV_DEG = 58.0
CAM_MIN_M = 0.28
CAM_MAX_M = 10.0

#: Coverage the generator insists on at every point of the seed: landmark
#: CENTRES inside a slightly shrunken frustum (so a landmark half out of
#: frame is not counted), at least one on each side of the camera's axis.
VISIBLE_MIN = 4
COUNT_MIN, COUNT_MAX = 10, 24

#: Sampling steps (metres) for the lower-bound distances. See module top.
STEP_M = 0.05


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _combine(d):
    """SDF of an axis-aligned slab intersection from per-axis distances."""
    outside = math.sqrt(sum(max(c, 0.0) ** 2 for c in d))
    inside = min(max(d), 0.0)
    return outside + inside


def sdf(lm, p):
    """Signed distance from point p (x, y, z) to landmark lm. Negative inside."""
    c = lm["center"]
    t = lm["type"]
    if t == "sphere":
        return _norm([p[i] - c[i] for i in range(3)]) - lm["radius"]
    if t == "cylinder":
        dxy = math.hypot(p[0] - c[0], p[1] - c[1]) - lm["radius"]
        dz = abs(p[2] - c[2]) - 0.5 * lm["height"]
        return _combine([dxy, dz])
    if t == "box":
        cy_, sy_ = math.cos(lm["yaw"]), math.sin(lm["yaw"])
        dx, dy = p[0] - c[0], p[1] - c[1]
        q = (cy_ * dx + sy_ * dy, -sy_ * dx + cy_ * dy, p[2] - c[2])
        return _combine([abs(q[i]) - 0.5 * lm["size"][i] for i in range(3)])
    raise ValueError(f"unknown landmark type {t!r}")


def sdf_xy(lm, p):
    """Horizontal distance from the vertical LINE through p to the landmark:
    what a column obstacle (a disc at every height) has to clear."""
    c = lm["center"]
    t = lm["type"]
    if t in ("sphere", "cylinder"):
        return math.hypot(p[0] - c[0], p[1] - c[1]) - lm["radius"]
    cy_, sy_ = math.cos(lm["yaw"]), math.sin(lm["yaw"])
    dx, dy = p[0] - c[0], p[1] - c[1]
    q = (cy_ * dx + sy_ * dy, -sy_ * dx + cy_ * dy)
    return _combine([abs(q[i]) - 0.5 * lm["size"][i] for i in range(2)])


def bounding_radius(lm):
    if lm["type"] == "sphere":
        return lm["radius"]
    if lm["type"] == "cylinder":
        return math.hypot(lm["radius"], 0.5 * lm["height"])
    return 0.5 * _norm(lm["size"])


def bottom(lm):
    if lm["type"] == "box":
        return lm["center"][2] - 0.5 * lm["size"][2]
    if lm["type"] == "cylinder":
        return lm["center"][2] - 0.5 * lm["height"]
    return lm["center"][2] - lm["radius"]


def _seg_point_dist(a, b, p):
    ab = [b[i] - a[i] for i in range(3)]
    L2 = sum(c * c for c in ab)
    u = 0.0 if L2 <= 0.0 else max(0.0, min(1.0, sum((p[i] - a[i]) * ab[i] for i in range(3)) / L2))
    return _norm([a[i] + u * ab[i] - p[i] for i in range(3)])


def seg_distance(lm, a, b, step=STEP_M, f=sdf):
    """LOWER bound on the distance from segment ab to the landmark."""
    # Cheap exit: the segment is far from the bounding sphere.
    far = _seg_point_dist(a, b, lm["center"]) - bounding_radius(lm)
    if far > 1.0:
        return far
    L = _norm([b[i] - a[i] for i in range(3)])
    n = max(1, int(math.ceil(L / step)))
    h = L / n
    best = math.inf
    for k in range(n + 1):
        u = k / n
        best = min(best, f(lm, [a[i] + u * (b[i] - a[i]) for i in range(3)]))
    return best - 0.5 * h


def polyline_distance(lm, pts, step=STEP_M, f=sdf):
    """LOWER bound on the distance from a polyline to the landmark."""
    if len(pts) == 1:
        return f(lm, pts[0])
    return min(seg_distance(lm, pts[i], pts[i + 1], step, f) for i in range(len(pts) - 1))


def bezier_points(cps, n):
    """n + 1 points of a Bezier's spatial part, uniform in its parameter."""
    out = []
    for k in range(n + 1):
        u = k / n
        pts = [list(r[:3]) for r in cps]
        while len(pts) > 1:
            pts = [[(1 - u) * a + u * b for a, b in zip(pts[i], pts[i + 1])]
                   for i in range(len(pts) - 1)]
        out.append(pts[0])
    return out


def curve_polyline(cps, step=STEP_M):
    """A Bezier's path as a polyline fine enough that no chord exceeds `step`
    much: doubled until it does not. The chord is a lower-bound sampler, and
    the bound is in the segments, so the polyline only needs to be close to
    the curve -- its deviation is added explicitly in `curve_slack`."""
    n = 64
    while True:
        pts = bezier_points(cps, n)
        longest = max(_norm([b[i] - a[i] for i in range(3)]) for a, b in zip(pts, pts[1:]))
        if longest <= step or n >= 16384:
            return pts, longest
        n *= 2


def curve_slack(longest_chord):
    """How far a curve can be from its chord polyline, bounded crudely by the
    chord length itself (a chord of length h between two points of a curve
    whose arc between them is about h never leaves a ball of radius h around
    either end). Subtracted from every distance to a sampled curve."""
    return longest_chord


# --------------------------------------------------------------------------
# what a scenario forbids
# --------------------------------------------------------------------------
class Field:
    """Everything a landmark has to keep clear of, for one scenario, in the
    sim frame: the paths (seed or plans), the obstacles' swept centre paths
    and the stations."""

    def __init__(self, scenario_sim, paths, reserve=0.0):
        self.name = scenario_sim["name"]
        self.reserve = float(reserve)
        self.paths = []           # [(label, polyline, slack)]
        for label, cps in paths:
            pts, chord = curve_polyline(cps)
            self.paths.append((label, pts, curve_slack(chord)))
        self.obstacles = []       # [(name, type, radius, polyline, slack)]
        for o in scenario_sim["obstacles"]:
            pts, chord = curve_polyline(o["control_points"])
            self.obstacles.append((o["name"], o["type"], float(o["radius"]), pts,
                                   curve_slack(chord)))
        self.stations = [list(st)[:3] for st in (scenario_sim.get("stations") or [])]
        # The sight fan, thinned: segments to path points at most STEP_M apart
        # cover every intermediate point to within STEP_M/2 at the far end,
        # which moves the segment by at most that much anywhere.
        self.fans = []            # [(label, station, [end points], slack)]
        for st in self.stations:
            for label, pts, slack in self.paths:
                self.fans.append((label, st, pts, slack + 0.5 * STEP_M))

    def violations(self, lm, first_only=False):
        """[(kind, against, distance, required)] for every rule `lm` breaks."""
        out = []
        r = self.reserve
        if bottom(lm) < -1e-9:
            out.append(("ground", "z=0", bottom(lm), 0.0))
            if first_only:
                return out
        for label, pts, slack in self.paths:
            d = polyline_distance(lm, pts) - slack
            if d < CORRIDOR_MARGIN_M + r:
                out.append(("corridor", label, d, CORRIDOR_MARGIN_M + r))
                if first_only:
                    return out
        for name, typ, rad, pts, slack in self.obstacles:
            f = sdf_xy if typ == "column" else sdf
            if typ == "column":
                # A column's swept volume is its xy track at every height;
                # flatten the track so the segment sampler works in the plane.
                pts = [[p[0], p[1], lm["center"][2]] for p in pts]
            d = polyline_distance(lm, pts, f=f) - slack - rad
            if d < OBSTACLE_MARGIN_M:
                out.append(("obstacle", name, d, OBSTACLE_MARGIN_M))
                if first_only:
                    return out
        for label, st, ends, slack in self.fans:
            # One segment per path point. The generator only needs yes/no and
            # stops at the first offender; the gate reports the true minimum.
            worst = math.inf
            for e in ends:
                d = seg_distance(lm, st, e) - slack
                worst = min(worst, d)
                if first_only and worst < SIGHT_MARGIN_M + r:
                    break
            if worst < SIGHT_MARGIN_M + r:
                out.append(("sight line", f"station{self.stations.index(st)}->{label}",
                            worst, SIGHT_MARGIN_M + r))
                if first_only:
                    return out
        return out


def separation_ok(lm, others):
    """Coarse: bounding spheres apart by SEPARATION_M. Only used to keep the
    generator from stacking landmarks; no claim rests on it."""
    for o in others:
        gap = (_norm([lm["center"][i] - o["center"][i] for i in range(3)])
               - bounding_radius(lm) - bounding_radius(o))
        if gap < SEPARATION_M:
            return False
    return True


# --------------------------------------------------------------------------
# visibility
# --------------------------------------------------------------------------
def visible(cam, lm, shrink=0.9):
    """Is the landmark's centre in the camera's frustum? Yaw 0, level."""
    c = lm["center"]
    dx, dy, dz = c[0] - cam[0], c[1] - cam[1], c[2] - cam[2]
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dx <= 0.0 or not (CAM_MIN_M <= rng <= CAM_MAX_M * (0.5 + 0.5 * shrink)):
        return False
    az = math.degrees(math.atan2(dy, dx))
    el = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return abs(az) <= 0.5 * CAM_HFOV_DEG * shrink and abs(el) <= 0.5 * CAM_VFOV_DEG * shrink


def coverage(cams, lms):
    """[(n visible, n left, n right)] at each camera position."""
    out = []
    for cam in cams:
        v = [lm for lm in lms if visible(cam, lm)]
        out.append((len(v), sum(lm["center"][1] > cam[1] for lm in v),
                    sum(lm["center"][1] < cam[1] for lm in v)))
    return out


def covered(c):
    return c[0] >= VISIBLE_MIN and c[1] >= 1 and c[2] >= 1


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------
def _r3(v):
    return round(float(v), 3)


#: Golden-ratio stratification of box yaws modulo a right angle: the n-th box
#: is turned (n * 0.618...) mod 1 of 90 deg, plus a random multiple of 90 and a
#: little jitter. Consecutive boxes then differ by 34 deg modulo 90 (less
#: jitter: >= 30), so no two present parallel faces, however few there are.
_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _candidate(rng, anchor, placed, reserve):
    """One landmark ahead of `anchor`, inside the camera's view of it.
    `placed` is the list accepted so far, which balances the types and
    stratifies the box yaws."""
    counts = {t: sum(lm["type"] == t for lm in placed) for t in ("box", "cylinder", "sphere")}
    # Mostly the least-used type (boxes weighted up: they carry the yaw
    # information the round shapes cannot), sometimes any.
    weight = {"box": counts["box"] / 1.5, "cylinder": counts["cylinder"], "sphere": counts["sphere"]}
    kind = (min(weight, key=lambda t: (weight[t], rng.random())) if rng.random() < 0.8
            else rng.choice(("box", "cylinder", "sphere")))
    az = math.radians(rng.uniform(-38.0, 38.0))
    d = rng.uniform(CORRIDOR_MARGIN_M + reserve + 0.8, 8.0)
    x, y = anchor[0] + d * math.cos(az), anchor[1] + d * math.sin(az)
    # Heights spread across the vertical field of view at that distance.
    zmax = anchor[2] + d * math.tan(math.radians(0.4 * CAM_VFOV_DEG))
    if kind == "box":
        sx, sy = rng.uniform(0.4, 1.8), rng.uniform(0.3, 1.4)
        sz = rng.uniform(0.4, max(0.6, min(3.0, zmax)))
        yaw = (((counts["box"] * _PHI) % 1.0) * 0.5 * math.pi
               + rng.choice((0.0, 0.5 * math.pi)) + rng.uniform(-0.03, 0.03))
        lm = {"type": "box", "center": [x, y, 0.5 * sz], "size": [sx, sy, sz],
              "yaw": yaw}
    elif kind == "cylinder":
        r = rng.uniform(0.15, 0.6)
        h = rng.uniform(0.8, max(1.0, min(3.5, zmax + 0.5)))
        lm = {"type": "cylinder", "center": [x, y, 0.5 * h], "radius": r, "height": h}
    else:
        r = rng.uniform(0.25, 0.7)
        z = rng.uniform(r + 0.2, max(r + 0.3, min(3.2, zmax)))
        lm = {"type": "sphere", "center": [x, y, z], "radius": r}
    # Round FIRST, so what is checked is exactly what is written.
    lm["center"] = [_r3(v) for v in lm["center"]]
    for key in ("radius", "yaw"):
        if key in lm:
            lm[key] = _r3(lm[key])
    # A standing height is rounded to 2 mm so that half of it is a whole
    # millimetre: the centre is then exactly half the height and the bottom
    # exactly 0. Rounded to 1 mm, half of 1.235 is 0.6175, written 0.617 or
    # 0.618, and half of all boxes either sank 0.5 mm or floated.
    if "size" in lm:
        lm["size"] = [_r3(v) for v in lm["size"]]
        lm["size"][2] = _r2mm(lm["size"][2])
        lm["center"][2] = _r3(0.5 * lm["size"][2])
    if lm["type"] == "cylinder":
        lm["height"] = _r2mm(lm["height"])
        lm["center"][2] = _r3(0.5 * lm["height"])
    return lm


def _r2mm(v):
    return round(round(float(v) * 500.0) / 500.0, 3)


def generate(scenario_sim, seed_cps):
    """The landmarks for one scenario: deterministic in the scenario's name
    and the seed plan, nothing else."""
    name = scenario_sim["name"]
    reserve = RESERVE_M[name]
    field = Field(scenario_sim, [("seed", seed_cps)], reserve=reserve)
    cams, _ = curve_polyline(seed_cps, step=0.25)
    rng = random.Random(zlib.crc32(f"landmarks:{name}".encode()))
    lms = []
    attempts = 0
    # A point of the seed where nothing can be placed in view (diverse's
    # first metre: its only left-of-axis view is swept by mover C) would
    # otherwise soak up every attempt. Give up on a point after this many.
    failed = [0] * len(cams)
    while len(lms) < COUNT_MAX and attempts < 6000:
        attempts += 1
        cov = coverage(cams, lms)
        if len(lms) >= COUNT_MIN and all(covered(c) for c in cov):
            break
        # The worst-covered point of the seed, ties broken by the generator.
        score = [c[0] + 2 * min(c[1], 1) + 2 * min(c[2], 1) if failed[i] < 300 else 1e9
                 for i, c in enumerate(cov)]
        worst = min(score)
        if worst >= 1e9:
            break
        pick = rng.choice([i for i, v in enumerate(score) if v == worst])
        lm = _candidate(rng, cams[pick], lms, reserve)
        if not separation_ok(lm, lms) or field.violations(lm, first_only=True):
            failed[pick] += 1
            continue
        lms.append(lm)
    for i, lm in enumerate(lms):
        lm["name"] = f"L{i}"
    cov = coverage(cams, lms)
    return lms, {"attempts": attempts, "min_visible": min(c[0] for c in cov),
                 "covered_frac": sum(covered(c) for c in cov) / len(cov)}


# --------------------------------------------------------------------------
# the file, in exactly the format the SLAM side reads
# --------------------------------------------------------------------------
def _num(v):
    s = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def render_yaml(lms, header_lines):
    lines = [f"# {h}" if h else "#" for h in header_lines]
    lines.append("landmarks:")
    for lm in lms:
        c = ", ".join(_num(v) for v in lm["center"])
        if lm["type"] == "box":
            s = ", ".join(_num(v) for v in lm["size"])
            lines.append(f"  - {{name: {lm['name']}, type: box, center: [{c}], "
                         f"size: [{s}], yaw: {_num(lm['yaw'])}}}")
        elif lm["type"] == "cylinder":
            lines.append(f"  - {{name: {lm['name']}, type: cylinder, center: [{c}], "
                         f"radius: {_num(lm['radius'])}, height: {_num(lm['height'])}}}")
        else:
            lines.append(f"  - {{name: {lm['name']}, type: sphere, center: [{c}], "
                         f"radius: {_num(lm['radius'])}}}")
    return "\n".join(lines) + "\n"


def parse_yaml(text):
    """Read a landmarks file back, validating the shape the SLAM side expects."""
    import yaml
    doc = yaml.safe_load(text) or {}
    out = []
    for i, e in enumerate(doc.get("landmarks") or []):
        t = e.get("type")
        need = {"box": ("center", "size", "yaw"), "cylinder": ("center", "radius", "height"),
                "sphere": ("center", "radius")}.get(t)
        if need is None:
            raise ValueError(f"landmark {i}: unknown type {t!r}")
        for k in ("name",) + need:
            if k not in e:
                raise ValueError(f"landmark {i} ({t}): missing {k}")
        lm = {"name": str(e["name"]), "type": t,
              "center": [float(v) for v in e["center"]]}
        if len(lm["center"]) != 3:
            raise ValueError(f"landmark {lm['name']}: center must be [x, y, z]")
        if t == "box":
            lm["size"] = [float(v) for v in e["size"]]
            lm["yaw"] = float(e["yaw"])
            if len(lm["size"]) != 3 or min(lm["size"]) <= 0:
                raise ValueError(f"landmark {lm['name']}: size must be 3 positive numbers")
        else:
            lm["radius"] = float(e["radius"])
            if lm["radius"] <= 0:
                raise ValueError(f"landmark {lm['name']}: radius must be positive")
            if t == "cylinder":
                lm["height"] = float(e["height"])
        out.append(lm)
    return out
