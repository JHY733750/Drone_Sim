"""Waypoints a person clicked, turned into a trajectory the controller can fly.

The viewer's drive mode: select the drone, right-click a spot, pick a height,
and it goes there. This module is the part with no ROS in it -- the queue,
the validation, and the trajectory -- so it is tested on the host.

THE FLIGHT, leg by leg:

  * Each leg is a quintic in time from the state the leg starts in (position,
    velocity, acceleration) to REST at the waypoint. A quintic matches all
    three at both ends, so replacing the queue mid-flight bends the path
    smoothly instead of stepping the velocity the controller is asked for.
    Its duration is the shortest one, found by search, whose sampled speed
    stays under v_max and acceleration under a_max.
  * The leg is a STRAIGHT line only when it starts from rest; replanned in
    motion it curves from the old direction onto the new one, which is what
    "go there instead" ought to look like.
  * On arrival the vehicle turns, in place, to the waypoint's heading
    (minimum-jerk in angle, the short way round, capped at yaw_rate), and
    only then starts the next leg. Between waypoints it keeps the heading of
    the one it left -- the depth camera looks where that waypoint pointed it,
    which is the point of giving a waypoint a heading at all.

A queue that empties while moving does not stop dead: it brakes to rest
along the current direction of travel, over the distance a_max allows.
Commanding zero velocity in one step would be a demand the vehicle cannot
meet, and the gap would show up as tracking error that is nobody's fault.
"""
from __future__ import annotations

import math

#: Defaults, all overridable. a_max is deliberately gentle: this is for
#: looking at things, and 1 m/s^2 is about 6 degrees of tilt.
A_MAX = 1.0
YAW_RATE = 1.0
DT = 0.02
#: A leg shorter than this, starting at rest, is not flown: the vehicle is
#: already there, to well inside its own tracking error.
MIN_LEG_M = 0.01

# Minimum-jerk rest-to-rest: peak speed 1.875 d/T, peak acceleration
# (10/sqrt(3)) d/T^2. Used as the starting guess for the duration search.
_MJ_V = 1.875
_MJ_A = 10.0 / math.sqrt(3.0)


class DriveError(ValueError):
    """A waypoint or command was refused. The message is safe to show."""


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def check_waypoint(p, yaw, bounds):
    """Refuse, never clamp: a clamped waypoint flies somewhere nobody chose.

    bounds = {"x": [lo, hi], "y": [lo, hi], "z": [lo, hi]}.
    """
    if len(p) != 3:
        raise DriveError(f"a waypoint is x, y, z; got {len(p)} numbers")
    for v in list(p) + [yaw]:
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            raise DriveError("waypoint coordinates and heading must be finite numbers")
    for axis, v in zip("xyz", p):
        lo, hi = bounds[axis]
        if not lo <= v <= hi:
            raise DriveError(f"{axis} = {v:.2f} m is outside [{lo:g}, {hi:g}]"
                             + (" -- the optical flow works between those heights"
                                if axis == "z" else " -- the edge of this world"))


# ---- one leg ----------------------------------------------------------------
def _quintic_axis(p0, v0, a0, p1, T):
    """Coefficients c0..c5 of p(t) = sum c_k t^k with p(0)=p0, p'(0)=v0,
    p''(0)=a0, p(T)=p1, p'(T)=0, p''(T)=0."""
    d = p1 - p0 - v0 * T - 0.5 * a0 * T * T
    e = -v0 - a0 * T
    f = -a0
    T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
    c3 = (10.0 * d - 4.0 * e * T + 0.5 * f * T2) / T3
    c4 = (-15.0 * d + 7.0 * e * T - f * T2) / T4
    c5 = (6.0 * d - 3.0 * e * T + 0.5 * f * T2) / T5
    return [p0, v0, 0.5 * a0, c3, c4, c5]


def _eval(c, t):
    p = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))))
    v = c[1] + t * (2 * c[2] + t * (3 * c[3] + t * (4 * c[4] + t * 5 * c[5])))
    a = 2 * c[2] + t * (6 * c[3] + t * (12 * c[4] + t * 20 * c[5]))
    return p, v, a


class Leg:
    """p0 (v0, a0) -> p1 at rest, in T seconds."""

    def __init__(self, p0, v0, a0, p1, T):
        self.T = T
        self.c = [_quintic_axis(p0[i], v0[i], a0[i], p1[i], T) for i in range(3)]

    def at(self, t):
        t = min(max(t, 0.0), self.T)
        out = [_eval(c, t) for c in self.c]
        return ([o[0] for o in out], [o[1] for o in out], [o[2] for o in out])

    def peaks(self, n=64):
        vmax = amax = 0.0
        for k in range(n + 1):
            _, v, a = self.at(self.T * k / n)
            vmax = max(vmax, math.sqrt(sum(x * x for x in v)))
            amax = max(amax, math.sqrt(sum(x * x for x in a)))
        return vmax, amax


def make_leg(p0, v0, a0, p1, v_max, a_max):
    """The shortest leg within v_max and a_max, or None if there is nothing
    to fly (already there, at rest)."""
    d = math.dist(p0, p1)
    speed0 = math.sqrt(sum(x * x for x in v0))
    if d < MIN_LEG_M and speed0 < 0.05:
        return None
    if not (v_max > 0.0 and a_max > 0.0):
        raise DriveError("v_max and a_max must be positive")
    # Start from the rest-to-rest minimum and grow. A leg starting in motion
    # may need longer (it has to shed that motion), never less than this.
    T = max(_MJ_V * d / v_max, math.sqrt(_MJ_A * d / a_max), 0.3,
            2.0 * speed0 / a_max)
    for _ in range(200):
        leg = Leg(p0, v0, a0, p1, T)
        vpk, apk = leg.peaks()
        if vpk <= v_max * 1.001 and apk <= a_max * 1.001:
            return leg
        T *= 1.08
    # Two hundred growths is a factor of 5e6: no leg needs that, so this is a
    # start state the limits cannot reach (e.g. v_max below the speed the
    # vehicle is already doing, with a_max too small to shed it). Say so.
    raise DriveError(f"no leg to {p1} keeps within v_max {v_max:g} and a_max {a_max:g}")


def _turn(y0, y1, rate):
    """Minimum-jerk turn from y0 to y1 the short way: (T, f(t) -> yaw, yaw_rate)."""
    dpsi = wrap(y1 - y0)
    if abs(dpsi) < 1e-6:
        return 0.0, None
    T = max(_MJ_V * abs(dpsi) / rate, 0.3)

    def f(t):
        s = min(max(t / T, 0.0), 1.0)
        shape = s ** 3 * (10 - 15 * s + 6 * s * s)
        dshape = 30 * s * s * (1 - s) ** 2 / T
        return wrap(y0 + dpsi * shape), dpsi * dshape
    return T, f


# ---- the whole queue --------------------------------------------------------
def plan(start, waypoints, v_max, a_max=A_MAX, yaw_rate=YAW_RATE, dt=DT):
    """Samples for the controller, and when each waypoint is DONE.

    start: {"p": [x,y,z], "v": [...], "a": [...], "yaw": rad}
    waypoints: [([x,y,z], yaw), ...]
    Returns (samples, done) -- samples [(t, p, v, a, yaw, yaw_rate)] from t=0
    at `dt`, ending on the final instant exactly; done[i] = the time waypoint
    i is reached AND turned to.
    """
    pieces = []          # (duration, f(t) -> (p, v, a, yaw, yaw_rate))
    done = []
    p, v, a, yaw = list(start["p"]), list(start["v"]), list(start["a"]), float(start["yaw"])
    t_end = 0.0

    if not waypoints and math.sqrt(sum(x * x for x in v)) > 0.05:
        # Brake along the direction of travel: the queue emptied in motion.
        speed = math.sqrt(sum(x * x for x in v))
        stop = [p[i] + v[i] / speed * speed * speed / (2.0 * a_max) for i in range(3)]
        waypoints = [(stop, yaw)]
        braking = True
    else:
        braking = False

    for wp, heading in waypoints:
        leg = make_leg(p, v, a, wp, max(v_max, math.sqrt(sum(x * x for x in v))), a_max)
        if leg is not None:
            held = yaw
            pieces.append((leg.T, lambda t, leg=leg, held=held: (*leg.at(t), held, 0.0)))
            t_end += leg.T
        p, v, a = list(wp), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        T, f = _turn(yaw, heading, yaw_rate)
        if f is not None:
            pieces.append((T, lambda t, f=f, p=p: (p, [0.0] * 3, [0.0] * 3, *f(t))))
            t_end += T
        yaw = heading
        done.append(t_end)

    if braking:
        done = []
    if not pieces:
        return [(0.0, p, [0.0] * 3, [0.0] * 3, yaw, 0.0)], done

    samples = []
    k = 0
    while True:
        t = k * dt
        if t >= t_end - 1e-9:
            break
        samples.append((t, *_piece_at(pieces, t)))
        k += 1
    samples.append((t_end, *_piece_at(pieces, t_end)))
    return samples, done


def _piece_at(pieces, t):
    for T, f in pieces:
        if t <= T:
            return f(t)
        t -= T
    T, f = pieces[-1]
    return f(T)


class Queue:
    """The waypoint queue and the plan flying it.

    `now` is simulated time throughout. Waypoints drop off the front as the
    plan passes their done time, so the viewer shows what is still to come.
    """

    def __init__(self, bounds, v_max=1.5, a_max=A_MAX, yaw_rate=YAW_RATE):
        self.bounds = bounds
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.yaw_rate = float(yaw_rate)
        self.waypoints = []        # [([x,y,z], yaw)]
        self.t0 = None             # sim time the current plan starts
        self.done = []             # absolute sim times, one per waypoint

    def command(self, op, p, yaw, v_max):
        """Apply one edit. Raises DriveError (and changes nothing) if refused."""
        if v_max is not None:
            if not (math.isfinite(v_max) and 0.1 <= v_max <= 5.0):
                raise DriveError(f"v_max {v_max:g} m/s is outside [0.1, 5]")
        if op in ("replace", "append"):
            check_waypoint(p, yaw, self.bounds)
        elif op != "clear":
            raise DriveError(f"unknown drive op {op!r}")
        if v_max is not None:
            self.v_max = float(v_max)
        if op == "replace":
            self.waypoints = [(list(p), wrap(yaw))]
        elif op == "append":
            self.waypoints.append((list(p), wrap(yaw)))
        else:
            self.waypoints = []

    def replan(self, start, t0):
        samples, done = plan(start, self.waypoints, self.v_max, self.a_max, self.yaw_rate)
        self.t0 = t0
        self.done = [t0 + d for d in done]
        return samples

    def advance(self, now):
        """Drop waypoints already reached. True if the queue changed."""
        n = 0
        while n < len(self.done) and now >= self.done[n]:
            n += 1
        if n:
            self.waypoints = self.waypoints[n:]
            self.done = self.done[n:]
        return n > 0
