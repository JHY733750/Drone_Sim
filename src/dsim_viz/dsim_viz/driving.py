"""Send the drone somewhere from the viewer: the /drive write path.

The FOURTH write path on this port, kept apart from the other three for the
reason they are kept apart from each other: each can express exactly one kind
of thing. This one can express a waypoint -- three coordinates and a heading
-- and a speed cap, and nothing else. It cannot arm, retune, launch, solve,
or reach Gazebo.

What a request may say:

    {"op": "replace" | "append", "p": [x, y, z], "yaw": rad, "v_max": m/s}
    {"op": "clear", "v_max": m/s}            (v_max optional everywhere)

It is checked HERE, against the bounds the drive node itself published, and
refused with a reason the page shows. Refused, never clamped: a clamped
waypoint flies somewhere nobody clicked. The same check runs again in the
drive node, which owns the queue; this copy exists so the person gets the
reason in the reply instead of an unexplained nothing.

Only a drive run has a drive node, and only a drive node publishes
/drone/drive_status, so a /drive request on any other run is refused for the
plain reason that there is nothing to drive.
"""
import json
import math

from dsim_msgs.msg import DriveCommand

try:
    from dsim_planner.drive import DriveError, check_waypoint
except ImportError:            # host tests without the planner package on the path
    DriveError = ValueError
    check_waypoint = None

OPS = {"replace": DriveCommand.OP_REPLACE, "append": DriveCommand.OP_APPEND,
       "clear": DriveCommand.OP_CLEAR}


class DriveRefused(ValueError):
    """The request was refused. The message is safe to show the user."""


def _finite(body, key):
    v = body.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise DriveRefused(f"{key} must be a finite number")
    return float(v)


class DriveClient:
    def __init__(self, node, enabled, is_drive_run=lambda: True):
        self.enabled = bool(enabled)
        # Read through to current.json: the drive node's last status is
        # latched and outlives its run, so "a status exists" alone would keep
        # accepting waypoints for a vehicle that is now flying a plan.
        self.is_drive_run = is_drive_run
        self.pub = node.create_publisher(DriveCommand, "/drone/drive_cmd", 10)
        self.status = None         # the drive node's latest /drone/drive_status

    def on_status(self, msg):
        try:
            self.status = json.loads(msg.data)
        except ValueError:
            self.status = None

    def command(self, body):
        if not self.enabled:
            raise DriveRefused("driving is disabled on this server")
        if self.status is None or not self.is_drive_run():
            raise DriveRefused("this run has no drive node -- pick a 'drive:' run")
        op = body.get("op")
        if op not in OPS:
            raise DriveRefused(f"op must be one of {sorted(OPS)}")
        m = DriveCommand()
        m.op = OPS[op]
        if "v_max" in body:
            m.v_max = _finite(body, "v_max")
            if not 0.1 <= m.v_max <= 5.0:
                raise DriveRefused(f"v_max {m.v_max:g} m/s is outside [0.1, 5]")
        if op != "clear":
            p = body.get("p")
            if not isinstance(p, list) or len(p) != 3:
                raise DriveRefused("p must be [x, y, z]")
            xyz = [_finite({"p": v}, "p") for v in p]
            yaw = _finite(body, "yaw")
            if check_waypoint is not None:
                try:
                    check_waypoint(xyz, yaw, self.status["bounds"])
                except DriveError as exc:
                    raise DriveRefused(str(exc)) from None
            m.position.x, m.position.y, m.position.z = xyz
            m.yaw = yaw
        self.pub.publish(m)
        return {"message": f"{op} sent", "drive": self.status}
