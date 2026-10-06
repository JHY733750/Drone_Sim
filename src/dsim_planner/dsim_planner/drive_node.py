"""Fly the waypoints a person clicks in the viewer.

    ros2 run dsim_planner drive_node --ros-args -p bounds:="[-8.,8.,-8.,8.,0.3,3.]"

The viewer server validates a click and publishes one dsim_msgs/DriveCommand
on /drone/drive_cmd. This node keeps the queue (dsim_planner/drive.py),
re-plans from wherever the controller's reference is, and publishes the
result as /drone/trajectory -- the same message a planner's plan arrives
as, so the controller cannot tell a person from a planner and needs no
changes.

It also publishes /drone/drive_status, latched: the bounds it accepts, the
queue still to fly, the speed cap and the last refusal. That is the one
account of the queue: the page draws it from here, never from its own
memory of what it clicked.

Time comes from the messages, not from /clock: the decimated truth carries
the simulated time at 30 Hz, which is all a planner of 0.02 s samples
needs, and an rclpy subscription to the 1 kHz clock costs about half a core
(see bridge_node.py).

On start-up, and after every reset, it takes off: one waypoint above where
the vehicle stands, at takeoff_z. A depth camera 12 cm off the floor sees
floor; the first thing anyone flying this wants is to be in the air.
"""
import json
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import String

from dsim_msgs.msg import DriveCommand, Trajectory, TrajectorySetpoint

from . import drive

OPS = {DriveCommand.OP_REPLACE: "replace", DriveCommand.OP_APPEND: "append",
       DriveCommand.OP_CLEAR: "clear"}


def _stamp(m):
    return m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec


def _yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DriveNode(Node):
    def __init__(self):
        super().__init__("dsim_drive")
        b = [float(v) for v in self.declare_parameter(
            "bounds", [-10.0, 10.0, -10.0, 10.0, 0.3, 3.0]).value]
        if len(b) != 6 or b[0] >= b[1] or b[2] >= b[3] or b[4] >= b[5]:
            raise RuntimeError(f"bounds must be [xlo, xhi, ylo, yhi, zlo, zhi], got {b}")
        self.bounds = {"x": b[0:2], "y": b[2:4], "z": b[4:6]}
        self.takeoff_z = float(self.declare_parameter("takeoff_z", 1.0).value)
        #: How far ahead of the newest known sim time a new plan starts. The
        #: reference it starts from is the newest DECIMATED setpoint, up to
        #: 1/30 s old, so a plan starting "now" would begin in the past.
        self.lead_s = float(self.declare_parameter("lead_s", 0.1).value)
        self.queue = drive.Queue(
            self.bounds,
            v_max=float(self.declare_parameter("v_max", 1.5).value),
            a_max=float(self.declare_parameter("a_max", drive.A_MAX).value),
            yaw_rate=float(self.declare_parameter("yaw_rate", drive.YAW_RATE).value))

        self.now = None          # newest simulated time seen
        self.pose = None         # newest truth Odometry (takeoff only)
        self.ref = None          # newest controller setpoint
        self.took_off = False
        self.error = ""

        self.traj_pub = self.create_publisher(Trajectory, "/drone/trajectory", 4)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/drone/drive_status", latched)
        self.create_subscription(Odometry, "/drone/viz/truth", self.on_truth,
                                 qos_profile_sensor_data)
        self.create_subscription(TrajectorySetpoint, "/drone/viz/setpoint", self.on_setpoint,
                                 qos_profile_sensor_data)
        self.create_subscription(DriveCommand, "/drone/drive_cmd", self.on_command, 10)
        self.create_timer(0.2, self.tick)
        self.publish_status()
        self.get_logger().info(
            f"drive up: bounds x{self.bounds['x']} y{self.bounds['y']} z{self.bounds['z']}, "
            f"v_max {self.queue.v_max:g} m/s, a_max {self.queue.a_max:g} m/s^2, "
            f"takeoff to {self.takeoff_z:g} m")

    # ---- inputs -------------------------------------------------------------
    def on_truth(self, m):
        t = _stamp(m)
        if self.now is not None and t < self.now - 0.5:
            # A reset: the queue and the plan belong to the run that ended.
            self.get_logger().info("simulated time went backwards: clearing the queue")
            self.queue = drive.Queue(self.bounds, self.queue.v_max, self.queue.a_max,
                                     self.queue.yaw_rate)
            self.took_off = False
            self.ref = None
            self.error = ""
            self.publish_status()
        self.now = t
        self.pose = m

    def on_setpoint(self, m):
        self.ref = m

    def on_command(self, m):
        op = OPS.get(m.op)
        p = [m.position.x, m.position.y, m.position.z]
        try:
            if op is None:
                raise drive.DriveError(f"unknown op {m.op}")
            self.queue.command(op, p, m.yaw, m.v_max if m.v_max > 0.0 else None)
        except drive.DriveError as exc:
            self.error = str(exc)
            self.get_logger().warn(f"refused {op} {p}: {exc}")
            self.publish_status()
            return
        self.error = ""
        self.get_logger().info(
            f"{op} {[round(v, 2) for v in p] if op != 'clear' else ''} "
            f"-> {len(self.queue.waypoints)} waypoint(s), v_max {self.queue.v_max:g}")
        self.replan()

    # ---- the plan -------------------------------------------------------------
    def start_state(self):
        """Where the new plan starts: the controller's own reference, so the
        trajectory is continuous with what it is flying now."""
        if self.ref is not None:
            r = self.ref
            return {"p": [r.position.x, r.position.y, r.position.z],
                    "v": [r.velocity.x, r.velocity.y, r.velocity.z],
                    "a": [r.acceleration.x, r.acceleration.y, r.acceleration.z],
                    "yaw": r.yaw}
        p, q = self.pose.pose.pose.position, self.pose.pose.pose.orientation
        return {"p": [p.x, p.y, p.z], "v": [0.0] * 3, "a": [0.0] * 3, "yaw": _yaw_of(q)}

    def replan(self):
        if self.now is None or self.pose is None:
            self.error = "no vehicle state yet"
            self.publish_status()
            return
        t0 = self.now + self.lead_s
        try:
            samples = self.queue.replan(self.start_state(), t0)
        except drive.DriveError as exc:
            self.error = str(exc)
            self.publish_status()
            return
        msg = Trajectory()
        sec = int(math.floor(t0))
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = min(int(round((t0 - sec) * 1e9)), 999_999_999)
        msg.header.frame_id = "world"
        for t, p, v, a, yaw, yaw_rate in samples:
            sp = TrajectorySetpoint()
            ts = int(math.floor(t))
            sp.time_from_start.sec = ts
            sp.time_from_start.nanosec = min(int(round((t - ts) * 1e9)), 999_999_999)
            sp.position.x, sp.position.y, sp.position.z = p
            sp.velocity.x, sp.velocity.y, sp.velocity.z = v
            sp.acceleration.x, sp.acceleration.y, sp.acceleration.z = a
            sp.yaw = yaw
            sp.yaw_rate = yaw_rate
            msg.points.append(sp)
        self.traj_pub.publish(msg)
        self.publish_status()

    def tick(self):
        if self.now is None:
            return
        if not self.took_off and self.pose is not None:
            p, q = self.pose.pose.pose.position, self.pose.pose.pose.orientation
            self.took_off = True
            z = min(max(self.takeoff_z, self.bounds["z"][0]), self.bounds["z"][1])
            x = min(max(p.x, self.bounds["x"][0]), self.bounds["x"][1])
            y = min(max(p.y, self.bounds["y"][0]), self.bounds["y"][1])
            self.queue.command("replace", [x, y, z], _yaw_of(q), None)
            self.get_logger().info(f"taking off to {z:g} m")
            self.replan()
            return
        if self.queue.advance(self.now):
            self.publish_status()

    def publish_status(self):
        doc = {
            "bounds": self.bounds,
            "queue": [[*p, yaw] for p, yaw in self.queue.waypoints],
            "v_max": self.queue.v_max,
            "error": self.error,
        }
        self.status_pub.publish(String(data=json.dumps(doc)))


def main(argv=None):
    rclpy.init(args=argv)
    node = DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
