"""The /drive write path (driving.py). Needs dsim_msgs: runs in the container."""
import json

import pytest

pytest.importorskip("dsim_msgs")
pytest.importorskip("dsim_planner")

from std_msgs.msg import String  # noqa: E402

from dsim_viz.driving import DriveClient, DriveRefused  # noqa: E402

STATUS = {"bounds": {"x": [-5, 5], "y": [-5, 5], "z": [0.3, 3.0]},
          "queue": [], "v_max": 1.5, "error": ""}


class FakeNode:
    def __init__(self):
        self.sent = []

    def create_publisher(self, _type, _topic, _qos):
        node = self

        class Pub:
            def publish(self, m):
                node.sent.append(m)
        return Pub()


def client(drive_run=True, status=STATUS):
    n = FakeNode()
    c = DriveClient(n, enabled=True, is_drive_run=lambda: drive_run)
    if status is not None:
        c.on_status(String(data=json.dumps(status)))
    return c, n


def test_a_valid_waypoint_is_published_as_given():
    # FAILS IF: the op, the point or the heading is altered on the way through.
    c, n = client()
    c.command({"op": "append", "p": [1.0, -2.0, 1.2], "yaw": 0.7, "v_max": 2.0})
    m = n.sent[-1]
    assert (m.op, m.position.x, m.position.y, m.position.z, m.yaw, m.v_max) == \
        (1, 1.0, -2.0, 1.2, 0.7, 2.0)


@pytest.mark.parametrize("body", [
    {"op": "replace", "p": [9.0, 0.0, 1.0], "yaw": 0.0},          # outside x
    {"op": "replace", "p": [0.0, 0.0, 3.4], "yaw": 0.0},          # above the band
    {"op": "replace", "p": [0.0, 0.0], "yaw": 0.0},               # not x,y,z
    {"op": "replace", "p": [0.0, 0.0, 1.0]},                      # no heading
    {"op": "replace", "p": [0.0, "1", 1.0], "yaw": 0.0},          # not a number
    {"op": "replace", "p": [0.0, 0.0, 1.0], "yaw": 0.0, "v_max": 50},
    {"op": "teleport", "p": [0.0, 0.0, 1.0], "yaw": 0.0},
])
def test_refused_not_clamped_and_nothing_sent(body):
    # FAILS IF: a bad request is clamped or passed on instead of refused.
    c, n = client()
    with pytest.raises(DriveRefused):
        c.command(body)
    assert n.sent == []


def test_no_drive_node_or_not_a_drive_run_refuses():
    # FAILS IF: waypoints are accepted for a vehicle flying a plan, because a
    # latched status from an earlier drive run is still cached.
    for c, n in (client(status=None), client(drive_run=False)):
        with pytest.raises(DriveRefused):
            c.command({"op": "clear"})
        assert n.sent == []
