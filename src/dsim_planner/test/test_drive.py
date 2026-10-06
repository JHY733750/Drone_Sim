"""The drive queue and the trajectory it flies. Pure Python, host-tested.

Each test says what wrong implementation it exists to catch.
"""
import math

import pytest

from dsim_planner import drive

BOUNDS = {"x": [-10, 10], "y": [-10, 10], "z": [0.3, 3.0]}
REST = {"p": [0.0, 0.0, 1.0], "v": [0.0, 0.0, 0.0], "a": [0.0, 0.0, 0.0], "yaw": 0.0}


def speed(v):
    return math.sqrt(sum(x * x for x in v))


def test_refused_not_clamped():
    # FAILS IF: an out-of-bounds waypoint is clamped onto the boundary (the
    # vehicle would fly somewhere nobody clicked) or accepted.
    q = drive.Queue(BOUNDS)
    for p in ([11, 0, 1], [0, -10.5, 1], [0, 0, 0.1], [0, 0, 3.5]):
        with pytest.raises(drive.DriveError):
            q.command("replace", p, 0.0, 1.5)
    with pytest.raises(drive.DriveError):
        q.command("append", [0, 0, float("nan")], 0.0, 1.5)
    with pytest.raises(drive.DriveError):
        q.command("replace", [1, 1, 1], 0.0, 99.0)       # v_max out of range
    assert q.waypoints == []
    q.command("replace", [10, -10, 3.0], 0.0, 1.5)       # the edges are inside
    assert len(q.waypoints) == 1


def test_replace_append_clear():
    # FAILS IF: append replaces, replace appends, or clear leaves anything.
    q = drive.Queue(BOUNDS)
    q.command("replace", [1, 0, 1], 0.0, None)
    q.command("append", [2, 0, 1], 0.5, None)
    q.command("append", [3, 0, 1], 1.0, None)
    assert [w[0][0] for w in q.waypoints] == [1, 2, 3]
    q.command("replace", [5, 5, 2], 0.0, None)
    assert [w[0] for w in q.waypoints] == [[5, 5, 2]]
    q.command("clear", None, 0.0, None)
    assert q.waypoints == []


def test_a_leg_arrives_at_rest_within_the_limits():
    # FAILS IF: the quintic's end conditions are wrong (it overshoots or
    # arrives moving), or the duration search ignores v_max / a_max.
    samples, done = drive.plan(REST, [([6.0, 2.0, 2.0], 0.0)], v_max=1.2, a_max=0.8)
    t, p, v, a, yaw, _ = samples[-1]
    assert p == pytest.approx([6.0, 2.0, 2.0], abs=1e-9)
    assert speed(v) == pytest.approx(0.0, abs=1e-9)
    assert max(speed(s[2]) for s in samples) <= 1.2 * 1.002
    assert max(speed(s[3]) for s in samples) <= 0.8 * 1.002
    assert max(speed(s[2]) for s in samples) > 0.9       # and it is not crawling
    assert done == [pytest.approx(t)]
    # From rest it is a straight line: every sample on the segment.
    for _, q, *_ in samples:
        cross = (q[0] * 2.0 - q[1] * 6.0)
        assert abs(cross) < 1e-9


def test_velocity_and_acceleration_are_the_derivatives_of_position():
    # FAILS IF: v or a is mis-scaled (e.g. per unit leg time, not per second)
    # -- two independent computations, finite differences of p against the
    # analytic derivatives, must agree.
    start = {"p": [0.0, 0.0, 1.0], "v": [0.8, -0.3, 0.0], "a": [0.1, 0.2, 0.0], "yaw": 0.0}
    samples, _ = drive.plan(start, [([3.0, 3.0, 1.5], 1.0), ([-2.0, 1.0, 1.0], -2.0)],
                            v_max=1.5, a_max=1.0, dt=0.005)
    n = 0
    for k in range(1, len(samples) - 1):
        t0, p0 = samples[k - 1][0], samples[k - 1][1]
        t1, p1 = samples[k + 1][0], samples[k + 1][1]
        if t1 - t0 > 0.011:
            continue
        fd = [(p1[i] - p0[i]) / (t1 - t0) for i in range(3)]
        assert fd == pytest.approx(samples[k][2], abs=2e-3)
        n += 1
    assert n > 500


def test_replanning_in_motion_starts_from_the_motion():
    # FAILS IF: a replan starts from rest (a velocity step the controller
    # cannot follow) instead of matching the state it starts in.
    start = {"p": [1.0, 2.0, 1.0], "v": [1.0, 0.5, 0.0], "a": [0.2, 0.0, 0.0], "yaw": 0.3}
    samples, _ = drive.plan(start, [([-3.0, 4.0, 1.0], 0.0)], v_max=1.5)
    _, p, v, a, yaw, _ = samples[0]
    assert p == pytest.approx(start["p"])
    assert v == pytest.approx(start["v"])
    assert a == pytest.approx(start["a"])
    assert yaw == pytest.approx(0.3)


def test_heading_is_held_on_the_leg_and_turned_on_arrival():
    # FAILS IF: the vehicle turns before it gets there, turns the long way, or
    # ends on the wrong heading. The user's rule: between waypoints keep the
    # heading of the one left; at a waypoint, turn to its own.
    start = dict(REST, yaw=0.2)
    samples, done = drive.plan(start, [([4.0, 0.0, 1.0], 3.0), ([4.0, 4.0, 1.0], -3.0)],
                               v_max=1.5, yaw_rate=1.0)
    t1, t2 = done
    first_leg = [s for s in samples if s[0] < t1 and math.dist(s[1], [4.0, 0.0, 1.0]) > 1e-6]
    assert first_leg and all(s[4] == pytest.approx(0.2) for s in first_leg)
    at_wp1 = [s for s in samples if s[0] <= t1 + 1e-9 and math.dist(s[1], [4.0, 0.0, 1.0]) < 1e-9]
    assert at_wp1[-1][4] == pytest.approx(3.0, abs=1e-4)   # last grid sample of the turn
    second_leg = [s for s in samples if t1 + 1e-6 < s[0] and math.dist(s[1], [4.0, 4.0, 1.0]) > 1e-6]
    assert second_leg and all(s[4] == pytest.approx(3.0) for s in second_leg)
    assert samples[-1][4] == pytest.approx(-3.0, abs=1e-9)
    # 3.0 -> -3.0 is 0.28 rad the short way through pi, not 6 rad back: the
    # turn is brief and its rate stays under the cap.
    turn2 = [s for s in samples if s[0] > t1 and math.dist(s[1], [4.0, 4.0, 1.0]) < 1e-9]
    assert turn2[-1][0] - turn2[0][0] < 1.0
    assert max(abs(s[5]) for s in samples) <= 1.0 * 1.002


def test_an_emptied_queue_brakes_instead_of_stopping_dead():
    # FAILS IF: clearing in motion commands zero velocity at once.
    start = {"p": [0.0, 0.0, 1.0], "v": [1.2, 0.0, 0.0], "a": [0.0, 0.0, 0.0], "yaw": 0.0}
    samples, done = drive.plan(start, [], v_max=1.5, a_max=1.0)
    assert done == []
    assert samples[0][2] == pytest.approx([1.2, 0.0, 0.0])
    assert speed(samples[-1][2]) == pytest.approx(0.0, abs=1e-9)
    assert max(speed(s[3]) for s in samples) <= 1.0 * 1.002
    assert samples[-1][1][0] > 0.5                       # it carried on, then stopped


def test_the_queue_drops_waypoints_as_they_are_reached():
    # FAILS IF: progress is measured from the wrong origin, or nothing drops.
    q = drive.Queue(BOUNDS)
    q.command("replace", [2, 0, 1], 0.0, 1.5)
    q.command("append", [2, 2, 1], 1.0, None)
    q.replan(REST, t0=100.0)
    d1, d2 = q.done
    assert d1 > 100.0 and d2 > d1
    assert not q.advance(d1 - 0.01)
    assert q.advance(d1 + 0.01) and len(q.waypoints) == 1
    assert q.advance(d2 + 0.01) and q.waypoints == []


def test_already_there_flies_nothing():
    samples, done = drive.plan(REST, [([0.0, 0.0, 1.0], 0.0)], v_max=1.5)
    assert len(samples) == 1 and done == [0.0]
