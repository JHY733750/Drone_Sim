"""The optical-flow display's geometry (overlay.flow_sensor). Host-tested."""
import math

import pytest

from dsim_viz import overlay

BAND = {"min_h": 0.1, "max_h": 3.0, "max_tilt": 0.52}
TOF = {"range": 1.5, "min": 0.03, "max": 4.0, "fov": 0.47}
FLOW = {"quality": 255, "ground_distance": 1.5}
LEVEL = {"p": [1.0, 2.0, 1.5], "q": [1.0, 0.0, 0.0, 0.0]}


def test_level_beam_lands_straight_below_at_the_measured_range():
    # FAILS IF: the beam is along body +z (up), uses the true altitude instead
    # of the reading, or the footprint radius ignores the field of view.
    f = overlay.flow_sensor(LEVEL, TOF, FLOW, {"flow_velocity_world": [0.3, 0.0, 0.0]},
                            [0.4, 0.0, 0.0], BAND)
    assert f["beam"][1] == pytest.approx([1.0, 2.0, 0.0])
    assert f["valid"] and f["why"] == ""
    assert f["height_m"] == pytest.approx(1.5)
    r = math.dist(f["footprint"][0][:2], [1.0, 2.0])
    assert r == pytest.approx(1.5 * math.tan(0.235), abs=2e-6)   # wire rounds to 1e-6
    # Measured and true ground motion, on the SAME scale, from the hit point.
    assert f["measured"][2] == pytest.approx(0.3) and f["truth"][2] == pytest.approx(0.4)
    m_len = math.dist(f["measured"][0], f["measured"][1])
    t_len = math.dist(f["truth"][0], f["truth"][1])
    assert m_len / t_len == pytest.approx(0.75)


def test_a_tilted_beam_leans_with_the_body():
    # FAILS IF: the beam is drawn world-down regardless of attitude, or the
    # height is the slant range. Pitched 20 deg, a slant reading of
    # 2/cos(20) must put the foot on the ground (z = 0), that far from the
    # vehicle, and report 2.0 m of height.
    a = math.radians(20.0)
    pose = {"p": [0.0, 0.0, 2.0], "q": [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]}
    tof = dict(TOF, range=2.0 / math.cos(a))
    f = overlay.flow_sensor(pose, tof, FLOW, None, None, BAND)
    assert math.dist(f["beam"][0], f["beam"][1]) == pytest.approx(tof["range"])
    assert f["beam"][1][2] == pytest.approx(0.0, abs=1e-9)
    assert f["height_m"] == pytest.approx(2.0)
    assert f["tilt_deg"] == pytest.approx(20.0)


def test_outside_the_envelope_it_says_why_and_shows_no_measurement():
    # FAILS IF: the beam stays "valid" above the band, or a stale accepted flow
    # reading is drawn as if the sensor still saw the ground.
    est = {"flow_velocity_world": [0.3, 0.0, 0.0]}
    high = overlay.flow_sensor(LEVEL, dict(TOF, range=3.4), FLOW, est, [0.4, 0, 0], BAND)
    assert not high["valid"] and "outside" in high["why"]
    assert high["measured"] is None and high["truth"] is not None
    none = overlay.flow_sensor(LEVEL, dict(TOF, range=float("inf")), FLOW, est, None, BAND)
    assert not none["in_range"] and "no rangefinder return" in none["why"]
    assert math.dist(none["beam"][0], none["beam"][1]) == pytest.approx(4.0)
    bad = overlay.flow_sensor(LEVEL, TOF, dict(FLOW, quality=0), est, None, BAND)
    assert not bad["valid"] and "unusable" in bad["why"]


def test_nothing_to_draw_without_a_pose_or_a_range():
    assert overlay.flow_sensor(None, TOF, FLOW, None, None, BAND) is None
    assert overlay.flow_sensor(LEVEL, None, FLOW, None, None, BAND) is None
