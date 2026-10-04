"""Scenarios AUTHORED in the sim, in SIM metres. Frame s = 1, c = 0.

Every other scenario is the planner's, imported verbatim from its SCENARIO_MAP
by scripts/import_scenarios.py and placed in the sim's world by frames.py. A
scenario here is the sim's own: written in the metres the vehicle flies, so
its frame is the identity BY CONSTRUCTION (dsim_planner/frames.py NATIVE).
import_scenarios.py renders it into scenarios/<name>.json in the same schema
as the imported ones, with "native": true, and `--check` holds that file to
the factory exactly as it holds an imported one to the planner.

REPLACED names the planner scenarios the sim deliberately does NOT import,
and why. `import_scenarios.py --check` prints every entry, and fails if the
planner stops defining one of them (the record would be stale) or if a file
for one appears in scenarios/ (it would be flyable and nobody would know
which design it was).
"""
from __future__ import annotations

import math

#: Cubic Bezier quarter-circle constant: the control arm is k * radius. A
#: cubic cannot close a circle, so a lap is four quarter arcs; k overshoots
#: the radius by 0.03 percent at the arc's middle, 0.3 mm on a 0.9 m orbit.
K_QUARTER = 0.5522847498


def scenario_loiter_sim(phase_deg: float | None = None) -> dict:
    """A body ORBITING the ground station, and a corridor its shadow slides ALONG.

    The planner's `loiter`, redesigned at the simulator's own size rather than
    shrunk into it. The design is the planner's and so are the three things it
    has to show; the numbers are new, because the sim has constraints the
    planner's 200 m scene never met:

    * **The corridor is at 2.25 m**, inside the optical flow's 0.1..3.0 m band
      (config/drone.yaml) and the rangefinder's 4 m, so `state:=est` has
      velocity aiding all the way.
    * **The vehicle is 0.3 m and the solver is told so** -- solve_plan.py
      grows every body by the vehicle radius before solving. The planner's
      scene scaled by 0.04 grew its bodies by 7.5 planner metres, which pushed
      the keep-out into the altitude band, and the run lost its occlusion
      certificate at N8_seg16. Here the GROWN body (0.2 + 0.3 = 0.5 m at
      1.5 m up) tops out at 2.0 m, under the 2.1 m floor of the band: the
      keep-out rows are present and never binding, as in the planner's.
    * **The whole scene is 12 m across**, so it fits the forward depth
      camera's world (0.28..10 m, best under ~4.5 m): the corridor is 12 m
      long and 1.35 m from the station, the orbit 0.9 m round it.

    What is kept from the planner's design, and why each piece is there:

    * **The corridor is TANGENT to the shadow ring.** At altitude z the
      body's shadow (the station as a point light) falls on a ring of radius
      orbit_r * z / body_z = 0.9 * 2.25 / 1.5 = 1.35 m, and the corridor is
      ON it (y = 1.35), so the spot arrives moving ALONG the corridor rather
      than across it and sits on the path instead of crossing it.
    * **The corridor's half-width (0.2 m) is under the spot's radius**
      (body_r * z / body_z = 0.3 m for the real body), so no lateral offset
      the corridor allows clears it. Same ratio as the planner's, 2/3.
    * **The band floor is above the body's reach** (above), which is also
      what makes the soil real: below the band the solver could dive into
      the shadow-free space a ground station cannot see past.
    * **One lap over the horizon** (T = 16 s), as a chain of four cubic
      quarter arcs on contiguous windows, so the shadow visits the corridor
      once and "wait for it to pass" means something.
    * **The phase is SCANNED, not derived** -- see PHASE_DEG.

    The solver knobs are this scene's, not the planner's (all measured
    2026-10-05 with the bodies grown by 0.3 m, as solve_plan.py solves):

    * ``trust_radius`` 0.18 -- the planner's 5.0 on 200 m times the 0.036
      that maps its corridor altitude onto this one -- and it is the knob
      that decides whether this scene certifies at all. The clip radius the
      shadow walls are built from grows with it (``reach = seg_radius + trust
      * sqrt(dim)`` in clip_band), so a larger one builds walls against more
      of the shadow than the run can reach. Phases 0..30 deg in 5 deg steps,
      N8_seg8 and N8_seg16, 14 solves each (scripts/loiter_pair.py
      --trust): trust 0.5 certified 0, 0.25 certified 2, 0.18 certified 9,
      0.12 certified 7. Every geometry tried at 0.5 before this one failed
      the same way, whatever its proportions, with the run's true line of
      sight often positive -- the walls, not the scene, were the obstacle.
    * The solve flags, SOLVE below: the arrival is FREE, priced by
      time_weight 10 and capped at 1.5 m/s -- the planner's paper
      configuration with the cap in this scene's units -- and the elastic
      weight starts at 1e5, the planner's registered weight for its loiter.
      Measured: a FIXED arrival (what solve_plan.py does for every other
      scenario) spreads the transit over the whole lap at 0.75 m/s, and over
      24 phases it never once had the baseline lose the link AND the
      constrained run certify. Starting at the solver's default weight of
      100, the chosen phase certifies in neither config. The flags travel
      with the scenario under ``solve`` so solve_plan.py and the viewer's
      solve button use them without being told.
    """
    orbit_r = 0.9          # m, radius of the loiter circle
    body_z = 1.5           # m, altitude of the loiter circle
    body_r = 0.2           # m, radius of the loitering body (the REAL one)
    corridor_z = 2.25      # m, altitude of the transit; centre of the band
    corridor_half = 0.2    # m, lateral half-width of the corridor
    band = (2.1, 2.4)      # m, altitude band; floor above the grown body (2.0)
    length = 12.0          # m, start to end
    T = 16.0               # s, one lap
    # Derived, not tuned: the shadow ring at the corridor's altitude.
    # Rounded to the nanometre only so the file reads 1.35, not 1.3499999999999999.
    corridor_y = round(orbit_r * corridor_z / body_z, 9)     # 1.35 m
    # PHASE_DEG unless a scan (scripts/loiter_pair.py --phase) asks for another.
    phase = math.radians(PHASE_DEG if phase_deg is None else float(phase_deg))
    k = K_QUARTER
    lap = [
        # (x, y) control points of each quarter, counterclockwise from (r, 0).
        [(orbit_r, 0.0), (orbit_r, orbit_r * k), (orbit_r * k, orbit_r), (0.0, orbit_r)],
        [(0.0, orbit_r), (-orbit_r * k, orbit_r), (-orbit_r, orbit_r * k), (-orbit_r, 0.0)],
        [(-orbit_r, 0.0), (-orbit_r, -orbit_r * k), (-orbit_r * k, -orbit_r), (0.0, -orbit_r)],
        [(0.0, -orbit_r), (orbit_r * k, -orbit_r), (orbit_r, -orbit_r * k), (orbit_r, 0.0)],
    ]
    cs, sn = math.cos(phase), math.sin(phase)
    quarter = T / 4.0
    obstacles = [
        {
            "name": f"orbit_q{i + 1}",
            "radius": body_r,
            "color": "#e74c3c",
            "control_points": [
                [round(x * cs - y * sn, 9), round(x * sn + y * cs, 9), body_z,
                 i * quarter + j * quarter / 3.0]
                for j, (x, y) in enumerate(arc)
            ],
        }
        for i, arc in enumerate(lap)
    ]
    half_len = 0.5 * length
    return {
        "name": "loiter_sim",
        "title": "Body Orbiting the Ground Station (sim-native)",
        "init_curve": {"mode": "straight"},
        "obstacles": obstacles,
        "start": [-half_len, corridor_y, corridor_z, 0.0],
        "end": [half_len, corridor_y, corridor_z, T],
        "stations": [[0.0, 0.0, 0.0]],
        # The corridor as a tube of HARD box rows, as in the planner's: the
        # lateral pair is what leaves timing as the only escape, the altitude
        # pair keeps the run above the body's reach and the soil real.
        "coord_bounds": [
            [round(-1.2 * half_len, 9), round(1.2 * half_len, 9)],
            [round(corridor_y - corridor_half, 9), round(corridor_y + corridor_half, 9)],
            [band[0], band[1]],
        ],
        "trust_radius": 0.18,
        "T": T,
    }


#: The phase of the orbit at t = 0, degrees counterclockwise from +x.
#:
#: SCANNED, and what it sets is WHEN the shadow reaches the tangency point
#: relative to the baseline's transit (the baseline is the same solve with the
#: station removed; at the 1.5 m/s cap it is at the tangency at t = 4 s). Three
#: conditions, each measured with the planner's own los_margin_at against the
#: REAL bodies, never with a certificate:
#:   (a) the constrained run certifies -- converged, keep-out and occlusion
#:       certificates at zero, no plane dropped -- and its sampled line of
#:       sight stays positive;
#:   (b) the baseline loses the link;
#:   (c) the constrained PATH flown on the baseline's SCHEDULE (same curve
#:       parameter, the other run's time coordinate) loses it too, so the
#:       retiming, not the path, is what keeps it.
#: Measured 2026-10-05 with scripts/loiter_pair.py --phase ..., the flags in
#: SOLVE, bodies grown 0.3 m:
#:   every 15 deg, N8_seg8 and N8_seg16 (48 solves): all three hold in 5,
#:     all between 345 and 30. From 60 to 300 the shadow reaches the corridor
#:     after the baseline has gone and (b) fails; elsewhere the run does not
#:     certify.
#:   every 5 deg from 340 to 40 (26 solves): all three hold in 15; both
#:     configs at once only at 15, 20 and 25.
#: 20 is chosen over 15 and 25 for ROBUSTNESS, measured: re-solved at
#: 19.9 / 19.99 / 20.01 / 20.1 deg, 9 of the 10 solves keep all three (15:
#: 7 of 10, 25: 8 of 10). The certificate here is sensitive -- a 3e-10 s
#: change in a window time has flipped N8_seg16 at 15 deg -- so a phase whose
#: neighbourhood certifies is worth more than the one cleanest point. The
#: cost: N8_seg8 sidesteps 0.15 m inside the 0.2 m corridor (at 15 deg it
#: was 0.026), which (c) still shows is not what keeps the link.
PHASE_DEG = 20.0

#: How this scenario is meant to be solved (see the docstring). solve_plan.py
#: applies these unless its command line says otherwise. v_max is sim m/s.
SOLVE = {"free_arrival_time": True, "time_weight": 10.0, "v_max": 1.5,
         "elastic_weight": 1.0e5}

#: name -> (factory, configs worth solving, solve flags). Same shape as the
#: planner's SCENARIO_MAP plus the flags.
SIM_SCENARIOS = {
    "loiter_sim": (scenario_loiter_sim, [(8, 8), (8, 16)], SOLVE),
}

#: Planner scenarios the sim does not import: name -> (replaced by, why).
REPLACED = {
    "loiter": (
        "loiter_sim",
        "the planner's 200 m / 62.5 m scene; flown at s = 0.04 the 0.3 m vehicle "
        "is 7.5 planner m, the grown bodies reach into the altitude band and the "
        "occlusion certificate was lost at N8_seg16. Redesigned at sim size in "
        "scripts/sim_scenarios.py (corridor 2.25 m, in the optical-flow band; "
        "12 m long, inside the depth camera's world).",
    ),
}
