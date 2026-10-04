# Flying a space-time Bezier plan

How a plan from [banana99ou/bezier-trajectory](https://github.com/banana99ou/bezier-trajectory/tree/integrate/rust-into-spacetime)
(`spacetime_bezier`) gets into this simulator, what the referee does with the
scenario it was solved in, and what a run proves.

```
the planner's SCENARIO_MAP  +  scripts/sim_scenarios.py (the sim's own: loiter_sim)
         │  scripts/import_scenarios.py          (--check catches drift)
         ▼
scenarios/<name>.json  ──────────────┬──> worlds/<name>.sdf             (ground, light, the vehicle at `start`)
  obstacles as lifted control         ├──> config/obstacles_<name>.yaml  (the referee's obstacle field)
  points in (x, y, z, t), the         ├──> web/scene.json                (colours and the camera's aim)
  active window intrinsic to          └──> plans/<name>_seed.json        (the straight seed, which mostly HITS)
  the first and last of them                          │
                                                      │
plans/<name>.json  (x, y, z, t) control points ───────┤
                                                      ▼
                          dsim_planner/bridge_node ── samples uniform in TIME,
                          v = p'/t',  a = (p'' t' - p' t'')/t'^3 ──> /drone/trajectory
                                                      │
                                    SE(3) controller (unchanged) ──> the vehicle
                                                      │
                    dsim_eval: signed clearance along each obstacle's Bezier,
                    inside its window, on the scenario clock ──> /drone/eval/clearance
                                                      │
                                    the browser draws THAT ──> what you see
```

Nothing is transcribed anywhere on that path. `import_scenarios.py --check`
fails if the planner's scenarios have moved, and `gen_assets.py --check` fails
if anything downstream has.

## The seven scenarios (six imported, one the sim's own)

```bash
python3 scripts/solve_plan.py --list     # inside the container
```

| | | |
|---|---|---|
| `original` | 2D+t | three moving obstacles |
| `curve` | 2D+t | one obstacle on a **cubic** path; the only non-convex tube |
| `diverse` | 2D+t | seven movers |
| `wall` | 2D+t | a wall that **stands until t=5 and is then gone** |
| `fence3d` | 3D+t | a moving fence; the demonstrated behaviour is the climb |
| `door3d` | 3D+t | a doorway in time; the demonstrated behaviour is **waiting** |
| `loiter_sim` | 3D+t | **sim-native** (not imported): the planner's `loiter` redesigned at sim size -- 12 m corridor at 2.25 m, T=16 s, four orbit pieces **and a station**. See [below](#loiter_sim-the-planners-loiter-redesigned-for-the-sim) |

Four are planned in **two** spatial dimensions plus time. They are solved that
way and the resulting curve is flown at 1.5 m; lifting them into 3D would hand
the solver an escape over the top that the 2D problem never had. Their
obstacles are therefore **columns** -- a disc at every altitude -- and not
balls at the flight altitude.

## Run it

```bash
make viz WORLD=fence3d PLAN=plans/fence3d_seed.json     # watch it in the browser
make plan PLAN=plans/fence3d_seed.json                  # headless, graded
```

The seed plan is the straight line from `start` to `end` -- the planner's own
initial guess -- and on six of the seven scenarios it flies straight through
something. Its file says what and when (`expect.first_hit`), and `make plan`
checks the referee agrees to within 0.3 s and 10 cm. A referee that lets the
seed through is broken, and this is how you would find out.

## The obstacles are not in the physics

They never were -- they have no collision geometry, because the referee scores
clearance analytically and a contact solver would only add a second,
differently shaped opinion. They are now not Gazebo bodies at all, because
nothing a rigid body can do expresses what they are:

* two scenarios move theirs along **cubics** (`curve`, `loiter_sim`), not straight
  lines;
* three switch obstacles **off partway through** (`wall` at t=5, `door3d` at
  t=5, `loiter_sim`'s four orbit pieces in sequence). Outside its window an obstacle
  does not exist. It is not far away and it is not parked at an endpoint: the
  planner solves a problem in which the thing is absent, and waiting for it to
  go is the behaviour `wall` and `door3d` demonstrate.

So `/drone/eval/clearance` **is** the obstacle field, and the browser draws
that. There is one account of where anything is rather than two that can drift.

An earlier version did put constant-velocity spheres in the world and checked
Gazebo's poses against the referee's arithmetic. That check read 0.00 mm, but
what it compared was the simulator against its own staging. The replacement is
stronger and lives in `check_planner.py`: the referee's live obstacle field
against the **planner's own** `obstacle_positions_at()` -- two implementations
of the same Bezier motion, in two languages, sampled 41 times across the
window, including whether each obstacle is active at that instant. Measured
agreement was exactly 0.0 when both sides read identical numbers; since the
frames change (below) the planner is evaluated on its own coordinates and its
ANSWER mapped, so the comparison now also tests the map. Offline (the
referee's obstacle.hpp on the generated yaml, check_planner's own comparison
code): <= 2.7e-15 m for the s = 1 scenarios, 3.4e-10 m for the planner's
loiter when it was flown at s = 0.04 (the yaml's 9-decimal rounding of 0.04 x),
against a 1e-6 bound; the
pre-frames yaml reads 6.5..56 m. Not yet re-measured on a live run.

## The scenario clock

The planner's t = 0 is when the plan starts executing. `sim.start_s` (5 s) is
when that happens in simulated seconds; the vehicle **spawns at the plan's
first point** and holds there until then, and the bridge stamps the trajectory
with `start_s` so the plan begins on the tick.

There is no transit from a pad. It is not part of any plan, nothing certifies
it, and the version that flew one collected five collisions before the run had
begun -- the fence swept through the pad while the vehicle sat on it.

## Where a scenario sits

`scenarios/*.json` are the planner's numbers, verbatim -- except the
sim-native ones (`"native": true`, today only `loiter_sim`), which are the
sim's own numbers in sim metres. Everything the sim flies, scores or draws is
in the SIM frame, `p_sim = s * (p - c)`, `t` unchanged, with one table and
one map in `src/dsim_planner/dsim_planner/frames.py`. For an imported
scenario `c` is the centre of the horizontal bounding box of start, end and
every obstacle control point (rounded to the cm; `gen_assets.py --check`
fails if the planner moves it), and `s = 1` for all six. For a sim-native
scenario the frame is the identity, `s = 1, c = 0`, by construction: its
origin is meaningful (loiter_sim's station stands on it), so it is not
re-centred. That is claimed in two places that must agree -- the file's
`"native": true` and `frames.NATIVE` -- and `check_table_centre` refuses
either half alone. The map is uniform (z scales too) because a sphere has to
stay a sphere.

There used to be one `s != 1`: the planner's 200 m `loiter` at `s = 0.04`.
It fitted the optical-flow band, but the 0.3 m vehicle became 7.5 planner
metres; grown by that, the bodies reached into the altitude band and the run
lost its occlusion certificate at N8_seg16. It is no longer imported (below).

A Bezier maps exactly under an affine map of its control points, so
`solve_plan.py` solves in planner coordinates and maps the answer once; its
`--inflate` and `--v-max` are in sim units. Every plan records its `frame`;
the launch and the bridge refuse one whose frame is not the scenario's.

## loiter_sim: the planner's loiter, redesigned for the sim

The planner's `loiter` is the paper's line-of-sight demo: a ground station at
the origin, a body orbiting ABOVE it and BELOW a transit corridor, the
corridor laid ON the body's shadow ring so the shadow slides along it, the
corridor narrower than the shadow so no sidestep clears it, and one lap over
the horizon. `scripts/sim_scenarios.py` keeps that design and re-picks every
number for this vehicle and these sensors; its docstring says why each one is
what it is.

| | | |
|---|---|---|
| corridor | y = 1.35 m, x -6..+6, z band 2.1..2.4 (2.25 m axis), +-0.2 m wide | flow band 0.1..3 m; 12 m fits the camera's world |
| body | r 0.2 m on a 0.9 m orbit at 1.5 m, one lap in 16 s | grown by 0.3 it tops out at 2.0 m, under the 2.1 m floor |
| shadow | ring 0.9 x 2.25 / 1.5 = 1.35 m (the corridor is ON it), spot r 0.3 m | half-width 0.2 < 0.3: no sidestep clears it |
| phase | 20 deg, scanned | see `PHASE_DEG` |
| solver | trust_radius 0.18; free arrival, time_weight 10, v_max 1.5, start weight 1e5 | each measured to matter -- docstring |

The solve flags travel with the scenario (`"solve"` in its file): the design
is the planner's free-arrival one, `solve_plan.py` and the viewer's solve
button apply them unless told otherwise, and the plan records what was used
and where it came from. Pinned at T, the 24-phase scan never once produced a
baseline that loses the link alongside a run that certifies.

`import_scenarios.py --check` prints why the planner's `loiter` is not
imported, fails if the planner stops defining it (the record would be stale)
and fails if a `scenarios/loiter.json` reappears. The new name is deliberate:
the planner keys solver knobs by scenario name (`SCENARIO_ELASTIC_WEIGHT`), and
two different problems under one name is how a 200 m-tuned number ends up
applied to a 12 m scene.

`scripts/loiter_pair.py` measures the design's three claims, in the container:
each config is solved twice, as `solve_plan.py` would and again with the
station removed (the baseline), and both are measured with the planner's own
`los_margin_at` against the REAL bodies. Measured 2026-10-05:

| | N8_seg8 | N8_seg16 |
|---|---|---|
| (a) certifies: converged, keep-out + occlusion certificates 0, 0 planes dropped | yes | yes |
| (a) sampled line of sight, constrained run | +0.652 m | +0.601 m |
| (b) baseline line of sight (blocked 3.62..3.92 s) | **-0.175 m** | **-0.175 m** |
| (c) constrained path on the baseline's schedule | **-0.048 m** | **-0.026 m** |
| reverse graft (baseline path, constrained schedule) | +0.605 m | +0.532 m |
| arrival, constrained / baseline | 13.46 / 8.00 s | 12.87 / 8.00 s |
| largest sidestep (corridor half-width 0.2) | 0.147 m | 0.180 m |
| line of sight while holding the goal until T | +0.899 m | +0.899 m |
| keep-out clearance for the 0.3 m vehicle | +1.628 m | +1.454 m |

The run waits about 5 s and keeps the link; the baseline goes at the cap and
loses it for 0.3 s; the constrained path flown on the baseline's clock loses
it too, so the retiming is what keeps it. Margins are small in absolute terms
because everything is: the body is 0.2 m.

**Not robust everywhere, and measured as such.** The occlusion certificate is
sensitive: re-solved at 19.9, 19.99, 20.01 and 20.1 deg, 9 of 10 solves keep
all three (20.1 deg N8_seg16 does not certify); a 3e-10 s change in a window
time has flipped N8_seg16 at 15 deg. The phase was picked for the
neighbourhood that certifies, not for the cleanest single point. Every
plan's demands are tiny -- 1.5 m/s, 0.28 m/s^2, 1.6 deg of tilt against 8.26
m/s^2 at the 40 deg clamp -- and the opening step is 0.23..0.26 m/s.
Not yet flown in the sim.

## SLAM landmarks

`config/landmarks_<name>.yaml`: static boxes, cylinders and spheres for the
depth camera, generated (seeded) by `gen_assets.py` via `scripts/landmarks.py`,
drawn by the viewer, and in neither the referee nor the physics. They keep
1.0 m from every plan, 0.5 m from every obstacle sweep and 0.5 m from every
station sight line; `scripts/check_landmarks.py` gates that against every
plan present, and `--self-test` proves it fails on planted ones.

After the horizon `T` the controller holds the final point. Every obstacle's
window ends at or before `T`, so past it the field is empty.

## The conversion (why the bridge exists)

The optimiser's curve is a Bezier in (x, y, z, t) with parameter tau. Time is
a coordinate of the curve, so dp/dtau is not a velocity. The bridge applies
the chain rule with tau eliminated,

    v = p' / t'                     a = (p'' t' - p' t'') / t'^3

and samples uniform in **time** by inverting t(tau) with bisection. Both
choices are tested against finite differences in time on a deliberately
clustered curve (`src/dsim_planner/test/test_spacetime.py`), which is the one
comparison an algebra slip cannot pass; `scripts/mutation_check.sh` injects
the slips -- dp/dtau as velocity, the t'' term dropped, tau-spaced samples --
and confirms each is caught.

The inversion is safe because the optimiser's hard row `t[i+1] - t[i] >=
min_dt` on the control points makes every derivative control point's time
component positive, and a convex combination of positives is positive: dt/dtau
> 0 on the whole curve, not just at the knots. The bridge checks the row
rather than trusting it.

## What the referee measures

Clearance is **signed** and counts the vehicle radius (0.3 m from
`config/drone.yaml`): `|p - centre(t)| - r - 0.3`, with `centre(t)` from de
Casteljau along the obstacle's control points and `+infinity` outside its
window. Negative is a collision.
Each excursion below zero is one hit, recorded with the vehicle position at
entry, the obstacle, the scenario time and the deepest penetration, and drawn
in the viewer as a red cross at that spot. Every run prints its hits.

This is the same quantity the planner certifies (`compute_min_clearance`),
except that the planner solves for a point and the referee for a vehicle. A
plan certified at +0.16 m for a point is 14 cm INTO the fence for this
vehicle. Either inflate every obstacle by 0.3 m before solving or expect the
referee to say so; the first is the planner's call, the second is a finding
about the plan, not a bug in either.

## Solving, in the container

The optimiser is a Rust extension (`bezier_opt`: pyo3 around clarabel) and the
planner has **no Python fallback** -- `optimize_spacetime()` raises without it.
The toolchain is in the image; the planner's source is a MacBook worktree
rsynced in:

```bash
make planner-sync    # rsync the MacBook worktree -> $PLANNER_REPO (/ws/planner_src)
make up              # (re)create the container so the mount takes
make planner         # build bezier_opt, install into /ws/install/planner_py, import it
make solve SCENARIO=fence3d N=8 NSEG=2 VMAX=3.0
make plan  PLAN=plans/fence3d_N8_seg2.json
```

`make planner` ends by importing the module **from the path it installed to**
and printing where it came from: maturin will build against a different
interpreter and report success, and the planner's own README records being
burned by exactly that. A fence3d solve takes **0.20 s**, a four-config sweep
1.4 s, so replanning in the loop is not off the table.

`scripts/solve_plan.py` hands the optimiser the SIM's `scenarios/<name>.json`,
which `import_scenarios.py` generated from the planner's own `SCENARIO_MAP`.
One source, so the obstacles the referee scores against and the problem the
solver reads are the same numbers. For a 2D scenario it drops the z column
before solving and puts it back afterwards, which is exact because the
altitude is a constant.

It also **inflates every obstacle by the vehicle radius** before solving, and
then states what the referee should measure. Without that the numbers do not
line up: fence3d N8_seg2 solved for a point certifies +0.1623 m, which is
**-0.1377 m** for a 0.3 m vehicle -- a plan the optimiser certifies and the
referee scores as a collision, with neither of them wrong. Inflating is exact
for a sphere. Solved that way it certifies +0.2209 m; the live run measured
+0.2444 m, and `check_planner.py` holds the two together.

### Two things that make a plan unflyable

The optimiser constrains **speed** (`--v-max`, a hard second-order cone on the
control-point differences) and nothing else. In particular:

1. **No acceleration cap.** Uncapped, fence3d N8_seg2 returns a curve
   demanding 15.05 m/s, 82.61 m/s² and **108.6° of tilt** against a 40° clamp:
   the solver loiters and then darts, because nothing costs it. A speed cap
   bounds it only indirectly and not reliably -- N10_seg8 at `--v-max 2.0`
   still demands 40 m/s² and 89°. `solve_plan.py` samples the plan with the
   same converter the bridge uses and prints the verdict before writing the
   file; `check_planner.py` refuses to grade an unflyable plan, because
   tracking error there measures the clamp rather than the plan.
2. **No boundary velocity.** Endpoint *positions* are pinned, velocities are
   not, so a plan opens at the speed cap from a standstill and ends still
   moving -- it is a segment of a flight, not a flight. The vehicle hovers at
   the start point and is handed a step: measured **92 cm** of opening error
   on a 3.00 m/s start, decaying to under 10 cm within 3 s. The catch-up
   distance `v0²/(2·a_max)` is 54 cm at full authority, so most of it is
   unavoidable given the formulation. Fixing it properly is a planner-side
   row (pin the first and last derivative control points); the sim could
   instead fly a lead-in, which is a different claim about what was tested.

### From the viewer

With a plan loaded, the control panel grows a **plan** box: degree N, segments,
v max, and a solve button. It POSTs to `/solve` -- a second write path, kept
separate from `/control` because `/control` is documented as unable to run a
command and solving runs the optimiser. The scenario and the output path are
**not** request fields (the Gazebo world is generated per scenario, so solving
another one would plan against obstacles that are not there), the three numbers
are range-checked, and the subprocess takes an argument list, never a shell.
The bridge notices the new file by mtime and reloads it, so a solve reaches the
flying vehicle in about a second with nothing restarted. A plan that does not
parse is refused and the old one keeps flying.

The **scenario** dropdown above it is a different hammer: it restarts the
simulator with whichever run you pick, in about four seconds, and the page
stays up throughout. Every scenario that some plan declares is offered, plus
the planner-free `hover` and `circle`; the server will only launch a name from
the list the page built its dropdown from. A solved plan is preferred over the
straight seed, because the seed exists to be flown *into* an obstacle. See
[VIEWER.md](VIEWER.md#the-scenario-dropdown).

## Line of sight

A scenario may name `stations`: fixed ground points the vehicle has to stay
visible from. Only `loiter_sim` has one, at the origin, and its orbiting
body is what can get in the way.

The referee measures the margin directly:

```
distance( obstacle centre at t, segment[station, vehicle] )  -  radius
```

positive exactly when the sight line clears the body, because a sphere blocks
a segment precisely when the segment passes within `radius` of its centre. The
foot of the perpendicular is **clamped to the segment** — a body behind the
station, or beyond the vehicle, is not between them and blocks nothing. An
obstacle outside its active window blocks nothing either, and a scenario with
no stations reports `+inf`, not `0`.

This is deliberately a second implementation of the planner's `los_margin_at()`
rather than a call into it. The solver never measures the sight line: it builds
a **convex occlusion relaxation** and certifies that. So there are three
things, not two — the certificate, the planner's own sampled measurement, and
the referee's — and `check_planner.py` holds the two measurements against each
other over the flown run. Measured on the planner's `loiter` at full scale
(since replaced by `loiter_sim`): **2.13e-14 m over 728 instants**, which is
float noise, against a 1e-6 bound. At s = 0.04 the offline replica read
2.2e-10 m over 801 instants (yaml rounding). Not yet re-measured on
`loiter_sim`.

That check earns its keep. It failed twice while being built, and both were
real defects in the referee:

* the report stamped `scenario_time_s` from the publish instant while the
  margins came from the last ground-truth sample. Two different instants, which
  only shows at a window boundary — the referee said "unconstrained at t=0"
  while the planner, asked about t=0, said +49.76 m. The report now describes
  one instant, and carries the position it was measured at;
* comparing against the viewer's pose instead of the referee's own sample read
  as 2.5 cm of disagreement at the planner loiter's 3.4 m/s, which is about 7 ms of flight.

Losing sight is recorded the way a collision is: edge-triggered, one
**blackout** per excursion, carrying which station lost it, to which obstacle,
where the vehicle was, for how long, and how far the sight line went into the
body. The viewer draws the station and the sight lines, red when blocked.

The gate grades against what the solver actually claimed. `occlusion_certified`
alone is not the claim — the builder **drops** occlusion planes when a station
lies inside the hull it is cutting from, and a dropped plane is not a satisfied
one, so all three of `occlusion_certified`, `occlusion_planes_dropped == 0` and
`occlusion_violation == 0` have to hold before a blackout counts as a failure.
Without them the run is reported and not graded, the same rule the clearance
check uses for a plan the optimiser could not certify.

**Not implemented: line of sight past a `column`.** The planner treats every
obstacle as a ball for occlusion, so there is no arithmetic on that side to
check a column against, and a number with nothing to check it against is worse
than no number. A scenario with both a column and a station is refused at load
with a message saying what would have to be built first. No 2D scenario has
stations today, so nothing hits it.

## The opening transient, measured

Every plan opens with a velocity STEP, because the optimiser pins endpoint
positions and not velocities. The peak tracking error that step causes is
proportional to the step, and the constant is a property of the control loop
rather than of any plan:

| plan | start speed | peak error | ratio |
|---|---|---|---|
| `wall` seed | 0.70 m/s | 0.237 m | 0.339 s |
| `fence3d` seed | 0.90 | 0.290 | 0.322 |
| `loiter` (planner's, retired) N8_seg16 | 1.60 | 0.522 | 0.326 |
| `diverse` N8_seg4 | 1.68 | 0.526 | 0.313 |
| `fence3d` N8_seg2 | 3.00 | 0.919 | 0.306 |
| `curve` N10_seg8 | 3.00 | 1.011 | 0.337 |

Least squares through the origin: **0.322 s**, residuals under 5 cm over a
4.3x range. So the gate allows `0.5 s x v0` with a 30 cm floor, and a
feedforward that stopped arriving would move that coefficient enough to fail.

The first version used the full-authority catch-up distance `v0^2/(2 a_max)`.
That is quadratic, and the loop is not acceleration-limited here but
bandwidth-limited: it under-predicted by 3x at the low end, passing `fence3d`
and failing `diverse` and `loiter` for reasons to do with neither plan.

## What the reset path taught

Resetting the world mid-plan (the viewer's reset button) exposed a controller
bug that no built-in reference could: with the plan dropped and no post-reset
state yet received, the position hold latched onto the PRE-reset state -- the
goal, nine metres away -- and commanded a 40° tilt from the pad for as long as
it took the bridge to republish. Measured before the fix: a 1.6 m lunge and
two hits before scenario t = 0. The controller now forgets its state on a
reset and idles until a fresh one arrives; three consecutive resets measured
under 2.1° of commanded tilt and 1 cm of travel in the first 2.5 s.

## What a run does not prove

* Nothing senses the obstacles. There is no lidar or depth in the loop and no
  avoidance; the trajectory is fixed before it is flown, and the obstacles are
  not in the physics, so nothing could sense them if it tried. The referee
  knows them because it was told, and a plan that clears them does so because
  the planner was told the same thing.
* The opening transient is the plan's standing start, not the loop: 29 cm for
  the 0.9 m/s seed, 92 cm for a 3.0 m/s optimised plan. The cruise figure --
  the last 40% of the plan, allowed 50 ms of lag at the plan's top speed -- is
  the one that says anything about tracking.
* Line of sight IS checked now. See below.
* The estimator is a LOW-ALTITUDE one. Its only velocity aiding is optical
  flow, which needs the ground between 0.1 and 3 m, plus a rangefinder good to
  4 m. The planner's `loiter` at full scale flew at 62.5 m: measured 48 m of
  tracking error on `state:=est` and 2.2 cm on `state:=truth`. Its
  replacement `loiter_sim` flies at 2.1..2.4 m, inside that band;
  `state:=est` there is NOT yet flight-tested.
* The lateral acceleration limit is ~8.3 m/s² at the 40° tilt clamp. That is
  the number the planner is missing a constraint for, and every plan here is
  kept under it by a speed cap chosen by hand rather than by anything in the
  formulation.
