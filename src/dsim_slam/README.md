# dsim_slam — depth-camera scan-to-map odometry

Position and orientation from a RealSense D435 point cloud and the IMU gyro,
with a covariance and a TRACKING / DEGRADED / LOST verdict. Written to run
**unchanged on a real drone**: it reads only what a real vehicle has, and
never `/drone/truth`, a scenario, an obstacle list or a landmark file — not
even to initialise.

```bash
ros2 launch dsim_slam slam.launch.py                       # hardware defaults
ros2 launch dsim_slam slam.launch.py use_sim_time:=true    # in the simulator
```

Not wired into `sim.launch.py`: replacing the controller's position source
is an integration decision.

## Interface

| Direction | Topic / frame | Type | Notes |
|---|---|---|---|
| in | `/camera/camera/depth/color/points` (`cloud_topic`) | `sensor_msgs/PointCloud2` | realsense-ros 4.x; float32 x/y/z, any extra fields, ordered or not; **optical frame** (z = depth) |
| in | `/drone/imu` (`imu_topic`) | `sensor_msgs/Imu` | body-frame gyro, 250 Hz; the accelerometer only levels the map at start |
| in | TF `base_link` → cloud frame | static | looked up once; clouds are dropped (status says why) until it exists |
| in | `~/set_pose` | `geometry_msgs/PoseWithCovarianceStamped` | restart the map at this pose |
| out | `/slam/odom` (`odom_topic`) | `nav_msgs/Odometry` | `map` → `base_link`, one per cloud, stamped with the cloud |
| out | `/slam/status` (`status_topic`) | `diagnostic_msgs/DiagnosticStatus` | per cloud **and** at `status_rate_hz` |
| out | TF `map` → `base_link` | | only with `publish_tf:=true` (default off) |

**Odometry.** Pose covariance is 6×6 in the nav_msgs convention: position in
`map`, rotation about the `map` axes through the body origin. Twist is in
`base_link` (REP-145) from a Kalman filter on position and velocity; an
unknown velocity (first frame, after LOST) is published as zero with
variance 1e6, angular rate is the mean gyro over the frame interval.

**Status.** `level` OK = TRACKING, WARN = DEGRADED, ERROR = LOST; `message` is
`STATE: reason`. `values` carry `state`, `reason`, `time_since_good_s`,
`source_points`, `correspondences`, `inlier_fraction`, `rms_residual_m`,
`chi2_per_dof`, `translation_min_eig` + `translation_weak_dir` (map frame),
`rotation_min_eig_m2`, `position_sigma_m`, `correspondence_threshold_m`,
`iterations`, `map_voxels`, `imu_prior_used`, `map_reset`, `compute_ms`.

| State | Means | Pose | Consumer |
|---|---|---|---|
| TRACKING | the scan constrained all six directions | registered | may close a position loop |
| DEGRADED | a direction rests on the motion prior (degenerate geometry), few points matched (< 10 %), or the map was just re-initialised | registered where observable, predicted elsewhere; covariance says which | should not trust the weak direction |
| LOST | too few points in range (< 60), too few matches (< 40), or no cloud for 0.5 s | position **held**, rotation from the gyro, covariance growing as `1 m/s × time lost` | must not close a position loop |

After 1 s LOST, the next scan that cannot be matched re-initialises the map
at the held pose (DEGRADED, `map_reset: true`): the position may then be off
from the old map by however far the vehicle moved, and the covariance — kept,
not reset — says so. A cloud stamp jumping back > 1 s (simulator RESET)
restarts everything.

## Algorithm

KISS-ICP's shape (voxel local map, adaptive correspondence threshold from
the motion model's own error, robust Gauss–Newton), changed for a narrow,
noisy stereo camera rather than a 360° lidar. kiss-icp itself was not
vendored: it fetches Sophus and tsl-robin-map at configure time, and the
build container has no network.

1. **Preprocess** (`scan_odometry.cpp`): depth gate 0.28–6 m, per-point
   variance `σz² = 0.005² + (0.0072 z²)²` (D435 at 424×240: subpixel 0.08 /
   (223 px × 50 mm)), to `base_link` via TF, one *raw* point per 0.2 m voxel
   (a centroid of points straddling an edge lies on no surface; that alone was
   a 6 cm bias).
2. **Map** (`voxel_map.cpp`): 0.25 m voxels holding an inverse-variance
   weighted mean; a voxel is used only once its mean is known to 10 cm; voxels
   > 12 m away are dropped (a local map: no loop closure).
3. **Plane per point**: orientation from the 6 nearest means, position from a
   noise-scaled window refit (selecting "nearest to the noisy point" made the
   anchor follow the noise and under-registered forward motion), rejected at
   edges, thin structures and outside the patch's own extent.
4. **Registration** (`registration.cpp`): point-to-plane — so a single wall
   leaves the translation Hessian rank-deficient and degeneracy is *visible*
   (point-to-point's translation block is `Σw·I` whatever the scene) —
   weighted by `1/σ²`, Geman–McClure kernel in metres (KISS's τ/3).
5. **Prior**: rotation from the gyro integrated in the body frame between
   cloud stamps (σ 0.5 mrad/frame — trusted, because with an 87° FOV a
   sideways shift and a yaw are nearly the same image change; with a 3 mrad
   floor the survey drifted 17–35° in yaw); translation from a Kalman filter
   whose velocity is Gauss–Markov (σ 1 m/s, τ 5 s).
6. **Degeneracy**: eigen-analysis of the weight-normalised Hessian
   (translation block; rotation via the Schur complement). Directions below
   threshold are projected out of the scan's information (solution remapping,
   Zhang et al. 2016) so the prior decides them, with hysteresis; reported in
   the status.
7. **Covariance**: posterior `(H/max(1,χ²) + prior)⁻¹` per frame, accumulated
   as a random walk transported through the motion, plus an empirical
   `(1 % × distance flown)²` drift term.

## Verification

`test/test_slam.cpp` (no ROS; also run by `scripts/mutation_check.sh`) flies
its own synthetic D435 — ray-cast 424×240, 87×58°, 0.28–10 m, σz ∝ z² — through
a world of rotated boxes, cylinders, spheres, a sloped slab and a floor, with
a biased noisy gyro. Every test states the bug it catches. Measured on the
host, `-O2`:

| Test | Measured |
|---|---|
| exact recovery (noise-free, coincident points) | 1.1e-6 m, 4.8e-7 rad |
| optical +z motion → `base_link` +x | 0.73 m for 0.6 m, y/z < 2 cm (see failure modes) |
| two independent noise-free renders, gyro rotation | 1.3 cm, 0.003° |
| 25 s survey, 22.9 m, 6-DOF, noisy | **1.22 %/m** final (seeds 1–4: 1.22, 2.46, 2.22, 1.17 %), max 0.96 m, 0.99°; error inside 3σ on 100 % of frames; TRACKING 72 % |
| compute, 424×240 cloud | 17 ms mean, 44 ms max (host -O2, single thread) |
| fast yaw (±86°, 9.4 rad/s, 15 Hz, pitched 12°) | gyro: 0.15 m, 2.2°, 0 lost; constant rate: 0.98 m, 179°, 10 lost |
| wall at 3.4 m, sliding along it | 188/188 DEGRADED, error inside 3σ on 188/188, σ² growing 80× faster than with features; TRACKING again on 121/121 once features appear |
| nothing in range for 1.5 s | 44/44 LOST, σ 0.08 → 2.54 m, TRACKING after, 0.14 m off |

## Known failure modes

- **A single plane 2–3 m away is reported TRACKING.** D435 depth noise
  (3–5 cm there) spreads the wall over neighbouring voxels; 6-mean patch
  normals scatter ~10°, which looks like 4–7 % of information *along* the
  wall — above any threshold that still passes real scenes. The pose then
  slides along the wall with a confident covariance. At 3.4 m the wall stops
  matching at all and the verdict is DEGRADED for that reason.
  `judge_wide_normals:=true` judges observability on 1 m plane fits and gets
  this case right (eigenvalue 0.001 vs 0.13), but small objects rarely
  support a 1 m fit, so the feature-rich survey then drifts ~4 %/m. Default
  off; `Registration.WideNormalsSeeANoisyWallIsDegenerate` pins both facts.
- **Motion straight at surfaces is over-estimated** (~20 % over the first
  second, noise-free too) while the map is thin; the noise-scaled windows
  straddle surface junctions.
- **Translation drift ~1–2.5 %/m** in a cluttered room-sized world; no loop
  closure, so it is never taken back. Single-seed numbers vary by 2×.
- **Far range is nearly useless**: beyond ~4.5 m (σ > 15 cm) points are mapped
  but not registered. Open or large spaces with nothing within ~4 m are LOST
  or DEGRADED.
- **Covariance is calibrated, not derived**: the `drift_per_metre` term (1 %)
  was measured on the synthetic survey; re-measure on real data. While
  hovering in a mapped area the random-walk part grows although the error
  does not (conservative).
- **Gyro-camera timing**: the rotation prior assumes cloud and IMU stamps on
  one clock. With an unsynchronised camera, raise `prior_rotation_floor_imu`
  to at least `max rate × time offset`.
- **No accelerometer in the translation prior** (only levelling at start),
  no gyro-bias estimation, no rolling-shutter deskew (the D435 depth imager
  is global-shutter).

## What was not verified

Nothing here has run against the simulator's own D435 cloud, a live graph,
or real hardware: the node (`slam_node.cpp`) is built and its PointCloud2
parser is tested (`test_cloud_conversion.cpp`), but TF lookup, QoS, gravity
alignment from a live IMU and the status timer are untested.
