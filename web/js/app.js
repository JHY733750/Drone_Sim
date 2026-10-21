// Drone_Sim remote viewer: wiring.
//
// Streams state over a same-origin event stream and renders it locally, rather
// than streaming video of a Gazebo GUI. Video would need a virtual display, a
// software GL rasteriser and an H.264 encoder on the sim host for a blurry
// laggy result; this is a few KB/s and renders crisply at whatever resolution
// the viewing device has.
//
// THE RULE: all logic lives in the ROS simulator and its scripts. This page
// only draws. Overlay geometry arrives from dsim_viz/overlay.py as world-space
// line segments already in metres, tagged with a `kind` and a `group`; the
// page projects them, picks colours, filters by checkbox and strokes them.
// There is no physics here -- no mass times acceleration, no frame rotation of
// a measured quantity, no opinion about what a force is.
//
// Layout of the code:
//   vec3.js       vector / quaternion maths for drawing   pure, tested in node
//   shapes.js     scene geometry as face lists            pure, tested in node
//   palette.js    overlay colours and labels              pure, tested in node
//   render.js     the only file that touches the canvas
//   telemetry.js  the event stream and its fallback
//   app.js        state, HUD, input, the frame loop       (this file)

import { add, scale } from "./vec3.js";
import { droneFaces, landmarkFaces, obstacleFaces, hitMarkers, sightLines, stationFaces }
  from "./shapes.js";
import { FLOW_COLOUR, FRUSTUM_COLOUR, GROUP_LABEL, LEGEND, PALETTE, TICK_COLOUR,
  colourFor, depthColour, widthFor } from "./palette.js";
import { clampHeight, click, headingAt, heightOnBar, hitsDrone, rayPlaneZ, screenRay }
  from "./drive.js";
import { Renderer } from "./render.js";
import { Telemetry } from "./telemetry.js";
import { extendTrail } from "./trail.js";
import { CAM_HOME, gridFor, homeFor } from "./framing.js";

const BACKGROUND = "#0d0f14";

const PREFS_KEY = "dsim.prefs";

const el = (id) => document.getElementById(id);

const S = {
  scene: null,
  world: "empty",
  pose: null,
  status: null,
  control: null,
  overlay: null,
  sim: null,
  solver: null,        // the planner panel: what can be solved, and what is loaded
  launcher: null,      // the run catalogue and what the simulator is doing with it
  clearance: null,     // the referee's obstacle report: live positions, hits
  planPath: null,      // the planner's whole path, before it is flown
  flown: [],
  planned: [],
  // Drive runs: the drive node's own account of its queue and bounds (null
  // on any other run), the flow-sensor geometry, and the depth preview.
  drive: null,
  flowViz: null,
  cloudSeq: 0,
  cloud: { seq: -1, n: 0, xyz: null, depth: null, frustum: null },
};

/// What a click means right now: see drive.js. Page-local on purpose -- a
/// selection is a fact about this tab, not about the vehicle.
let pick = { mode: "idle" };
let cursor = null;          // last pointer position, backing-store pixels

/// How far the follow camera closes on the vehicle each frame. Not a snap:
/// at 3 m/s the target would jitter with every noisy pose, and at loiter's
/// speeds the whole scene would twitch. 0.15 settles in about a fifth of a
/// second and rides out a single bad sample.
const FOLLOW_LERP = 0.15;
/// Pulled in to this when follow is switched on, if the camera was further
/// out than it. loiter frames at 300 m, where the vehicle is a dot -- which is
/// the whole reason this mode exists.
const FOLLOW_MAX_DIST_M = 18;

const prefs = {
  gain: 1,
  follow: false,
  show: Object.fromEntries(Object.keys(GROUP_LABEL).map((g) => [g, true])),
};

let cam = { ...CAM_HOME, target: CAM_HOME.target.slice() };
let hudError = "";
// True between asking for a run and the server reporting one. It stops the
// dropdown being overwritten by the still-running old scenario while the new
// one comes up, which would silently snap the user's choice back.
let switching = false;

const renderer = new Renderer(el("view"));
const telemetry = new Telemetry({ onSnapshot: ingest });

// ---- preferences ---------------------------------------------------------
// Wrapped because localStorage throws outright in some privacy modes rather
// than returning null, and a viewer that refuses to start because it could not
// remember a checkbox would be a poor trade.
function loadPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFS_KEY) || "null");
    if (!saved) return;
    if (typeof saved.gain === "number") prefs.gain = saved.gain;
    if (typeof saved.follow === "boolean") prefs.follow = saved.follow;
    if (saved.show) Object.assign(prefs.show, saved.show);
    // NOT the world: that follows the running simulator now (see adoptRun),
    // and a remembered one would fight it on every load.
  } catch (_) { /* private browsing, or a stale format */ }
}

function savePrefs() {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(
      { gain: prefs.gain, show: prefs.show, follow: prefs.follow }));
  } catch (_) { /* not important enough to surface */ }
}

// ---- incoming state ------------------------------------------------------

let lastStreamT = null;
function ingest(snap) {
  if (!snap) return;
  // Simulated time going backwards is a reset, whoever asked for it -- this
  // page's button, another tab, or a gate script. The trails belong to the
  // run that just ended. Clearing only on this page's own button left a
  // reset from anywhere else drawing the old run's path under the new one.
  //
  // This is necessary and NOT sufficient: it fires on the frame the clock
  // comes back, and whatever has not come back yet lands on the cleared
  // trail. extendTrail() is what makes the trail honest either way.
  const prevT = lastStreamT;
  if (typeof snap.t === "number" && lastStreamT !== null && snap.t < lastStreamT - 0.5) {
    S.flown = [];
    S.planned = [];
  }
  if (typeof snap.t === "number") lastStreamT = snap.t;
  const dt = (typeof snap.t === "number" && prevT !== null) ? snap.t - prevT : 0;
  const r = snap.overlay && snap.overlay.readout;
  if (snap.pose) {
    S.pose = snap.pose;
    extendTrail(S.flown, snap.pose.p, dt, r && r.speed_mps);
  }
  if (snap.setpoint) {
    extendTrail(S.planned, snap.setpoint, dt, r && r.cmd_speed_mps);
  }
  if (snap.status) S.status = snap.status;
  if (snap.control) S.control = snap.control;
  if (snap.clearance !== undefined) S.clearance = snap.clearance;
  if (snap.plan_path !== undefined) S.planPath = snap.plan_path;
  if (snap.overlay !== undefined) S.overlay = snap.overlay;
  if (snap.sim) S.sim = snap.sim;
  if (snap.solver !== undefined) S.solver = snap.solver;
  if (snap.launcher !== undefined) adoptRun(snap.launcher);
  if (snap.drive !== undefined) {
    S.drive = snap.drive;
    if (!S.drive) pick = { mode: "idle" };
  }
  if (snap.flow !== undefined) S.flowViz = snap.flow;
  if (typeof snap.cloud_seq === "number") S.cloudSeq = snap.cloud_seq;
}

// ---- the depth preview ---------------------------------------------------
// Fetched on its own, at most 10 Hz and only when a new cloud exists and the
// layer is shown: thousands of points do not belong in the 30 Hz stream.
const b64 = (s, Type) => {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
};
let cloudInflight = false;
function pollCloud() {
  if (cloudInflight || prefs.show.cloud === false || S.cloudSeq === S.cloud.seq) return;
  cloudInflight = true;
  fetch("/cloud").then((r) => r.json()).then((c) => {
    const n = c.n || 0;
    const cm = n ? b64(c.xyz, Int16Array) : new Int16Array(0);
    const xyz = new Float32Array(cm.length);
    for (let i = 0; i < cm.length; i++) xyz[i] = cm[i] / 100;
    const mm = n ? b64(c.depth, Uint16Array) : new Uint16Array(0);
    S.cloud = { seq: c.seq, n, xyz, depth: mm, frustum: c.frustum };
  }).catch(() => { /* the next poll tries again */ })
    .finally(() => { cloudInflight = false; });
}
setInterval(pollCloud, 100);

/// Follow the simulator that is actually up.
///
/// The world decides which colours the obstacles are drawn in and where the
/// camera goes home to, and it is taken from the RUNNING sim rather than from
/// a saved preference. A viewer that remembered a world across a restart would
/// paint one scenario's palette over another scenario's obstacles, which is
/// the failure the old cosmetic selector could produce on purpose.
function adoptRun(lc) {
  S.launcher = lc;
  const world = (lc.current && lc.current.world) || null;
  // Which dropdown entry is flying. NOT the world: hover and circle share the
  // empty world, so a selector set from the world would sit on neither of them
  // while one was running. The server works this out; see current_run_name().
  const run = lc.current_run;
  if (world && world !== S.world && S.scene && S.scene.worlds[world]) {
    S.world = world;
    cam = homeFor(S.scene.worlds[world]);
    // Whatever was on screen belonged to the run that just ended.
    S.flown = [];
    S.planned = [];
    S.clearance = null;
    S.planPath = null;
  }
  const sel = el("world");
  // Keyed on the NAMES, not the count: a catalogue that swapped one run for
  // another would keep its length, and the dropdown would go on offering a run
  // the server no longer has.
  const key = lc.runs.map((r) => r.name).join("|");
  if (sel && sel.dataset.filled !== key) fillRuns(sel, lc.runs, key);
  // Not while a switch is in flight: current.json still names the OLD run
  // until the new launch has accepted its arguments and written it, so setting
  // the selector from it here would visibly snap the user's choice back for a
  // second and then jump forward again.
  const settled = !switching && lc.status !== "starting";
  if (sel && settled && document.activeElement !== sel && run) sel.value = run;
}

/// Build the dropdown from the catalogue the SERVER offers.
///
/// Not from scene.json: the server will only launch a name out of this list,
/// so taking the list from anywhere else would let the page offer a run that
/// is then refused. Empty means the server has nothing to launch, and the
/// control says so instead of being an empty box.
function fillRuns(sel, runs, key) {
  sel.textContent = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.name;
    o.textContent = r.title || r.name;
    sel.appendChild(o);
  }
  if (!runs.length) {
    const o = document.createElement("option");
    o.textContent = "no runs on offer";
    sel.appendChild(o);
  }
  sel.dataset.filled = key;
}

// ---- frame rate ----------------------------------------------------------
// Measured here rather than guessed. The viewer's own cost is the one thing a
// remote user can see directly, and "it feels slower" is not something you can
// act on -- a number is.
const fps = { frames: 0, last: 0, value: 0 };
function tickFps(now) {
  fps.frames++;
  if (!fps.last) { fps.last = now; return; }
  const dt = now - fps.last;
  if (dt >= 500) {
    fps.value = (fps.frames * 1000) / dt;
    fps.frames = 0;
    fps.last = now;
  }
}

// ---- overlay: filter, zoom, colour --------------------------------------
/// Scale an arrow about its own anchor. This is a display zoom -- the same
/// class of operation as the camera's -- so it cannot change the ratio between
/// two arrows, which is what makes their lengths comparable.
function zoomAbout(anchor, point, gain) {
  return add(anchor, scale([point[0] - anchor[0], point[1] - anchor[1],
                            point[2] - anchor[2]], gain));
}

function overlayToDraw() {
  if (!S.overlay) return { arrows: [], ticks: [] };
  const g = prefs.gain;
  // An arrow whose group has no checkbox is drawn rather than hidden: if the
  // ROS side starts publishing something new, it should show up, not vanish.
  const arrows = S.overlay.arrows
    .filter((a) => prefs.show[a.group] !== false)
    .map((a) => ({
      from: a.from,
      to: zoomAbout(a.from, a.to, g),
      colour: colourFor(a),
      width: widthFor(a),
      label: a.label,
      dashed: a.dashed === true,
      clamped: a.clamped === true,
    }));
  const ticks = (prefs.show.motors === false ? [] : S.overlay.ticks)
    .map((t) => ({
      a: zoomAbout(t.anchor, t.a, g),
      b: zoomAbout(t.anchor, t.b, g),
      colour: TICK_COLOUR,
    }));
  return { arrows, ticks };
}

// ---- frame ---------------------------------------------------------------
function render(now) {
  // The reschedule sits OUTSIDE the try on purpose: this loop re-arms itself,
  // so one exception inside would stop the picture permanently.
  try {
    tickFps(now || performance.now());
    renderOnce();
  } catch (e) {
    hudError = `render: ${e && e.message ? e.message : e}`;
  }
  requestAnimationFrame(render);
}

/// Keep the vehicle in frame.
///
/// Only the camera's TARGET moves; orbit angle and zoom stay where the user
/// put them, so following does not take the controls away. Without this,
/// loiter is unwatchable: it covers 200 m and 62.5 m of altitude, so a camera
/// framed on the whole course draws the drone about a pixel wide.
function followVehicle() {
  if (!prefs.follow || !S.pose) return;
  const p = S.pose.p;
  for (let i = 0; i < 3; i++) cam.target[i] += FOLLOW_LERP * (p[i] - cam.target[i]);
}

function renderOnce() {
  followVehicle();
  renderer.begin(cam, BACKGROUND);
  if (!S.scene) return;
  // The grid follows the world, so the action is on it rather than in one
  // corner of a fixed patch. See gridFor().
  const g = gridFor(S.scene.worlds[S.world]);
  renderer.grid(g.lo, g.hi, g.step);

  // The obstacles come from the REFEREE, live, not from the scene file: it is
  // the only account of where they are, and the only one that knows which of
  // them exist at this instant. scene.json supplies the colours and nothing
  // else about them.
  const world = S.scene.worlds[S.world] || {};
  const faces = obstacleFaces(S.clearance, world.colors, cam.az);
  // Static scenery for the depth camera's SLAM, from scene.json: the one
  // account of them, since they never move and the referee never sees them.
  faces.push(...landmarkFaces(world.landmarks, cam.az));
  faces.push(...stationFaces(S.clearance));
  faces.push(...droneFaces(S.pose, S.scene.drone));
  renderer.faces(faces);

  // The depth cloud lands ON the obstacles, landmarks and floor; drawn after
  // the faces so the points on a near surface are not painted over by it.
  if (prefs.show.cloud !== false && S.cloud.n) {
    const d = S.cloud.depth;
    renderer.points(S.cloud.xyz, S.cloud.n, (i) => depthColour(d[i] / 1000));
  }
  if (prefs.show.cloud !== false && S.cloud.frustum) {
    const [apex, ...c] = S.cloud.frustum;
    renderer.segments(c.map((p) => ({ a: apex, b: p, colour: FRUSTUM_COLOUR })));
    renderer.polyline([...c, c[0]], FRUSTUM_COLOUR, 1.5, false);
  }
  if (prefs.show.flow !== false && S.flowViz) drawFlow(S.flowViz);
  drawDrive();

  // The sight lines, under the trails so a blocked one does not hide the path
  // that blocked it. Red means a station cannot see the vehicle -- a failure
  // that is invisible in a picture of clearances, because the vehicle can be
  // nowhere near an obstacle and still be behind it.
  renderer.segments(sightLines(S.clearance, S.pose));

  if (S.planPath) renderer.polyline(S.planPath, "rgba(250,204,21,.7)", 1.5, true);
  renderer.polyline(S.planned, "rgba(248,113,113,.85)", 2, true);
  renderer.polyline(S.flown, "rgba(96,165,250,.95)", 2, false);
  if (S.pose) renderer.envelope(S.pose, S.scene.drone.radius);
  renderer.segments(hitMarkers(S.clearance));

  const draw = overlayToDraw();
  renderer.segments(draw.ticks);
  renderer.arrows(draw.arrows);
}

/// The downward sensors: the rangefinder beam (green while the flow sensor is
/// inside its envelope, grey outside it), the patch it covers, and the ground
/// motion the estimator accepted next to the true motion. All geometry from
/// overlay.flow_sensor on the ROS side.
function drawFlow(f) {
  const beam = f.valid ? FLOW_COLOUR.beam_ok : FLOW_COLOUR.beam_bad;
  renderer.polyline(f.beam, beam, 2, !f.in_range);
  renderer.polyline(f.footprint, beam, 1.5, false);
  const arrows = [];
  if (f.truth) arrows.push({ from: f.truth[0], to: f.truth[1], colour: FLOW_COLOUR.truth,
    width: 2, dashed: true, label: `true ${f.truth[2].toFixed(2)} m/s` });
  if (f.measured) arrows.push({ from: f.measured[0], to: f.measured[1],
    colour: FLOW_COLOUR.measured, width: 3, label: `flow ${f.measured[2].toFixed(2)} m/s` });
  renderer.arrows(arrows);
  if (!f.valid && f.why) renderer.label(f.beam[1], f.why, FLOW_COLOUR.beam_bad);
}

const DRIVE_COLOUR = { sel: "rgba(250,204,21,.95)", queue: "rgba(167,139,250,.9)",
  bar: "rgba(250,204,21,.55)", cand: "rgba(250,204,21,.95)" };

/// Selection, the queue still to fly (from the drive node), and height mode.
function drawDrive() {
  if (!S.drive) return;
  const p = S.pose && S.pose.p;
  if (pick.mode !== "idle" && p) renderer.ring([p[0], p[1], p[2]], 0.45, DRIVE_COLOUR.sel, 2.5);
  const q = S.drive.queue || [];
  if (q.length && p) {
    renderer.polyline([p, ...q.map((w) => [w[0], w[1], w[2]])], DRIVE_COLOUR.queue, 2, true);
    q.forEach((w, i) => {
      const c = [w[0], w[1], w[2]];
      renderer.ring(c, 0.15, DRIVE_COLOUR.queue, 2);
      renderer.segments([{ a: c, b: [c[0] + 0.5 * Math.cos(w[3]), c[1] + 0.5 * Math.sin(w[3]), c[2]],
        colour: DRIVE_COLOUR.queue }, { a: [c[0], c[1], 0.005], b: c, colour: "rgba(167,139,250,.35)" }]);
      renderer.label(c, String(i + 1), DRIVE_COLOUR.queue);
    });
  }
  if (pick.mode === "height" || pick.mode === "heading") {
    const b = S.drive.bounds;
    const x = pick.x, y = pick.y;
    renderer.segments([{ a: [x, y, 0], b: [x, y, b.z[1]], colour: DRIVE_COLOUR.bar }]);
    const ticks = [];
    for (let z = Math.ceil(b.z[0] * 2) / 2; z <= b.z[1] + 1e-9; z += 0.5) {
      ticks.push({ a: [x - 0.12, y, z], b: [x + 0.12, y, z], colour: DRIVE_COLOUR.bar });
    }
    renderer.segments(ticks);
    const c = [x, y, pick.z];
    renderer.ring(c, 0.2, DRIVE_COLOUR.cand, 2.5);
    // The heading bar only on the heading step: on the height step there is
    // no heading yet, and drawing one would suggest there were.
    if (pick.mode === "heading") {
      renderer.arrows([{ from: c, to: [x + 0.8 * Math.cos(pick.heading),
        y + 0.8 * Math.sin(pick.heading), pick.z], colour: DRIVE_COLOUR.cand, width: 3 }]);
    }
    if (p) renderer.polyline([p, c], DRIVE_COLOUR.bar, 1.5, true);
    const hdg = pick.mode === "heading"
      ? `${(pick.heading * 180 / Math.PI).toFixed(0)}°` : "—";
    renderer.label(c, `x ${x.toFixed(2)}  y ${y.toFixed(2)}  z ${pick.z.toFixed(2)}  hdg ${hdg}`,
      DRIVE_COLOUR.cand);
  }
}

// ---- HUD -----------------------------------------------------------------
function updateHud() {
  try {
    updateHudOnce();
  } catch (e) {
    hudError = e && e.message ? e.message : String(e);
  }
  setTimeout(updateHud, 100);
}

/// A speed, or an em dash when the simulator has not said yet. Distinct from
/// fmt() because "unknown" and "1.00x" must never look the same on screen.
const num = (v, digits = 2) =>
  (typeof v === "number" && Number.isFinite(v)) ? v.toFixed(digits) : "—";

const speedLabel = () => {
  el("speedval").textContent = `${Number(el("speed").value).toFixed(2)}x`;
};

/// The slider's bounds come from the simulator, so this page holds no copy of
/// them; index.html's attributes are only a placeholder until the control node
/// reports. Applied once: re-applying every frame would drag the handle out
/// from under a user who is holding it.
let speedRangeSet = false;
// The gust slider's ceiling is the control node's cap, for the same reason the
// speed slider's is: a page that offered a force the node refuses would make
// the user debug their own UI.
let gustRangeSet = false;
function applyGustRange(sim) {
  const g = el("gustn");
  const known = sim.enabled && typeof sim.max_gust_n === "number";
  g.disabled = !known;
  for (const b of document.querySelectorAll("button.gust")) { b.disabled = !known; }
  el("gustclear").disabled = !known;
  if (!known || gustRangeSet) { return; }
  g.max = sim.max_gust_n;
  gustLabel();
  gustRangeSet = true;
}

function gustLabel() {
  el("gustval").textContent = `${Number(el("gustn").value).toFixed(1)} N`;
}

function applySpeedRange(sim) {
  const speed = el("speed");
  const known = sim.enabled &&
    typeof sim.min_speed === "number" && typeof sim.max_speed === "number";
  speed.disabled = !known;
  if (!known || speedRangeSet) { return; }
  speed.min = sim.min_speed;
  speed.max = sim.max_speed;
  speed.value = sim.requested_speed ?? 1;
  speedLabel();
  speedRangeSet = true;
}

const fmt = (v, unit, digits = 2) =>
  (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits) + unit);

function updateHudOnce() {
  const t = telemetry.status;
  el("link").className = `dot ${t.connected ? "up" : "down"}`;
  el("linktext").textContent = hudError
    ? `hud error: ${hudError}`
    : t.connected
      ? `live · ${t.via}`
      : `disconnected — ${t.via}${t.note ? ` (${t.note})` : ""}`;

  if (S.pose) el("alt").textContent = fmt(S.pose.p[2], " m");

  // Every control number is read straight from the overlay's readout: it was
  // computed once, on the ROS side, from the message that produced the arrows.
  // A second calculation here could disagree with the picture.
  const r = S.overlay && S.overlay.readout;
  const c = S.control;
  if (r) {
    el("spd").textContent = fmt(r.speed_mps, " m/s");
    el("tilt").textContent = fmt(r.tilt_deg, "°", 1);
    // Millinewton-metres, matching the torque arrows' labels: a steady turn
    // demands about 0.017 N·m, and two decimals of that is one significant
    // figure. The HUD and the picture must not print the same quantity in
    // different units.
    el("tau").textContent = fmt(1e3 * r.torque_nm, " mN·m");
    el("aero").textContent = r.aero_n === null ? "—" : fmt(r.aero_n, " N");
    // What each loop of the cascade is asking for, next to what the vehicle
    // is doing. On screen even when an arrow is too short to carry a label.
    el("ctrack").textContent = fmt(r.cmd_track_m, " m");
    el("cspd").textContent = fmt(r.cmd_speed_mps, " m/s");
    el("ctilt").textContent = fmt(r.tilt_deg, "°");
    el("ctilt").className = r.tilt_clamped ? "bad" : "";
    // The integral term: near zero in undisturbed flight, growing to cancel a
    // held disturbance. "held" means anti-windup has frozen it because the
    // demand is already clamped.
    el("integ").textContent = fmt(r.integral_n, " N") +
      (r.integral_held ? " · held" : "");
    el("integ").className = r.integral_held ? "bad" : "";
  }
  if (c) {
    el("thr").textContent = fmt(c.realised_thrust_n, " N");
    el("rot").textContent = c.rotor_thrust_n.map((f) => f.toFixed(2)).join("  ");
    el("tilt").className = c.tilt_clamped ? "bad" : "";
    el("sat").textContent = c.saturated ? "SATURATED" : (c.armed ? "ok" : "disarmed");
    el("sat").className = c.saturated || !c.armed ? "bad" : "good";
  }

  el("fps").textContent = fps.value ? `${fps.value.toFixed(0)} /s` : "—";

  // Simulator pause and speed, as OBSERVED by the server, not as requested.
  const sim = S.sim;
  if (sim) {
    if (!sim.enabled) {
      el("simstate").textContent = "controls off";
      el("simstate").className = "";
    } else if (sim.error) {
      el("simstate").textContent = sim.error;
      el("simstate").className = "bad";
    } else {
      // Two numbers, because they disagree whenever the machine cannot keep
      // up. Showing only the requested one would be a lie at exactly the
      // moment it matters; showing only the achieved one would make the
      // slider look broken.
      const want = sim.requested_speed;
      const got = sim.achieved_speed;
      el("simstate").textContent = sim.paused
        ? "PAUSED"
        : `${num(want)}x asked · ${num(got)}x real`;
      el("simstate").className = sim.paused ? "bad" : "good";
    }
    el("resets").textContent = `${sim.resets ?? 0}` +
      (sim.step_errors ? ` (${sim.step_errors} step errors)` : "");
    el("pause").textContent = sim.paused ? "resume" : "pause";
    el("pause").disabled = !sim.enabled;
    el("simreset").disabled = !sim.enabled;
    applySpeedRange(sim);
    applyGustRange(sim);
    // What the airframe is FEELING, from the control node -- not what was
    // asked for. A capped gust shows the smaller number here, so the HUD and
    // the physics cannot disagree.
    const g = sim.gust_force_n;
    const mag = g ? Math.hypot(g[0], g[1], g[2]) : 0;
    el("gust").textContent = mag < 1e-6
      ? "none"
      : `${mag.toFixed(2)} N` +
        (sim.gust_remaining_s > 0 ? ` · ${sim.gust_remaining_s.toFixed(1)} s` : " · held");
    el("gust").className = mag < 1e-6 ? "" : "bad";
  }

  updateRunPanel();
  updateSolvePanel();
  updateDrivePanel();

  const st = S.status;
  if (st) {
    // Already in centimetres when it arrives: the page does not convert
    // physical quantities. See _status_of() in viz_server.py.
    el("err").textContent = fmt(st.tracking_error_cm, " cm", 1);
    el("rmse").textContent = fmt(st.tracking_rmse_cm, " cm", 1);
    const clr = st.min_obstacle_clearance_m;
    el("clr").textContent = clr === null ? "n/a" : fmt(clr, " m");
    el("clr").className = clr !== null && clr < 0 ? "bad" : "";
    el("col").textContent = st.collided ? `HIT (${st.collision_count})` : "clear";
    el("col").className = st.collided ? "bad" : "good";
    el("ela").textContent = fmt(st.elapsed_s, " s", 1);
  }

  const cl = S.clearance;
  if (cl) {
    // Live clearance and who is nearest, so a shrinking margin can be
    // watched before it goes negative.
    el("clrnow").textContent = cl.clearance_m === null
      ? "n/a" : `${fmt(cl.clearance_m, " m")} · ${cl.nearest}`;
    el("clrnow").className = cl.clearance_m !== null && cl.clearance_m < 0 ? "bad" : "";
    const last = cl.hits[cl.hits.length - 1];
    el("hit").textContent = last
      ? `${last.with} @ (${last.pos.map((v) => v.toFixed(2)).join(", ")}) t=${last.t.toFixed(2)} s`
      : "none";
    el("hit").className = last ? "bad" : "good";
    // How much of the obstacle field exists right now. `wall` and `door3d`
    // are built on obstacles that switch off, so "18 of 23" is the scenario
    // working rather than a fault -- and 0 of anything means the referee was
    // given no obstacles and every clearance below is vacuously infinite.
    const n = cl.obstacles.length;
    const live = cl.obstacles.filter((o) => o.active).length;
    el("obst").textContent = n === 0 ? "none" : `${live} of ${n}`;
    el("obst").className = n > 0 && live === 0 ? "bad" : "";
    el("stime").textContent = fmt(cl.scenario_time_s, " s", 1) +
      (cl.duration_s ? ` of ${cl.duration_s.toFixed(0)}` : "");
    // Line of sight. "n/a" for a scenario with no stations: nothing constrains
    // visibility there, and a 0 would read as "only just visible".
    const stations = cl.stations || [];
    if (!stations.length) {
      el("los").textContent = "n/a";
      el("los").className = "";
      el("dark").textContent = "—";
      el("dark").className = "";
    } else {
      const m = cl.los_margin_m;
      el("los").textContent = m === null
        ? "clear" : `${fmt(m, " m")}${cl.los_blocker ? ` · ${cl.los_blocker}` : ""}`;
      el("los").className = m !== null && m < 0 ? "bad" : "good";
      const dark = cl.blackouts || [];
      const lastDark = dark[dark.length - 1];
      el("dark").textContent = lastDark
        ? `${lastDark.station} · ${lastDark.with} @ t=${lastDark.t.toFixed(2)} s`
        : "never";
      el("dark").className = dark.length ? "bad" : "good";
    }
  }
}

/// What the simulator is doing with the run that was asked for.
///
/// Written to be able to say NO. "starting" carries the seconds it has been
/// starting for, and a launch that dies shows the tail of its own log in the
/// page -- because the alternative, discovered the hard way, is a dropdown
/// that changes and then nothing happens, with the reason sitting in a file on
/// the host that the person looking at the page cannot read.
function updateRunPanel() {
  const lc = S.launcher;
  const line = el("runline");
  const note = el("runnote");
  const log = el("runlog");
  const sel = el("world");
  if (!lc) {
    line.textContent = "waiting for the server";
    return;
  }
  sel.disabled = !lc.enabled || lc.status === "starting";
  if (lc.status === "starting") {
    switching = false;               // the server owns the state from here
    line.textContent = `starting ${lc.requested}… ${lc.elapsed_s ?? 0} s`;
    line.className = "planline warn";
    log.hidden = true;
  } else if (lc.status === "failed") {
    line.textContent = `${lc.requested || "the run"} did not start`;
    line.className = "planline bad";
    log.hidden = false;
    log.textContent = lc.detail || "no detail";
  } else if (lc.status === "running") {
    const cur = lc.current || {};
    const bits = [cur.world || "?"];
    if (cur.plan) bits.push(cur.plan.replace(/\.json$/, ""));
    else if (cur.reference && cur.reference !== "none") bits.push(cur.reference);
    if (cur.state === "truth") bits.push("on truth");
    line.textContent = `flying ${bits.join(" · ")}`;
    line.className = "planline";
    log.hidden = true;
  } else {
    // No telemetry and nothing starting. The server is up -- you are reading
    // its page -- so this is specifically "no simulator", not "no viewer".
    line.textContent = switching ? "asking…" : "no simulator running";
    line.className = "planline bad";
    log.hidden = !lc.detail;
    if (lc.detail) log.textContent = lc.detail;
  }
  // Anything the chosen run needs said about it: an altitude outside the
  // sensing envelope, or a plan the vehicle cannot fly.
  const run = lc.runs.find((r) => r.name === sel.value);
  const why = run && (run.state_reason || run.warning);
  note.hidden = !why;
  if (why) note.textContent = why;
  if (!lc.enabled) {
    line.textContent = "switching runs is disabled on this server";
    line.className = "planline";
  }
}

/// The plan panel: shown only when this run is actually flying a plan.
/// Describing the loaded plan is the point -- "certified +0.22 m, needs 10 deg
/// of tilt" is what makes the solve knobs mean something, and a plan the
/// vehicle cannot fly says so in red rather than being discovered as a
/// mysterious 90 cm of tracking error.
function updateSolvePanel() {
  const sv = S.solver;
  const box = el("solvebox");
  if (!sv || !sv.enabled) { box.hidden = true; return; }
  box.hidden = false;
  el("solvescen").textContent = `· ${sv.scenario}`;
  const p = sv.plan;
  const line = el("planline");
  if (!p) {
    line.textContent = "no plan loaded";
    line.className = "planline";
    return;
  }
  const bits = [];
  if (p.config) bits.push(p.config);
  if (p.predicted_clearance_m !== null && p.predicted_clearance_m !== undefined) {
    bits.push(`clearance ${p.predicted_clearance_m >= 0 ? "+" : ""}${p.predicted_clearance_m.toFixed(3)} m`);
  }
  if (p.max_speed_mps !== null && p.max_speed_mps !== undefined) {
    bits.push(`${p.max_speed_mps.toFixed(2)} m/s`);
  }
  if (p.max_tilt_deg !== null && p.max_tilt_deg !== undefined) {
    bits.push(`${p.max_tilt_deg.toFixed(1)}° tilt` +
      (p.tilt_limit_deg ? ` of ${p.tilt_limit_deg.toFixed(0)}°` : ""));
  }
  if (p.flyable === false) bits.push("NOT FLYABLE");
  line.textContent = bits.join(" · ");
  line.className = p.flyable === false ? "planline bad" : "planline";
}

function updateDrivePanel() {
  const v = Number(el("sV").value);
  el("vmaxval").textContent = v > 0 ? `${v.toFixed(1)} m/s` : "no cap";
  const box = el("drivebox");
  if (!S.drive) { box.hidden = true; return; }
  box.hidden = false;
  const q = S.drive.queue || [];
  const what = { height: "pick height", heading: "pick heading", selected: "selected" }[pick.mode]
    || "click the drone";
  el("driveline").textContent = `${what} · ${q.length} waypoint(s) queued · cap ${S.drive.v_max.toFixed(1)} m/s`;
  if (S.drive.error) el("driveerr").textContent = S.drive.error;
}

// ---- controls ------------------------------------------------------------
function buildControls() {
  const box = el("groups");
  for (const [group, label] of Object.entries(GROUP_LABEL)) {
    const row = document.createElement("label");
    row.className = "check";
    row.innerHTML =
      `<input type="checkbox"${prefs.show[group] ? " checked" : ""}>` +
      `<span>${label}</span>`;
    box.appendChild(row);
    row.querySelector("input").onchange = (e) => {
      prefs.show[group] = e.target.checked;
      savePrefs();
    };
  }

  const slider = el("gain");
  slider.value = String(prefs.gain);
  const showGain = () => { el("gainval").textContent = `${prefs.gain.toFixed(1)}x`; };
  showGain();
  slider.oninput = (e) => {
    prefs.gain = Number(e.target.value);
    showGain();
    savePrefs();
  };

  // The legend is built from the same palette the arrows use, so it cannot
  // end up telling a different story from the picture.
  const legend = el("legend");
  for (const [name, kind] of LEGEND) {
    const row = document.createElement("div");
    row.innerHTML = `<i style="background:${PALETTE[kind]}"></i>${name}`;
    legend.appendChild(row);
  }
  const note = document.createElement("div");
  note.className = "note";
  note.textContent = "rotor arrows: blue below hover, orange above";
  legend.appendChild(note);
}

function bindInput() {
  const canvas = el("view");
  window.addEventListener("resize", () => renderer.resize());

  let drag = null;
  // Backing-store pixels, the units project() and the picking work in.
  const toCanvas = (e) => {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * renderer.px, (e.clientY - r.top) * renderer.px];
  };
  canvas.addEventListener("pointerdown", (e) => {
    // Left drag orbits; right or middle drag pans, and so does shift+left
    // for a trackpad with one button. Shift alone was the only way before,
    // and nothing on the page said so. A press that does not MOVE is a click,
    // and on a drive run clicks select and send waypoints (see drive.js).
    drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, moved: false,
             pan: (e.shiftKey && e.button === 0) || e.button === 1 || e.button === 2 };
    canvas.classList.add("dragging");
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointermove", (e) => {
    cursor = toCanvas(e);
    if (pick.mode === "height" || pick.mode === "heading") updateHeightPick();
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 5) return;
    drag.moved = true;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.pan) {
      const k = cam.dist * 0.0016;
      const v = renderer.view || { right: [1, 0, 0], up: [0, 0, 1] };
      cam.target = add(cam.target, add(scale(v.right, -dx * k), scale(v.up, dy * k)));
    } else {
      cam.az -= dx * 0.006;
      cam.el = Math.max(-1.45, Math.min(1.45, cam.el + dy * 0.006));
    }
  });
  const endDrag = () => { drag = null; canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup", (e) => {
    const wasClick = drag && !drag.moved;
    endDrag();
    if (wasClick && S.drive) driveClick(e.button, e.shiftKey, toCanvas(e));
  });
  canvas.addEventListener("pointercancel", endDrag);
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && (pick.mode === "height" || pick.mode === "heading")) {
      pick = { mode: "selected" };
    }
  });

  const zoom = (factor) => {
    cam.dist = Math.max(0.8, Math.min(60, cam.dist * factor));
  };
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom(Math.exp(e.deltaY * 0.0013));
  }, { passive: false });

  let pinch = null;
  const spread = (e) => Math.hypot(
    e.touches[0].clientX - e.touches[1].clientX,
    e.touches[0].clientY - e.touches[1].clientY);
  const centre = (e) => [
    0.5 * (e.touches[0].clientX + e.touches[1].clientX),
    0.5 * (e.touches[0].clientY + e.touches[1].clientY)];
  canvas.addEventListener("touchstart", (e) => {
    if (e.touches.length === 2) pinch = { d: spread(e), c: centre(e) };
  });
  canvas.addEventListener("touchmove", (e) => {
    if (e.touches.length === 2 && pinch) {
      // Two fingers: the spread zooms, the centroid pans. Before, two fingers
      // could only zoom and there was no way to pan on a phone at all.
      const d = spread(e);
      const c = centre(e);
      zoom(pinch.d / d);
      const k = cam.dist * 0.0016;
      const v = renderer.view || { right: [1, 0, 0], up: [0, 0, 1] };
      cam.target = add(cam.target,
        add(scale(v.right, -(c[0] - pinch.c[0]) * k), scale(v.up, (c[1] - pinch.c[1]) * k)));
      pinch = { d, c };
      e.preventDefault();
    }
  }, { passive: false });
  canvas.addEventListener("touchend", () => { pinch = null; });

  // ---- simulator controls ------------------------------------------------
  // The page sends intent only. Validation, the range check and the actual
  // service call all live in dsim_viz/simcontrol.py, and the reply reports
  // what the simulator was observed to do afterwards.
  const post = (body) => fetch("/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => r.json()).then((r) => {
    if (r.error) { hudError = r.error; } else { S.sim = r; hudError = ""; }
  }).catch((e) => { hudError = `control: ${e.message}`; });

  // The press, not a computed state. Working out the new value here would
  // mean deciding it from a cached copy that can be stale, and two open tabs
  // would fight over it; the control node holds the truth and flips it.
  el("pause").onclick = () => post({ toggle_pause: true });
  el("simreset").onclick = () => {
    post({ reset: true });
    S.flown = [];
    S.planned = [];
  };

  const speed = el("speed");
  speedLabel();
  speed.oninput = speedLabel;
  // On release, not on every drag pixel: each change is a service call into
  // the simulator, and forty of them while dragging is forty round trips.
  speed.onchange = () => post({ speed: Number(speed.value) });

  // One axis at a time, deliberately: a drag-a-vector control would need a
  // projection from screen to world, which is physics in the browser, and the
  // whole viewer is built the other way round. Six buttons say exactly what
  // they do in the frame the number is in.
  gustLabel();
  el("gustn").oninput = gustLabel;
  for (const b of document.querySelectorAll("button.gust")) {
    b.onclick = () => {
      const n = Number(el("gustn").value) * Number(b.dataset.sign);
      const force = { x: [n, 0, 0], y: [0, n, 0], z: [0, 0, n] }[b.dataset.ax];
      // Held gusts last until cleared; a timed one is measured in SIMULATED
      // seconds by the node, so it is the same push at any playback speed.
      post({ gust: force, duration_s: el("gusthold").checked ? 0 : 1.5 });
    };
  }
  el("gustclear").onclick = () => post({ clear_gust: true });

  el("drivestop").onclick = () => sendDrive({ op: "clear" });

  el("reset").onclick = () => {
    cam = homeFor(S.scene && S.scene.worlds[S.world]);
  };
  const followBox = el("follow");
  followBox.checked = prefs.follow;
  followBox.onchange = () => {
    prefs.follow = followBox.checked;
    // Switching it on from a whole-course framing would otherwise follow the
    // vehicle from 300 m away, which looks exactly like it is not working.
    if (prefs.follow) cam.dist = Math.min(cam.dist, FOLLOW_MAX_DIST_M);
    savePrefs();
  };
  el("clear").onclick = () => { S.flown = []; S.planned = []; };

  // Solving goes to its OWN endpoint, not /control. The simulator's write
  // path is documented as unable to run a command; this one runs the
  // optimiser, and keeping them apart is what keeps that sentence true.
  el("solve").onclick = () => {
    const btn = el("solve");
    const log = el("solvelog");
    btn.disabled = true;
    btn.textContent = "solving…";
    log.hidden = false;
    log.textContent = "solving…";
    fetch("/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        N: Number(el("sN").value),
        n_seg: Number(el("sSeg").value),
        v_max: Number(el("sV").value),
      }),
    }).then((r) => r.json()).then((r) => {
      if (r.error) {
        log.textContent = r.error;
        hudError = "solve refused";
      } else {
        S.solver = r;
        hudError = "";
        // The trails belong to the plan that was flying; the new one is a
        // different path through the same fence.
        S.flown = [];
        S.planned = [];
        log.textContent = (r.last && r.last.log) || r.message || "solved";
      }
    }).catch((e) => { log.textContent = `solve: ${e.message}`; })
      .finally(() => { btn.disabled = false; btn.textContent = "solve & fly"; });
  };
  // The scenario dropdown RESTARTS the simulator. It goes to its own endpoint
  // for the same reason /solve does: /control is documented as unable to run a
  // command, and this one runs a launch.
  el("world").onchange = (e) => {
    const name = e.target.value;
    const sel = e.target;
    switching = true;
    sel.disabled = true;
    el("runline").textContent = `asking for ${name}…`;
    el("runline").className = "planline warn";
    el("runlog").hidden = true;
    fetch("/launch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run: name }),
    }).then((r) => r.json()).then((r) => {
      if (r.error) {
        switching = false;
        el("runline").textContent = "refused";
        el("runline").className = "planline bad";
        el("runlog").hidden = false;
        el("runlog").textContent = r.error;
        hudError = "switch refused";
      } else {
        // The old run's path is not this run's path, and its obstacles are
        // not this run's obstacles. Clear them now rather than let the last
        // frame of the previous scenario hang under the new one.
        S.flown = [];
        S.planned = [];
        S.clearance = null;
        S.planPath = null;
        S.launcher = r;
        hudError = "";
      }
    }).catch((err) => {
      switching = false;
      el("runline").textContent = `switch: ${err.message}`;
      el("runline").className = "planline bad";
    }).finally(() => { sel.disabled = false; });
  };
}

// ---- drive: clicks to waypoints -----------------------------------------
const yawOf = (q) => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));

/// The cursor drives the step in progress: on the HEIGHT step it is read off
/// the vertical bar; on the HEADING step the height is fixed and the heading
/// points from the bar toward the cursor at that height. One quantity per
/// step, so moving to aim the heading cannot disturb the height.
function updateHeightPick() {
  if (!cursor || !renderer.view || !S.drive) return;
  const ray = screenRay(renderer.view, cursor[0], cursor[1]);
  if (pick.mode === "height") {
    const z = heightOnBar(ray, pick.x, pick.y);
    if (z !== null) pick.z = clampHeight(z, S.drive.bounds);
  } else if (pick.mode === "heading") {
    pick.heading = headingAt(ray, pick.x, pick.y, pick.z, pick.heading);
  }
}

function driveClick(button, shift, at) {
  const ray = screenRay(renderer.view, at[0], at[1]);
  const ground = rayPlaneZ(ray, 0);
  const onDrone = hitsDrone((p) => renderer.project(p), S.pose, at[0], at[1], 28 * renderer.px);
  const z0 = clampHeight(S.pose ? S.pose.p[2] : 1.0, S.drive.bounds);
  const out = click(pick, { button, shift, onDrone, ground, height: z0,
    heading: S.pose ? yawOf(S.pose.q) : 0 });
  pick = out.state;
  if (pick.mode === "height" || pick.mode === "heading") updateHeightPick();
  if (out.send) sendDrive(out.send);
}

function sendDrive(body) {
  const v = Number(el("sV").value);
  if (v > 0) body.v_max = v;
  fetch("/drive", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => r.json()).then((r) => {
    el("driveerr").textContent = r.error || "";
    if (r.drive) S.drive = r.drive;
  }).catch((e) => { el("driveerr").textContent = `drive: ${e.message}`; });
}

// ---- boot ----------------------------------------------------------------
/// The static half of the scene: the drone's geometry, and per-world palettes
/// and camera framing.
///
/// It no longer fills the dropdown or reads current.json. Both of those are
/// live facts about a simulator that now starts and stops underneath this
/// page, so they arrive on the stream (see adoptRun) where they can change
/// without a reload. This file is only the things that are true of a world
/// whether or not anything is running in it.
function loadScene() {
  return fetch("scene.json")
    .then((r) => r.json())
    .then((scene) => {
      S.scene = scene;
      if (scene.worlds[S.world]) cam = homeFor(scene.worlds[S.world]);
    })
    .catch(() => {
      el("linktext").textContent = "scene.json missing";
    });
}

// A read-only handle for driving the page from a test harness (headless
// Chrome over CDP): where things are drawn and what the click state is. It
// exposes nothing that can write -- writes go through the endpoints like
// any click does.
window.dsimDebug = {
  project: (p) => { const q = renderer.project(p); return [q.x / renderer.px, q.y / renderer.px]; },
  pose: () => S.pose,
  pick: () => ({ ...pick }),
  drive: () => S.drive,
  cloud: () => ({ n: S.cloud.n, seq: S.cloud.seq }),
  flow: () => S.flowViz,
};

loadPrefs();
buildControls();
bindInput();
loadScene();
telemetry.start();
updateHud();
render();
