// Overlay colours and labels. Presentation only.
//
// The geometry these colours are painted onto is computed on the ROS side
// (dsim_viz/overlay.py) and arrives as world-space line segments tagged with a
// `kind`. This file decides what each kind looks like, which is the browser's
// job and nobody else's -- so a palette change is a one-file change and can
// never alter what is being drawn.

export const PALETTE = {
  thrust: "#34d399",
  weight: "#94a3b8",
  aero: "#fb923c",
  velocity: "#60a5fa",
  torque: "#facc15",
  cmd_axis: "#e879f9",
  body_axis: "#22d3ee",
  // The cascade's commands share one colour family, deliberately: they are
  // three levels of one controller, and reading them as a set is the point.
  cmd_position: "#f472b6",
  cmd_velocity: "#f472b6",
};

export const TICK_COLOUR = "rgba(230,232,239,.6)";

export const GROUP_LABEL = {
  motors: "motor thrust",
  forces: "forces",
  velocity: "velocity",
  attitude: "attitude cmd",
  torque: "torque",
  command: "commands (pos/vel/att)",
  cloud: "depth cloud + frustum",
  flow: "flow sensor + rangefinder",
};


export const LEGEND = [
  ["thrust", "thrust"],
  ["weight", "weight"],
  ["aero (measured)", "aero"],
  ["velocity", "velocity"],
  ["torque", "torque"],
  ["attitude cmd", "cmd_axis"],
  ["body z", "body_axis"],
  ["cmd position / velocity", "cmd_position"],
];

const BELOW = [59, 130, 246];      // blue   -- below hover thrust
const NEUTRAL = [230, 232, 239];   // white  -- exactly hover
const ABOVE = [249, 115, 22];      // orange -- above hover

const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

/// Diverging colour for a rotor's deviation from hover, already computed and
/// clamped to [-1, 1] by overlay.py. A quadrotor holds a bank by splitting
/// thrust across a diagonal, so this is what makes the control action readable
/// at a glance: two rotors go orange, two go blue, and the split grows with
/// the manoeuvre.
export function rotorColour(norm) {
  const t = typeof norm === "number" ? norm : 0;
  return rgb(t < 0 ? mix(NEUTRAL, BELOW, -t) : mix(NEUTRAL, ABOVE, t));
}

export function colourFor(arrow) {
  return arrow.kind === "rotor" ? rotorColour(arrow.norm)
    : (PALETTE[arrow.kind] || "#e6e8ef");
}

const WIDTH = {
  rotor: 3.5, thrust: 4, weight: 4, body_axis: 2, cmd_axis: 2,
  cmd_position: 2.5, cmd_velocity: 2.5,
};
export const widthFor = (arrow) => WIDTH[arrow.kind] || 3;

/// The depth cloud's colour by DEPTH (the camera's own optical z, sent with
/// each point): near warm, far cool, over the range the camera is worth
/// anything at. Past `far` it stays at the far colour rather than wrapping.
const DEPTH_STOPS = [
  [0.0, [248, 113, 113]],   // red    -- right in front
  [0.25, [251, 191, 36]],   // amber
  [0.5, [74, 222, 128]],    // green
  [0.75, [56, 189, 248]],   // sky
  [1.0, [129, 140, 248]],   // indigo -- as far as it usefully sees
];
export function depthColour(depth, far = 6.0) {
  const t = Math.max(0, Math.min(1, depth / far));
  for (let i = 1; i < DEPTH_STOPS.length; i++) {
    const [t1, c1] = DEPTH_STOPS[i];
    if (t <= t1) {
      const [t0, c0] = DEPTH_STOPS[i - 1];
      return rgb(mix(c0, c1, (t - t0) / (t1 - t0)));
    }
  }
  return rgb(DEPTH_STOPS[DEPTH_STOPS.length - 1][1]);
}

export const FLOW_COLOUR = {
  beam_ok: "rgba(74,222,128,.9)",
  beam_bad: "rgba(148,163,184,.55)",
  measured: "#22d3ee",
  truth: "rgba(230,232,239,.85)",
};
export const FRUSTUM_COLOUR = "rgba(250,204,21,.45)";
