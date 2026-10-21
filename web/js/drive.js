// Drive mode: RTS-style waypoints, the clicking half. Pure, tested in node.
//
//   left-click the drone        select it
//   left-click empty space      deselect
//   right-click (selected)      pick x,y on the ground -> HEIGHT step: a
//                                 vertical bar over that spot, the cursor
//                                 picks the height on it
//   right-click (height)        fix the height -> HEADING step: a bar at
//                                 the point turns toward the cursor
//   right-click (heading)       send: the waypoint REPLACES the queue, or
//                                 with shift held is APPENDED to it
//   left-click (height/heading) cancel that waypoint
//
// This file decides what a click MEANS and turns a screen position into a
// world point. It does not decide whether the point is allowed: the server
// does, against the bounds the drive node published, and the page shows its
// reason. The bounds are used here only to keep the height bar drawn over
// the band the server will accept, so the cursor cannot pick a height that
// is certain to be refused.
//
// Picking is the inverse of render.js's projection, from the same view
// basis (eye, right, up, fwd, f), so a point picked under the cursor is the
// point drawn under the cursor.

import { add, scale, sub, dot } from "./vec3.js";

/// The world ray through a canvas pixel (backing-store pixels, as project()
/// returns them): {o, d}, d not normalised.
export function screenRay(view, x, y) {
  const d = add(view.fwd, add(scale(view.right, (x - view.w * 0.5) / view.f),
                              scale(view.up, (view.h * 0.5 - y) / view.f)));
  return { o: view.eye, d };
}

/// Where the ray meets the horizontal plane at height z, or null if it
/// never does (parallel, or the plane is behind the eye).
export function rayPlaneZ(ray, z) {
  if (Math.abs(ray.d[2]) < 1e-9) return null;
  const t = (z - ray.o[2]) / ray.d[2];
  if (t <= 0) return null;
  return add(ray.o, scale(ray.d, t));
}

/// The height on the vertical line through (x, y) closest to the ray: the
/// standard closest-points-of-two-lines solve, with the vertical line as
/// L(s) = (x, y, s). Null when the ray is itself vertical (looking straight
/// down the bar, no height can be read).
export function heightOnBar(ray, x, y) {
  const w0 = sub(ray.o, [x, y, 0]);
  const a = dot(ray.d, ray.d);
  const b = ray.d[2];                 // d . (0,0,1)
  const c = 1;
  const d = dot(ray.d, w0);
  const e = w0[2];
  const den = a * c - b * b;
  if (den < 1e-9) return null;
  const t = (b * e - c * d) / den;    // along the ray
  const s = (a * e - b * d) / den;    // along the bar
  return t > 0 ? s : null;
}

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/// Heading from the bar toward where the cursor's ray crosses the chosen
/// height. Inside `deadband` metres of the bar the direction is noise, so the
/// previous heading is kept.
export function headingAt(ray, x, y, z, previous, deadband = 0.15) {
  const q = rayPlaneZ(ray, z);
  if (!q) return previous;
  const dx = q[0] - x;
  const dy = q[1] - y;
  return Math.hypot(dx, dy) < deadband ? previous : wrap(Math.atan2(dy, dx));
}

/// True if the canvas pixel (x, y) is within `radius` pixels of the drone as
/// drawn. `project` is the renderer's own projection.
export function hitsDrone(project, pose, x, y, radius) {
  if (!pose) return false;
  const q = project(pose.p);
  if (q.depth <= 0.02) return false;
  return Math.hypot(q.x - x, q.y - y) <= radius;
}

/// The click state machine. `mode` is "idle" | "selected" | "height" |
/// "heading".
/// Returns {state, send} where `send` is a /drive request body or null.
export function click(state, ev) {
  // ev: {button: 0|2, shift, onDrone, ground: [x,y,z]|null, height, heading}
  if (ev.button === 0) {
    if (state.mode === "height" || state.mode === "heading") {
      return { state: { mode: "selected" }, send: null };
    }
    return { state: { mode: ev.onDrone ? "selected" : "idle" }, send: null };
  }
  if (ev.button !== 2) return { state, send: null };
  if (state.mode === "selected") {
    if (!ev.ground) return { state, send: null };
    return {
      state: { mode: "height", x: ev.ground[0], y: ev.ground[1],
               z: ev.height, heading: ev.heading },
      send: null,
    };
  }
  if (state.mode === "height") {
    return { state: { ...state, mode: "heading" }, send: null };
  }
  if (state.mode === "heading") {
    return {
      state: { mode: "selected" },
      send: { op: ev.shift ? "append" : "replace",
              p: [state.x, state.y, state.z], yaw: state.heading },
    };
  }
  return { state, send: null };     // idle: a right-click is only a pan
}

/// Clamp a picked height into the band the server accepts, for DRAWING the
/// candidate. (The request still carries it and the server still checks.)
export function clampHeight(z, bounds) {
  if (z === null || z === undefined || !bounds) return z;
  return Math.max(bounds.z[0], Math.min(bounds.z[1], z));
}
