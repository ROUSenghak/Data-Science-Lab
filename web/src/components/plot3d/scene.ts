/**
 * Pure geometry for 3D plots: how data coordinates map into the scene, where the camera sits, and
 * which box edges carry the tick labels. No three.js, no React, so it can be unit-tested.
 *
 * Convention (matches the 2D charts when seen from above): data x → scene +X (right), data y → scene
 * −Z (away from the viewer, i.e. "up" in a top view), data z → scene +Y (up). The floor is z = z₀.
 */
import { scaleLinear } from "d3-scale";

export type Range = [number, number];
export type V3 = [number, number, number];

export interface BoxSpec {
  x: Range;
  y: Range;
  z: Range;
  /** One unit of x as long as one unit of y on the floor (true angles, as an equal-aspect 2D chart). */
  equalXY?: boolean;
  /** Same unit length on all three axes (data whose features share a scale, e.g. clustering). */
  equalXYZ?: boolean;
  /** Box height in scene units; the longer floor side is 2. Ignored with equalXYZ. */
  height?: number;
}

export interface BoxMap {
  spec: BoxSpec;
  /** Half-extents of the floor in scene units. */
  hx: number;
  hz: number;
  height: number;
  toScene(x: number, y: number, z: number): V3;
  /** Scene floor coordinates (X, Z) → data (x, y). */
  toData(X: number, Z: number): [number, number];
  /** Data z → scene height. */
  zToY(z: number): number;
}

const span = (r: Range) => r[1] - r[0] || 1;
const mid = (r: Range) => (r[0] + r[1]) / 2;

export function boxMap(spec: BoxSpec): BoxMap {
  let height = spec.height ?? 1.1;
  let sx: number;
  let sy: number;
  if (spec.equalXYZ) {
    // One unit is the same length along every axis: distances in the picture are true distances.
    const s = 2 / Math.max(span(spec.x), span(spec.y), span(spec.z));
    sx = s;
    sy = s;
    height = span(spec.z) * s;
  } else if (spec.equalXY) {
    const s = 2 / Math.max(span(spec.x), span(spec.y));
    sx = s;
    sy = s;
  } else {
    sx = 2 / span(spec.x);
    sy = 2 / span(spec.y);
  }
  const mx = mid(spec.x);
  const my = mid(spec.y);
  const zToY = (z: number) => ((z - spec.z[0]) / span(spec.z)) * height;
  return {
    spec,
    hx: (span(spec.x) * sx) / 2,
    hz: (span(spec.y) * sy) / 2,
    height,
    toScene: (x, y, z) => [(x - mx) * sx, zToY(z), -(y - my) * sy],
    toData: (X, Z) => [mx + X / sx, my - Z / sy],
    zToY,
  };
}

/* ------------------------------------------------------------------ camera */

export interface ViewAngles {
  /** Radians; 0 looks from the front (low y), positive turns the camera towards +x. */
  azimuth: number;
  /** Radians above the floor plane; π/2 is a top view. */
  elevation: number;
}

export const DEFAULT_VIEW: ViewAngles = { azimuth: (30 * Math.PI) / 180, elevation: (30 * Math.PI) / 180 };
/** Top view, oriented exactly like the 2D chart (x to the right, y up). */
export const TOP_VIEW: ViewAngles = { azimuth: 0, elevation: Math.PI / 2 - 1e-4 };
/** Never look from below the floor; never flip over the top. */
export const ELEVATION_LIMITS: Range = [(4 * Math.PI) / 180, Math.PI / 2 - 1e-4];

export function clampElevation(e: number): number {
  return Math.min(ELEVATION_LIMITS[1], Math.max(ELEVATION_LIMITS[0], e));
}

/** Camera position for the given angles and distance around a target. */
export function orbitPosition(view: ViewAngles, distance: number, target: V3): V3 {
  const ce = Math.cos(view.elevation);
  return [target[0] + distance * ce * Math.sin(view.azimuth), target[1] + distance * Math.sin(view.elevation), target[2] + distance * ce * Math.cos(view.azimuth)];
}

/** Screen-plane axes (right, up) of an orthographic camera looking at the target from `view`. */
export function viewBasis(view: ViewAngles): { right: V3; up: V3 } {
  const ce = Math.cos(view.elevation);
  const d: V3 = [ce * Math.sin(view.azimuth), Math.sin(view.elevation), ce * Math.cos(view.azimuth)];
  const right: V3 = [Math.cos(view.azimuth), 0, -Math.sin(view.azimuth)];
  const up: V3 = [d[1] * right[2] - d[2] * right[1], d[2] * right[0] - d[0] * right[2], d[0] * right[1] - d[1] * right[0]];
  return { right, up };
}

/**
 * Orthographic zoom (pixels per scene unit) at which the whole box, seen from `view` and centred on
 * `target`, fits the canvas with `pad` pixels left for tick labels around it.
 */
export function fitZoom(map: BoxMap, view: ViewAngles, target: V3, width: number, height: number, pad: [number, number] = [34, 26]): number {
  const { right, up } = viewBasis(view);
  let mx = 1e-6;
  let my = 1e-6;
  for (const X of [-map.hx, map.hx])
    for (const Y of [0, map.height])
      for (const Z of [-map.hz, map.hz]) {
        const c: V3 = [X - target[0], Y - target[1], Z - target[2]];
        mx = Math.max(mx, Math.abs(c[0] * right[0] + c[1] * right[1] + c[2] * right[2]));
        my = Math.max(my, Math.abs(c[0] * up[0] + c[1] * up[1] + c[2] * up[2]));
      }
  return Math.max(1, Math.min((width - 2 * pad[0]) / (2 * mx), (height - 2 * pad[1]) / (2 * my)));
}

/* ------------------------------------------------------------------ axes */

/** Signs of the floor edges nearest to the camera: tick labels go there so the box never hides them. */
export function nearSides(camX: number, camZ: number): { x: 1 | -1; z: 1 | -1 } {
  return { x: camX >= 0 ? 1 : -1, z: camZ >= 0 ? 1 : -1 };
}

export function niceTicks(r: Range, count: number): number[] {
  return scaleLinear().domain(r).ticks(count);
}

/** Where each axis' tick labels sit, in scene coordinates, for the given near sides. */
export function tickAnchor(map: BoxMap, axis: "x" | "y" | "z", value: number, sides: { x: 1 | -1; z: 1 | -1 }, offset = 0.11): V3 {
  if (axis === "x") {
    const [X] = map.toScene(value, mid(map.spec.y), map.spec.z[0]);
    return [X, 0, sides.z * (map.hz + offset)];
  }
  if (axis === "y") {
    const [, , Z] = map.toScene(mid(map.spec.x), value, map.spec.z[0]);
    return [sides.x * (map.hx + offset), 0, Z];
  }
  // Vertical axis on the near corner opposite the y-axis labels (where matplotlib puts it).
  return [-sides.x * (map.hx + offset * 0.7), map.zToY(value), sides.z * (map.hz + offset * 0.7)];
}

/** Corner of the floor that carries the vertical axis. */
export const verticalAxisCorner = (map: BoxMap, sides: { x: 1 | -1; z: 1 | -1 }): [number, number] => [-sides.x * map.hx, sides.z * map.hz];
