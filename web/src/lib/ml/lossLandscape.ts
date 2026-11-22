/**
 * Geometry of the MSE landscape L(w, b), shared by the 2D contour view and the 3D surface view so
 * that both always show the same window, the same contour levels and the same numbers.
 * Pure functions, no rendering. Every height is an exact evaluation of mse(); nothing is fitted.
 */
import { leastSquares, mse, type Dataset } from "./linreg";

export interface ParamDomain {
  w: [number, number];
  b: [number, number];
}

export type Pt = [number, number];

const finite = (p: readonly number[]) => p.every(Number.isFinite);

/**
 * Window used when the caller gives no fixed domain: centred on θ*, wide enough to contain the anchor
 * (e.g. the initial parameters) and the first point of the trajectory, with 35% margin.
 */
export function autoDomain(data: Dataset, anchor?: Pt | null, first?: Pt | null): ParamDomain {
  const opt = leastSquares(data);
  const pts = [[opt.w, opt.b], ...(anchor ? [anchor] : []), ...(first ? [first] : [])].filter(finite);
  const rw = 1.35 * Math.max(1, ...pts.map((p) => Math.abs(p[0] - opt.w)));
  const rb = 1.35 * Math.max(1, ...pts.map((p) => Math.abs(p[1] - opt.b)));
  return { w: [opt.w - rw, opt.w + rw], b: [opt.b - rb, opt.b + rb] };
}

/** L is a convex quadratic, so its maximum over a rectangle is attained at one of the four corners. */
export function maxLossOnDomain(data: Dataset, d: ParamDomain): number {
  return Math.max(mse(d.w[0], d.b[0], data), mse(d.w[0], d.b[1], data), mse(d.w[1], d.b[0], data), mse(d.w[1], d.b[1], data));
}

/** Ratio between consecutive contour levels' excess loss L − L*. */
export const LEVEL_RATIO = 0.6;
export const LEVEL_COUNT = 14;

/**
 * Contour levels L* + E·0.6ᵏ (k = 0 … count−1), with E the largest excess loss in the window.
 * Geometric spacing puts as many level sets near the minimum as on the steep walls.
 */
export function contourLevels(data: Dataset, d: ParamDomain, count = LEVEL_COUNT): number[] {
  const opt = leastSquares(data);
  const maxEx = maxLossOnDomain(data, d) - opt.loss;
  return Array.from({ length: count }, (_, k) => opt.loss + maxEx * LEVEL_RATIO ** k);
}

/**
 * Shade of a loss value in [0, 1]: the fraction of contour levels that lie above it, as a continuous
 * number. The 2D view fills every level set with the same translucent colour, so a point inside k of
 * them is k layers deep; this is that k / count, without the steps. 0 at the highest level, 1 at the
 * lowest level and below.
 */
export function levelShade(L: number, Lstar: number, maxEx: number, count = LEVEL_COUNT): number {
  if (!Number.isFinite(L) || !(maxEx > 0)) return 0;
  const ex = L - Lstar;
  if (ex <= 0) return 1;
  const k = Math.log(ex / maxEx) / Math.log(LEVEL_RATIO);
  return Math.min(1, Math.max(0, k / (count - 1)));
}

export interface SurfaceGrid {
  /** Nodes per side. */
  res: number;
  w: number[];
  b: number[];
  /** L at node (i, j) = (w[i], b[j]), row-major: L[j * res + i]. Exact mse() values. */
  L: Float64Array;
  /** Largest L in the window (attained at a corner). */
  max: number;
  /** Least-squares minimum. */
  opt: { w: number; b: number; loss: number };
}

/** Exact loss at the nodes of a res × res grid over the window. */
export function surfaceGrid(data: Dataset, d: ParamDomain, res = 72): SurfaceGrid {
  const n = Math.max(2, Math.floor(res));
  const w = Array.from({ length: n }, (_, i) => d.w[0] + ((d.w[1] - d.w[0]) * i) / (n - 1));
  const b = Array.from({ length: n }, (_, j) => d.b[0] + ((d.b[1] - d.b[0]) * j) / (n - 1));
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) L[j * n + i] = mse(w[i], b[j], data);
  return { res: n, w, b, L, max: maxLossOnDomain(data, d), opt: leastSquares(data) };
}

/**
 * Largest gap between the drawn surface and the true loss, in loss units. The surface is a mesh of
 * triangles through exact node values; along any direction u the error of linear interpolation is at
 * most h²·uᵀHu / 8 over a cell of diagonal h, and uᵀHu ≤ λ_max. Used to state the drawing accuracy.
 */
export function interpolationErrorBound(d: ParamDomain, res: number, lambdaMax: number): number {
  const hw = (d.w[1] - d.w[0]) / (res - 1);
  const hb = (d.b[1] - d.b[0]) / (res - 1);
  return ((hw * hw + hb * hb) * lambdaMax) / 8;
}

/** Liang–Barsky: the part of segment p→q inside the window, as parameters [t0, t1] ⊂ [0, 1], or null. */
function clipSegment(p: Pt, q: Pt, d: ParamDomain): [number, number] | null {
  let t0 = 0;
  let t1 = 1;
  const dx = q[0] - p[0];
  const dy = q[1] - p[1];
  const checks: [number, number][] = [
    [-dx, p[0] - d.w[0]],
    [dx, d.w[1] - p[0]],
    [-dy, p[1] - d.b[0]],
    [dy, d.b[1] - p[1]],
  ];
  for (const [pk, qk] of checks) {
    if (pk === 0) {
      if (qk < 0) return null;
      continue;
    }
    const r = qk / pk;
    if (pk < 0) {
      if (r > t1) return null;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return null;
      if (r < t1) t1 = r;
    }
  }
  return [t0, t1];
}

/** Point at parameter t on p→q; the endpoints are returned exactly (iterates keep their computed values). */
const at = (p: Pt, q: Pt, t: number): Pt => (t <= 0 ? [p[0], p[1]] : t >= 1 ? [q[0], q[1]] : [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);

/**
 * The parts of a polyline inside the window, split where it leaves and re-enters, with the exact
 * crossing points added. Non-finite vertices break the line.
 */
export function clipPolyline(pts: readonly Pt[], d: ParamDomain): Pt[][] {
  const out: Pt[][] = [];
  let cur: Pt[] = [];
  const flush = () => {
    if (cur.length > 1) out.push(cur);
    cur = [];
  };
  if (pts.length === 1 && finite(pts[0]) && inDomain(pts[0], d)) return [[pts[0]]];
  for (let k = 0; k + 1 < pts.length; k++) {
    const p = pts[k];
    const q = pts[k + 1];
    if (!finite(p) || !finite(q)) {
      flush();
      continue;
    }
    const c = clipSegment(p, q, d);
    if (!c) {
      flush();
      continue;
    }
    const a = at(p, q, c[0]);
    const b = at(p, q, c[1]);
    if (c[0] > 0 || !cur.length) {
      flush();
      cur.push(a);
    }
    cur.push(b);
    if (c[1] < 1) flush();
  }
  flush();
  return out;
}

export const inDomain = (p: readonly number[], d: ParamDomain) => finite(p) && p[0] >= d.w[0] && p[0] <= d.w[1] && p[1] >= d.b[0] && p[1] <= d.b[1];

/**
 * Inserts `perSegment − 1` evenly spaced points into every segment, so that a straight path in
 * (w, b) can be drawn on the curved surface: each inserted point gets its own exact height L(w, b).
 * The inserted points are drawing positions, not iterations.
 */
export function densify(pts: readonly Pt[], perSegment: number): Pt[] {
  const m = Math.max(1, Math.floor(perSegment));
  if (pts.length < 2 || m === 1) return pts.map((p) => [p[0], p[1]]);
  const out: Pt[] = [[pts[0][0], pts[0][1]]];
  for (let k = 0; k + 1 < pts.length; k++) for (let s = 1; s <= m; s++) out.push(at(pts[k], pts[k + 1], s / m));
  return out;
}

/**
 * Top of the plotted loss range for a 3D view. The window's corners are usually far higher than
 * anything an optimizer visits, so a box up to the corner maximum squashes the descent into its
 * bottom few percent. Instead the box ends `headroom` × the highest starting excess loss above L*,
 * never above the window's maximum. Without starting points the whole window is shown.
 */
export function plotTop(data: Dataset, d: ParamDomain, starts: readonly (Pt | null | undefined)[], headroom = 1.3): number {
  const top = maxLossOnDomain(data, d);
  const opt = leastSquares(data).loss;
  const ex = starts.filter((p): p is Pt => !!p && inDomain(p, d)).map((p) => mse(p[0], p[1], data) - opt);
  if (!ex.length) return top;
  const want = opt + Math.max(...ex) * headroom;
  // A start at (or extremely near) the minimum still gets a visible bowl.
  return Math.min(top, Math.max(want, opt + 0.04 * (top - opt)));
}

/**
 * Splits a lifted polyline [(w, b, L)] where it rises above `zTop`, keeping the parts at or below it.
 * Crossing points are interpolated between neighbouring drawn points (densify first for accuracy).
 */
export function clipAbove(pts: readonly [number, number, number][], zTop: number): [number, number, number][][] {
  const out: [number, number, number][][] = [];
  let cur: [number, number, number][] = [];
  const cross = (p: [number, number, number], q: [number, number, number]): [number, number, number] => {
    const t = (zTop - p[2]) / (q[2] - p[2]);
    return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, zTop];
  };
  for (let k = 0; k < pts.length; k++) {
    const p = pts[k];
    const inside = p[2] <= zTop;
    const prev = k > 0 ? pts[k - 1] : null;
    if (inside) {
      if (prev && prev[2] > zTop) cur.push(cross(prev, p));
      cur.push(p);
    } else if (prev && prev[2] <= zTop) {
      cur.push(cross(prev, p));
      if (cur.length > 1) out.push(cur);
      cur = [];
    }
  }
  if (cur.length > 1) out.push(cur);
  return out;
}

/** (w, b) → (w, b, L(w, b)), exact. */
export const lift = (pts: readonly Pt[], data: Dataset): [number, number, number][] => pts.map(([w, b]) => [w, b, mse(w, b, data)]);
