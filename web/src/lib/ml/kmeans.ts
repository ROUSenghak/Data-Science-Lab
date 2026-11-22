/**
 * K-Means (Lloyd's algorithm) for points with any number of features; the Voronoi cells and the
 * synthetic datasets are two-dimensional. Pure functions; verified against the
 * NumPy experiment (api/app/experiments/templates/kmeans_numpy.py) in kmeans.test.ts.
 *
 * Conventions shared with the NumPy template:
 *  - squared Euclidean distances; argmin ties go to the lowest centroid index;
 *  - an empty cluster keeps its previous centroid;
 *  - stop after the first iteration in which no point changes cluster (scikit-learn, tol = 0).
 */
import { createRng, mulberry32 } from "./random";

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
/** A point with any number of features. Lloyd's algorithm does not depend on the dimension. */
export type Vec = number[];
export type InitMethod = "random" | "kmeans++" | "uniform";

/** Largest K in the interactive labs (one colour and one marker shape per cluster). */
export const MAX_K = 8;

export class KMeansInputError extends Error {}

export function validateInput(X: readonly Vec[], k: number): void {
  if (!Number.isInteger(k) || k < 1) throw new KMeansInputError(`K must be a positive integer (got ${k}).`);
  if (X.length === 0) throw new KMeansInputError("No data points.");
  if (k > X.length) throw new KMeansInputError(`K = ${k} exceeds the number of points (${X.length}).`);
  if (!X.every((p) => p.length === X[0].length && p.every(Number.isFinite))) throw new KMeansInputError("Points must have finite coordinates and the same number of features.");
}

/** ‖a − b‖² = Σ_d (a_d − b_d)², over every feature. */
export function sqDist(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let d = 0; d < a.length; d++) s += (a[d] - b[d]) ** 2;
  return s;
}

/** n × k matrix of squared distances ‖xᵢ − μⱼ‖². */
export const distanceMatrix = (X: readonly Vec[], C: readonly Vec[]) => X.map((x) => C.map((c) => sqDist(x, c)));

/** Index of the smallest value; ties go to the lowest index (as np.argmin). */
export function argmin(v: number[]): number {
  let best = 0;
  for (let j = 1; j < v.length; j++) if (v[j] < v[best]) best = j;
  return best;
}

export const assign = (X: readonly Vec[], C: readonly Vec[]) => distanceMatrix(X, C).map(argmin);

/** μⱼ = mean of the points with label j. Empty clusters keep their previous centroid. */
export function updateCentroids<V extends Vec>(X: readonly V[], labels: number[], prev: readonly V[]) {
  const k = prev.length;
  const dim = prev[0]?.length ?? 0;
  const counts = new Array<number>(k).fill(0);
  const sums = Array.from({ length: k }, () => new Array<number>(dim).fill(0) as V);
  X.forEach((x, i) => {
    const j = labels[i];
    counts[j]++;
    for (let d = 0; d < dim; d++) sums[j][d] += x[d];
  });
  const centroids = prev.map((c, j) => (counts[j] > 0 ? sums[j].map((v) => v / counts[j]) : [...c]) as V);
  const empty = counts.flatMap((c, j) => (c === 0 ? [j] : []));
  return { centroids, counts, sums, empty };
}

/** J = Σᵢ ‖xᵢ − μ_{cᵢ}‖², the within-cluster sum of squares (scikit-learn's inertia). */
export function objective(X: readonly Vec[], C: readonly Vec[], labels: number[]): number {
  let s = 0;
  for (let i = 0; i < X.length; i++) s += sqDist(X[i], C[labels[i]]);
  return s;
}

/** Jⱼ = Σ_{i: cᵢ = j} ‖xᵢ − μⱼ‖², each cluster's share of J (Σⱼ Jⱼ = J). */
export function clusterObjectives(X: readonly Vec[], C: readonly Vec[], labels: number[]): number[] {
  const out = new Array<number>(C.length).fill(0);
  for (let i = 0; i < X.length; i++) out[labels[i]] += sqDist(X[i], C[labels[i]]);
  return out;
}

export function boundingBox<V extends Vec>(X: readonly V[]): { lo: V; hi: V } {
  const dim = X[0]?.length ?? 2;
  const lo = new Array<number>(dim).fill(Infinity) as V;
  const hi = new Array<number>(dim).fill(-Infinity) as V;
  for (const p of X)
    for (let d = 0; d < dim; d++) {
      lo[d] = Math.min(lo[d], p[d]);
      hi[d] = Math.max(hi[d], p[d]);
    }
  return { lo, hi };
}

/**
 * Initial centroids.
 *  - random:   k distinct data points chosen uniformly (Forgy);
 *  - kmeans++: first point uniform, each next one with probability ∝ D(x)², the squared distance
 *              to the nearest centroid chosen so far. If every D(x) = 0 (fewer distinct points
 *              than k), the next one is drawn uniformly;
 *  - uniform:  uniform positions in the bounding box of the data (one draw per feature, in order).
 * `chosen` lists the data indices used (null for uniform).
 */
export function initCentroids<V extends Vec>(X: readonly V[], k: number, method: InitMethod, seed: number): { centroids: V[]; chosen: number[] | null } {
  validateInput(X, k);
  const next = mulberry32(seed);
  const n = X.length;
  if (method === "uniform") {
    const { lo, hi } = boundingBox(X);
    return { centroids: Array.from({ length: k }, () => lo.map((l, d) => l + (hi[d] - l) * next()) as V), chosen: null };
  }
  let chosen: number[];
  if (method === "random") {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < k; i++) {
      const j = i + Math.floor(next() * (n - i));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    chosen = idx.slice(0, k);
  } else {
    chosen = [Math.floor(next() * n)];
    const D2 = X.map((x) => sqDist(x, X[chosen[0]]));
    while (chosen.length < k) {
      const total = D2.reduce((a, v) => a + v, 0);
      let pick: number;
      if (total > 0) {
        let r = next() * total;
        pick = D2.findIndex((d) => (r -= d) < 0);
        if (pick < 0) pick = D2.findLastIndex((d) => d > 0); // floating-point remainder
      } else pick = Math.floor(next() * n);
      chosen.push(pick);
      X.forEach((x, i) => (D2[i] = Math.min(D2[i], sqDist(x, X[pick]))));
    }
  }
  return { centroids: chosen.map((i) => [...X[i]] as V), chosen };
}

/** One recorded half-step of Lloyd's algorithm. */
export interface Phase<V extends Vec = Vec2> {
  kind: "init" | "assign" | "update";
  /** 0 for the initialization; t = 1, 2, … for the iteration this phase belongs to. */
  iteration: number;
  /** Centroids after this phase. */
  centroids: V[];
  /** Labels after this phase (null before the first assignment). */
  labels: number[] | null;
  /** Objective after this phase: J(c⁽ᵗ⁾, μ⁽ᵗ⁻¹⁾) after an assignment, J(c⁽ᵗ⁾, μ⁽ᵗ⁾) after an update. */
  J: number | null;
  /** Assignment: points whose label changed. */
  changed: number[];
  /** Update: centroids before the update, cluster sizes, empty clusters, distance each centroid moved. */
  prevCentroids?: V[];
  counts?: number[];
  empty?: number[];
  shift?: number[];
  /** True on the update that completes the converged iteration. */
  converged?: boolean;
}

export interface KMeansRun<V extends Vec = Vec2> {
  phases: Phase<V>[];
  /** Number of completed iterations (scikit-learn's n_iter_). */
  iterations: number;
  converged: boolean;
  /** Labels re-assigned to the returned centroids, and their inertia. */
  labels: number[];
  centroids: V[];
  inertia: number;
}

export function lloyd<V extends Vec>(X: readonly V[], init: readonly V[], maxIter = 100): KMeansRun<V> {
  validateInput(X, init.length);
  let C: V[] = init.map((c) => [...c] as V);
  let labels: number[] = new Array(X.length).fill(-1);
  const phases: Phase<V>[] = [{ kind: "init", iteration: 0, centroids: C, labels: null, J: null, changed: [] }];
  let converged = false;
  let t = 0;
  while (t < maxIter) {
    t++;
    const D = distanceMatrix(X, C);
    const next = D.map(argmin);
    const changed = next.flatMap((l, i) => (l !== labels[i] ? [i] : []));
    labels = next;
    phases.push({ kind: "assign", iteration: t, centroids: C, labels, J: next.reduce((s, l, i) => s + D[i][l], 0), changed });
    const up = updateCentroids(X, labels, C);
    const shift = up.centroids.map((c, j) => Math.sqrt(sqDist(c, C[j])));
    const prev = C;
    C = up.centroids;
    converged = changed.length === 0;
    phases.push({ kind: "update", iteration: t, centroids: C, labels, J: objective(X, C, labels), changed: [], prevCentroids: prev, counts: up.counts, empty: up.empty, shift, converged });
    if (converged) break;
  }
  const finalLabels = assign(X, C);
  return { phases, iterations: t, converged, labels: finalLabels, centroids: C, inertia: objective(X, C, finalLabels) };
}

export function runKMeans<V extends Vec>(X: readonly V[], k: number, opts: { method: InitMethod; seed: number; maxIter?: number }) {
  const init = initCentroids(X, k, opts.method, opts.seed);
  return { ...lloyd(X, init.centroids, opts.maxIter ?? 100), chosen: init.chosen };
}

/**
 * Mean silhouette coefficient s = mean over i of (bᵢ − aᵢ) / max(aᵢ, bᵢ), with aᵢ the mean distance to
 * the own cluster and bᵢ the smallest mean distance to another cluster; sᵢ = 0 in singleton clusters
 * (scikit-learn's convention). NaN unless 2 ≤ #clusters ≤ n − 1.
 */
export function silhouette(X: Vec2[], labels: number[]): number {
  const ids = [...new Set(labels)];
  if (ids.length < 2 || ids.length > X.length - 1) return NaN;
  const size = new Map<number, number>();
  for (const l of labels) size.set(l, (size.get(l) ?? 0) + 1);
  let total = 0;
  for (let i = 0; i < X.length; i++) {
    const sum = new Map<number, number>();
    for (let m = 0; m < X.length; m++) if (m !== i) sum.set(labels[m], (sum.get(labels[m]) ?? 0) + Math.sqrt(sqDist(X[i], X[m])));
    const own = size.get(labels[i])!;
    if (own === 1) continue;
    const a = (sum.get(labels[i]) ?? 0) / (own - 1);
    let b = Infinity;
    for (const l of ids) if (l !== labels[i]) b = Math.min(b, (sum.get(l) ?? 0) / size.get(l)!);
    total += (b - a) / Math.max(a, b);
  }
  return total / X.length;
}

/** Best inertia over `restarts` k-means++ runs for K = 1 … kMax, and the silhouette of that best run. */
export function elbow(X: Vec2[], kMax: number, restarts = 8, seed = 0) {
  const out: { k: number; inertia: number; silhouette: number }[] = [];
  for (let k = 1; k <= Math.min(kMax, X.length); k++) {
    let best: KMeansRun | null = null;
    for (let r = 0; r < restarts; r++) {
      const run = runKMeans(X, k, { method: "kmeans++", seed: seed + 7919 * r + k });
      if (!best || run.inertia < best.inertia) best = run;
    }
    out.push({ k, inertia: best!.inertia, silhouette: k > 1 ? silhouette(X, best!.labels) : NaN });
  }
  return out;
}

/**
 * Exact Voronoi cell of centroid j inside a box: the region where μⱼ is the nearest centroid,
 * i.e. the intersection of the half-planes 2(μᵢ − μⱼ)·p ≤ ‖μᵢ‖² − ‖μⱼ‖². Coincident centroids
 * follow the tie rule: the higher index gets an empty cell.
 */
export function voronoiCell(C: Vec2[], j: number, box: { lo: Vec2; hi: Vec2 }): Vec2[] {
  let poly: Vec2[] = [
    [box.lo[0], box.lo[1]],
    [box.hi[0], box.lo[1]],
    [box.hi[0], box.hi[1]],
    [box.lo[0], box.hi[1]],
  ];
  const cj = C[j];
  for (let i = 0; i < C.length && poly.length; i++) {
    if (i === j) continue;
    const ci = C[i];
    const a = 2 * (ci[0] - cj[0]);
    const b = 2 * (ci[1] - cj[1]);
    const c = ci[0] ** 2 + ci[1] ** 2 - cj[0] ** 2 - cj[1] ** 2;
    if (a === 0 && b === 0) {
      if (i < j) return [];
      continue;
    }
    const inside = (p: Vec2) => a * p[0] + b * p[1] <= c;
    const out: Vec2[] = [];
    for (let m = 0; m < poly.length; m++) {
      const p = poly[m];
      const q = poly[(m + 1) % poly.length];
      const pin = inside(p);
      const qin = inside(q);
      if (pin) out.push(p);
      if (pin !== qin) {
        const t = (c - a * p[0] - b * p[1]) / (a * (q[0] - p[0]) + b * (q[1] - p[1]));
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
    poly = out;
  }
  return poly;
}

export function polygonArea(poly: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

/* ------------------------------------------------------------------ datasets */

export type ClusterDataKind = "blobs" | "overlapping" | "elongated" | "unequal" | "rings" | "moons" | "uniform";

export const DATASETS: { kind: ClusterDataKind; label: string; k: number; note: string }[] = [
  { kind: "blobs", label: "Separated blobs", k: 3, note: "Three round, equal-size Gaussian clusters: the case K-Means is built for." },
  { kind: "overlapping", label: "Overlapping blobs", k: 3, note: "The blobs overlap, so the boundary cuts through dense regions and some labels are ambiguous." },
  { kind: "elongated", label: "Elongated clusters", k: 3, note: "Three long parallel clusters. J only measures squared distance to a centre, so cutting across them can score better than following them." },
  { kind: "unequal", label: "Unequal sizes", k: 3, note: "One wide cluster and two tight ones. Boundaries are perpendicular bisectors between centroids, so the wide cluster loses its edge to its neighbours." },
  { kind: "rings", label: "Concentric rings", k: 2, note: "Non-convex clusters. Every K-Means cell is a convex polygon, so a ring can never be one cluster." },
  { kind: "moons", label: "Two moons", k: 2, note: "Interleaved half-circles. A straight bisector cannot separate them." },
  { kind: "uniform", label: "No structure", k: 3, note: "Uniform noise. K-Means still returns K clusters: it never says there are none." },
];

/** Synthetic 2-D datasets with known generating groups. `spread` scales the noise. */
export function generateClusters(kind: ClusterDataKind, n: number, seed: number, spread = 1): { X: Vec2[]; truth: number[] } {
  const rng = createRng(seed);
  const X: Vec2[] = [];
  const truth: number[] = [];
  const push = (p: Vec2, g: number) => {
    X.push(p);
    truth.push(g);
  };
  const split = (weights: number[]) => {
    const tot = weights.reduce((a, b) => a + b, 0);
    const sizes = weights.map((w) => Math.floor((n * w) / tot));
    for (let i = 0; sizes.reduce((a, b) => a + b, 0) < n; i++) sizes[i % sizes.length]++;
    return sizes;
  };
  const gauss = (centers: Vec2[], std: Vec2[], weights = centers.map(() => 1)) => {
    split(weights).forEach((m, g) => {
      for (let i = 0; i < m; i++) push([centers[g][0] + rng.normal(0, std[g][0] * spread), centers[g][1] + rng.normal(0, std[g][1] * spread)], g);
    });
  };
  switch (kind) {
    case "blobs":
      gauss([[-3, -2], [3, -1.5], [0, 3]], [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]]);
      break;
    case "overlapping":
      gauss([[-1.3, -0.8], [1.3, -0.8], [0, 1.4]], [[1, 1], [1, 1], [1, 1]]);
      break;
    case "elongated": {
      gauss([[0, -2.4], [0, 0], [0, 2.4]], [[2.6, 0.35], [2.6, 0.35], [2.6, 0.35]]);
      const th = Math.PI / 7;
      for (const p of X) {
        const [a, b] = p;
        p[0] = Math.cos(th) * a - Math.sin(th) * b;
        p[1] = Math.sin(th) * a + Math.cos(th) * b;
      }
      break;
    }
    case "unequal":
      gauss([[-1.6, 0], [2.6, 1.6], [2.6, -1.6]], [[1.7, 1.7], [0.3, 0.3], [0.3, 0.3]], [6, 1, 1]);
      break;
    case "rings":
      split([1, 2]).forEach((m, g) => {
        const r = g === 0 ? 1.2 : 3.6;
        for (let i = 0; i < m; i++) {
          const a = rng.uniform(0, 2 * Math.PI);
          push([r * Math.cos(a) + rng.normal(0, 0.18 * spread), r * Math.sin(a) + rng.normal(0, 0.18 * spread)], g);
        }
      });
      break;
    case "moons":
      split([1, 1]).forEach((m, g) => {
        for (let i = 0; i < m; i++) {
          const a = rng.uniform(0, Math.PI);
          const p: Vec2 = g === 0 ? [2 * Math.cos(a) - 1, 2 * Math.sin(a) - 0.5] : [1 - 2 * Math.cos(a), 0.5 - 2 * Math.sin(a)];
          push([p[0] + rng.normal(0, 0.15 * spread), p[1] + rng.normal(0, 0.15 * spread)], g);
        }
      });
      break;
    case "uniform":
      for (let i = 0; i < n; i++) push([rng.uniform(-3, 3), rng.uniform(-3, 3)], 0);
      break;
  }
  return { X, truth };
}

/* ------------------------------------------------------------------ three features */

/**
 * Centres of the three generating groups of `generateClusters3D`. Groups 0 and 2 have the same
 * (x₁, x₂) and differ only in x₃: projected onto the first two features they are one blob.
 */
export const HIDDEN_CENTERS: Vec3[] = [
  [-1.8, -0.6, -1.9],
  [1.9, 0.8, 0],
  [-1.8, -0.6, 1.9],
];

/** Three round Gaussian groups in R³ (equal sizes, σ = 0.6 · spread per feature). */
export function generateClusters3D(n: number, seed: number, spread = 1): { X: Vec3[]; truth: number[] } {
  const rng = createRng(seed);
  const X: Vec3[] = [];
  const truth: number[] = [];
  for (let i = 0; i < n; i++) {
    const g = i % 3;
    const c = HIDDEN_CENTERS[g];
    X.push([c[0] + rng.normal(0, 0.6 * spread), c[1] + rng.normal(0, 0.6 * spread), c[2] + rng.normal(0, 0.6 * spread)]);
    truth.push(g);
  }
  return { X, truth };
}

/** Fraction of points whose cluster matches the generating group, under the best relabelling (K ≤ 4). */
export function agreement(labels: number[], truth: number[], k: number): number {
  const perms = (a: number[]): number[][] => (a.length <= 1 ? [a] : a.flatMap((v, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map((p) => [v, ...p])));
  let best = 0;
  for (const p of perms(Array.from({ length: k }, (_, j) => j))) {
    let hits = 0;
    for (let i = 0; i < labels.length; i++) if (p[labels[i]] === truth[i]) hits++;
    best = Math.max(best, hits / labels.length);
  }
  return best;
}
