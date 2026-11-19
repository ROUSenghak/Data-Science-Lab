import { describe, expect, it } from "vitest";
import fixtures from "./__fixtures__/kmeans-numpy.json";
import {
  argmin,
  assign,
  DATASETS,
  distanceMatrix,
  elbow,
  generateClusters,
  initCentroids,
  KMeansInputError,
  lloyd,
  objective,
  polygonArea,
  runKMeans,
  silhouette,
  sqDist,
  updateCentroids,
  voronoiCell,
  type InitMethod,
  type Vec2,
} from "./kmeans";

const close = (a: number, b: number, rel = 1e-10) => expect(Math.abs(a - b)).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(b)));

describe("K-Means against the NumPy experiment", () => {
  for (const fx of fixtures) {
    it(`reproduces every iteration for ${JSON.stringify(fx.params)}`, () => {
      const X = fx.X as Vec2[];
      const run = lloyd(X, fx.centroids_init as Vec2[], 20);
      const assigns = run.phases.filter((p) => p.kind === "assign");
      const updates = run.phases.filter((p) => p.kind === "update");
      expect(run.iterations).toBe(fx.iterations.length);
      fx.iterations.forEach((ref, t) => {
        expect(assigns[t].labels).toEqual(ref.labels);
        expect(assigns[t].changed.length).toBe(ref.n_changed);
        close(assigns[t].J!, ref.J_assign);
        expect(updates[t].counts).toEqual(ref.counts);
        updates[t].centroids.forEach((c, j) => {
          close(c[0], ref.centroids[j][0], 1e-12);
          close(c[1], ref.centroids[j][1], 1e-12);
        });
        close(updates[t].J!, ref.J_update);
      });
      expect(run.labels).toEqual(fx.final.labels);
      close(run.inertia, fx.final.inertia);
    });
  }

  it("silhouette matches scikit-learn's silhouette_score", () => {
    for (const fx of fixtures) if (fx.silhouette != null) close(silhouette(fx.X as Vec2[], fx.final.labels), fx.silhouette, 1e-10);
  });

  it("the fixture with uniform initialization exercises the empty-cluster policy", () => {
    const fx = fixtures.find((f) => f.params.init_method === 2)!;
    expect(fx.iterations.some((it) => it.counts.includes(0))).toBe(true);
  });
});

describe("K-Means steps", () => {
  const X: Vec2[] = [
    [0, 0],
    [1, 0],
    [0, 1],
    [5, 5],
    [6, 5],
  ];

  it("assigns each point to its nearest centroid, ties to the lowest index", () => {
    expect(assign(X, [[0, 0], [5, 5]])).toEqual([0, 0, 0, 1, 1]);
    // (1, 0) is equidistant from (0, 0) and (2, 0): the lower index wins
    expect(assign([[1, 0]], [[2, 0], [0, 0]])).toEqual([0]);
    expect(argmin([3, 1, 1])).toBe(1);
    expect(distanceMatrix([[1, 2]], [[4, 6]])).toEqual([[25]]);
  });

  it("moves centroids to the mean of their points and keeps empty clusters in place", () => {
    const up = updateCentroids(X, [0, 0, 0, 1, 1], [[9, 9], [9, 9], [-4, 7]]);
    expect(up.centroids[0][0]).toBeCloseTo(1 / 3, 15);
    expect(up.centroids[0][1]).toBeCloseTo(1 / 3, 15);
    expect(up.centroids[1]).toEqual([5.5, 5]);
    expect(up.centroids[2]).toEqual([-4, 7]);
    expect(up.counts).toEqual([3, 2, 0]);
    expect(up.empty).toEqual([2]);
  });

  it("computes the objective as the within-cluster sum of squares", () => {
    const C: Vec2[] = [[1 / 3, 1 / 3], [5.5, 5]];
    const labels = [0, 0, 0, 1, 1];
    const direct = X.reduce((s, x, i) => s + sqDist(x, C[labels[i]]), 0);
    expect(objective(X, C, labels)).toBe(direct);
    expect(direct).toBeCloseTo(2 * (1 / 9 + 1 / 9) + (4 / 9 + 4 / 9) + 0.5, 12);
  });

  it("the mean minimizes the within-cluster sum of squares (guarantee)", () => {
    const pts: Vec2[] = generateClusters("blobs", 30, 3).X;
    const m = updateCentroids(pts, pts.map(() => 0), [[0, 0]]).centroids[0];
    const J = (c: Vec2) => pts.reduce((s, p) => s + sqDist(p, c), 0);
    for (const d of [[1e-3, 0], [0, -1e-3], [0.5, 0.5]] as Vec2[]) expect(J([m[0] + d[0], m[1] + d[1]])).toBeGreaterThan(J(m));
  });
});

describe("Lloyd's algorithm", () => {
  const methods: InitMethod[] = ["random", "kmeans++", "uniform"];

  it("never increases the objective, at any half-step (guarantee, 150 random cases)", () => {
    let cases = 0;
    for (const d of DATASETS)
      for (const method of methods)
        for (let seed = 0; seed < 7; seed++) {
          const k = 1 + (seed % 8);
          const { X } = generateClusters(d.kind, 40 + seed * 9, seed, 0.6 + 0.2 * seed);
          const run = runKMeans(X, k, { method, seed });
          const Js = run.phases.filter((p) => p.J != null).map((p) => p.J!);
          for (let t = 1; t < Js.length; t++) expect(Js[t]).toBeLessThanOrEqual(Js[t - 1] * (1 + 1e-12));
          cases++;
        }
    expect(cases).toBeGreaterThanOrEqual(147);
  });

  it("stops after the first iteration without label changes, when centroids no longer move", () => {
    const { X } = generateClusters("overlapping", 90, 5);
    const run = runKMeans(X, 4, { method: "random", seed: 2 });
    expect(run.converged).toBe(true);
    const last = run.phases.at(-1)!;
    const lastAssign = run.phases.at(-2)!;
    expect(lastAssign.changed).toEqual([]);
    expect(last.shift!.every((s) => s === 0)).toBe(true);
    expect(run.inertia).toBe(last.J);
    run.phases.filter((p) => p.kind === "assign").slice(0, -1).forEach((p) => expect(p.changed.length).toBeGreaterThan(0));
  });

  it("reports non-convergence when max_iter is reached, and re-assigns the final labels", () => {
    const { X } = generateClusters("uniform", 120, 1);
    const run = runKMeans(X, 6, { method: "uniform", seed: 1, maxIter: 1 });
    expect(run.iterations).toBe(1);
    expect(run.converged).toBe(false);
    expect(run.labels).toEqual(assign(X, run.centroids));
    expect(run.inertia).toBeLessThanOrEqual(run.phases.at(-1)!.J!);
  });

  it("K = 1 gives the mean and the total sum of squares", () => {
    const { X } = generateClusters("moons", 50, 4);
    const run = runKMeans(X, 1, { method: "random", seed: 0 });
    const mx = X.reduce((s, p) => s + p[0], 0) / X.length;
    const my = X.reduce((s, p) => s + p[1], 0) / X.length;
    close(run.centroids[0][0], mx, 1e-12);
    close(run.inertia, X.reduce((s, p) => s + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0), 1e-12);
    close(elbow(X, 1)[0].inertia, run.inertia, 1e-12);
  });

  it("handles duplicate points: coincident centroids leave one cluster empty, no NaN", () => {
    const X: Vec2[] = [...Array(6).fill([0, 0]), ...Array(4).fill([4, 1])] as Vec2[];
    const run = lloyd(X, [[0, 0], [0, 0], [4, 1]], 10);
    const first = run.phases[1];
    expect(first.labels).toEqual([0, 0, 0, 0, 0, 0, 2, 2, 2, 2]);
    expect(run.phases[2].empty).toEqual([1]);
    expect(run.centroids).toEqual([[0, 0], [0, 0], [4, 1]]);
    expect(run.inertia).toBe(0);
    // k-means++ with fewer distinct points than k falls back to uniform draws instead of dividing by zero
    const pp = initCentroids(X, 4, "kmeans++", 3);
    expect(pp.centroids.flat().every(Number.isFinite)).toBe(true);
    expect(new Set(initCentroids(X, 10, "random", 1).chosen).size).toBe(10);
  });

  it("is reproducible with a fixed seed and sensitive to it", () => {
    const { X } = generateClusters("blobs", 60, 8);
    for (const method of methods) {
      expect(initCentroids(X, 3, method, 11)).toEqual(initCentroids(X, 3, method, 11));
      expect(initCentroids(X, 3, method, 11)).not.toEqual(initCentroids(X, 3, method, 12));
    }
    expect(generateClusters("rings", 50, 2)).toEqual(generateClusters("rings", 50, 2));
  });

  it("rejects invalid parameters", () => {
    const X: Vec2[] = [[0, 0], [1, 1]];
    expect(() => lloyd(X, [])).toThrow(KMeansInputError);
    expect(() => initCentroids(X, 3, "random", 0)).toThrow(KMeansInputError);
    expect(() => initCentroids(X, 1.5, "random", 0)).toThrow(KMeansInputError);
    expect(() => initCentroids([], 1, "random", 0)).toThrow(KMeansInputError);
    expect(() => initCentroids([[0, NaN]], 1, "kmeans++", 0)).toThrow(KMeansInputError);
  });
});

describe("Voronoi cells and datasets", () => {
  const box = { lo: [-5, -5] as Vec2, hi: [5, 5] as Vec2 };
  const C: Vec2[] = [[-2, -1], [3, 0.5], [0, 3], [0.5, -3]];

  it("cells contain exactly the points nearest to their centroid and tile the box", () => {
    const cells = C.map((_, j) => voronoiCell(C, j, box));
    close(cells.reduce((s, p) => s + polygonArea(p), 0), 100, 1e-12);
    const inside = (poly: Vec2[], p: Vec2) => {
      let w = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const [xi, yi] = poly[i];
        const [xj, yj] = poly[j];
        if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) w = !w;
      }
      return w;
    };
    for (let a = -4.9; a < 5; a += 0.37)
      for (let b = -4.9; b < 5; b += 0.41) {
        const j = assign([[a, b]], C)[0];
        expect(inside(cells[j], [a, b])).toBe(true);
      }
  });

  it("gives an empty cell to the higher index of two coincident centroids", () => {
    expect(voronoiCell([[1, 1], [1, 1]], 1, box)).toEqual([]);
    close(polygonArea(voronoiCell([[1, 1], [1, 1]], 0, box)), 100, 1e-12);
  });

  it("generates the requested number of points for every dataset", () => {
    for (const d of DATASETS) {
      const { X, truth } = generateClusters(d.kind, 77, 1);
      expect(X).toHaveLength(77);
      expect(truth).toHaveLength(77);
      expect(X.flat().every(Number.isFinite)).toBe(true);
    }
  });

  it("silhouette is NaN for a single cluster and high for separated blobs", () => {
    const { X } = generateClusters("blobs", 60, 0);
    expect(silhouette(X, X.map(() => 0))).toBeNaN();
    const run = runKMeans(X, 3, { method: "kmeans++", seed: 0 });
    expect(silhouette(X, run.labels)).toBeGreaterThan(0.6);
  });
});
