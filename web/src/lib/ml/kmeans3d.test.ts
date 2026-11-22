import { describe, expect, it } from "vitest";
import { agreement, assign, generateClusters3D, HIDDEN_CENTERS, lloyd, objective, runKMeans, sqDist, type Vec2, type Vec3 } from "./kmeans";

describe("K-Means with three features", () => {
  const { X, truth } = generateClusters3D(120, 4);

  it("the distance uses every feature", () => {
    expect(sqDist([0, 0, 0], [1, 2, 2])).toBe(9);
    expect(sqDist([1, 2], [1, 2])).toBe(0);
  });

  it("each half-step computes the exact 3D quantities and J never increases", () => {
    const run = lloyd(X, [X[0], X[1], X[2]]);
    let prevJ = Infinity;
    for (const ph of run.phases) {
      if (ph.kind === "init") continue;
      expect(ph.J!).toBeLessThanOrEqual(prevJ + 1e-9);
      prevJ = ph.J!;
      expect(ph.J!).toBeCloseTo(objective(X, ph.centroids, ph.labels!), 9);
      if (ph.kind === "assign") expect(ph.labels).toEqual(assign(X, ph.centroids));
      if (ph.kind === "update")
        ph.centroids.forEach((c, j) => {
          const members = X.filter((_, i) => ph.labels![i] === j);
          if (!members.length) return;
          for (let d = 0; d < 3; d++) expect(c[d]).toBeCloseTo(members.reduce((s, p) => s + p[d], 0) / members.length, 12);
        });
    }
    expect(run.converged).toBe(true);
    expect(run.centroids[0]).toHaveLength(3);
  });

  it("two generating groups share their (x₁, x₂) centre", () => {
    expect(HIDDEN_CENTERS[0].slice(0, 2)).toEqual(HIDDEN_CENTERS[2].slice(0, 2));
  });

  // Empirical, for this seed: what the lesson figure claims.
  it("all three features recover the groups; the first two cannot (seed 4, best of 10 starts)", () => {
    const best = <V extends number[]>(P: V[]) =>
      Array.from({ length: 10 }, (_, s) => runKMeans(P, 3, { method: "kmeans++", seed: s })).reduce((a, b) => (b.inertia < a.inertia ? b : a));
    const in3d = best<Vec3>(X);
    const in2d = best<Vec2>(X.map((p): Vec2 => [p[0], p[1]]));
    expect(agreement(in3d.labels, truth, 3)).toBeGreaterThan(0.97);
    expect(agreement(in2d.labels, truth, 3)).toBeLessThan(0.8);
  });

  it("agreement is invariant to renumbering clusters", () => {
    expect(agreement([2, 2, 0, 1], [0, 0, 1, 2], 3)).toBe(1);
    expect(agreement([0, 1, 0, 1], [0, 0, 1, 1], 2)).toBe(0.5);
  });
});
