import { describe, expect, it } from "vitest";
import type { KMeansState, Snapshot } from "../types";
import { changedPoints, foldKMeans, iterationOf, objectiveHistory, shownLabels } from "./trace";

const snap = (index: number, step: string, state: KMeansState): Snapshot<KMeansState> => ({ index, step, lines: [index + 1, index + 1], state, locals: {}, removed: [] });

describe("K-Means trace helpers", () => {
  const frames = foldKMeans([
    snap(0, "data", { X: [[0, 0], [1, 0], [5, 5]] }),
    snap(1, "init", { centroids: [[0, 0], [1, 0]], centroids_init: [[0, 0], [1, 0]] }),
    snap(2, "distances", { iteration: 0, sq_dist: [[0, 1], [1, 0], [50, 41]] }),
    snap(3, "assign", { prev_labels: [-1, -1, -1], labels: [0, 1, 1], n_changed: 3, J: 41 }),
    snap(4, "update", { centroids: [[0, 0], [3, 2.5]], counts: [1, 2] }),
    snap(5, "objective", { J: 12.5 }),
    snap(6, "distances", { iteration: 1, sq_dist: [[0, 15.25], [1, 10.25], [50, 10.25]] }),
    snap(7, "assign", { prev_labels: [0, 1, 1], labels: [0, 0, 1], n_changed: 1, J: 10.25 }),
    snap(8, "evaluate", { inertia: 1 }),
  ]);

  it("numbers iterations from 1 inside the loop only", () => {
    expect(frames.map(iterationOf)).toEqual([null, null, 1, 1, 1, 1, 2, 2, null]);
  });

  it("separates J after assignment from J after a completed iteration", () => {
    expect(objectiveHistory(frames, 8)).toEqual([
      { iteration: 1, phase: "assign", J: 41 },
      { iteration: 1, phase: "update", J: 12.5 },
      { iteration: 2, phase: "assign", J: 10.25 },
    ]);
    // the update snapshot itself carries no new J (it still holds the assignment value)
    expect(objectiveHistory(frames, 4)).toHaveLength(1);
  });

  it("reports changed points only after the first assignment", () => {
    expect(changedPoints(frames[3].state)).toEqual([]);
    expect(changedPoints(frames[7].state)).toEqual([1]);
    expect(shownLabels(frames[2].state)).toBeNull();
    expect(shownLabels(frames[3].state)).toEqual([0, 1, 1]);
  });
});
