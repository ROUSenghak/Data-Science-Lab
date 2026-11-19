import { describe, expect, it } from "vitest";
import { findStep, foldSnapshots, lossHistory, paramPath, trainingHistory } from "./fold";
import type { Snapshot } from "./types";

const snap = (index: number, step: string, state: Snapshot["state"], locals: Snapshot["locals"] = {}, removed: string[] = []): Snapshot => ({
  index,
  step,
  lines: [index + 1, index + 1],
  state,
  locals,
  removed,
});

describe("foldSnapshots", () => {
  const snaps = [
    snap(0, "data", { x: [1, 2], y: [3, 5] }, { x: { type: "ndarray", value: "[1, 2]" } }),
    snap(1, "init", { w: 0, b: 0 }),
    snap(2, "forward", { y_hat: [0, 0], epoch: 0 }),
    snap(3, "loss", { loss: 17 }),
    snap(4, "update", { w: 1, b: 0.5 }, { tmp: { type: "int", value: "1" } }),
    snap(5, "forward", { y_hat: [1.5, 2.5], epoch: 1 }, {}, ["tmp"]),
    snap(6, "loss", { loss: 3.25 }),
  ];
  const frames = foldSnapshots(snaps);

  it("accumulates diffs into full states", () => {
    expect(frames[3].state).toEqual({ x: [1, 2], y: [3, 5], w: 0, b: 0, y_hat: [0, 0], epoch: 0, loss: 17 });
    expect(frames[4].prev.w).toBe(0);
    expect(frames[4].state.w).toBe(1);
  });

  it("tracks the parameters that produced the current predictions", () => {
    expect(frames[4].forwardParams).toEqual({ w: 0, b: 0 }); // after update, y_hat is still from (0, 0)
    expect(frames[5].forwardParams).toEqual({ w: 1, b: 0.5 });
  });

  it("applies removed locals", () => {
    expect(frames[4].locals.tmp).toBeDefined();
    expect(frames[5].locals.tmp).toBeUndefined();
    expect(frames[5].locals.x).toBeDefined();
  });

  it("extracts loss history and parameter path up to a frame", () => {
    expect(lossHistory(frames, 6)).toEqual([{ epoch: 0, loss: 17 }, { epoch: 1, loss: 3.25 }]);
    expect(lossHistory(frames, 4)).toHaveLength(1);
    expect(paramPath(frames, 6)).toEqual([[0, 0], [1, 0.5]]);
  });

  it("aligns parameters, loss and gradient of each epoch, and records the final evaluation as epoch T", () => {
    const gd = foldSnapshots([
      snap(0, "data", { x: [1, 2], y: [3, 5] }),
      snap(1, "init", { w: 0, b: 0 }),
      snap(2, "forward", { y_hat: [0, 0], epoch: 0 }),
      snap(3, "loss", { loss: 17 }),
      snap(4, "gradient", { dw: -13, db: -8 }),
      snap(5, "update", { w: 1.3, b: 0.8 }),
      snap(6, "evaluate", { y_hat: [2.1, 3.4], loss: 1.625 }),
    ]);
    expect(trainingHistory(gd, 6)).toEqual([
      { epoch: 0, w: 0, b: 0, loss: 17, dw: -13, db: -8 },
      { epoch: 1, w: 1.3, b: 0.8, loss: 1.625, dw: null, db: null },
    ]);
    // before the gradient step, epoch 0 has no gradient yet
    expect(trainingHistory(gd, 3)[0]).toMatchObject({ loss: 17, dw: null });
    expect(gd[6].forwardParams).toEqual({ w: 1.3, b: 0.8 });
    expect(lossHistory(gd, 6).map((r) => r.loss)).toEqual([17, 1.625]);
  });

  it("finds a step within the current epoch", () => {
    expect(findStep(frames, 6, "forward")).toBe(5); // epoch 1
    expect(findStep(frames, 3, "forward")).toBe(2); // epoch 0
    expect(findStep(frames, 0, "loss")).toBe(3); // no epoch yet: first occurrence
    expect(findStep(frames, 0, "missing")).toBe(-1);
  });

  it("refreshes forwardParams at a forward step even when y_hat is unchanged", () => {
    const f = foldSnapshots([snap(0, "init", { w: 0, b: 0, y_hat: [1] }), snap(1, "update", { w: 2 }), snap(2, "forward", { epoch: 1 })]);
    expect(f[2].forwardParams).toEqual({ w: 2, b: 0 });
  });
});
