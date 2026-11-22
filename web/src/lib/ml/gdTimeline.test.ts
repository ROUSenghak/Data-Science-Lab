import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/linreg-numpy.json";
import { buildGDTimeline, eigenErrors, frameOfEpoch, GD_STAGES, regimeOf, stepFactors, summarizeGD } from "./gdTimeline";
import { centerX, generateDataset, gradient, gradientDescent, leastSquares, mse, predict, stability } from "./linreg";

const data = { x: fixture.x, y: fixture.y };
const stab = stability(data);
const run = (lr: number, epochs = 60, w0 = 0, b0 = -1) => gradientDescent(data, { lr, epochs, w0, b0 });

describe("gradient-descent timeline", () => {
  const steps = run(0.08);

  it("epoch granularity: one frame per computed epoch, showing θ_t", () => {
    const frames = buildGDTimeline(steps, false);
    expect(frames).toHaveLength(steps.length);
    frames.forEach((f, t) => expect([f.epoch, f.w, f.b]).toEqual([t, steps[t].w, steps[t].b]));
  });

  it("detailed granularity: forward, loss, gradient, update for every epoch, in order", () => {
    const frames = buildGDTimeline(steps, true);
    expect(frames).toHaveLength(4 * steps.length);
    frames.forEach((f, k) => {
      expect(f.stage).toBe(GD_STAGES[k % 4]);
      expect(f.epoch).toBe(Math.floor(k / 4));
    });
  });

  it("every frame refers to computed values: the update shows θ_{t+1}, which the next forward pass uses", () => {
    const frames = buildGDTimeline(steps, true);
    for (let k = 0; k < frames.length; k++) {
      const f = frames[k];
      if (f.stage === "update") {
        expect([f.w, f.b]).toEqual([f.step.wNext, f.step.bNext]);
        const next = frames[k + 1];
        if (next) expect([next.w, next.b]).toEqual([f.w, f.b]);
      } else expect([f.w, f.b]).toEqual([f.step.w, f.step.b]);
    }
  });

  it("the numbers each stage displays are consistent with one another", () => {
    for (const s of steps.slice(0, 10)) {
      const r = predict(s.w, s.b, data.x).map((v, i) => v - data.y[i]);
      expect(r.reduce((a, v) => a + v * v, 0) / r.length).toBeCloseTo(s.loss, 12);
      const g = gradient(s.w, s.b, data);
      expect(g.dw).toBe(s.dw);
      expect(g.db).toBe(s.db);
      expect(s.wNext).toBeCloseTo(s.w - 0.08 * s.dw, 14);
    }
  });

  it("switching granularity keeps the epoch", () => {
    const detailed = buildGDTimeline(steps, true);
    expect(detailed[frameOfEpoch(detailed, 7)]).toMatchObject({ epoch: 7, stage: "forward" });
    expect(frameOfEpoch(buildGDTimeline(steps, false), 7)).toBe(7);
  });
});

describe("convergence regimes", () => {
  const lrMono = 0.8 / stab.lambdaMax;
  const lrZig = 1.6 / stab.lambdaMax;
  const lrDiv = 2.1 / stab.lambdaMax;

  it("classifies by ηλ_max", () => {
    expect(regimeOf(lrMono, stab.lambdaMax)).toBe("monotone");
    expect(regimeOf(lrZig, stab.lambdaMax)).toBe("zigzag");
    expect(regimeOf(lrDiv, stab.lambdaMax)).toBe("diverge");
    expect(regimeOf(stab.criticalLR, stab.lambdaMax)).toBe("diverge");
  });

  it("each eigen-component of the error is multiplied by exactly 1 − ηλ per step", () => {
    for (const lr of [lrMono, lrZig, lrDiv]) {
      const f = stepFactors(lr, stab.lambdaMin, stab.lambdaMax);
      for (const s of run(lr, 15)) {
        const e0 = eigenErrors(s.w, s.b, data);
        const e1 = eigenErrors(s.wNext, s.bNext, data);
        expect(e1.flat).toBeCloseTo(f.flat * e0.flat, 8);
        expect(e1.steep).toBeCloseTo(f.steep * e0.steep, 8);
      }
    }
  });

  it("monotone: the steep component never changes sign", () => {
    const e = run(lrMono, 40).map((s) => eigenErrors(s.w, s.b, data).steep);
    // Until the component reaches round-off level, where its sign is noise.
    for (let t = 1; t < e.length && Math.abs(e[t - 1]) > 1e-9; t++) expect(e[t] * e[t - 1]).toBeGreaterThanOrEqual(0);
  });

  it("zigzag: the steep component flips sign every step, yet the loss never increases", () => {
    const steps = run(lrZig, 40);
    const e = steps.map((s) => eigenErrors(s.w, s.b, data).steep);
    for (let t = 1; t < 20; t++) expect(e[t] * e[t - 1]).toBeLessThan(0);
    const sum = summarizeGD(steps, data, lrZig, stab.lambdaMax);
    expect(sum.regime).toBe("zigzag");
    expect(sum.increases).toBe(0);
  });

  it("diverge: the loss grows and the summary reports it", () => {
    const sum = summarizeGD(run(lrDiv, 200), data, lrDiv, stab.lambdaMax);
    expect(sum.regime).toBe("diverge");
    expect(sum.increases).toBeGreaterThan(0);
    expect(sum.gap).toBeGreaterThan(1);
    expect(sum.epochsToTol).toBeNull();
  });

  it("overflow is reported, not crashed", () => {
    const lr = 0.5;
    const wide = generateDataset({ n: 30, noise: 1, seed: 7, xMax: 40 });
    const steps = gradientDescent(wide, { lr, epochs: 400, w0: 0, b0: 0 });
    const sum = summarizeGD(steps, wide, lr, stability(wide).lambdaMax);
    expect(sum.regime).toBe("diverge");
    expect(sum.overflowed).toBe(true);
    expect(sum.epochsRun).toBeLessThan(400);
    expect(Number.isNaN(sum.final.loss)).toBe(true);
  });

  it("summary of a converged run: final loss belongs to the final parameters", () => {
    const steps = run(0.08, 400);
    const sum = summarizeGD(steps, data, 0.08, stab.lambdaMax);
    const last = steps.at(-1)!;
    expect(sum.final.loss).toBe(mse(last.wNext, last.bNext, data));
    expect(sum.distance).toBeLessThan(1e-3);
    expect(sum.epochsToTol).not.toBeNull();
    const ls = leastSquares(data);
    const losses = steps.map((s) => s.loss);
    expect(losses[sum.epochsToTol!] - ls.loss).toBeLessThanOrEqual(1e-3 * (losses[0] - ls.loss));
    expect(losses[sum.epochsToTol! - 1] - ls.loss).toBeGreaterThan(1e-3 * (losses[0] - ls.loss));
  });

  it("centering decouples the parameters: the eigenvectors become the axes", () => {
    const c = centerX(data);
    const e = eigenErrors(1, 0, c);
    const ls = leastSquares(c);
    // With x̄ = 0, the error along w is purely one eigen-direction.
    expect(Math.abs(Math.abs(e.flat) + Math.abs(e.steep) - Math.abs(1 - ls.w) - Math.abs(ls.b))).toBeLessThan(1e-9);
  });
});
