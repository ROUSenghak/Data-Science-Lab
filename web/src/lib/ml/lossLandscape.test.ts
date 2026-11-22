import { describe, expect, it } from "vitest";
import { generateDataset, gradientDescent, leastSquares, mse, stability } from "./linreg";
import {
  autoDomain,
  clipAbove,
  clipPolyline,
  contourLevels,
  densify,
  inDomain,
  interpolationErrorBound,
  levelShade,
  lift,
  maxLossOnDomain,
  plotTop,
  surfaceGrid,
  type ParamDomain,
  type Pt,
} from "./lossLandscape";

const data = generateDataset({ n: 30, noise: 1, seed: 7 });
const domain: ParamDomain = { w: [-3.5, 6.5], b: [-4.5, 6.5] };

describe("surface grid", () => {
  it("stores the exact loss at every node", () => {
    const g = surfaceGrid(data, domain, 17);
    expect(g.w[0]).toBe(domain.w[0]);
    expect(g.w[16]).toBeCloseTo(domain.w[1], 12);
    for (let j = 0; j < g.res; j++) for (let i = 0; i < g.res; i++) expect(g.L[j * g.res + i]).toBe(mse(g.w[i], g.b[j], data));
  });

  it("the window's maximum is at a corner (convexity), so no node exceeds it", () => {
    const g = surfaceGrid(data, domain, 41);
    expect(Math.max(...g.L)).toBeLessThanOrEqual(g.max);
    // and it is attained: the corners are grid nodes
    expect(Math.max(...g.L)).toBe(g.max);
    // random interior points never exceed it either
    for (let k = 0; k < 200; k++) {
      const w = domain.w[0] + ((domain.w[1] - domain.w[0]) * ((k * 37) % 101)) / 100;
      const b = domain.b[0] + ((domain.b[1] - domain.b[0]) * ((k * 53) % 97)) / 96;
      expect(mse(w, b, data)).toBeLessThanOrEqual(maxLossOnDomain(data, domain) + 1e-9);
    }
  });

  it("the stated interpolation bound holds for the drawn mesh", () => {
    const res = 25;
    const g = surfaceGrid(data, domain, res);
    const bound = interpolationErrorBound(domain, res, stability(data).lambdaMax);
    // Midpoint of each cell diagonal: the mesh shows the mean of the two corner values.
    for (let j = 0; j + 1 < res; j += 3)
      for (let i = 0; i + 1 < res; i += 3) {
        const drawn = (g.L[j * res + i] + g.L[(j + 1) * res + i + 1]) / 2;
        const exact = mse((g.w[i] + g.w[i + 1]) / 2, (g.b[j] + g.b[j + 1]) / 2, data);
        expect(drawn - exact).toBeGreaterThanOrEqual(-1e-12); // chords of a convex function lie above it
        expect(drawn - exact).toBeLessThanOrEqual(bound + 1e-12);
      }
  });
});

describe("contour levels shared by the 2D and 3D views", () => {
  it("are L* + E·0.6^k with E the largest excess in the window", () => {
    const opt = leastSquares(data);
    const E = maxLossOnDomain(data, domain) - opt.loss;
    const lv = contourLevels(data, domain);
    expect(lv).toHaveLength(14);
    lv.forEach((l, k) => expect(l).toBeCloseTo(opt.loss + E * 0.6 ** k, 12));
  });

  it("shade counts the levels above a value: 0 at the top level, 1 at the lowest", () => {
    const opt = leastSquares(data);
    const E = maxLossOnDomain(data, domain) - opt.loss;
    const lv = contourLevels(data, domain);
    expect(levelShade(lv[0], opt.loss, E)).toBeCloseTo(0, 12);
    expect(levelShade(lv[13], opt.loss, E)).toBeCloseTo(1, 12);
    expect(levelShade(lv[6], opt.loss, E)).toBeCloseTo(6 / 13, 12);
    expect(levelShade(opt.loss, opt.loss, E)).toBe(1);
    expect(levelShade(NaN, opt.loss, E)).toBe(0);
    // monotone: lower loss, deeper shade
    expect(levelShade(lv[3], opt.loss, E)).toBeLessThan(levelShade(lv[4], opt.loss, E));
  });
});

describe("auto domain", () => {
  it("is centred on θ* and contains the anchor and the first iterate", () => {
    const opt = leastSquares(data);
    const d = autoDomain(data, [-2, 7], [0, -1]);
    expect((d.w[0] + d.w[1]) / 2).toBeCloseTo(opt.w, 12);
    expect((d.b[0] + d.b[1]) / 2).toBeCloseTo(opt.b, 12);
    expect(inDomain([-2, 7], d)).toBe(true);
    expect(inDomain([0, -1], d)).toBe(true);
  });
  it("ignores non-finite points", () => {
    const d = autoDomain(data, [NaN, 1], [Infinity, 0]);
    expect(d.w.every(Number.isFinite) && d.b.every(Number.isFinite)).toBe(true);
  });
});

describe("clipping a path to the window", () => {
  const box: ParamDomain = { w: [0, 1], b: [0, 1] };
  it("keeps a path that is inside unchanged", () => {
    const p: Pt[] = [
      [0.1, 0.1],
      [0.5, 0.2],
      [0.9, 0.9],
    ];
    expect(clipPolyline(p, box)).toEqual([p]);
  });
  it("splits where the path leaves and re-enters, with exact crossing points", () => {
    const parts = clipPolyline(
      [
        [0.5, 0.5],
        [1.5, 0.5],
        [1.5, 0.25],
        [0.5, 0.25],
      ],
      box,
    );
    expect(parts).toEqual([
      [
        [0.5, 0.5],
        [1, 0.5],
      ],
      [
        [1, 0.25],
        [0.5, 0.25],
      ],
    ]);
  });
  it("drops segments that miss the window and breaks at non-finite points", () => {
    expect(clipPolyline([[2, 2], [3, 3]], box)).toEqual([]);
    const parts = clipPolyline([[0.1, 0.1], [0.2, 0.2], [NaN, 0], [0.3, 0.3], [0.4, 0.4]], box);
    expect(parts).toHaveLength(2);
  });
  it("clips a diverging gradient-descent run to the window", () => {
    const steps = gradientDescent(data, { lr: 0.2, epochs: 30, w0: 0, b0: -1 });
    const path: Pt[] = steps.map((s) => [s.w, s.b]);
    const parts = clipPolyline(path, domain);
    for (const part of parts) for (const p of part) expect(inDomain(p, { w: [domain.w[0] - 1e-9, domain.w[1] + 1e-9], b: [domain.b[0] - 1e-9, domain.b[1] + 1e-9] })).toBe(true);
    expect(parts.length).toBeGreaterThan(0);
  });
});

describe("drawing a path on the surface", () => {
  it("densify keeps every iterate and inserts points on the straight segments", () => {
    const p: Pt[] = [
      [0, 0],
      [4, 2],
      [4, 6],
    ];
    const d = densify(p, 4);
    expect(d).toHaveLength(9);
    expect(d[0]).toEqual([0, 0]);
    expect(d[4]).toEqual([4, 2]);
    expect(d[8]).toEqual([4, 6]);
    expect(d[2]).toEqual([2, 1]);
  });
  it("lift gives each point its exact loss", () => {
    const pts: Pt[] = [
      [1, 2],
      [2.5, -1],
    ];
    expect(lift(pts, data)).toEqual([
      [1, 2, mse(1, 2, data)],
      [2.5, -1, mse(2.5, -1, data)],
    ]);
  });
});

describe("plotted loss range", () => {
  it("ends 30% above the highest starting excess loss, never above the window maximum", () => {
    const opt = leastSquares(data);
    const start: Pt = [0, -1];
    const top = plotTop(data, domain, [start]);
    expect(top).toBeCloseTo(opt.loss + 1.3 * (mse(0, -1, data) - opt.loss), 12);
    expect(top).toBeLessThan(maxLossOnDomain(data, domain));
    // the starting point is inside the box
    expect(mse(0, -1, data)).toBeLessThan(top);
    // no start, or a start outside the window: the whole window
    expect(plotTop(data, domain, [])).toBe(maxLossOnDomain(data, domain));
    expect(plotTop(data, domain, [[100, 100]])).toBe(maxLossOnDomain(data, domain));
    // a start at the minimum still leaves a visible bowl
    expect(plotTop(data, domain, [[opt.w, opt.b]])).toBeGreaterThan(opt.loss);
  });

  it("clipAbove keeps the parts of a lifted path at or below the top, with crossing points on it", () => {
    const parts = clipAbove(
      [
        [0, 0, 1],
        [1, 0, 3],
        [2, 0, 1],
        [3, 0, 0.5],
      ],
      2,
    );
    expect(parts).toEqual([
      [
        [0, 0, 1],
        [0.5, 0, 2],
      ],
      [
        [1.5, 0, 2],
        [2, 0, 1],
        [3, 0, 0.5],
      ],
    ]);
    expect(clipAbove([[0, 0, 5], [1, 1, 6]], 2)).toEqual([]);
  });

  it("every point of a clipped, lifted gradient-descent path is on the surface and inside the box", () => {
    const steps = gradientDescent(data, { lr: 0.17, epochs: 40, w0: 0, b0: -1 });
    const top = plotTop(data, domain, [[0, -1]]);
    const lifted = clipPolyline(densify(steps.map((s): Pt => [s.w, s.b]), 10), domain).flatMap((part) => clipAbove(lift(part, data), top));
    for (const part of lifted)
      for (const [w, b, L] of part) {
        expect(L).toBeLessThanOrEqual(top + 1e-9);
        if (L < top) expect(L).toBe(mse(w, b, data)); // interior points carry their exact loss
      }
  });
});
