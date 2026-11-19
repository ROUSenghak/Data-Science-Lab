import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/linreg-numpy.json";
import {
  generateDataset,
  gradient,
  gradientDescent,
  hessian,
  leastSquares,
  lossEllipse,
  mse,
  stability,
} from "./linreg";
import { createRng } from "./random";

const data = { x: fixture.x, y: fixture.y };

describe("linear regression math", () => {
  it("matches the gradient-descent trajectory produced by the NumPy experiment", () => {
    const { learning_rate, epochs } = fixture.params;
    const steps = gradientDescent(data, { lr: learning_rate, epochs, w0: 0, b0: 0 });
    expect(steps).toHaveLength(fixture.steps.length);
    steps.forEach((s, i) => {
      const ref = fixture.steps[i];
      expect(s.loss).toBeCloseTo(ref.loss, 10);
      expect(s.dw).toBeCloseTo(ref.dw, 10);
      expect(s.db).toBeCloseTo(ref.db, 10);
      expect(s.wNext).toBeCloseTo(ref.w, 10);
      expect(s.bNext).toBeCloseTo(ref.b, 10);
    });
  });

  it("matches scikit-learn's LinearRegression in closed form", () => {
    const ls = leastSquares(data);
    expect(ls.w).toBeCloseTo(fixture.sklearn.w, 10);
    expect(ls.b).toBeCloseTo(fixture.sklearn.b, 10);
    expect(ls.loss).toBeCloseTo(fixture.sklearn.mse, 10);
  });

  it("analytic gradient equals a central finite difference", () => {
    const h = 1e-6;
    for (const [w, b] of [[0, 0], [1.3, -0.4], [3, 2]]) {
      const g = gradient(w, b, data);
      expect(g.dw).toBeCloseTo((mse(w + h, b, data) - mse(w - h, b, data)) / (2 * h), 5);
      expect(g.db).toBeCloseTo((mse(w, b + h, data) - mse(w, b - h, data)) / (2 * h), 5);
    }
  });

  it("gradient vanishes at the least-squares solution", () => {
    const ls = leastSquares(data);
    const g = gradient(ls.w, ls.b, data);
    expect(Math.abs(g.dw)).toBeLessThan(1e-10);
    expect(Math.abs(g.db)).toBeLessThan(1e-10);
  });

  it("recovers the true line exactly without noise", () => {
    const d = generateDataset({ n: 20, noise: 0, seed: 1, trueW: -1.5, trueB: 4 });
    const ls = leastSquares(d);
    expect(ls.w).toBeCloseTo(-1.5, 12);
    expect(ls.b).toBeCloseTo(4, 12);
  });

  it("stability bound 2/λ_max separates convergence from divergence", () => {
    const { criticalLR } = stability(data);
    const ls = leastSquares(data);
    const below = gradientDescent(data, { lr: 0.95 * criticalLR, epochs: 3000, w0: 0, b0: 0 }).at(-1)!;
    const above = gradientDescent(data, { lr: 1.05 * criticalLR, epochs: 3000, w0: 0, b0: 0 }).at(-1)!;
    expect(below.wNext).toBeCloseTo(ls.w, 6);
    expect(Math.abs(above.loss)).toBeGreaterThan(1e6);
  });

  it("loss ellipse points lie on the requested level set", () => {
    const level = leastSquares(data).loss + 3;
    for (const [w, b] of lossEllipse(data, level, 24)) expect(mse(w, b, data)).toBeCloseTo(level, 9);
  });

  it("hessian equals second finite differences", () => {
    const H = hessian(data);
    const h = 1e-3;
    const d2w = (mse(1 + h, 0, data) - 2 * mse(1, 0, data) + mse(1 - h, 0, data)) / h ** 2;
    expect(H[0][0]).toBeCloseTo(d2w, 5);
  });
});

describe("seeded random numbers", () => {
  it("is reproducible and roughly standard normal", () => {
    const a = createRng(42);
    const b = createRng(42);
    const xs = Array.from({ length: 20000 }, () => a.normal(0, 1));
    expect(xs.slice(0, 5)).toEqual(Array.from({ length: 5 }, () => b.normal(0, 1)));
    const m = xs.reduce((s, v) => s + v, 0) / xs.length;
    const v = xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length;
    expect(Math.abs(m)).toBeLessThan(0.03);
    expect(Math.abs(v - 1)).toBeLessThan(0.05);
  });
});
