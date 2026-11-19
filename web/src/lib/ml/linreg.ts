/**
 * Simple linear regression  ŷ = w·x + b  with mean squared error.
 * Pure functions; verified against NumPy in linreg.test.ts.
 */
import { createRng } from "./random";

export interface Dataset {
  x: number[];
  y: number[];
}

export interface DataConfig {
  n: number;
  noise: number;
  seed: number;
  xMin?: number;
  xMax?: number;
  trueW?: number;
  trueB?: number;
  /** Failure case: ground truth is curved, y = 0.4x² + 0.4x + 1 + ε. */
  nonlinear?: boolean;
  /** Failure case: add one gross outlier at the right edge. */
  outlier?: boolean;
}

export function generateDataset(cfg: DataConfig): Dataset {
  const { n, noise, seed, xMin = 0, xMax = 4, trueW = 2, trueB = 1 } = cfg;
  const rng = createRng(seed);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const xi = rng.uniform(xMin, xMax);
    const f = cfg.nonlinear ? 0.4 * xi * xi + 0.4 * xi + 1 : trueW * xi + trueB;
    x.push(xi);
    y.push(f + rng.normal(0, noise));
  }
  if (cfg.outlier) {
    x.push(xMax * 0.95);
    y.push(-2);
  }
  return { x, y };
}

export function centerX(d: Dataset): Dataset {
  const m = mean(d.x);
  return { x: d.x.map((v) => v - m), y: d.y };
}

export const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;

export const predict = (w: number, b: number, x: number[]) => x.map((xi) => w * xi + b);

export function mse(w: number, b: number, d: Dataset): number {
  let s = 0;
  for (let i = 0; i < d.x.length; i++) {
    const r = w * d.x[i] + b - d.y[i];
    s += r * r;
  }
  return s / d.x.length;
}

/** ∂L/∂w = (2/n) Σ rᵢ xᵢ,  ∂L/∂b = (2/n) Σ rᵢ  with residual rᵢ = ŷᵢ − yᵢ. */
export function gradient(w: number, b: number, d: Dataset): { dw: number; db: number } {
  const n = d.x.length;
  let sw = 0;
  let sb = 0;
  for (let i = 0; i < n; i++) {
    const r = w * d.x[i] + b - d.y[i];
    sw += r * d.x[i];
    sb += r;
  }
  return { dw: (2 * sw) / n, db: (2 * sb) / n };
}

/** One full-batch gradient-descent step, with every intermediate quantity. */
export interface GDStep {
  epoch: number;
  w: number;
  b: number;
  loss: number;
  dw: number;
  db: number;
  wNext: number;
  bNext: number;
}

export function gradientDescent(
  d: Dataset,
  opts: { lr: number; epochs: number; w0: number; b0: number },
): GDStep[] {
  const steps: GDStep[] = [];
  let w = opts.w0;
  let b = opts.b0;
  for (let epoch = 0; epoch < opts.epochs; epoch++) {
    const loss = mse(w, b, d);
    const { dw, db } = gradient(w, b, d);
    const wNext = w - opts.lr * dw;
    const bNext = b - opts.lr * db;
    steps.push({ epoch, w, b, loss, dw, db, wNext, bNext });
    if (!Number.isFinite(wNext) || !Number.isFinite(bNext)) break;
    w = wNext;
    b = bNext;
  }
  return steps;
}

/** Closed-form least squares: w* = Cov(x, y) / Var(x),  b* = ȳ − w* x̄. */
export function leastSquares(d: Dataset): { w: number; b: number; loss: number } {
  const mx = mean(d.x);
  const my = mean(d.y);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < d.x.length; i++) {
    sxy += (d.x[i] - mx) * (d.y[i] - my);
    sxx += (d.x[i] - mx) ** 2;
  }
  const w = sxy / sxx;
  const b = my - w * mx;
  return { w, b, loss: mse(w, b, d) };
}

/** Hessian of the MSE in (w, b) — constant because the loss is quadratic. */
export function hessian(d: Dataset): [[number, number], [number, number]] {
  const mx = mean(d.x);
  const mxx = mean(d.x.map((v) => v * v));
  return [
    [2 * mxx, 2 * mx],
    [2 * mx, 2],
  ];
}

/** Eigen-decomposition of a symmetric 2×2 matrix (ascending eigenvalues). */
export function eigSym2(m: [[number, number], [number, number]]) {
  const [[a, c], [, d]] = m;
  const tr = a + d;
  const disc = Math.sqrt(((a - d) / 2) ** 2 + c * c);
  const l1 = tr / 2 - disc;
  const l2 = tr / 2 + disc;
  // eigenvector for eigenvalue l: (c, l − a) or (l − d, c) if c = 0
  const vec = (l: number): [number, number] => {
    let v: [number, number] = Math.abs(c) > 1e-12 ? [c, l - a] : Math.abs(a - l) < Math.abs(d - l) ? [1, 0] : [0, 1];
    const n = Math.hypot(v[0], v[1]);
    v = [v[0] / n, v[1] / n];
    return v;
  };
  return { values: [l1, l2] as [number, number], vectors: [vec(l1), vec(l2)] as [[number, number], [number, number]] };
}

/**
 * Gradient descent on a quadratic converges iff 0 < η < 2 / λ_max(H).
 * Returns that bound and the condition number κ = λ_max / λ_min.
 */
export function stability(d: Dataset) {
  const { values } = eigSym2(hessian(d));
  return { lambdaMin: values[0], lambdaMax: values[1], criticalLR: 2 / values[1], condition: values[1] / values[0] };
}

/**
 * Level set {θ : L(θ) = level} of the quadratic loss, as a closed polyline in (w, b).
 * L(θ) = L* + ½ (θ − θ*)ᵀ H (θ − θ*), so level sets are ellipses aligned with H's eigenvectors.
 */
export function lossEllipse(d: Dataset, level: number, segments = 96): [number, number][] {
  const opt = leastSquares(d);
  const excess = level - opt.loss;
  if (excess <= 0) return [];
  const { values, vectors } = eigSym2(hessian(d));
  const r1 = Math.sqrt((2 * excess) / values[0]);
  const r2 = Math.sqrt((2 * excess) / values[1]);
  const pts: [number, number][] = [];
  for (let k = 0; k <= segments; k++) {
    const t = (2 * Math.PI * k) / segments;
    const a = r1 * Math.cos(t);
    const c = r2 * Math.sin(t);
    pts.push([opt.w + a * vectors[0][0] + c * vectors[1][0], opt.b + a * vectors[0][1] + c * vectors[1][1]]);
  }
  return pts;
}
