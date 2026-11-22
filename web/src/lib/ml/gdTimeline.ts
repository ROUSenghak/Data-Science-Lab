/**
 * Gradient descent as a sequence of displayable states. Every number here comes from
 * linreg.ts (gradientDescent, mse, the Hessian); nothing is interpolated or invented.
 *
 *  - "epoch" granularity: one frame per epoch t, showing θ_t, L(θ_t), ∇L(θ_t) and the pending update.
 *  - "detailed" granularity: four frames per epoch, one per computation in the loop body:
 *      forward  ŷ = w_t x + b_t and residuals r = ŷ − y
 *      loss     L(θ_t) = mean(r²)
 *      gradient ∇L(θ_t)
 *      update   θ_{t+1} = θ_t − η∇L(θ_t)   (this frame shows θ_{t+1})
 */
import { eigSym2, hessian, leastSquares, mse, type Dataset, type GDStep } from "./linreg";

export type GDStage = "forward" | "loss" | "gradient" | "update";
export const GD_STAGES: readonly GDStage[] = ["forward", "loss", "gradient", "update"];

export interface GDFrame {
  epoch: number;
  /** "epoch" in epoch granularity; otherwise the computation this frame shows. */
  stage: GDStage | "epoch";
  step: GDStep;
  /** Parameters drawn in this frame: θ_{t+1} at an update, θ_t otherwise. */
  w: number;
  b: number;
}

export function buildGDTimeline(steps: GDStep[], detailed: boolean): GDFrame[] {
  if (!detailed) return steps.map((step) => ({ epoch: step.epoch, stage: "epoch", step, w: step.w, b: step.b }));
  return steps.flatMap((step) =>
    GD_STAGES.map((stage): GDFrame => ({
      epoch: step.epoch,
      stage,
      step,
      w: stage === "update" ? step.wNext : step.w,
      b: stage === "update" ? step.bNext : step.b,
    })),
  );
}

/** Index of the first frame of `epoch` (to keep the position when switching granularity). */
export function frameOfEpoch(frames: GDFrame[], epoch: number): number {
  const k = frames.findIndex((f) => f.epoch === epoch);
  return k < 0 ? Math.max(0, frames.length - 1) : k;
}

/**
 * What theory predicts for fixed-step GD on this quadratic loss. In the eigenbasis of H each error
 * component is multiplied by (1 − ηλ) per step:
 *  - monotone: η ≤ 1/λ_max, every factor in [0, 1): the path approaches θ* without overshooting;
 *  - zigzag:   1/λ_max < η < 2/λ_max, the steep factor is in (−1, 0): the steep component flips sign
 *              every step but shrinks, so the loss still falls;
 *  - diverge:  η ≥ 2/λ_max, |1 − ηλ_max| ≥ 1: the steep component never shrinks.
 */
export type Regime = "monotone" | "zigzag" | "diverge";

export function regimeOf(lr: number, lambdaMax: number): Regime {
  const r = lr * lambdaMax;
  // Relative tolerance so that η = 2/λ_max itself (which rounds to ηλ = 1.999…) counts as non-convergent.
  if (r >= 2 * (1 - 1e-12)) return "diverge";
  if (r > 1) return "zigzag";
  return "monotone";
}

/** Per-step multipliers 1 − ηλ along the flat (λ_min) and steep (λ_max) directions. */
export function stepFactors(lr: number, lambdaMin: number, lambdaMax: number) {
  return { flat: 1 - lr * lambdaMin, steep: 1 - lr * lambdaMax };
}

/**
 * Coordinates of the error θ − θ* along H's eigenvectors (flat = λ_min, steep = λ_max).
 * Exact dynamics: e_k(t+1) = (1 − ηλ_k) e_k(t).
 */
export function eigenErrors(w: number, b: number, data: Dataset) {
  const opt = leastSquares(data);
  const { vectors } = eigSym2(hessian(data));
  const e = [w - opt.w, b - opt.b];
  return { flat: vectors[0][0] * e[0] + vectors[0][1] * e[1], steep: vectors[1][0] * e[0] + vectors[1][1] * e[1] };
}

export interface GDSummary {
  regime: Regime;
  /** Epochs actually run (fewer than requested if the values overflowed). */
  epochsRun: number;
  overflowed: boolean;
  /** θ_T and L(θ_T) after the last update (NaN after an overflow). */
  final: { w: number; b: number; loss: number };
  /** L(θ_T) − L*. */
  gap: number;
  /** ‖θ_T − θ*‖. */
  distance: number;
  /** First t with L(θ_t) − L* ≤ tol · (L(θ_0) − L*); null if never within the run. */
  epochsToTol: number | null;
  /** Number of epochs whose loss is higher than the epoch before. */
  increases: number;
}

export function summarizeGD(steps: GDStep[], data: Dataset, lr: number, lambdaMax: number, tol = 1e-3): GDSummary {
  const opt = leastSquares(data);
  const lastStep = steps[steps.length - 1];
  const overflowed = !Number.isFinite(lastStep.wNext) || !Number.isFinite(lastStep.bNext);
  const final = { w: lastStep.wNext, b: lastStep.bNext, loss: overflowed ? NaN : mse(lastStep.wNext, lastStep.bNext, data) };
  const losses = [...steps.map((s) => s.loss), final.loss];
  const gap0 = losses[0] - opt.loss;
  let epochsToTol: number | null = null;
  for (let t = 0; t < losses.length; t++) {
    if (Number.isFinite(losses[t]) && losses[t] - opt.loss <= tol * gap0) {
      epochsToTol = t;
      break;
    }
  }
  let increases = 0;
  for (let t = 1; t < losses.length; t++) if (!(losses[t] <= losses[t - 1])) increases++;
  return {
    regime: regimeOf(lr, lambdaMax),
    epochsRun: steps.length,
    overflowed,
    final,
    gap: final.loss - opt.loss,
    distance: Math.hypot(final.w - opt.w, final.b - opt.b),
    epochsToTol,
    increases,
  };
}
