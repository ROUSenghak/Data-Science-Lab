"use client";

import { Button } from "@/components/ui/controls";
import { fmt } from "@/lib/format";
import { leastSquares, stability } from "@/lib/ml/linreg";
import { trainingHistory, type EpochRecord, type Frame } from "./fold";
import type { RunResponse } from "./types";

/** Colours assigned to a run when it is recorded, so a run keeps its colour while it is listed. */
export const RUN_COLORS = ["var(--viz-model)", "var(--viz-run-1)", "var(--viz-run-2)", "var(--viz-run-3)"] as const;
export const MAX_RUNS = RUN_COLORS.length;

export interface RunSummary {
  serial: number;
  color: string;
  params: Record<string, number>;
  history: EpochRecord[];
  /** Least-squares minimum and stability limit of this run's own (traced) data. */
  lStar: number;
  critLR: number;
  initial: number | null;
  final: number | null;
  /** Fraction of the initial excess loss L₀ − L* removed by training. */
  removed: number | null;
  outcome: "diverged" | "converged" | "not converged";
}

/** Summarize a gradient-descent run from its traced values (no values are invented). */
export function summarizeRun(run: RunResponse, frames: Frame[], serial: number): RunSummary | null {
  const st = frames[frames.length - 1]?.state;
  if (!st?.x || !st?.y) return null;
  const data = { x: st.x, y: st.y };
  const ls = leastSquares(data);
  const history = trainingHistory(frames, frames.length - 1);
  const initial = history[0]?.loss ?? null;
  const final = history[history.length - 1]?.loss ?? null;
  const ok = (v: number | null): v is number => v != null && Number.isFinite(v);
  let removed: number | null = null;
  let outcome: RunSummary["outcome"] = "not converged";
  if (!ok(final) || (ok(initial) && final > initial)) outcome = "diverged";
  else if (ok(initial)) {
    const excess0 = initial - ls.loss;
    removed = excess0 > 0 ? 1 - (final - ls.loss) / excess0 : 1;
    if (removed >= 0.999) outcome = "converged";
  }
  return {
    serial,
    color: RUN_COLORS[serial % RUN_COLORS.length],
    params: run.params,
    history,
    lStar: ls.loss,
    critLR: stability(data).criticalLR,
    initial,
    final,
    removed,
    outcome,
  };
}

const OUTCOME = {
  diverged: { tone: "text-bad", icon: "✗" },
  converged: { tone: "text-ok", icon: "✓" },
  "not converged": { tone: "text-warn", icon: "…" },
} as const;

export function RunComparison({ runs, currentSerial, onClear }: { runs: RunSummary[]; currentSerial: number | null; onClear: () => void }) {
  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12px] tabular-nums">
          <thead className="text-[11px] text-fg-3">
            <tr>
              <th className="py-1 pr-2 font-medium">run</th>
              <th className="py-1 pr-2 font-medium">η</th>
              <th className="py-1 pr-2 font-medium" title="Learning rate relative to the stability limit 2/λmax of that run's data">η / η<sub>crit</sub></th>
              <th className="py-1 pr-2 font-medium">epochs</th>
              <th className="py-1 pr-2 font-medium">(w₀, b₀)</th>
              <th className="py-1 pr-2 font-medium" title="Loss of the final parameters θ_T">L(θ<sub>T</sub>)</th>
              <th className="py-1 pr-2 font-medium" title="Least-squares minimum of that run's data">L*</th>
              <th className="py-1 font-medium" title="Share of the initial excess loss L₀ − L* removed by training">outcome</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line font-mono text-fg-2">
            {[...runs].reverse().map((r) => {
              const p = r.params;
              const o = OUTCOME[r.outcome];
              const current = r.serial === currentSerial;
              return (
                <tr key={r.serial} className={current ? "text-fg" : ""}>
                  <td className="py-1 pr-2 font-sans">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="inline-block h-0.5 w-3.5" style={{ background: r.color }} aria-hidden />#{r.serial + 1}
                      {current && <span className="text-[10.5px] text-fg-3">current</span>}
                    </span>
                  </td>
                  <td className="py-1 pr-2">{fmt(p.learning_rate, 4)}</td>
                  <td className="py-1 pr-2">{fmt(p.learning_rate / r.critLR, 2)}</td>
                  <td className="py-1 pr-2">{p.epochs}</td>
                  <td className="py-1 pr-2">
                    ({fmt(p.w_init, 1)}, {fmt(p.b_init, 1)})
                  </td>
                  <td className="py-1 pr-2">{fmt(r.final, 5)}</td>
                  <td className="py-1 pr-2">{fmt(r.lStar, 5)}</td>
                  <td className={`py-1 font-sans ${o.tone}`}>
                    <span aria-hidden>{o.icon}</span> {r.outcome}
                    {r.removed != null && r.outcome !== "diverged" && <span className="text-fg-3"> · {fmt(100 * Math.min(1, Math.max(0, r.removed)), r.removed > 0.99 ? 2 : 0)}%</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11.5px] text-fg-3">
        <span>
          Last {MAX_RUNS} runs. All values are traced from Python; L* and η<sub>crit</sub> = 2/λ<sub>max</sub> are recomputed in the browser from each run&apos;s traced data. “Converged” = ≥ 99.9% of L₀ − L* removed.
        </span>
        <Button variant="ghost" onClick={onClear} disabled={runs.length <= 1}>
          Clear earlier runs
        </Button>
      </div>
    </div>
  );
}
