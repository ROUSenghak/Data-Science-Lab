"use client";

import { useState } from "react";
import { SeriesPlot, type RefLine, type Series } from "@/components/plot/SeriesPlot";
import { Segmented } from "@/components/ui/controls";
import type { EpochRecord } from "./fold";
import type { RunSummary } from "./RunComparison";

type Metric = "loss" | "params" | "grads";

const pts = (h: EpochRecord[], key: keyof EpochRecord): [number, number | null][] => h.map((r) => [r.epoch, r[key]]);

/**
 * Loss, parameters or gradients per epoch. The current run is revealed up to the selected
 * snapshot; earlier runs (loss only) are shown in full for comparison.
 */
export function TrainingHistory({
  history,
  current,
  epochs,
  color,
  optimum,
  earlier,
  diverged,
}: {
  /** Current run, up to the selected snapshot. */
  history: EpochRecord[];
  /** Epoch of the selected snapshot (null before the loop). */
  current: number | null;
  epochs: number;
  color: string;
  /** Least-squares solution of the current data. */
  optimum: { w: number; b: number; loss: number } | null;
  earlier: RunSummary[];
  /** The current run diverged (judged on the full run, not the revealed part). */
  diverged: boolean;
}) {
  const [metric, setMetric] = useState<Metric>("loss");
  const [scale, setScale] = useState<"lin" | "log">("lin");
  const maxEpoch = Math.max(epochs, ...earlier.map((r) => r.params.epochs ?? 0));
  const xDomain: [number, number] = [0, Math.max(1, maxEpoch)];
  const anyDiverged = diverged || earlier.some((r) => r.outcome === "diverged");
  const log = metric === "loss" && (scale === "log" || anyDiverged);

  let series: Series[];
  let refs: RefLine[] = [];
  let yLabel: string;
  if (metric === "loss") {
    series = [
      { label: earlier.length ? "current run" : "L(θₜ)", color, points: pts(history, "loss") },
      ...earlier.map((r) => ({ label: `run #${r.serial + 1} (η = ${r.params.learning_rate})`, color: r.color, points: pts(r.history, "loss"), muted: true })),
    ];
    if (optimum) refs = [{ label: "least-squares L*", value: optimum.loss, color: "var(--viz-alt)" }];
    yLabel = "MSE";
  } else if (metric === "params") {
    series = [
      { label: "w", color: "var(--viz-model)", points: pts(history, "w") },
      { label: "b", color: "var(--viz-residual)", points: pts(history, "b") },
    ];
    if (optimum)
      refs = [
        { label: "w*", value: optimum.w, color: "var(--viz-model)" },
        { label: "b*", value: optimum.b, color: "var(--viz-residual)" },
      ];
    yLabel = "parameter value";
  } else {
    series = [
      { label: "∂L/∂w", color: "var(--viz-model)", points: pts(history, "dw") },
      { label: "∂L/∂b", color: "var(--viz-residual)", points: pts(history, "db") },
    ];
    yLabel = "gradient";
  }

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <Segmented
          ariaLabel="Quantity"
          value={metric}
          onChange={setMetric}
          options={[
            { value: "loss", label: "loss" },
            { value: "params", label: "w, b" },
            { value: "grads", label: "gradients" },
          ]}
        />
        {metric === "loss" && (
          <Segmented
            ariaLabel="Loss axis scale"
            value={log ? "log" : "lin"}
            onChange={setScale}
            options={[
              { value: "lin", label: "linear" },
              { value: "log", label: "log" },
            ]}
          />
        )}
      </div>
      <SeriesPlot
        series={series}
        refs={refs}
        xLabel="epoch"
        yLabel={yLabel}
        log={log}
        includeZero={metric !== "params"}
        current={current}
        xDomain={xDomain}
        height={240}
        ariaLabel={`Training history: ${metric} per epoch`}
      />
      <p className="mt-1 text-[11.5px] text-fg-3">
        Epoch t shows θ<sub>t</sub>, L(θ<sub>t</sub>) and ∇L(θ<sub>t</sub>), all at the same parameters; the last point is the final evaluation of θ<sub>T</sub>.
        {metric === "loss" && anyDiverged && scale === "lin" && " Log scale is forced because a run diverged."}
      </p>
    </div>
  );
}
