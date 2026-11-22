"use client";

import { useRef, useState } from "react";
import { FocusProvider } from "@/components/anim/Focus";
import { Button } from "@/components/ui/controls";
import { Panel } from "@/components/ui/primitives";
import { fmt } from "@/lib/format";
import { leastSquares, stability } from "@/lib/ml/linreg";
import { DataFitPlot, LossSurfacePlot, type FitLine } from "@/labs/linear-regression/plots";
import { findStep, foldSnapshots, paramPath, trainingHistory, type Frame } from "./fold";
import { Inspector } from "./Inspector";
import { MAX_RUNS, RunComparison, summarizeRun, type RunSummary } from "./RunComparison";
import { CodePanel, OfflineNotice, OutputPanel, ReplayToolbar, RunCallouts, VariablesPanel } from "./shell";
import { TrainingHistory } from "./TrainingHistory";
import type { LinRegState } from "./types";
import { useCodeLab } from "./useCodeLab";

const FRAMEWORKS = [
  { id: "linreg-numpy", key: "numpy", label: "NumPy", file: "linreg_numpy.py" },
  { id: "linreg-pytorch", key: "pytorch", label: "PyTorch", file: "linreg_pytorch.py" },
  { id: "linreg-sklearn", key: "sklearn", label: "scikit-learn", file: "linreg_sklearn.py" },
] as const;
type FrameworkKey = (typeof FRAMEWORKS)[number]["key"];

/** The plots share a focus (hover a point to see its residual and squared error). */
export function CodeLab() {
  return (
    <FocusProvider>
      <CodeLabInner />
    </FocusProvider>
  );
}

function CodeLabInner() {
  /** Recent gradient-descent runs per experiment, newest last (for comparison). */
  const [history, setHistory] = useState<Record<string, RunSummary[]>>({});
  const serials = useRef<Record<string, number>>({});
  const [point, setPoint] = useState(0);

  const lab = useCodeLab<FrameworkKey, LinRegState, Frame>({
    frameworks: FRAMEWORKS,
    fold: foldSnapshots,
    onResult: (id, res) => {
      if (!("epochs" in res.params)) return;
      const serial = serials.current[id] ?? 0;
      serials.current[id] = serial + 1;
      const summary = summarizeRun(res, foldSnapshots(res.snapshots), serial);
      if (summary) setHistory((h) => ({ ...h, [id]: [...(h[id] ?? []), summary].slice(-MAX_RUNS) }));
    },
  });
  const { fw, exp, meta, run, vals, frames, frame, i, last, running } = lab;

  const jumpToStep = (step: string) => {
    const k = findStep(frames, i, step);
    if (k >= 0) lab.goTo(k);
  };

  const nextEpoch = () => {
    for (let k = i + 1; k < frames.length; k++) {
      if (frames[k].state.epoch !== frame?.state.epoch && frames[k].state.epoch != null) return lab.goTo(k);
    }
    lab.goTo(last);
  };

  /* ---------------------------------------------------------------- render */

  if (lab.loadError?.offline) return <OfflineNotice onRetry={lab.retry} />;

  const stepInfo = meta?.steps.find((s) => s.id === frame?.snap.step);
  const stepTitle = stepInfo?.title ?? frame?.snap.step;
  const st = frame?.state;
  const data = st?.x && st?.y ? { x: st.x, y: st.y } : null;
  const ls = data ? leastSquares(data) : null;
  const stab = data ? stability(data) : null;
  const isGD = fw !== "sklearn";
  const epochs = run?.params.epochs ?? 1;
  const runHistory = history[exp.id] ?? [];
  const currentRun = runHistory[runHistory.length - 1] ?? null;
  const earlierRuns = runHistory.slice(0, -1);
  /** Epoch of the selected snapshot; the final evaluation counts as epoch T. */
  const curEpoch = frame?.snap.step === "evaluate" ? (st?.epoch ?? -1) + 1 : (st?.epoch ?? null);

  const runWithLR = (lr: number) => lab.runWith({ learning_rate: lr });
  const lrPresets = stab
    ? [
        { label: "too small", lr: 0.05 * stab.criticalLR, title: "η = 0.05 · 2/λmax: stable but slow" },
        { label: "no overshoot", lr: 1 / stab.lambdaMax, title: "η = 1/λmax: the steep direction is solved in one step; the flat one stays slow" },
        { label: "fastest fixed η", lr: 2 / (stab.lambdaMin + stab.lambdaMax), title: "η = 2/(λmin + λmax): best worst-case rate; oscillates along the steep direction" },
        { label: "diverges", lr: 1.05 * stab.criticalLR, title: "η = 1.05 · 2/λmax: just above the stability limit" },
      ].map((p) => ({ ...p, lr: Math.min(0.5, Math.max(0.0001, Math.round(p.lr * 1e4) / 1e4)) }))
    : [];

  const fitLines: FitLine[] = [];
  if (frame && st?.w != null && st?.b != null) {
    const fp = frame.forwardParams;
    if (fp && (fp.w !== st.w || fp.b !== st.b)) {
      fitLines.push({ ...fp, label: "model used for ŷ", color: "var(--viz-model)" });
      fitLines.push({ w: st.w, b: st.b, label: "after update", color: "var(--viz-model)", dashed: true });
    } else fitLines.push({ w: st.w, b: st.b, label: `model ŷ = ${fmt(st.w, 3)}x + ${fmt(st.b, 3)}`, color: "var(--viz-model)" });
  }
  const yHat = st?.y_hat && st.y_hat.every((v) => v != null) ? (st.y_hat as number[]) : null;
  const showNext = (frame?.snap.step === "gradient" || frame?.snap.step === "backward") && st?.w != null && st?.b != null && st?.dw != null && st?.db != null && st?.lr != null;

  return (
    <div className="space-y-4">
      <ReplayToolbar
        lab={lab}
        onSelectStep={jumpToStep}
        extraButtons={
          isGD && (
            <Button variant="ghost" onClick={nextEpoch} disabled={!run || i >= last} title="Jump to the next epoch">
              Next epoch
            </Button>
          )
        }
        status={
          curEpoch != null && (
            <span>
              epoch {curEpoch}
              <span className="text-fg-3">/{epochs}</span>
            </span>
          )
        }
      />

      <RunCallouts lab={lab} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,11fr)_minmax(0,13fr)]">
        {/* A. code */}
        <div className="min-w-0 space-y-3 xl:sticky xl:top-[9.5rem] xl:self-start">
          <CodePanel lab={lab} />
          {isGD && stab && run && meta && vals && (
            <Panel title="Learning-rate experiments">
              <p className="text-[12.5px] text-fg-2">
                Stability limit of this data: <span className="font-mono tabular-nums text-fg">η<sub>crit</sub> = 2/λ<sub>max</sub> = {fmt(stab.criticalLR, 4)}</span>
                {" · "}condition number <span className="font-mono tabular-nums">κ = {fmt(stab.condition, 1)}</span>
                {" · "}last run: <span className="font-mono tabular-nums text-fg">η = {fmt(run.params.learning_rate, 4)} = {fmt(run.params.learning_rate / stab.criticalLR, 2)} η<sub>crit</sub></span>
              </p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {lrPresets.map((p) => (
                  <Button key={p.label} onClick={() => runWithLR(p.lr)} disabled={running} title={p.title}>
                    {p.label} <span className="font-mono text-[11.5px] text-fg-3">η = {p.lr}</span>
                  </Button>
                ))}
              </div>
              <p className="mt-2 text-[11.5px] text-fg-3">
                Each button sets <code className="font-mono">learning_rate</code> and runs the script. λ are the eigenvalues of the Hessian H = (2/n)AᵀA, computed in the browser from the traced x. Compare runs under Training history.
              </p>
            </Panel>
          )}
          {run && <OutputPanel run={run} framework={fw} />}
        </div>

        {/* B + C. inspector and visualizations */}
        <div className="min-w-0 space-y-4">
          <Panel title={<>Mathematics{stepTitle ? <> · <span className="text-fg">{stepTitle}</span></> : null}</>}>
            {frame ? (
              <>
                {stepInfo?.summary && <p className="mb-1 text-[13px] text-fg-2">{stepInfo.summary}</p>}
                <Inspector frame={frame} framework={fw} point={point} />
              </>
            ) : (
              <div className="py-6 text-center text-sm text-fg-3">{running ? "Executing Python…" : "Run the script to record snapshots."}</div>
            )}
          </Panel>

          {data && (
            <>
              <Panel title="Data and model">
                <DataFitPlot data={data} lines={fitLines} residualFrom={yHat} squares={frame?.snap.step === "loss" ? true : "focus"} selected={point} onSelect={setPoint} height={280} />
              </Panel>
              <div className="grid gap-4 lg:grid-cols-2">
                <Panel title="Parameter space (w, b)">
                  <LossSurfacePlot
                    data={data}
                    trajectory={paramPath(frames, i)}
                    current={st?.w != null && st?.b != null ? [st.w, st.b] : null}
                    next={showNext ? [st!.w! - st!.lr! * st!.dw!, st!.b! - st!.lr! * st!.db!] : null}
                    components={showNext}
                    levelThrough={showNext}
                    anchor={run?.params.w_init != null ? [run.params.w_init, run.params.b_init] : undefined}
                    height={260}
                  />
                  <p className="mt-1 text-[11.5px] text-fg-3">Level sets are computed exactly in the browser from the traced x, y; the path is the traced (w, b).</p>
                </Panel>
                {isGD ? (
                  <Panel title="Training history">
                    <TrainingHistory
                      history={trainingHistory(frames, i)}
                      current={curEpoch}
                      epochs={epochs}
                      color={currentRun?.color ?? "var(--viz-model)"}
                      optimum={ls}
                      earlier={earlierRuns}
                      diverged={currentRun?.outcome === "diverged"}
                    />
                  </Panel>
                ) : (
                  <Panel title="Result">
                    <table className="w-full font-mono text-[12.5px] tabular-nums">
                      <thead className="text-left text-fg-3">
                        <tr>
                          <th className="py-1 font-normal" />
                          <th className="py-1 font-normal">w</th>
                          <th className="py-1 font-normal">b</th>
                        </tr>
                      </thead>
                      <tbody className="text-fg">
                        <tr className="border-t border-line">
                          <td className="py-1 font-sans text-fg-2">scikit-learn</td>
                          <td>{fmt(st?.w, 6)}</td>
                          <td>{fmt(st?.b, 6)}</td>
                        </tr>
                        <tr className="border-t border-line">
                          <td className="py-1 font-sans text-fg-2">normal equations</td>
                          <td>{st?.theta ? fmt(st.theta[0], 6) : "—"}</td>
                          <td>{st?.theta ? fmt(st.theta[1], 6) : "—"}</td>
                        </tr>
                      </tbody>
                    </table>
                    <p className="mt-2 text-[12px] text-fg-3">Both rows come from the Python run. The normal-equations row appears at the final step.</p>
                  </Panel>
                )}
              </div>
              {isGD && runHistory.length > 0 && (
                <Panel title="Run comparison">
                  <RunComparison runs={runHistory} currentSerial={currentRun?.serial ?? null} onClear={() => setHistory((h) => ({ ...h, [exp.id]: runHistory.slice(-1) }))} />
                </Panel>
              )}
            </>
          )}

          {frame && <VariablesPanel frame={frame} />}
        </div>
      </div>
    </div>
  );
}
