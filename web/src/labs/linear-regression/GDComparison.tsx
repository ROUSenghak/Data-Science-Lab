"use client";

import { useMemo, useState } from "react";
import { usePlayback } from "@/components/anim/hooks";
import { PlaybackBar } from "@/components/anim/PlaybackBar";
import { PredictPrompt } from "@/components/anim/PredictPrompt";
import { Tex } from "@/components/math/Tex";
import { SeriesPlot } from "@/components/plot/SeriesPlot";
import { Button, Slider } from "@/components/ui/controls";
import type { SpeedDef } from "@/lib/anim/playback";
import { fmt } from "@/lib/format";
import { summarizeGD, type GDSummary } from "@/lib/ml/gdTimeline";
import { gradientDescent, leastSquares, stability, type Dataset } from "@/lib/ml/linreg";
import { LR_RANGE, paramDomain, REGIME_LABEL } from "./GradientDescentExplorer";
import { LossSurfacePlot } from "./plots";

const SPEEDS: Record<"slow" | "normal" | "fast", SpeedDef> = {
  slow: { label: "slow", interval: 500, stride: 1 },
  normal: { label: "normal", interval: 160, stride: 1 },
  fast: { label: "fast", interval: 50, stride: 2 },
};
const RUNS = [
  { key: "A", color: "var(--viz-run-3)" },
  { key: "B", color: "var(--viz-run-1)" },
] as const;
const TOL = 1e-3;
const START = { w0: 0, b0: -1 };

const round3 = (v: number) => {
  const m = 10 ** (Math.floor(Math.log10(v)) - 2);
  return Math.min(LR_RANGE[1], Math.max(LR_RANGE[0], Math.round(v / m) * m));
};

/**
 * Two learning rates on the same data from the same θ₀, replayed epoch by epoch in sync.
 * Every number comes from gradientDescent / summarizeGD on that data.
 */
export function GDComparison({ data }: { data: Dataset }) {
  const stab = useMemo(() => stability(data), [data]);
  const ls = useMemo(() => leastSquares(data), [data]);
  const domain = useMemo(() => paramDomain(data), [data]);
  const presets = useMemo(
    () => [
      { label: "just below vs just above 2/λmax", lr: [0.9 * stab.criticalLR, 1.05 * stab.criticalLR] },
      { label: "no overshoot vs fastest fixed η", lr: [1 / stab.lambdaMax, 2 / (stab.lambdaMin + stab.lambdaMax)] },
      { label: "tiny vs moderate", lr: [0.05 * stab.criticalLR, 0.5 * stab.criticalLR] },
    ].map((p) => ({ ...p, lr: p.lr.map(round3) as [number, number] })),
    [stab],
  );
  const [lrs, setLrs] = useState<[number, number]>(() => presets[1].lr);
  const [epochs, setEpochs] = useState(60);

  const runs = useMemo(() => lrs.map((lr) => gradientDescent(data, { lr, epochs, ...START })), [data, lrs, epochs]);
  const sums = useMemo(() => runs.map((r, k) => summarizeGD(r, data, lrs[k], stab.lambdaMax, TOL)), [runs, data, lrs, stab]);
  const T = Math.max(...runs.map((r) => r.length)) + 1; // epochs 0 … T−1, the last being θ_T after the final update
  const pb = usePlayback(T, { speeds: SPEEDS, initialSpeed: "normal", resetKey: runs });
  const t = pb.index;

  /** θ_t of run k, or its last finite value if it stopped (overflow). */
  const thetaAt = (k: number, tt: number): [number, number] => {
    const r = runs[k];
    if (tt < r.length) return [r[tt].w, r[tt].b];
    const last = r[r.length - 1];
    return [last.wNext, last.bNext];
  };
  const pathTo = (k: number, tt: number) => Array.from({ length: Math.min(tt, runs[k].length) + 1 }, (_, i) => thetaAt(k, i));
  const excess = (k: number) => {
    const pts: [number, number | null][] = [...runs[k].map((s) => [s.epoch, s.loss - ls.loss] as [number, number]), [runs[k].length, sums[k].gap]];
    return pts.map(([e, v]) => [e, e <= t && Number.isFinite(v) && v! > 0 ? v : null] as [number, number | null]);
  };
  const allExcess = runs.flatMap((r, k) => [...r.map((s) => s.loss - ls.loss), sums[k].gap]).filter((v) => Number.isFinite(v) && v > 0);
  const yDomain: [number, number] | undefined = allExcess.length
    ? [10 ** Math.floor(Math.log10(Math.min(...allExcess))), 10 ** Math.ceil(Math.log10(Math.max(...allExcess) * 1.01))]
    : undefined;

  const winner = firstToTol(sums);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-[13px] text-fg-2">
        Presets:
        {presets.map((p) => (
          <Button key={p.label} onClick={() => setLrs(p.lr)} variant={p.lr[0] === lrs[0] && p.lr[1] === lrs[1] ? "primary" : "default"}>
            {p.label}
          </Button>
        ))}
      </div>
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-3">
        {RUNS.map((run, k) => (
          <Slider
            key={run.key}
            label={
              <span className="inline-flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: run.color }} aria-hidden />
                run {run.key}: <Tex>{"\\eta"}</Tex>
              </span>
            }
            value={lrs[k]}
            min={LR_RANGE[0]}
            max={LR_RANGE[1]}
            step={0.001}
            scale="log"
            onChange={(v) => setLrs((cur) => (k === 0 ? [v, cur[1]] : [cur[0], v]))}
            format={(v) => `${fmt(v, 3)} = ${fmt(v / stab.criticalLR, 2)} η꜀`}
          />
        ))}
        <Slider label="epochs" value={epochs} min={10} max={300} step={10} onChange={setEpochs} hint={<>same data, same θ₀ = ({START.w0}, {START.b0}); η꜀ = 2/λ<sub>max</sub> = {fmt(stab.criticalLR, 3)}</>} />
      </div>

      <PlaybackBar pb={pb} speeds={(["slow", "normal", "fast"] as const).map((v) => ({ value: v, label: v }))} status={`epoch ${t} / ${T - 1}`} scrubLabel="Epoch" />
      <PredictPrompt
        question={<>Which run gets within {TOL * 100}% of the initial excess loss first?</>}
        options={[
          { value: "A", label: "run A" },
          { value: "B", label: "run B" },
          { value: "tie", label: "same epoch" },
          { value: "neither", label: "neither, within the epochs" },
        ]}
        answer={winner}
        revealed={t >= pb.last}
        resetKey={`${lrs[0]}|${lrs[1]}|${epochs}`}
        explanation={
          <>
            Two factors per run, each applied to one eigen-direction of the error every epoch (smaller |·| is faster; ≥ 1 never converges). Steep direction |1 − ηλ<sub>max</sub>|:{" "}
            {RUNS.map((r, k) => `${r.key} ${fmt(Math.abs(1 - lrs[k] * stab.lambdaMax), 3)}`).join(", ")}. Flat direction |1 − ηλ<sub>min</sub>|: {RUNS.map((r, k) => `${r.key} ${fmt(Math.abs(1 - lrs[k] * stab.lambdaMin), 3)}`).join(", ")}. The steep direction usually holds most of the initial loss, so it decides the early drop; the slower of the two factors decides the final slope. That is why the curves can cross.
          </>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="min-w-0">
          <LossSurfacePlot
            data={data}
            trajectory={[]}
            domain={domain}
            equalAspect
            paths={RUNS.map((r, k) => ({ label: `run ${r.key}`, color: r.color, points: pathTo(k, t), current: thetaAt(k, Math.min(t, runs[k].length)) }))}
            height={300}
          />
        </div>
        <div className="min-w-0">
          <SeriesPlot
            series={RUNS.map((r, k) => ({ label: `run ${r.key}`, color: r.color, points: excess(k) }))}
            xLabel="epoch"
            yLabel="excess loss L(θₜ) − L*"
            log
            current={t}
            xDomain={[0, T - 1]}
            yDomain={yDomain}
            height={300}
            ariaLabel="Excess loss of both runs per epoch, log scale"
          />
          <p className="mt-1 text-[11.5px] text-fg-3">On a log axis a geometric rate is a straight line: its slope is the per-epoch factor. A rising line diverges.</p>
        </div>
      </div>

      <ComparisonTable sums={sums} lrs={lrs} data={data} />
    </div>
  );
}

function firstToTol(sums: GDSummary[]): "A" | "B" | "tie" | "neither" {
  const [a, b] = sums.map((s) => s.epochsToTol);
  if (a == null && b == null) return "neither";
  if (a == null) return "B";
  if (b == null) return "A";
  return a === b ? "tie" : a < b ? "A" : "B";
}

function ComparisonTable({ sums, lrs, data }: { sums: GDSummary[]; lrs: [number, number]; data: Dataset }) {
  const ls = leastSquares(data);
  const rows: { label: string; value: (s: GDSummary, k: number) => string }[] = [
    { label: "learning rate η", value: (_, k) => fmt(lrs[k], 4) },
    { label: "behaviour (theory)", value: (s) => REGIME_LABEL[s.regime] },
    { label: "final L(θ_T)", value: (s) => (s.overflowed ? "overflow" : fmt(s.final.loss, 5)) },
    { label: "L(θ_T) − L*", value: (s) => (s.overflowed ? "—" : fmt(s.gap, 3)) },
    { label: "‖θ_T − θ*‖", value: (s) => (s.overflowed ? "—" : fmt(s.distance, 3)) },
    { label: `epochs to ${TOL * 100}% of initial excess`, value: (s) => (s.epochsToTol == null ? "not reached" : String(s.epochsToTol)) },
    { label: "epochs where the loss rose", value: (s) => String(s.increases) },
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[12.5px] tabular-nums">
        <thead className="text-[11px] text-fg-3">
          <tr>
            <th className="py-1 pr-3 font-medium" />
            {RUNS.map((r) => (
              <th key={r.key} className="py-1 pr-3 font-medium">
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: r.color }} aria-hidden />
                  run {r.key}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((row) => (
            <tr key={row.label}>
              <td className="py-1 pr-3 text-fg-2">{row.label}</td>
              {sums.map((s, k) => (
                <td key={k} className="py-1 pr-3 font-mono text-fg">
                  {row.value(s, k)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[11.5px] text-fg-3">
        θ* = ({fmt(ls.w, 4)}, {fmt(ls.b, 4)}), L* = {fmt(ls.loss, 4)}. θ_T is the result of the last update and L(θ_T) is evaluated at it.
      </p>
    </div>
  );
}
