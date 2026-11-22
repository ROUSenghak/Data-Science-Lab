"use client";

import { useMemo, useState } from "react";
import { FocusProvider, LinkedMath, LTexBlock, useFocus } from "@/components/anim/Focus";
import { usePlayback, useTween } from "@/components/anim/hooks";
import { PlaybackBar } from "@/components/anim/PlaybackBar";
import { PredictPrompt } from "@/components/anim/PredictPrompt";
import { Tex } from "@/components/math/Tex";
import { Segmented, Slider, Toggle } from "@/components/ui/controls";
import { Callout, Expand } from "@/components/ui/primitives";
import { focusedResidual, hl } from "@/lib/anim/focus";
import type { SpeedDef } from "@/lib/anim/playback";
import { tweenDuration } from "@/lib/anim/tween";
import { clamp, fmt, texNum, texSq } from "@/lib/format";
import { buildGDTimeline, eigenErrors, frameOfEpoch, regimeOf, stepFactors, type GDFrame, type GDStage, type Regime } from "@/lib/ml/gdTimeline";
import { gradientDescent, leastSquares, predict, stability, type Dataset } from "@/lib/ml/linreg";
import { LossLandscapeView } from "./LossLandscapeView";
import { DataFitPlot, LossCurvePlot } from "./plots";

/* Ranges shared by the sliders, the draggable θ₀ and the stability gauge. */
export const LR_RANGE: [number, number] = [0.001, 0.5];
export const W0_RANGE: [number, number] = [-3, 6];
export const B0_RANGE: [number, number] = [-4, 6];

const EPOCH_SPEEDS: Record<"slow" | "normal" | "fast", SpeedDef> = {
  slow: { label: "slow", interval: 900, stride: 1 },
  normal: { label: "normal", interval: 380, stride: 1 },
  fast: { label: "fast", interval: 60, stride: 3 },
};
const DETAIL_SPEEDS: typeof EPOCH_SPEEDS = {
  slow: { label: "slow", interval: 1600, stride: 1 },
  normal: { label: "normal", interval: 900, stride: 1 },
  fast: { label: "fast", interval: 380, stride: 1 },
};
const SPEED_OPTIONS = (["slow", "normal", "fast"] as const).map((v) => ({ value: v, label: v }));

export const STAGE_INFO: Record<GDStage, { n: number; name: string; what: string }> = {
  forward: { n: 1, name: "predict", what: "Predict every point with the current line and measure each residual." },
  loss: { n: 2, name: "loss", what: "Square the residuals and average them. Each square's area is one rᵢ²." },
  gradient: { n: 3, name: "gradient", what: "Differentiate the loss: which way, and how steeply, it rises in w and in b." },
  update: { n: 4, name: "update", what: "Step against the gradient, scaled by η. The line moves to its new parameters." },
};

export const REGIME_LABEL: Record<Regime, string> = {
  monotone: "glides straight in",
  zigzag: "zigzags, but converges",
  diverge: "blows up",
};

/** Default plot window in parameter space: the slider box plus the optimum. */
export function paramDomain(data: Dataset) {
  const ls = leastSquares(data);
  const w: [number, number] = [Math.min(W0_RANGE[0], ls.w) - 0.5, Math.max(W0_RANGE[1], ls.w) + 0.5];
  const b: [number, number] = [Math.min(B0_RANGE[0], ls.b) - 0.5, Math.max(B0_RANGE[1], ls.b) + 0.5];
  return { w, b };
}

export function GradientDescentExplorer(props: { data: Dataset; defaultLR?: number; defaultDetailed?: boolean; predictions?: boolean }) {
  return (
    <FocusProvider>
      <GDExplorerInner {...props} />
    </FocusProvider>
  );
}

function GDExplorerInner({ data, defaultLR = 0.08, defaultDetailed = false, predictions = true }: { data: Dataset; defaultLR?: number; defaultDetailed?: boolean; predictions?: boolean }) {
  const [lr, setLR] = useState(defaultLR);
  const [epochs, setEpochs] = useState(60);
  const [w0, setW0] = useState(0);
  const [b0, setB0] = useState(-1);
  const [detailed, setDetailed] = useState(defaultDetailed);
  const [log, setLog] = useState(false);
  const [allSquares, setAllSquares] = useState(false);
  const [axes, setAxes] = useState(false);
  const [ask, setAsk] = useState(predictions);
  const [pinned, setPinned] = useState<number | null>(null);
  const { focus } = useFocus();

  const steps = useMemo(() => gradientDescent(data, { lr, epochs, w0, b0 }), [data, lr, epochs, w0, b0]);
  const frames = useMemo(() => buildGDTimeline(steps, detailed), [steps, detailed]);
  const stab = useMemo(() => stability(data), [data]);
  const ls = useMemo(() => leastSquares(data), [data]);
  const domain = useMemo(() => paramDomain(data), [data]);
  const pb = usePlayback(frames.length, { speeds: detailed ? DETAIL_SPEEDS : EPOCH_SPEEDS, initialSpeed: "normal", resetKey: steps });

  const f = frames[pb.index];
  const s = f.step;
  const stage = f.stage;
  const tween = useTween([f.w, f.b], { singleStep: pb.singleStep, duration: tweenDuration(pb.speedDef.interval) });
  const [dw, db] = tween.value;
  const regime = regimeOf(lr, stab.lambdaMax);
  const overflow = !Number.isFinite(f.w) || !Number.isFinite(f.b);

  // What each view shows for this computed state.
  const showResiduals = stage !== "update" && !overflow;
  // Squares: the focused residual's square; all of them at the loss stage or on request.
  const showSquares = !showResiduals ? false : stage === "loss" || allSquares ? true : ("focus" as const);
  const showArrow = (stage === "epoch" || stage === "gradient") && !tween.moving;
  const yHat = showResiduals ? predict(dw, db, data.x) : null;
  const lossKnown = stage !== "forward";

  // Path of computed parameters up to this frame (the moving point extends it while it glides).
  const path: [number, number][] = [];
  for (let t = 0; t <= f.epoch; t++) path.push([steps[t].w, steps[t].b]);
  if (stage === "update") path.push([s.wNext, s.bNext]);
  if (tween.moving) path.splice(path.length - 1, 1, [dw, db]);

  const losses = steps.slice(0, f.epoch + (lossKnown ? 1 : 0)).map((st) => ({ epoch: st.epoch, loss: st.loss }));
  const r = predict(s.w, s.b, data.x).map((v, i) => v - data.y[i]);
  const hovered = focusedResidual(focus);
  const worst = r.reduce((best, v, i) => (Math.abs(v) > Math.abs(r[best]) ? i : best), 0);
  const point = hovered ?? pinned ?? worst;

  const setStart = (w: number, b: number) => {
    setW0(Math.round(clamp(w, ...W0_RANGE) * 10) / 10);
    setB0(Math.round(clamp(b, ...B0_RANGE) * 10) / 10);
  };
  const switchGranularity = (d: boolean) => {
    const next = buildGDTimeline(steps, d);
    setDetailed(d);
    pb.seek(frameOfEpoch(next, f.epoch), next.length);
  };

  const status = overflow
    ? `epoch ${f.epoch} · overflow`
    : stage === "epoch"
      ? `epoch ${f.epoch} / ${steps.length - 1}`
      : `epoch ${f.epoch} · ${STAGE_INFO[stage].n} ${STAGE_INFO[stage].name}`;

  return (
    <div className="space-y-4">
      {/* The one control that matters first: η, placed on its stability scale. */}
      <div className="grid gap-x-8 gap-y-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)] md:items-end">
        <Slider
          label={<Tex>{"\\text{learning rate } \\eta"}</Tex>}
          value={lr}
          min={LR_RANGE[0]}
          max={LR_RANGE[1]}
          step={0.001}
          scale="log"
          onChange={setLR}
          format={(v) => fmt(v, 3)}
          hint={<>stable iff η &lt; 2/λ<sub>max</sub> = {fmt(stab.criticalLR, 3)}</>}
        />
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline justify-between gap-2 text-[12px]">
            <span className="text-fg-3">where η sits (click to choose)</span>
            <span className={regime === "diverge" ? "text-bad" : regime === "zigzag" ? "text-warn" : "text-ok"}>θ {REGIME_LABEL[regime]}</span>
          </div>
          <StabilityGauge lr={lr} lambdaMax={stab.lambdaMax} onPick={setLR} />
        </div>
      </div>
      <details className="group text-[12.5px] text-fg-2">
        <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 text-fg-3 hover:text-fg [&::-webkit-details-marker]:hidden">
          <svg width="10" height="10" viewBox="0 0 12 12" className="transition-transform group-open:rotate-90" aria-hidden>
            <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.6" fill="none" />
          </svg>
          Run settings: start θ₀ = ({fmt(w0, 1)}, {fmt(b0, 1)}), {epochs} epochs
        </summary>
        <div className="mt-2 grid gap-x-6 gap-y-3 sm:grid-cols-3">
          <Slider label={<Tex>{"w_0"}</Tex>} value={w0} min={W0_RANGE[0]} max={W0_RANGE[1]} step={0.1} onChange={setW0} format={(v) => fmt(v, 1)} hint="or click / drag θ₀ on the loss plot" />
          <Slider label={<Tex>{"b_0"}</Tex>} value={b0} min={B0_RANGE[0]} max={B0_RANGE[1]} step={0.1} onChange={setB0} format={(v) => fmt(v, 1)} />
          <Slider label="epochs" value={epochs} min={5} max={400} step={5} onChange={setEpochs} />
        </div>
      </details>

      <div className="space-y-2">
        <PlaybackBar
          pb={pb}
          speeds={SPEED_OPTIONS}
          status={status}
          scrubLabel={detailed ? "Computation step" : "Epoch"}
          extra={
            <Segmented
              ariaLabel="Granularity"
              value={detailed ? "detailed" : "epoch"}
              onChange={(v) => switchGranularity(v === "detailed")}
              options={[
                { value: "epoch", label: "per epoch" },
                { value: "detailed", label: "inside an epoch" },
              ]}
            />
          }
        />
        {detailed && stage !== "epoch" && <StageStrip stage={stage} />}
        {ask && (
          <PredictPrompt
            question={<>With η = {fmt(lr, 3)}, how will θ approach the minimum?</>}
            options={(["monotone", "zigzag", "diverge"] as const).map((v) => ({ value: v, label: REGIME_LABEL[v] }))}
            answer={regime}
            revealed={pb.index >= pb.last}
            resetKey={lr}
            onDismiss={() => setAsk(false)}
            explanation={<RegimeExplanation regime={regime} lr={lr} lambdaMax={stab.lambdaMax} />}
          />
        )}
      </div>

      <div className="grid gap-x-6 gap-y-5 lg:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1.5 text-[12px] font-medium text-fg-2">Data and the model at this step</div>
          <DataFitPlot
            data={data}
            residualFrom={yHat}
            squares={showSquares}
            selected={pinned}
            onSelect={(i) => setPinned((p) => (p === i ? null : i))}
            lines={[
              ...(stage === "update" && !overflow ? [{ w: s.w, b: s.b, label: `before: θ${sub(f.epoch)}`, color: "var(--fg-3)", dashed: true }] : []),
              { w: dw, b: db, label: stage === "update" ? `after: θ${sub(f.epoch + 1)}` : `model at epoch ${f.epoch}`, color: "var(--viz-model)" },
            ]}
            height={300}
          />
          <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-fg-3">
            <Toggle label="all squared errors" checked={allSquares} onChange={setAllSquares} />
            <span>Hover a point or a term in the equations to see its residual and square; click to pin.</span>
          </div>
        </div>
        <div className="min-w-0">
          <LossLandscapeView
            data={data}
            trajectory={path}
            current={[dw, db]}
            currentLabel={`θ${sub(stage === "update" ? f.epoch + 1 : f.epoch)}`}
            emphasizeHeight={stage === "loss"}
            next={showArrow ? [s.wNext, s.bNext] : null}
            components={showArrow}
            levelThrough={!tween.moving && !overflow}
            eigenAxes={axes}
            domain={domain}
            equalAspect
            start={[w0, b0]}
            onStartChange={setStart}
            height={300}
            note2d={
              <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-fg-3">
                <Toggle label="steep / flat directions" checked={axes} onChange={setAxes} />
                <span>Equal axis scales, so −∇L is truly perpendicular to the level set. A glide follows the exact segment θₜ → θₜ₊₁; positions in between are not iterations.</span>
              </div>
            }
          />
        </div>
      </div>

      <div className="grid gap-x-6 gap-y-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <GDEquations frame={f} data={data} lr={lr} r={r} point={point} pointSource={hovered != null ? "hovered" : pinned != null ? "pinned" : "largest"} />
        </div>
        <div className="min-w-0">
          <div className="mb-1 flex items-center justify-between gap-2">
            <span className="text-[12px] font-medium text-fg-2">Loss per epoch</span>
            <Segmented ariaLabel="Loss axis scale" value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ value: "lin", label: "linear" }, { value: "log", label: "log" }]} />
          </div>
          <LossCurvePlot losses={losses} totalEpochs={epochs} minimum={ls.loss} current={lossKnown ? f.epoch : null} log={log || regime === "diverge"} height={250} />
        </div>
      </div>

      {overflow && (
        <Callout tone="bad" title="The values overflowed">
          At epoch {f.epoch} the update produced a non-finite number, so the run stops here. Each step multiplied the steep-direction error by |1 − ηλ<sub>max</sub>| = {fmt(Math.abs(1 - lr * stab.lambdaMax), 2)}.
        </Callout>
      )}

      <EigenPanel lr={lr} stab={stab} frame={f} data={data} />
    </div>
  );
}

const sub = (t: number) => String(t).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]);

function StageStrip({ stage }: { stage: GDStage }) {
  return (
    <ol className="grid grid-cols-2 gap-1.5 text-[12px] sm:grid-cols-4" aria-label="Computations in one epoch">
      {(Object.keys(STAGE_INFO) as GDStage[]).map((k) => {
        const on = k === stage;
        return (
          <li key={k} aria-current={on ? "step" : undefined} className={`rounded-md border px-2 py-1 ${on ? "border-[var(--viz-model)] bg-accent-soft text-fg" : "border-line text-fg-3"}`}>
            <span className="font-medium">
              {STAGE_INFO[k].n} · {STAGE_INFO[k].name}
            </span>
            {on && <span className="block leading-snug text-fg-2">{STAGE_INFO[k].what}</span>}
          </li>
        );
      })}
    </ol>
  );
}

/* ------------------------------------------------------------------ equations, linked to the plots */

function GDEquations({ frame, data, lr, r, point, pointSource }: { frame: GDFrame; data: Dataset; lr: number; r: number[]; point: number; pointSource: "hovered" | "pinned" | "largest" }) {
  const { epoch: t, step: s, stage } = frame;
  const n = data.x.length;
  const i = Math.min(point, n - 1);
  const yHat = s.w * data.x[i] + s.b;
  const R = (k: number) => hl({ kind: "residual", i: k }, texSq(r[k], 2));
  const terms = n <= 4 ? r.map((_, k) => R(k)).join(" + ") : `${R(0)} + ${R(1)} + \\cdots + ${R(n - 1)}`;
  const active = (k: GDStage) => stage === "epoch" || stage === k;

  const rows: { k: GDStage; tex: string }[] = [
    {
      k: "forward",
      tex: `\\begin{aligned} \\hat y_{${i + 1}} &= w_{${t}}\\,x_{${i + 1}} + b_{${t}} = ${texNum(s.w, 3)}\\cdot${texNum(data.x[i], 3, true)} + ${texNum(s.b, 3, true)} = ${texNum(yHat, 3)} \\\\ r_{${i + 1}} &= \\hat y_{${i + 1}} - y_{${i + 1}} = ${hl({ kind: "residual", i }, texNum(r[i], 3))} \\end{aligned}`,
    },
    { k: "loss", tex: `L(\\theta_{${t}}) = \\frac{1}{${n}}\\Big(${terms}\\Big) = ${hl({ kind: "loss" }, texNum(s.loss, 4))}` },
    {
      k: "gradient",
      tex: `\\begin{aligned} \\tfrac{\\partial L}{\\partial w} &= \\tfrac{2}{n}\\textstyle\\sum_i r_i x_i = ${hl({ kind: "grad", axis: "w" }, texNum(s.dw, 4))} \\\\ \\tfrac{\\partial L}{\\partial b} &= \\tfrac{2}{n}\\textstyle\\sum_i r_i = ${hl({ kind: "grad", axis: "b" }, texNum(s.db, 4))} \\end{aligned}`,
    },
    {
      k: "update",
      tex: `\\begin{aligned} w_{${t + 1}} &= w_{${t}} - \\eta\\,\\tfrac{\\partial L}{\\partial w} = ${texNum(s.w, 4)} - ${texNum(lr, 3)}\\cdot${texNum(s.dw, 4, true)} = ${hl({ kind: "step", axis: "w" }, texNum(s.wNext, 4))} \\\\ b_{${t + 1}} &= b_{${t}} - \\eta\\,\\tfrac{\\partial L}{\\partial b} = ${texNum(s.b, 4)} - ${texNum(lr, 3)}\\cdot${texNum(s.db, 4, true)} = ${hl({ kind: "step", axis: "b" }, texNum(s.bNext, 4))} \\end{aligned}`,
    },
  ];

  return (
    <LinkedMath className="rounded-md border border-line bg-surface-2 px-3 py-2 text-[14px]">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-[12px]">
        <span className="font-medium text-fg-2">Epoch {t}: the numbers of this run</span>
        <span className="text-fg-3">
          point {i + 1} ({pointSource === "largest" ? "largest |rᵢ|" : pointSource})
        </span>
      </div>
      <ol className="mt-1 space-y-0.5">
        {rows.map(({ k, tex }) => (
          <li
            key={k}
            className={`grid grid-cols-[4.5rem_minmax(0,1fr)] items-center gap-2 rounded border-l-2 pl-2 transition-opacity ${stage === k ? "border-[var(--viz-model)] bg-surface" : "border-transparent"} ${active(k) ? "" : "opacity-45"}`}
          >
            <span className="text-[11.5px] text-fg-3">
              {STAGE_INFO[k].n} · {STAGE_INFO[k].name}
            </span>
            <LTexBlock className="min-w-0 [&_.katex-display]:my-1">{tex}</LTexBlock>
          </li>
        ))}
      </ol>
    </LinkedMath>
  );
}

/* ------------------------------------------------------------------ why it converges, zigzags or diverges */

function RegimeExplanation({ regime, lr, lambdaMax }: { regime: Regime; lr: number; lambdaMax: number }) {
  const f = 1 - lr * lambdaMax;
  return (
    <>
      ηλ<sub>max</sub> = {fmt(lr * lambdaMax, 2)}, so every step multiplies the error along the steep direction by 1 − ηλ<sub>max</sub> = {fmt(f, 2)}.{" "}
      {regime === "monotone" && <>That factor is between 0 and 1: the error shrinks without changing sign, so θ never overshoots the valley floor.</>}
      {regime === "zigzag" && <>That factor is negative but larger than −1: θ overshoots and lands on the other side of the valley each step, yet closer than before, so the loss still falls.</>}
      {regime === "diverge" && <>Its magnitude is at least 1: each overshoot lands further away than the last, and the loss grows geometrically.</>}
    </>
  );
}

/** Why the path glides, zigzags or diverges: the error along H's eigen-directions at this step. */
function EigenPanel({ lr, stab, frame, data }: { lr: number; stab: ReturnType<typeof stability>; frame: GDFrame; data: Dataset }) {
  const regime = regimeOf(lr, stab.lambdaMax);
  const fac = stepFactors(lr, stab.lambdaMin, stab.lambdaMax);
  const finite = Number.isFinite(frame.w) && Number.isFinite(frame.b);
  const e = finite ? eigenErrors(frame.w, frame.b, data) : null;
  const slowSteps = Math.abs(fac.flat) < 1 ? Math.ceil(Math.log(10) / -Math.log(Math.abs(fac.flat))) : Infinity;
  return (
    <Expand title="Why it glides, zigzags or diverges: the error along H's eigen-directions">
      <p>
        Near the minimum the error θ − θ* splits into a component along the steep direction of the bowl (eigenvalue λ<sub>max</sub> of the Hessian H) and one along the flat direction (λ<sub>min</sub>). Each epoch multiplies them by exactly 1 − ηλ. Turn on <i>steep / flat directions</i> in the 2D view to see the two axes.
      </p>
      <table className="mt-2 w-full max-w-lg font-mono text-[12.5px] tabular-nums">
        <thead className="font-sans text-[11px] text-fg-3">
          <tr>
            <th className="py-0.5 text-left font-medium">direction</th>
            <th className="py-0.5 text-right font-medium">error eₖ now</th>
            <th className="py-0.5 text-right font-medium">× per step</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="py-0.5 font-sans">steep (λ = {fmt(stab.lambdaMax, 2)})</td>
            <td className="py-0.5 text-right">{fmt(e?.steep, 3)}</td>
            <td className={`py-0.5 text-right ${Math.abs(fac.steep) >= 1 ? "text-bad" : fac.steep < 0 ? "text-warn" : ""}`}>{fmt(fac.steep, 3)}</td>
          </tr>
          <tr>
            <td className="py-0.5 font-sans">flat (λ = {fmt(stab.lambdaMin, 2)})</td>
            <td className="py-0.5 text-right">{fmt(e?.flat, 3)}</td>
            <td className="py-0.5 text-right">{fmt(fac.flat, 3)}</td>
          </tr>
        </tbody>
      </table>
      <p>
        <RegimeExplanation regime={regime} lr={lr} lambdaMax={stab.lambdaMax} />{" "}
        {regime !== "diverge" && Number.isFinite(slowSteps) && <>The flat direction needs ≈ {slowSteps} epochs per 10× reduction (κ = {fmt(stab.condition, 1)}).</>}
      </p>
    </Expand>
  );
}

/** η on a log axis, coloured by what fixed-step GD does on this loss. Click to set η. */
export function StabilityGauge({ lr, lambdaMax, onPick, height = 44 }: { lr: number; lambdaMax: number; onPick?: (lr: number) => void; height?: number }) {
  const [lo, hi] = LR_RANGE;
  const W = 600;
  const pad = 8;
  const x = (v: number) => pad + ((Math.log10(clamp(v, lo, hi)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (W - 2 * pad);
  const inv = (px: number) => 10 ** (Math.log10(lo) + ((px - pad) / (W - 2 * pad)) * (Math.log10(hi) - Math.log10(lo)));
  const a = 1 / lambdaMax;
  const b = 2 / lambdaMax;
  const regions = [
    { from: lo, to: a, color: "var(--viz-alt)", label: "no overshoot" },
    { from: a, to: b, color: "var(--viz-run-1)", label: "zigzag" },
    { from: b, to: hi, color: "var(--bad)", label: "diverges" },
  ].filter((r) => r.to > lo && r.from < hi);
  const barY = 6;
  const barH = 14;
  return (
    <svg
      viewBox={`0 0 ${W} ${height}`}
      className="block h-auto w-full"
      role="img"
      aria-label={`Learning rate ${fmt(lr, 3)}; no overshoot below ${fmt(a, 3)}, zigzag up to ${fmt(b, 3)}, divergence above.`}
      style={{ cursor: onPick ? "pointer" : undefined }}
      onClick={(e) => {
        if (!onPick) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const v = inv(((e.clientX - rect.left) / rect.width) * W);
        onPick(Math.round(clamp(v, lo, hi) * 1000) / 1000 || lo);
      }}
    >
      {regions.map((r) => (
        <g key={r.label}>
          <rect x={x(Math.max(lo, r.from))} y={barY} width={Math.max(0, x(Math.min(hi, r.to)) - x(Math.max(lo, r.from)))} height={barH} fill={r.color} fillOpacity={0.22} />
          {x(Math.min(hi, r.to)) - x(Math.max(lo, r.from)) > 70 && (
            <text x={(x(Math.max(lo, r.from)) + x(Math.min(hi, r.to))) / 2} y={barY + barH / 2} dy="0.35em" textAnchor="middle" fontSize={11} fill="var(--fg-2)">
              {r.label}
            </text>
          )}
        </g>
      ))}
      {[a, b].map((v, k) => (v > lo && v < hi ? (
        <g key={k}>
          <line x1={x(v)} x2={x(v)} y1={barY - 2} y2={barY + barH + 2} stroke="var(--fg-3)" />
          {/* 1/λmax is labelled to the left of its tick, 2/λmax to the right, so they never collide. */}
          <text x={x(v) + (k === 0 ? -4 : 4)} y={barY + barH + 13} textAnchor={k === 0 ? "end" : "start"} fontSize={10.5} fill="var(--fg-3)">
            {k === 0 ? "1/λmax" : "2/λmax"} = {fmt(v, 3)}
          </text>
        </g>
      ) : null))}
      <path transform={`translate(${x(lr)},${barY - 1})`} d="M-6,-5 L6,-5 L0,3z" fill="var(--fg)" />
      <line x1={x(lr)} x2={x(lr)} y1={barY} y2={barY + barH} stroke="var(--fg)" strokeWidth={2} />
    </svg>
  );
}
