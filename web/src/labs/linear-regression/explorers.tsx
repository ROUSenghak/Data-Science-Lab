"use client";

import { useEffect, useMemo, useState } from "react";
import { Tex, TexBlock } from "@/components/math/Tex";
import { Chart } from "@/components/plot/Chart";
import { Button, Icon, Segmented, Slider } from "@/components/ui/controls";
import { Callout } from "@/components/ui/primitives";
import { fmt, texNum } from "@/lib/format";
import { gradientDescent, leastSquares, mse, predict, stability, type Dataset } from "@/lib/ml/linreg";
import { DataFitPlot, LossCurvePlot, LossSurfacePlot } from "./plots";

/* ------------------------------------------------------------------ fit by hand */

export function FitExplorer({ data, compact = false }: { data: Dataset; compact?: boolean }) {
  const [w, setW] = useState(0.5);
  const [b, setB] = useState(0.5);
  const [showLS, setShowLS] = useState(false);
  const ls = useMemo(() => leastSquares(data), [data]);
  const yHat = predict(w, b, data.x);
  const loss = mse(w, b, data);
  const n = data.x.length;
  const r = yHat.map((v, i) => v - data.y[i]);

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="min-w-0">
        <DataFitPlot
          data={data}
          residualFrom={yHat}
          lines={[
            { w, b, label: "your line ŷ = wx + b", color: "var(--viz-model)" },
            ...(showLS ? [{ w: ls.w, b: ls.b, label: "least-squares fit", color: "var(--viz-alt)", dashed: true }] : []),
          ]}
          height={compact ? 260 : 300}
        />
      </div>
      <div className="min-w-0 space-y-3">
        <div className="grid grid-cols-2 gap-4">
          <Slider label={<Tex>{"\\text{slope } w"}</Tex>} value={w} min={-2} max={5} step={0.01} onChange={setW} format={(v) => fmt(v, 2)} />
          <Slider label={<Tex>{"\\text{intercept } b"}</Tex>} value={b} min={-3} max={5} step={0.01} onChange={setB} format={(v) => fmt(v, 2)} />
        </div>
        <div className="rounded-md border border-line bg-surface-2 px-3 py-1 text-[15px]">
          <TexBlock>{`\\begin{aligned} L(w,b) &= \\frac{1}{${n}}\\sum_{i=1}^{${n}} r_i^2 \\\\ &= \\frac{1}{${n}}\\big(${texNum(r[0], 2, true)}^2 + ${texNum(r[1], 2, true)}^2 + \\cdots\\big) = ${texNum(loss, 4)} \\end{aligned}`}</TexBlock>
          <div className="pb-1 text-[12px] text-fg-3">
            Residual <Tex>{"r_i=\\hat y_i-y_i"}</Tex> (orange). Least-squares minimum: <span className="font-mono">L* = {fmt(ls.loss, 4)}</span>
            {" · "}you are {fmt(loss - ls.loss, 3)} above it.
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => setShowLS((s) => !s)}>{showLS ? "Hide" : "Show"} least-squares line</Button>
          <Button
            variant="ghost"
            onClick={() => {
              setW(Math.round(ls.w * 100) / 100);
              setB(Math.round(ls.b * 100) / 100);
            }}
          >
            Snap to least squares
          </Button>
        </div>
        {!compact && <LossSurfacePlot data={data} trajectory={[]} current={[w, b]} height={240} anchor={[w, b]} />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ gradient descent */

export function GradientDescentExplorer({ data, defaultLR = 0.08 }: { data: Dataset; defaultLR?: number }) {
  const [lr, setLR] = useState(defaultLR);
  const [epochs, setEpochs] = useState(100);
  const [w0, setW0] = useState(0);
  const [b0, setB0] = useState(-1);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [log, setLog] = useState(false);
  const [speed, setSpeed] = useState<"1" | "4" | "16">("4");

  const steps = useMemo(() => gradientDescent(data, { lr, epochs, w0, b0 }), [data, lr, epochs, w0, b0]);
  const stab = useMemo(() => stability(data), [data]);
  const ls = useMemo(() => leastSquares(data), [data]);
  const last = steps.length - 1;
  const ti = Math.min(t, last);
  const s = steps[ti];

  // Reset the animation when the experiment changes (state adjustment during render).
  const [prevSteps, setPrevSteps] = useState(steps);
  if (prevSteps !== steps) {
    setPrevSteps(steps);
    setT(0);
    setPlaying(false);
  }

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setT((v) => {
        if (v >= last) {
          setPlaying(false);
          return v;
        }
        return Math.min(last, v + Number(speed));
      });
    }, 60);
    return () => window.clearInterval(id);
  }, [playing, last, speed]);

  const trajectory = steps.slice(0, ti + 1).map((st) => [st.w, st.b] as [number, number]);
  const diverged = !Number.isFinite(steps[last].wNext) || Math.abs(steps[last].loss) > 1e8;
  const ratio = lr / stab.criticalLR;
  const fastFactor = Math.abs(1 - lr * stab.lambdaMax);
  const slowFactor = Math.abs(1 - lr * stab.lambdaMin);
  const stepsTo10x = slowFactor < 1 ? Math.ceil(Math.log(10) / -Math.log(slowFactor)) : Infinity;
  const yHat = predict(s.w, s.b, data.x);

  return (
    <div className="space-y-4">
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
        <Slider
          label={<Tex>{"\\text{learning rate } \\eta"}</Tex>}
          value={lr}
          min={0.001}
          max={0.5}
          step={0.001}
          scale="log"
          onChange={setLR}
          format={(v) => fmt(v, 3)}
          hint={<>stable iff η &lt; 2/λ<sub>max</sub> = {fmt(stab.criticalLR, 3)}</>}
        />
        <Slider label="epochs" value={epochs} min={5} max={400} step={5} onChange={setEpochs} />
        <Slider label={<Tex>{"w_0"}</Tex>} value={w0} min={-3} max={6} step={0.1} onChange={setW0} format={(v) => fmt(v, 1)} />
        <Slider label={<Tex>{"b_0"}</Tex>} value={b0} min={-4} max={6} step={0.1} onChange={setB0} format={(v) => fmt(v, 1)} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" onClick={() => (ti >= last ? (setT(0), setPlaying(true)) : setPlaying((p) => !p))} ariaLabel={playing ? "Pause" : "Play"}>
          {playing ? Icon.pause : Icon.play} {playing ? "Pause" : ti >= last ? "Replay" : "Play"}
        </Button>
        <Button onClick={() => setT((v) => Math.max(0, v - 1))} disabled={ti === 0} ariaLabel="Step back">
          {Icon.back}
        </Button>
        <Button onClick={() => setT((v) => Math.min(last, v + 1))} disabled={ti >= last} ariaLabel="Step forward">
          {Icon.step} Step
        </Button>
        <Button variant="ghost" onClick={() => (setPlaying(false), setT(0))} ariaLabel="Reset">
          {Icon.reset} Reset
        </Button>
        <Segmented ariaLabel="Playback speed" value={speed} onChange={setSpeed} options={[{ value: "1", label: "1×" }, { value: "4", label: "4×" }, { value: "16", label: "16×" }]} />
        <input
          type="range"
          aria-label="Epoch"
          className="min-w-32 flex-1"
          min={0}
          max={last}
          value={ti}
          onChange={(e) => (setPlaying(false), setT(Number(e.target.value)))}
        />
        <span className="font-mono text-[12px] tabular-nums text-fg-2">
          epoch {s.epoch} / {epochs - 1}
        </span>
      </div>

      {ratio >= 1 ? (
        <Callout tone="bad" title="Diverging: the learning rate exceeds the stability limit">
          η = {fmt(lr, 3)} &gt; 2/λ<sub>max</sub> = {fmt(stab.criticalLR, 3)}. Along the steepest direction of the loss, each step multiplies the error by |1 − ηλ<sub>max</sub>| = {fmt(fastFactor, 2)} &gt; 1, so it grows geometrically.
          {diverged && last < epochs - 1 && <> Stopped at epoch {last}: values overflowed.</>}
        </Callout>
      ) : ratio > 0.5 ? (
        <Callout tone="warn" title="Oscillating along the steep direction">
          1 − ηλ<sub>max</sub> = {fmt(1 - lr * stab.lambdaMax, 2)} &lt; 0: each step overshoots the valley floor and flips sides, while still shrinking the error.
        </Callout>
      ) : stepsTo10x > epochs ? (
        <Callout tone="note" title="Slow convergence along the flat direction">
          The loss surface is elongated (condition number κ = {fmt(stab.condition, 1)}). Along its flattest direction the error shrinks by only {fmt(slowFactor, 3)} per step, so a 10× reduction needs ≈ {stepsTo10x} epochs.
        </Callout>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <DataFitPlot data={data} residualFrom={yHat} lines={[{ w: s.w, b: s.b, label: `model at epoch ${s.epoch}`, color: "var(--viz-model)" }]} height={270} />
        <LossSurfacePlot data={data} trajectory={trajectory} current={[s.w, s.b]} next={[s.wNext, s.bNext]} anchor={[w0, b0]} height={270} />
        <div className="min-w-0">
          <div className="mb-1 flex justify-end">
            <Segmented ariaLabel="Loss axis scale" value={log ? "log" : "lin"} onChange={(v) => setLog(v === "log")} options={[{ value: "lin", label: "linear" }, { value: "log", label: "log" }]} />
          </div>
          <LossCurvePlot losses={steps.slice(0, ti + 1).map((st) => ({ epoch: st.epoch, loss: st.loss }))} totalEpochs={epochs} minimum={ls.loss} current={s.epoch} log={log || ratio >= 1} height={240} />
        </div>
        <div className="min-w-0 rounded-md border border-line bg-surface-2 px-3 py-2 text-[14px]">
          <div className="text-[12px] font-medium text-fg-2">Update at epoch {s.epoch} (values from this run)</div>
          <TexBlock>{`\\frac{\\partial L}{\\partial w} = \\frac{2}{n}\\sum_i r_i x_i = ${texNum(s.dw)},\\quad \\frac{\\partial L}{\\partial b} = \\frac{2}{n}\\sum_i r_i = ${texNum(s.db)}`}</TexBlock>
          <TexBlock>{`w \\leftarrow ${texNum(s.w)} - ${texNum(lr, 3)}\\cdot${texNum(s.dw, 4, true)} = ${texNum(s.wNext)}`}</TexBlock>
          <TexBlock>{`b \\leftarrow ${texNum(s.b)} - ${texNum(lr, 3)}\\cdot${texNum(s.db, 4, true)} = ${texNum(s.bNext)}`}</TexBlock>
          <div className="text-[12px] text-fg-3">
            L = {fmt(s.loss, 5)} · Hessian eigenvalues λ = {fmt(stab.lambdaMin, 3)}, {fmt(stab.lambdaMax, 3)} · κ = {fmt(stab.condition, 1)}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ residual diagnostics */

export function ResidualPlot({ data, height = 220 }: { data: Dataset; height?: number }) {
  const ls = leastSquares(data);
  const r = data.x.map((xi, i) => data.y[i] - (ls.w * xi + ls.b));
  const m = Math.max(1, ...r.map(Math.abs)) * 1.15;
  const xmax = Math.max(...data.x);
  return (
    <Chart
      x={{ domain: [Math.min(0, ...data.x), xmax + 0.2], label: "feature x" }}
      y={{ domain: [-m, m], label: "residual y − ŷ" }}
      height={height}
      ariaLabel="Residuals of the least-squares fit against x"
      hover={{ points: data.x.map((xi, i) => ({ x: xi, y: r[i], content: <div className="font-mono tabular-nums">x = {fmt(xi, 3)}<br />y − ŷ = {fmt(r[i], 3)}</div> })) }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          <line x1={0} x2={s.width} y1={s.y(0)} y2={s.y(0)} stroke="var(--viz-axis)" />
          {data.x.map((xi, i) => (
            <circle key={i} cx={s.x(xi)} cy={s.y(r[i])} r={4} fill="var(--viz-residual)" stroke="var(--surface)" strokeWidth={1.5} />
          ))}
        </g>
      )}
    </Chart>
  );
}
