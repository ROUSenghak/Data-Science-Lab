"use client";

import { useMemo, useState } from "react";
import { FocusProvider, LinkedMath, LTexBlock } from "@/components/anim/Focus";
import { Tex } from "@/components/math/Tex";
import { Chart } from "@/components/plot/Chart";
import { Button, Slider, Toggle } from "@/components/ui/controls";
import { hl } from "@/lib/anim/focus";
import { fmt, texNum, texSq } from "@/lib/format";
import { leastSquares, mse, predict, type Dataset } from "@/lib/ml/linreg";
import { DataFitPlot, LossSurfacePlot } from "./plots";

export { GradientDescentExplorer } from "./GradientDescentExplorer";

/* ------------------------------------------------------------------ fit by hand */

export function FitExplorer(props: { data: Dataset; compact?: boolean }) {
  return (
    <FocusProvider>
      <FitExplorerInner {...props} />
    </FocusProvider>
  );
}

function FitExplorerInner({ data, compact = false }: { data: Dataset; compact?: boolean }) {
  const [allSquares, setAllSquares] = useState(false);
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
          squares={allSquares ? true : "focus"}
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
        <LinkedMath className="rounded-md border border-line bg-surface-2 px-3 py-1 text-[15px]">
          <LTexBlock>{`\\begin{aligned} L(w,b) &= \\frac{1}{${n}}\\sum_{i=1}^{${n}} r_i^2 \\\\ &= \\frac{1}{${n}}\\big(${[0, 1, 2].filter((i) => i < n).map((i) => hl({ kind: "residual", i }, texSq(r[i], 2))).join(" + ")} + \\cdots\\big) = ${hl({ kind: "loss" }, texNum(loss, 4))} \\end{aligned}`}</LTexBlock>
          <div className="pb-1 text-[12px] text-fg-3">
            Residual <Tex>{"r_i=\\hat y_i-y_i"}</Tex> (orange); each square has area <Tex>{"\\propto r_i^2"}</Tex>, so <Tex>L</Tex> is their mean area. Hover a term to find its point. Least-squares minimum:{" "}
            <span className="font-mono">L* = {fmt(ls.loss, 4)}</span>
            {" · "}you are {fmt(loss - ls.loss, 3)} above it.
          </div>
        </LinkedMath>
        <div className="flex flex-wrap items-center gap-2">
          <Toggle label="all squared errors" checked={allSquares} onChange={setAllSquares} />
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
