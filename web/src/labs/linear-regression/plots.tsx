"use client";

import { useMemo } from "react";
import { Chart, linePath, type LegendItem } from "@/components/plot/Chart";
import { fmt } from "@/lib/format";
import { leastSquares, lossEllipse, mse, type Dataset } from "@/lib/ml/linreg";

const pad = ([a, b]: [number, number], f = 0.08): [number, number] => {
  const r = b - a || 1;
  return [a - r * f, b + r * f];
};
const extent = (v: number[]): [number, number] => [Math.min(...v), Math.max(...v)];

export interface FitLine {
  w: number;
  b: number;
  label: string;
  color: string;
  dashed?: boolean;
}

/** Scatter of (x, y) with model lines and residual segments. */
export function DataFitPlot({
  data,
  lines,
  residualFrom,
  selected,
  onSelect,
  height = 300,
}: {
  data: Dataset;
  lines: FitLine[];
  /** Predictions ŷ used to draw residuals (from the model, or from a traced forward pass). */
  residualFrom?: number[] | null;
  selected?: number | null;
  onSelect?: (i: number) => void;
  height?: number;
}) {
  const xDom = useMemo(() => pad(extent([0, ...data.x])), [data.x]);
  const yDom = useMemo(() => pad(extent([0, ...data.y]), 0.12), [data.y]);
  const legend: LegendItem[] = [
    { label: "data (xᵢ, yᵢ)", color: "var(--viz-data)", kind: "dot" },
    ...lines.map((l) => ({ label: l.label, color: l.color, kind: l.dashed ? ("dashed" as const) : ("line" as const) })),
    ...(residualFrom ? [{ label: "residual ŷᵢ − yᵢ", color: "var(--viz-residual)", kind: "line" as const }] : []),
  ];
  const hoverPts = data.x.map((xi, i) => ({
    x: xi,
    y: data.y[i],
    content: (
      <div className="font-mono tabular-nums">
        <div className="font-sans font-medium">point i = {i + 1}</div>
        <div>x = {fmt(xi, 3)}</div>
        <div>y = {fmt(data.y[i], 3)}</div>
        {residualFrom && (
          <>
            <div>ŷ = {fmt(residualFrom[i], 3)}</div>
            <div>r = {fmt(residualFrom[i] - data.y[i], 3)}</div>
          </>
        )}
        {onSelect && <div className="mt-0.5 font-sans text-fg-3">click to inspect</div>}
      </div>
    ),
  }));

  return (
    <Chart
      x={{ domain: xDom, label: "feature x" }}
      y={{ domain: yDom, label: "target y" }}
      height={height}
      legend={legend}
      ariaLabel={`Scatter plot of ${data.x.length} data points with fitted line`}
      hover={{ points: hoverPts }}
      onSelect={onSelect}
      table={{
        columns: residualFrom ? ["i", "x", "y", "ŷ", "residual"] : ["i", "x", "y"],
        rows: data.x.map((xi, i) => [i + 1, fmt(xi, 4), fmt(data.y[i], 4), ...(residualFrom ? [fmt(residualFrom[i], 4), fmt(residualFrom[i] - data.y[i], 4)] : [])]),
      }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          {residualFrom &&
            data.x.map((xi, i) => (
              <line key={`r${i}`} x1={s.x(xi)} x2={s.x(xi)} y1={s.y(data.y[i])} y2={s.y(residualFrom[i])} stroke="var(--viz-residual)" strokeWidth={1.5} opacity={selected == null || selected === i ? 1 : 0.45} />
            ))}
          {lines.map((l) => (
            <path
              key={l.label}
              d={linePath([[xDom[0], l.w * xDom[0] + l.b], [xDom[1], l.w * xDom[1] + l.b]], s)}
              stroke={l.color}
              strokeWidth={2}
              strokeDasharray={l.dashed ? "6 4" : undefined}
              fill="none"
            />
          ))}
          {data.x.map((xi, i) => (
            <circle key={i} cx={s.x(xi)} cy={s.y(data.y[i])} r={selected === i ? 5.5 : 4} fill="var(--viz-data)" stroke="var(--surface)" strokeWidth={1.5} />
          ))}
          {selected != null && selected < data.x.length && (
            <circle cx={s.x(data.x[selected])} cy={s.y(data.y[selected])} r={9} fill="none" stroke="var(--fg)" strokeWidth={1.5} />
          )}
        </g>
      )}
    </Chart>
  );
}

/** Training loss per epoch, with the least-squares minimum L* as reference. */
export function LossCurvePlot({
  losses,
  totalEpochs,
  minimum,
  current,
  log,
  height = 240,
}: {
  losses: { epoch: number; loss: number | null }[];
  totalEpochs: number;
  minimum?: number;
  current?: number | null;
  log?: boolean;
  height?: number;
}) {
  const finite = losses.filter((l): l is { epoch: number; loss: number } => l.loss != null && Number.isFinite(l.loss));
  const vals = finite.map((l) => l.loss);
  if (minimum != null) vals.push(minimum);
  let yDom: [number, number];
  if (log) {
    const pos = vals.filter((v) => v > 0);
    const lo = pos.length ? Math.min(...pos) : 0.1;
    const hi = pos.length ? Math.max(...pos) : 10;
    yDom = [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi * 1.01))];
    if (yDom[0] === yDom[1]) yDom[1] *= 10;
  } else {
    yDom = [0, (vals.length ? Math.max(...vals) : 1) * 1.08 || 1];
  }
  const xDom: [number, number] = [0, Math.max(1, totalEpochs - 1)];
  const cur = current != null ? finite.find((l) => l.epoch === current) : undefined;
  const legend: LegendItem[] = [{ label: "training loss L(w, b)", color: "var(--viz-model)", kind: "line" }];
  if (minimum != null) legend.push({ label: `least-squares minimum L* = ${fmt(minimum, 4)}`, color: "var(--viz-alt)", kind: "dashed" });

  return (
    <Chart
      x={{ domain: xDom, label: "epoch", format: (v) => String(Math.round(v)) }}
      y={{ domain: yDom, label: log ? "MSE (log scale)" : "MSE", log }}
      height={height}
      legend={legend}
      ariaLabel="Training loss by epoch"
      hover={{
        mode: "x",
        points: finite.map((l) => ({
          x: l.epoch,
          y: l.loss,
          content: (
            <div className="font-mono tabular-nums">
              epoch {l.epoch}
              <br />L = {fmt(l.loss, 5)}
            </div>
          ),
        })),
      }}
      table={{ columns: ["epoch", "loss"], rows: finite.map((l) => [l.epoch, fmt(l.loss, 6)]) }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          {minimum != null && (!log || minimum > 0) && <line x1={0} x2={s.width} y1={s.y(minimum)} y2={s.y(minimum)} stroke="var(--viz-alt)" strokeWidth={1.5} strokeDasharray="5 4" />}
          <path d={linePath(finite.map((l) => [l.epoch, l.loss]), s)} stroke="var(--viz-model)" strokeWidth={2} fill="none" />
          {cur && <circle cx={s.x(cur.epoch)} cy={s.y(cur.loss)} r={4.5} fill="var(--viz-model)" stroke="var(--surface)" strokeWidth={2} />}
        </g>
      )}
    </Chart>
  );
}

/**
 * Exact loss landscape in parameter space. Level sets of the quadratic MSE are ellipses,
 * computed analytically from the data (see lossEllipse). Darker bands = lower loss.
 */
export function LossSurfacePlot({
  data,
  trajectory,
  current,
  next,
  anchor,
  height = 300,
}: {
  data: Dataset;
  trajectory: [number, number][];
  current?: [number, number] | null;
  /** Upcoming update θ − η∇L, drawn as an arrow. */
  next?: [number, number] | null;
  /** Point that must stay in view (e.g. initial parameters). */
  anchor?: [number, number];
  height?: number;
}) {
  const opt = useMemo(() => leastSquares(data), [data]);
  const { wDom, bDom } = useMemo(() => {
    const pts = [[opt.w, opt.b], ...(anchor ? [anchor] : []), ...trajectory.slice(0, 1)].filter((p) => p.every(Number.isFinite));
    let rw = Math.max(1, ...pts.map((p) => Math.abs(p[0] - opt.w)));
    let rb = Math.max(1, ...pts.map((p) => Math.abs(p[1] - opt.b)));
    rw *= 1.35;
    rb *= 1.35;
    return { wDom: [opt.w - rw, opt.w + rw] as [number, number], bDom: [opt.b - rb, opt.b + rb] as [number, number] };
  }, [opt, anchor, trajectory]);

  const levels = useMemo(() => {
    const corners = [mse(wDom[0], bDom[0], data), mse(wDom[0], bDom[1], data), mse(wDom[1], bDom[0], data), mse(wDom[1], bDom[1], data)];
    const maxEx = Math.max(...corners) - opt.loss;
    return Array.from({ length: 14 }, (_, k) => opt.loss + maxEx * 0.6 ** k);
  }, [data, wDom, bDom, opt]);
  const ellipses = useMemo(() => levels.map((l) => ({ level: l, pts: lossEllipse(data, l) })), [levels, data]);

  const legend: LegendItem[] = [
    { label: "loss level sets", color: "color-mix(in srgb, var(--viz-model) 45%, transparent)", kind: "area" },
    { label: "least-squares θ*", color: "var(--viz-alt)", kind: "dot" },
    { label: "path of (w, b)", color: "var(--fg-2)", kind: "line" },
  ];
  if (current) legend.push({ label: "current (w, b)", color: "var(--fg)", kind: "dot" });
  if (next) legend.push({ label: "next update −η∇L", color: "var(--viz-residual)", kind: "arrow" });

  const finiteTraj = trajectory.filter((p) => p.every(Number.isFinite));
  return (
    <Chart
      x={{ domain: wDom, label: "slope w" }}
      y={{ domain: bDom, label: "intercept b" }}
      height={height}
      legend={legend}
      ariaLabel="Loss surface over slope and intercept with optimization path"
      hover={{
        points: [
          { x: opt.w, y: opt.b, content: <div className="font-mono tabular-nums"><div className="font-sans font-medium">least squares θ*</div>w = {fmt(opt.w)}<br />b = {fmt(opt.b)}<br />L* = {fmt(opt.loss)}</div> },
          ...finiteTraj.map(([w, b], i) => ({
            x: w,
            y: b,
            content: (
              <div className="font-mono tabular-nums">
                <div className="font-sans font-medium">step {i}</div>w = {fmt(w)}
                <br />b = {fmt(b)}
                <br />L = {fmt(mse(w, b, data))}
              </div>
            ),
          })),
        ],
      }}
    >
      {(s) => {
        let arrow = null;
        if (current && next && current.every(Number.isFinite) && next.every(Number.isFinite)) {
          const x1 = s.x(current[0]), y1 = s.y(current[1]), x2 = s.x(next[0]), y2 = s.y(next[1]);
          const len = Math.hypot(x2 - x1, y2 - y1);
          if (len > 4) {
            const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
            const hx = x2 - ux * 8, hy = y2 - uy * 8;
            arrow = (
              <g>
                <line x1={x1} y1={y1} x2={hx} y2={hy} stroke="var(--viz-residual)" strokeWidth={2} />
                <path d={`M${x2},${y2} L${hx - uy * 4.5},${hy + ux * 4.5} L${hx + uy * 4.5},${hy - ux * 4.5}z`} fill="var(--viz-residual)" />
              </g>
            );
          }
        }
        return (
          <g clipPath={`url(#${s.clipId})`}>
            {ellipses.map((e) => (
              <path key={e.level} d={linePath(e.pts, s) + "Z"} fill="var(--viz-model)" fillOpacity={0.07} stroke="var(--viz-model)" strokeOpacity={0.3} strokeWidth={1} />
            ))}
            <path d={linePath(finiteTraj, s)} stroke="var(--fg-2)" strokeWidth={1.5} fill="none" />
            {finiteTraj.length < 260 && finiteTraj.map(([w, b], i) => <circle key={i} cx={s.x(w)} cy={s.y(b)} r={2} fill="var(--fg-2)" />)}
            <path transform={`translate(${s.x(opt.w)},${s.y(opt.b)})`} d="M0,-6 L6,0 L0,6 L-6,0z" fill="var(--viz-alt)" stroke="var(--surface)" strokeWidth={1.5} />
            {arrow}
            {current && current.every(Number.isFinite) && <circle cx={s.x(current[0])} cy={s.y(current[1])} r={5.5} fill="var(--fg)" stroke="var(--surface)" strokeWidth={2} />}
          </g>
        );
      }}
    </Chart>
  );
}
