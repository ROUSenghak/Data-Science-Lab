"use client";

import { useMemo } from "react";
import { useFocus } from "@/components/anim/Focus";
import { Chart, linePath, type LegendItem, type Scales } from "@/components/plot/Chart";
import { focusedAxis, focusedResidual } from "@/lib/anim/focus";
import { fmt } from "@/lib/format";
import { eigSym2, hessian, leastSquares, lossEllipse, mse, type Dataset } from "@/lib/ml/linreg";
import { autoDomain, contourLevels } from "@/lib/ml/lossLandscape";

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
  squares = false,
  height = 300,
}: {
  data: Dataset;
  lines: FitLine[];
  /** Predictions ŷ used to draw residuals (from the model, or from a traced forward pass). */
  residualFrom?: number[] | null;
  selected?: number | null;
  onSelect?: (i: number) => void;
  /**
   * Draw squared residuals as squares of side |rᵢ| (in y pixels), so area ∝ rᵢ².
   * "focus": only the hovered/focused residual's square (all of them when the loss is focused).
   */
  squares?: boolean | "focus";
  height?: number;
}) {
  const { focus, setFocus } = useFocus();
  const fr = residualFrom ? focusedResidual(focus) : null;
  const lossFocus = focus?.kind === "loss";
  const xDom = useMemo(() => pad(extent([0, ...data.x])), [data.x]);
  const yDom = useMemo(() => pad(extent([0, ...data.y]), 0.12), [data.y]);
  const legend: LegendItem[] = [
    { label: "data (xᵢ, yᵢ)", color: "var(--viz-data)", kind: "dot" },
    ...lines.map((l) => ({ label: l.label, color: l.color, kind: l.dashed ? ("dashed" as const) : ("line" as const) })),
    ...(residualFrom ? [{ label: "residual ŷᵢ − yᵢ", color: "var(--viz-residual)", kind: "line" as const }] : []),
    ...(residualFrom && squares === true ? [{ label: "squared residual (area ∝ rᵢ²)", color: "color-mix(in srgb, var(--viz-residual) 30%, transparent)", kind: "area" as const }] : []),
  ];
  const dim = (i: number) => (fr == null ? (selected == null || selected === i ? 1 : 0.45) : fr === i ? 1 : 0.25);
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
      onHoverChange={residualFrom ? (i) => setFocus(i == null ? null : { kind: "residual", i }) : undefined}
      onSelect={onSelect}
      table={{
        columns: residualFrom ? ["i", "x", "y", "ŷ", "residual"] : ["i", "x", "y"],
        rows: data.x.map((xi, i) => [i + 1, fmt(xi, 4), fmt(data.y[i], 4), ...(residualFrom ? [fmt(residualFrom[i], 4), fmt(residualFrom[i] - data.y[i], 4)] : [])]),
      }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          {residualFrom && squares && <ResidualSquares data={data} yHat={residualFrom} s={s} focus={fr} all={lossFocus} only={squares === "focus"} />}
          {residualFrom &&
            data.x.map((xi, i) => (
              <line key={`r${i}`} x1={s.x(xi)} x2={s.x(xi)} y1={s.y(data.y[i])} y2={s.y(residualFrom[i])} stroke="var(--viz-residual)" strokeWidth={fr === i ? 3 : 1.5} opacity={dim(i)} />
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
          {fr != null && fr < data.x.length && fr !== selected && <circle cx={s.x(data.x[fr])} cy={s.y(data.y[fr])} r={8} fill="none" stroke="var(--viz-residual)" strokeWidth={1.5} />}
        </g>
      )}
    </Chart>
  );
}

/**
 * Squares on the residual segments. The side is the residual's length in y pixels, so every
 * square is drawn at the same scale and its area is proportional to rᵢ²: the loss is their mean area.
 */
function ResidualSquares({ data, yHat, s, focus, all, only }: { data: Dataset; yHat: number[]; s: Scales; focus: number | null; all: boolean; only: boolean }) {
  if (only && focus == null && !all) return null;
  return (
    <g>
      {data.x.map((xi, i) => {
        const px = s.x(xi);
        const y1 = s.y(data.y[i]);
        const y2 = s.y(yHat[i]);
        const side = Math.abs(y2 - y1);
        if (!Number.isFinite(side) || side < 0.5) return null;
        if (only && !all && focus !== i) return null;
        // Open towards the side with more room, so squares near the right edge stay visible.
        const left = px + side > s.width && px - side >= 0;
        const on = focus === i || all;
        return (
          <rect
            key={`sq${i}`}
            x={left ? px - side : px}
            y={Math.min(y1, y2)}
            width={side}
            height={side}
            fill="var(--viz-residual)"
            fillOpacity={focus === i ? 0.3 : on ? 0.07 : focus == null ? 0.06 : 0.03}
            stroke="var(--viz-residual)"
            strokeOpacity={focus === i ? 0.95 : on ? 0.5 : focus == null ? 0.3 : 0.1}
            strokeWidth={1}
          />
        );
      })}
    </g>
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
export interface ParamPath {
  points: [number, number][];
  color: string;
  label: string;
  /** Current position on this path. */
  current?: [number, number] | null;
}

export function LossSurfacePlot({
  data,
  trajectory,
  current,
  next,
  anchor,
  height = 300,
  domain,
  equalAspect = false,
  components = false,
  levelThrough = false,
  eigenAxes = false,
  paths,
  start,
  onStartChange,
}: {
  data: Dataset;
  trajectory: [number, number][];
  current?: [number, number] | null;
  /** Upcoming update θ − η∇L, drawn as an arrow. */
  next?: [number, number] | null;
  /** Point that must stay in view (e.g. initial parameters). */
  anchor?: [number, number];
  height?: number;
  /** Fixed axes (so the view does not rescale while parameters change). */
  domain?: { w: [number, number]; b: [number, number] };
  /** One unit of w as long as one unit of b: angles are true, so −∇L is visibly perpendicular to the level sets. */
  equalAspect?: boolean;
  /** Split the update arrow into its w and b components (−η ∂L/∂w, −η ∂L/∂b). */
  components?: boolean;
  /** Highlight the level set through the current point. */
  levelThrough?: boolean;
  /** Draw H's eigen-directions through θ* (steep = λ_max, flat = λ_min). */
  eigenAxes?: boolean;
  /** Further optimization paths on the same surface (comparisons). */
  paths?: ParamPath[];
  /** Draggable initial point θ₀. */
  start?: [number, number];
  onStartChange?: (w: number, b: number) => void;
}) {
  const { focus } = useFocus();
  const axisFocus = focusedAxis(focus);
  const opt = useMemo(() => leastSquares(data), [data]);
  const eig = useMemo(() => eigSym2(hessian(data)), [data]);
  const first = trajectory[0];
  const { w: wDom, b: bDom } = useMemo(() => domain ?? autoDomain(data, anchor, first), [data, anchor, first, domain]);
  const levels = useMemo(() => contourLevels(data, { w: wDom, b: bDom }), [data, wDom, bDom]);
  const ellipses = useMemo(() => levels.map((l) => ({ level: l, pts: lossEllipse(data, l) })), [levels, data]);
  const curLevel = levelThrough && current && current.every(Number.isFinite) ? mse(current[0], current[1], data) : null;
  const curEllipse = useMemo(() => (curLevel != null ? lossEllipse(data, curLevel, 160) : []), [curLevel, data]);

  const legend: LegendItem[] = [
    { label: "loss level sets", color: "color-mix(in srgb, var(--viz-model) 45%, transparent)", kind: "area" },
    { label: "least-squares θ*", color: "var(--viz-alt)", kind: "dot" },
    { label: "path of (w, b)", color: "var(--fg-2)", kind: "line" },
  ];
  if (current) legend.push({ label: "current (w, b)", color: "var(--fg)", kind: "dot" });
  if (curEllipse.length) legend.push({ label: "level set through current θ", color: "var(--viz-model)", kind: "line" });
  if (next) legend.push({ label: "next update −η∇L", color: "var(--viz-residual)", kind: "arrow" });
  for (const p of paths ?? []) legend.push({ label: p.label, color: p.color, kind: "line" });
  if (start && onStartChange) legend.push({ label: "θ₀ (drag me)", color: "var(--fg)", kind: "ring" });

  const finiteTraj = trajectory.filter((p) => p.every(Number.isFinite));
  return (
    <Chart
      x={{ domain: wDom, label: "slope w" }}
      y={{ domain: bDom, label: "intercept b" }}
      height={height}
      legend={legend}
      aspect={equalAspect ? "equal" : undefined}
      ariaLabel="Loss surface over slope and intercept with optimization path"
      onPlotClick={onStartChange ? (w, b) => onStartChange(w, b) : undefined}
      drag={start && onStartChange ? { points: [{ x: start[0], y: start[1] }], onDrag: (_, w, b) => onStartChange(w, b) } : undefined}
      hover={{
        points: [
          { x: opt.w, y: opt.b, content: <div className="font-mono tabular-nums"><div className="font-sans font-medium">least squares θ*</div>w = {fmt(opt.w)}<br />b = {fmt(opt.b)}<br />L* = {fmt(opt.loss)}</div> },
          ...finiteTraj.map(([w, b], i) => ({
            x: w,
            y: b,
            content: (
              <div className="font-mono tabular-nums">
                <div className="font-sans font-medium">θ at epoch {i}</div>w = {fmt(w)}
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
          const emph = axisFocus == null ? 1 : 0.35;
          arrow = (
            <g>
              {components && (
                <>
                  <Arrow x1={x1} y1={y1} x2={x2} y2={y1} color="var(--viz-model)" width={axisFocus === "w" ? 3 : 1.5} dash={axisFocus === "w" ? undefined : "4 3"} opacity={axisFocus === "b" ? 0.3 : 1} label="−η ∂L/∂w" labelPos="below" />
                  <Arrow x1={x2} y1={y1} x2={x2} y2={y2} color="var(--viz-model)" width={axisFocus === "b" ? 3 : 1.5} dash={axisFocus === "b" ? undefined : "4 3"} opacity={axisFocus === "w" ? 0.3 : 1} label="−η ∂L/∂b" labelPos="side" />
                </>
              )}
              <Arrow x1={x1} y1={y1} x2={x2} y2={y2} color="var(--viz-residual)" width={2.25} opacity={emph} />
            </g>
          );
        }
        const axisLen = Math.hypot(wDom[1] - wDom[0], bDom[1] - bDom[0]);
        return (
          <g clipPath={`url(#${s.clipId})`}>
            {ellipses.map((e) => (
              <path key={e.level} d={linePath(e.pts, s) + "Z"} fill="var(--viz-model)" fillOpacity={0.07} stroke="var(--viz-model)" strokeOpacity={0.3} strokeWidth={1} />
            ))}
            {eigenAxes &&
              eig.vectors.map((v, k) => (
                <g key={`ax${k}`}>
                  <line
                    x1={s.x(opt.w - v[0] * axisLen)}
                    y1={s.y(opt.b - v[1] * axisLen)}
                    x2={s.x(opt.w + v[0] * axisLen)}
                    y2={s.y(opt.b + v[1] * axisLen)}
                    stroke="var(--fg-3)"
                    strokeWidth={1}
                    strokeDasharray="2 4"
                  />
                </g>
              ))}
            {curEllipse.length > 0 && <path d={linePath(curEllipse, s) + "Z"} fill="none" stroke="var(--viz-model)" strokeWidth={1.75} />}
            {paths?.map((p) => {
              const fin = p.points.filter((q) => q.every(Number.isFinite));
              return (
                <g key={p.label}>
                  <path d={linePath(fin, s)} stroke={p.color} strokeWidth={1.75} fill="none" />
                  {fin.length < 260 && fin.map(([w, b], i) => <circle key={i} cx={s.x(w)} cy={s.y(b)} r={1.8} fill={p.color} />)}
                  {p.current && p.current.every(Number.isFinite) && <circle cx={s.x(p.current[0])} cy={s.y(p.current[1])} r={5.5} fill={p.color} stroke="var(--surface)" strokeWidth={2} />}
                </g>
              );
            })}
            <path d={linePath(finiteTraj, s)} stroke="var(--fg-2)" strokeWidth={1.5} fill="none" />
            {finiteTraj.length < 260 && finiteTraj.map(([w, b], i) => <circle key={i} cx={s.x(w)} cy={s.y(b)} r={2} fill="var(--fg-2)" />)}
            <path transform={`translate(${s.x(opt.w)},${s.y(opt.b)})`} d="M0,-6 L6,0 L0,6 L-6,0z" fill="var(--viz-alt)" stroke="var(--surface)" strokeWidth={1.5} />
            {arrow}
            {start && onStartChange && start.every(Number.isFinite) && <circle cx={s.x(start[0])} cy={s.y(start[1])} r={8} fill="none" stroke="var(--fg)" strokeWidth={1.5} strokeDasharray="3 2" />}
            {current && current.every(Number.isFinite) && <circle cx={s.x(current[0])} cy={s.y(current[1])} r={5.5} fill="var(--fg)" stroke="var(--surface)" strokeWidth={2} />}
            {eigenAxes && <EigenLabels opt={opt} vectors={eig.vectors} s={s} />}
          </g>
        );
      }}
    </Chart>
  );
}

/** Arrow in pixel coordinates, with an optional label. Skipped when shorter than its head. */
function Arrow({
  x1,
  y1,
  x2,
  y2,
  color,
  width,
  dash,
  opacity = 1,
  label,
  labelPos,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  width: number;
  dash?: string;
  opacity?: number;
  label?: string;
  labelPos?: "below" | "side";
}) {
  const len = Math.hypot(x2 - x1, y2 - y1);
  if (!Number.isFinite(len) || len < 6) return null;
  const ux = (x2 - x1) / len;
  const uy = (y2 - y1) / len;
  const hx = x2 - ux * 8;
  const hy = y2 - uy * 8;
  const showLabel = label && len > 70;
  return (
    <g opacity={opacity}>
      <line x1={x1} y1={y1} x2={hx} y2={hy} stroke={color} strokeWidth={width} strokeDasharray={dash} />
      <path d={`M${x2},${y2} L${hx - uy * 4.5},${hy + ux * 4.5} L${hx + uy * 4.5},${hy - ux * 4.5}z`} fill={color} />
      {showLabel && (
        <text
          x={labelPos === "side" ? x1 + 7 : (x1 + x2) / 2}
          y={labelPos === "side" ? (y1 + y2) / 2 : y1 + 15}
          textAnchor={labelPos === "side" ? "start" : "middle"}
          dy={labelPos === "side" ? "0.32em" : undefined}
          fontSize={11}
          fill={color}
          stroke="var(--surface)"
          strokeWidth={3}
          paintOrder="stroke"
        >
          {label}
        </text>
      )}
    </g>
  );
}

function EigenLabels({ opt, vectors, s }: { opt: { w: number; b: number }; vectors: [[number, number], [number, number]]; s: Scales }) {
  // Place each label where its axis meets the plot border, on the side with more room.
  return (
    <g>
      {vectors.map((v, k) => {
        const cx = s.x(opt.w);
        const cy = s.y(opt.b);
        const dx = s.x(opt.w + v[0]) - cx;
        const dy = s.y(opt.b + v[1]) - cy;
        const n = Math.hypot(dx, dy) || 1;
        const ux = dx / n;
        const uy = dy / n;
        let t = Infinity;
        if (ux > 0) t = Math.min(t, (s.width - 8 - cx) / ux);
        if (ux < 0) t = Math.min(t, (8 - cx) / ux);
        if (uy > 0) t = Math.min(t, (s.height - 8 - cy) / uy);
        if (uy < 0) t = Math.min(t, (8 - cy) / uy);
        if (!Number.isFinite(t) || t < 30) return null;
        const r = Math.max(30, t - 34);
        return (
          <text key={k} x={cx + ux * r} y={cy + uy * r} textAnchor="middle" dy="0.32em" fontSize={10.5} fill="var(--fg-3)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">
            {k === 0 ? "flat (λmin)" : "steep (λmax)"}
          </text>
        );
      })}
    </g>
  );
}
