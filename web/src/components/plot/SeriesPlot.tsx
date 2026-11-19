"use client";

import { Chart, linePath, type LegendItem } from "./Chart";
import { fmt } from "@/lib/format";

export interface Series {
  label: string;
  color: string;
  /** (x, y) pairs; null or non-finite y breaks the line. */
  points: [number, number | null][];
  dashed?: boolean;
  /** Draw thinner and lighter (e.g. earlier runs in a comparison). */
  muted?: boolean;
}

export interface RefLine {
  label: string;
  value: number;
  color: string;
}

/**
 * Line chart of one or more series over a shared x axis (epochs, iterations, …).
 * One y axis only; series with different units belong in separate plots.
 */
export function SeriesPlot({
  series,
  refs = [],
  xLabel,
  yLabel,
  log = false,
  includeZero = false,
  current,
  height = 260,
  ariaLabel,
  digits = 5,
  xDomain,
}: {
  series: Series[];
  refs?: RefLine[];
  xLabel: string;
  yLabel: string;
  log?: boolean;
  includeZero?: boolean;
  /** x value of the selected step, drawn as a vertical marker. */
  current?: number | null;
  height?: number;
  ariaLabel: string;
  digits?: number;
  /** Fixed x range, so the axis does not rescale while a series is revealed step by step. */
  xDomain?: [number, number];
}) {
  const finite = (v: number | null | undefined): v is number => v != null && Number.isFinite(v) && (!log || v > 0);
  const ys = [...series.flatMap((s) => s.points.map((p) => p[1])), ...refs.map((r) => r.value)].filter(finite);
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);

  let yDom: [number, number];
  if (log) {
    const lo = ys.length ? Math.min(...ys) : 0.1;
    const hi = ys.length ? Math.max(...ys) : 10;
    yDom = [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi * 1.01))];
    if (yDom[0] === yDom[1]) yDom[1] *= 10;
  } else {
    let lo = ys.length ? Math.min(...ys) : 0;
    let hi = ys.length ? Math.max(...ys) : 1;
    if (includeZero) {
      lo = Math.min(0, lo);
      hi = Math.max(0, hi);
    }
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
    yDom = [lo === 0 ? 0 : lo - pad, hi + pad];
  }
  const xDom: [number, number] = xDomain ?? [xs[0] ?? 0, Math.max((xs[0] ?? 0) + 1, xs[xs.length - 1] ?? 1)];

  const legend: LegendItem[] = [
    ...series.map((s) => ({ label: s.label, color: s.color, kind: s.dashed ? ("dashed" as const) : ("line" as const) })),
    ...refs.map((r) => ({ label: `${r.label} = ${fmt(r.value, 4)}`, color: r.color, kind: "dashed" as const })),
  ];

  const valueAt = series.map((s) => new Map(s.points));
  const anchor = series[0];
  const hoverPoints = xs.flatMap((x) => {
    const y = valueAt[0]?.get(x) ?? valueAt.map((m) => m.get(x)).find(finite);
    if (!finite(y)) return [];
    return [
      {
        x,
        y,
        content: (
          <div className="font-mono tabular-nums">
            <div className="font-sans font-medium">
              {xLabel} {x}
            </div>
            {series.map((s, k) => (
              <div key={s.label} className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-3" style={{ background: s.color }} aria-hidden />
                <span className="font-sans text-fg-2">{s.label}</span> {fmt(valueAt[k].get(x), digits)}
              </div>
            ))}
          </div>
        ),
      },
    ];
  });

  return (
    <Chart
      x={{ domain: xDom, label: xLabel, format: (v) => String(Math.round(v)) }}
      y={{ domain: yDom, label: log ? `${yLabel} (log scale)` : yLabel, log }}
      height={height}
      legend={series.length > 1 || refs.length ? legend : undefined}
      ariaLabel={ariaLabel}
      hover={{ mode: "x", points: hoverPoints }}
      table={{
        columns: [xLabel, ...series.map((s) => s.label)],
        rows: xs.map((x) => [x, ...valueAt.map((m) => fmt(m.get(x), 6))]),
      }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          {includeZero && !log && <line x1={0} x2={s.width} y1={s.y(0)} y2={s.y(0)} stroke="var(--viz-axis)" />}
          {refs.map((r) => (!log || r.value > 0 ? <line key={r.label} x1={0} x2={s.width} y1={s.y(r.value)} y2={s.y(r.value)} stroke={r.color} strokeWidth={1.5} strokeDasharray="5 4" /> : null))}
          {current != null && <line x1={s.x(current)} x2={s.x(current)} y1={0} y2={s.height} stroke="var(--viz-axis)" strokeDasharray="3 3" />}
          {[...series].reverse().map((ser) => (
            <path
              key={ser.label}
              d={linePath(ser.points.map(([x, y]) => [x, finite(y) ? y : NaN]), s)}
              stroke={ser.color}
              strokeWidth={ser.muted ? 1.5 : 2}
              strokeOpacity={ser.muted ? 0.85 : 1}
              strokeDasharray={ser.dashed ? "6 4" : undefined}
              fill="none"
            />
          ))}
          {current != null &&
            anchor &&
            (() => {
              const y = valueAt[0].get(current);
              return finite(y) ? <circle cx={s.x(current)} cy={s.y(y)} r={4.5} fill={anchor.color} stroke="var(--surface)" strokeWidth={2} /> : null;
            })()}
        </g>
      )}
    </Chart>
  );
}
