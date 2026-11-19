"use client";

import { scaleLinear, scaleLog } from "d3-scale";
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";

export interface AxisSpec {
  domain: [number, number];
  label: string;
  log?: boolean;
  ticks?: number;
  format?: (v: number) => string;
}

export interface Scales {
  x: (v: number) => number;
  y: (v: number) => number;
  width: number;
  height: number;
  clipId: string;
}

export interface HoverPoint {
  x: number;
  y: number;
  content: ReactNode;
}

export interface LegendItem {
  label: string;
  color: string;
  kind: "line" | "dashed" | "dot" | "ring" | "area" | "arrow";
}

const M = { top: 12, right: 14, bottom: 42, left: 52 };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg-2" aria-label="Legend">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          <svg width="18" height="10" aria-hidden>
            {it.kind === "line" && <line x1="1" x2="17" y1="5" y2="5" stroke={it.color} strokeWidth="2" />}
            {it.kind === "dashed" && <line x1="1" x2="17" y1="5" y2="5" stroke={it.color} strokeWidth="2" strokeDasharray="4 3" />}
            {it.kind === "arrow" && (
              <>
                <line x1="1" x2="13" y1="5" y2="5" stroke={it.color} strokeWidth="2" />
                <path d="M12 1.5 L17 5 L12 8.5z" fill={it.color} />
              </>
            )}
            {it.kind === "dot" && <circle cx="9" cy="5" r="4" fill={it.color} />}
            {it.kind === "ring" && <circle cx="9" cy="5" r="3.5" fill="none" stroke={it.color} strokeWidth="1.8" />}
            {it.kind === "area" && <rect x="2" y="1" width="14" height="8" rx="1.5" fill={it.color} />}
          </svg>
          {it.label}
        </li>
      ))}
    </ul>
  );
}

export function Chart({
  x,
  y,
  height = 280,
  ariaLabel,
  legend,
  hover,
  onSelect,
  table,
  aspect,
  onPlotClick,
  children,
}: {
  x: AxisSpec;
  y: AxisSpec;
  height?: number;
  ariaLabel: string;
  legend?: LegendItem[];
  hover?: { points: HoverPoint[]; mode?: "nearest" | "x" };
  onSelect?: (index: number) => void;
  table?: { columns: string[]; rows: (string | number)[][]; caption?: string };
  /** "equal": one data unit has the same length on both axes (the smaller range is widened). */
  aspect?: "equal";
  /** Click anywhere in the plot area, in data coordinates. Takes precedence over onSelect. */
  onPlotClick?: (x: number, y: number) => void;
  children: (s: Scales) => ReactNode;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const clipId = "clip" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const [hi, setHi] = useState<number | null>(null);

  const innerW = Math.max(0, width - M.left - M.right);
  const innerH = height - M.top - M.bottom;

  const { xs, ys } = useMemo(() => {
    let xd = x.domain;
    let yd = y.domain;
    if (aspect === "equal" && innerW > 0 && innerH > 0 && !x.log && !y.log) {
      const ux = (xd[1] - xd[0]) / innerW;
      const uy = (yd[1] - yd[0]) / innerH;
      const widen = (d: [number, number], len: number): [number, number] => [(d[0] + d[1]) / 2 - len / 2, (d[0] + d[1]) / 2 + len / 2];
      if (ux > uy) yd = widen(yd, ux * innerH);
      else xd = widen(xd, uy * innerW);
    }
    const xs = (x.log ? scaleLog() : scaleLinear()).domain(xd).range([0, innerW]);
    const ys = (y.log ? scaleLog() : scaleLinear()).domain(yd).range([innerH, 0]);
    return { xs, ys };
  }, [x.domain, x.log, y.domain, y.log, innerW, innerH, aspect]);

  const xTicks = xs.ticks(x.ticks ?? Math.max(2, Math.floor(innerW / 80)));
  const yTicks = y.log ? ys.ticks().filter((t) => Math.abs(Math.log10(t) - Math.round(Math.log10(t))) < 1e-9) : ys.ticks(y.ticks ?? Math.max(2, Math.floor(innerH / 50)));
  const xFmt = x.format ?? xs.tickFormat(xTicks.length);
  const yFmt = y.format ?? (y.log ? (v: number) => (v >= 1e4 || v < 1e-2 ? `1e${Math.round(Math.log10(v))}` : String(v)) : ys.tickFormat(yTicks.length));

  const clampPx = (v: number, lo: number, hi: number) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : NaN);
  const scales: Scales = {
    x: (v) => clampPx(xs(v), -1e4, 1e4),
    y: (v) => clampPx(ys(v), -1e4, 1e4),
    width: innerW,
    height: innerH,
    clipId,
  };

  const findNearest = (px: number, py: number) => {
    if (!hover?.points.length) return null;
    let best = -1;
    let bestD = Infinity;
    hover.points.forEach((p, i) => {
      const dx = scales.x(p.x) - px;
      const dy = hover.mode === "x" ? 0 : scales.y(p.y) - py;
      const d = dx * dx + dy * dy;
      if (Number.isFinite(d) && d < bestD) {
        bestD = d;
        best = i;
      }
    });
    const limit = hover.mode === "x" ? Infinity : 28 * 28;
    return best >= 0 && bestD <= limit ? best : null;
  };

  const local = (e: React.MouseEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };

  const hp = hi !== null && hover ? hover.points[hi] : null;

  return (
    <figure className="min-w-0">
      {legend && legend.length > 0 && (
        <div className="mb-1.5">
          <Legend items={legend} />
        </div>
      )}
      <div ref={ref} className="relative w-full" style={{ height }}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible select-none">
            <defs>
              <clipPath id={clipId}>
                <rect x={0} y={0} width={innerW} height={innerH} />
              </clipPath>
            </defs>
            <g transform={`translate(${M.left},${M.top})`}>
              {yTicks.map((t) => (
                <g key={`y${t}`} transform={`translate(0,${ys(t)})`}>
                  <line x2={innerW} stroke="var(--viz-grid)" />
                  <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--fg-3)" className="tabular-nums">
                    {yFmt(t)}
                  </text>
                </g>
              ))}
              {xTicks.map((t) => (
                <g key={`x${t}`} transform={`translate(${xs(t)},0)`}>
                  <line y2={innerH} stroke="var(--viz-grid)" />
                  <text y={innerH + 16} textAnchor="middle" fontSize={11} fill="var(--fg-3)" className="tabular-nums">
                    {xFmt(t)}
                  </text>
                </g>
              ))}
              <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--viz-axis)" />
              <line x1={0} x2={0} y1={0} y2={innerH} stroke="var(--viz-axis)" />
              <text x={innerW / 2} y={innerH + 35} textAnchor="middle" fontSize={12} fill="var(--fg-2)">
                {x.label}
              </text>
              <text transform={`translate(${-40},${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--fg-2)">
                {y.label}
              </text>
              {children(scales)}
              {hp && (
                <g pointerEvents="none">
                  {hover?.mode === "x" && <line x1={scales.x(hp.x)} x2={scales.x(hp.x)} y1={0} y2={innerH} stroke="var(--viz-axis)" strokeDasharray="3 3" />}
                  <circle cx={scales.x(hp.x)} cy={scales.y(hp.y)} r={6} fill="none" stroke="var(--fg)" strokeWidth={1.5} />
                </g>
              )}
              {hover && (
                <rect
                  width={innerW}
                  height={innerH}
                  fill="transparent"
                  style={{ cursor: onPlotClick ? "copy" : onSelect ? "pointer" : "crosshair" }}
                  onPointerMove={(e) => setHi(findNearest(...local(e)))}
                  onPointerLeave={() => setHi(null)}
                  onClick={(e) => {
                    const [px, py] = local(e);
                    if (onPlotClick) return onPlotClick(xs.invert(px), ys.invert(py));
                    const i = findNearest(px, py);
                    if (i !== null) onSelect?.(i);
                  }}
                />
              )}
            </g>
          </svg>
        )}
        {hp && Number.isFinite(scales.x(hp.x)) && Number.isFinite(scales.y(hp.y)) && (
          <div
            className="pointer-events-none absolute z-10 rounded-md border border-line bg-surface px-2 py-1.5 text-[11.5px] leading-snug shadow-md"
            style={{
              left: Math.min(M.left + scales.x(hp.x) + 10, width - 170),
              top: Math.max(0, M.top + scales.y(hp.y) - 10),
            }}
          >
            {hp.content}
          </div>
        )}
      </div>
      {table && (
        <details className="mt-1 text-[12px] text-fg-3">
          <summary className="cursor-pointer select-none hover:text-fg-2">Data table</summary>
          <div className="mt-1 max-h-48 overflow-auto rounded border border-line">
            <table className="w-full text-left font-mono text-[11.5px] tabular-nums">
              {table.caption && <caption className="sr-only">{table.caption}</caption>}
              <thead className="sticky top-0 bg-surface-2 text-fg-2">
                <tr>
                  {table.columns.map((c) => (
                    <th key={c} className="px-2 py-1 font-medium">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((r, i) => (
                  <tr key={i} className="border-t border-line text-fg-2">
                    {r.map((c, j) => (
                      <td key={j} className="px-2 py-0.5">
                        {c}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  );
}

/** Polyline path; breaks the line at non-finite values. */
export function linePath(pts: [number, number][], s: Scales): string {
  let d = "";
  let pen = false;
  for (const [a, b] of pts) {
    const px = s.x(a);
    const py = s.y(b);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${px.toFixed(2)},${py.toFixed(2)}`;
    pen = true;
  }
  return d;
}
