"use client";

import { useMemo, type ReactNode } from "react";
import { Chart, Legend, linePath, type LegendItem } from "@/components/plot/Chart";
import { fmt } from "@/lib/format";
import { sqDist, voronoiCell, type Vec2 } from "@/lib/ml/kmeans";

/* ------------------------------------------------------------------ cluster identity: colour + shape */

/** Fixed categorical order; a cluster keeps its colour for the whole run. */
export const clusterColor = (j: number) => `var(--viz-cluster-${(j % 8) + 1})`;
const SHAPE_NAMES = ["circle", "square", "triangle", "diamond", "inverted triangle", "plus", "cross", "star"] as const;
export const clusterShapeName = (j: number) => SHAPE_NAMES[j % 8];

const polygon = (pts: Vec2[]) => "M" + pts.map(([a, b]) => `${a.toFixed(2)},${b.toFixed(2)}`).join("L") + "Z";
const rotate = (pts: Vec2[], deg: number): Vec2[] => {
  const t = (deg * Math.PI) / 180;
  return pts.map(([a, b]) => [a * Math.cos(t) - b * Math.sin(t), a * Math.sin(t) + b * Math.cos(t)]);
};
const regular = (m: number, r: number, offsetDeg: number): Vec2[] =>
  Array.from({ length: m }, (_, i) => {
    const t = ((offsetDeg + (360 * i) / m) * Math.PI) / 180;
    return [r * Math.cos(t), r * Math.sin(t)];
  });

/**
 * SVG path of cluster j's marker, centred at the origin, with roughly equal visual area for every shape.
 * Shape is the secondary encoding: beyond three clusters, hue alone is not reliably distinguishable.
 */
export function markerPath(j: number, r: number): string {
  switch (j % 8) {
    case 0:
      return `M${r},0A${r},${r} 0 1 0 ${-r},0A${r},${r} 0 1 0 ${r},0Z`;
    case 1:
      return polygon(regular(4, r * 1.25, 45));
    case 2:
      return polygon(regular(3, r * 1.45, -90));
    case 3:
      return polygon(regular(4, r * 1.35, 0));
    case 4:
      return polygon(regular(3, r * 1.45, 90));
    case 5:
    case 6: {
      const w = r * 0.45;
      const L = r * 1.3;
      const plus: Vec2[] = [[-w, -L], [w, -L], [w, -w], [L, -w], [L, w], [w, w], [w, L], [-w, L], [-w, w], [-L, w], [-L, -w], [-w, -w]];
      return polygon(j % 8 === 5 ? plus : rotate(plus, 45));
    }
    default: {
      const pts: Vec2[] = [];
      for (let i = 0; i < 10; i++) {
        const t = ((-90 + 36 * i) * Math.PI) / 180;
        const rr = i % 2 ? r * 0.62 : r * 1.45;
        pts.push([rr * Math.cos(t), rr * Math.sin(t)]);
      }
      return polygon(pts);
    }
  }
}

export function ClusterSwatch({ j, size = 12 }: { j: number; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-7 -7 14 14" aria-hidden className="inline-block shrink-0 align-[-2px]">
      <path d={markerPath(j, 4.6)} fill={clusterColor(j)} />
    </svg>
  );
}

/** "cluster 2" with its marker, for tables and text. */
export function ClusterTag({ j }: { j: number }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap">
      <ClusterSwatch j={j} />
      {j + 1}
    </span>
  );
}

function ClusterLegend({ k, extra }: { k: number; extra: ReactNode }) {
  return (
    <div className="mb-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-fg-2" aria-label="Legend">
      <span className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
        {Array.from({ length: k }, (_, j) => (
          <span key={j} className="inline-flex items-center gap-1" title={`cluster ${j + 1}: ${clusterShapeName(j)}`}>
            <ClusterSwatch j={j} /> {j + 1}
          </span>
        ))}
      </span>
      {extra}
    </div>
  );
}

/** Bounding box of points (and optional extra points such as every centroid of a run), padded. */
export function plotDomain(points: Vec2[], pad = 0.08): { lo: Vec2; hi: Vec2 } {
  const fin = points.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]));
  if (!fin.length) return { lo: [-1, -1], hi: [1, 1] };
  let lo: Vec2 = [Math.min(...fin.map((p) => p[0])), Math.min(...fin.map((p) => p[1]))];
  let hi: Vec2 = [Math.max(...fin.map((p) => p[0])), Math.max(...fin.map((p) => p[1]))];
  const r = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1);
  lo = [lo[0] - r * pad, lo[1] - r * pad];
  hi = [hi[0] + r * pad, hi[1] + r * pad];
  return { lo, hi };
}

/* ------------------------------------------------------------------ the main scatter */

export function ClusterPlot({
  X,
  labels,
  k,
  centroids,
  prevCentroids,
  cells,
  changed,
  selected,
  onSelect,
  onAddPoint,
  distancesFrom,
  domain,
  height = 360,
  ariaLabel,
  pointNote,
  changedLabel = "changed cluster",
}: {
  X: Vec2[];
  /** Cluster of each point; null = not assigned yet (grey). */
  labels: number[] | null;
  k: number;
  centroids?: Vec2[] | null;
  /** Centroids before an update: drawn as hollow markers with an arrow to the new position. */
  prevCentroids?: Vec2[] | null;
  /** Draw the Voronoi cells of these centroids (the regions the next assignment uses). */
  cells?: Vec2[] | null;
  /** Points that just changed cluster (ringed). */
  changed?: number[];
  selected?: number | null;
  onSelect?: (i: number) => void;
  onAddPoint?: (p: Vec2) => void;
  /** Draw the distance segments from this point to every centroid; the nearest is solid. */
  distancesFrom?: number | null;
  domain: { lo: Vec2; hi: Vec2 };
  height?: number;
  ariaLabel: string;
  /** Extra tooltip line for point i. */
  pointNote?: (i: number) => ReactNode;
  /** Legend text for the ringed points. */
  changedLabel?: string;
}) {
  const xDom = useMemo<[number, number]>(() => [domain.lo[0], domain.hi[0]], [domain]);
  const yDom = useMemo<[number, number]>(() => [domain.lo[1], domain.hi[1]], [domain]);
  const changedSet = useMemo(() => new Set(changed ?? []), [changed]);
  const polys = useMemo(() => {
    if (!cells || cells.length < 2) return null;
    const span = Math.max(domain.hi[0] - domain.lo[0], domain.hi[1] - domain.lo[1]);
    const box = { lo: [domain.lo[0] - 3 * span, domain.lo[1] - 3 * span] as Vec2, hi: [domain.hi[0] + 3 * span, domain.hi[1] + 3 * span] as Vec2 };
    return cells.map((_, j) => voronoiCell(cells, j, box));
  }, [cells, domain]);

  const own = (i: number) => (labels && centroids && labels[i] != null && labels[i] >= 0 ? sqDist(X[i], centroids[labels[i]]) : null);
  const hover = X.map((p, i) => ({
    x: p[0],
    y: p[1],
    content: (
      <div className="font-mono tabular-nums">
        <div className="font-sans font-medium">point i = {i + 1}</div>
        <div>
          x = ({fmt(p[0], 3)}, {fmt(p[1], 3)})
        </div>
        {labels && labels[i] != null && labels[i] >= 0 ? (
          <div className="flex items-center gap-1 font-sans">
            cluster <ClusterTag j={labels[i]} />
          </div>
        ) : (
          <div className="font-sans text-fg-3">not assigned yet</div>
        )}
        {own(i) != null && <div>‖x − μ‖² = {fmt(own(i), 4)}</div>}
        {pointNote?.(i)}
        {onSelect && !onAddPoint && <div className="mt-0.5 font-sans text-fg-3">click to inspect</div>}
      </div>
    ),
  }));

  const extra = (
    <>
      {labels == null && <Legend items={[{ label: "data point (unassigned)", color: "var(--viz-data)", kind: "dot" }]} />}
      {centroids && (
        <span className="inline-flex items-center gap-1.5">
          <svg width="16" height="14" viewBox="-8 -7 16 14" aria-hidden>
            <path d={markerPath(0, 4.5)} fill="var(--fg-3)" stroke="var(--fg)" strokeWidth={1.6} />
          </svg>
          centroid μⱼ
        </span>
      )}
      {prevCentroids && (
        <Legend items={[{ label: "previous centroid → move", color: "var(--fg-2)", kind: "arrow" }]} />
      )}
      {polys && <Legend items={[{ label: "nearest-centroid region (Voronoi cell)", color: "var(--viz-axis)", kind: "dashed" }]} />}
      {changed && changed.length > 0 && <Legend items={[{ label: `${changedLabel} (${changed.length})`, color: "var(--fg)", kind: "ring" }]} />}
    </>
  );

  return (
    <div className="min-w-0">
      <ClusterLegend k={labels || centroids ? k : 0} extra={extra} />
      <Chart
        x={{ domain: xDom, label: "feature x₁" }}
        y={{ domain: yDom, label: "feature x₂" }}
        aspect="equal"
        height={height}
        ariaLabel={ariaLabel}
        hover={{ points: hover }}
        onSelect={onSelect}
        onPlotClick={onAddPoint ? (a, b) => onAddPoint([a, b]) : undefined}
        table={{
          columns: ["i", "x₁", "x₂", "cluster"],
          rows: X.map((p, i) => [i + 1, fmt(p[0], 4), fmt(p[1], 4), labels && labels[i] >= 0 ? labels[i] + 1 : "—"]),
        }}
      >
        {(s) => (
          <g clipPath={`url(#${s.clipId})`}>
            {polys?.map((poly, j) =>
              poly.length ? (
                <path key={`c${j}`} d={linePath(poly, s) + "Z"} fill={clusterColor(j)} fillOpacity={0.07} stroke="var(--viz-axis)" strokeWidth={1} strokeDasharray="4 3" />
              ) : null,
            )}
            {distancesFrom != null && centroids && X[distancesFrom] && (() => {
              const p = X[distancesFrom];
              const d = centroids.map((c) => sqDist(p, c));
              const best = d.indexOf(Math.min(...d));
              return centroids.map((c, j) => (
                <line
                  key={`d${j}`}
                  x1={s.x(p[0])}
                  y1={s.y(p[1])}
                  x2={s.x(c[0])}
                  y2={s.y(c[1])}
                  stroke={j === best ? "var(--fg)" : "var(--fg-3)"}
                  strokeWidth={j === best ? 1.75 : 1}
                  strokeDasharray={j === best ? undefined : "3 3"}
                />
              ));
            })()}
            {X.map((p, i) => {
              const l = labels?.[i];
              const assigned = l != null && l >= 0;
              return (
                <path
                  key={i}
                  transform={`translate(${s.x(p[0]).toFixed(2)},${s.y(p[1]).toFixed(2)})`}
                  d={markerPath(assigned ? l : 0, 4.2)}
                  fill={assigned ? clusterColor(l) : "var(--viz-data)"}
                  fillOpacity={assigned ? 0.9 : 0.75}
                  stroke="var(--surface)"
                  strokeWidth={1}
                />
              );
            })}
            {[...changedSet].map((i) =>
              X[i] ? <circle key={`ch${i}`} cx={s.x(X[i][0])} cy={s.y(X[i][1])} r={8} fill="none" stroke="var(--fg)" strokeWidth={1.25} /> : null,
            )}
            {selected != null && X[selected] && <circle cx={s.x(X[selected][0])} cy={s.y(X[selected][1])} r={10} fill="none" stroke="var(--fg)" strokeWidth={2} />}
            {prevCentroids &&
              centroids &&
              prevCentroids.map((c, j) => {
                const n = centroids[j];
                if (!n) return null;
                const x1 = s.x(c[0]);
                const y1 = s.y(c[1]);
                const x2 = s.x(n[0]);
                const y2 = s.y(n[1]);
                const len = Math.hypot(x2 - x1, y2 - y1);
                const ux = (x2 - x1) / (len || 1);
                const uy = (y2 - y1) / (len || 1);
                const hx = x2 - ux * 14;
                const hy = y2 - uy * 14;
                return (
                  <g key={`m${j}`}>
                    <path transform={`translate(${x1},${y1})`} d={markerPath(j, 7)} fill="none" stroke="var(--fg-2)" strokeWidth={1.25} strokeDasharray="2.5 2" />
                    {len > 22 && (
                      <>
                        <line x1={x1 + ux * 9} y1={y1 + uy * 9} x2={hx} y2={hy} stroke="var(--fg-2)" strokeWidth={1.75} />
                        <path d={`M${x2 - ux * 9},${y2 - uy * 9} L${hx - uy * 4},${hy + ux * 4} L${hx + uy * 4},${hy - ux * 4}z`} fill="var(--fg-2)" />
                      </>
                    )}
                  </g>
                );
              })}
            {centroids?.map((c, j) =>
              Number.isFinite(c[0]) && Number.isFinite(c[1]) ? (
                <g key={`mu${j}`} transform={`translate(${s.x(c[0]).toFixed(2)},${s.y(c[1]).toFixed(2)})`}>
                  <path d={markerPath(j, 7.5)} fill={clusterColor(j)} stroke="var(--fg)" strokeWidth={1.75} />
                  <text x={11} y={-9} fontSize={12} fontWeight={600} fill="var(--fg)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke" className="select-none">
                    μ{j + 1}
                  </text>
                </g>
              ) : null,
            )}
          </g>
        )}
      </Chart>
    </div>
  );
}

/* ------------------------------------------------------------------ objective over half-steps */

export interface ObjectivePoint {
  iteration: number;
  phase: "assign" | "update";
  J: number;
}

/**
 * J after every half-step. Iteration t's assignment is drawn at t − ½ (hollow: labels new, centroids old),
 * its update at t (filled: the iteration is complete; scikit-learn's inertia).
 */
export function ObjectivePlot({
  points,
  current,
  maxIteration,
  yMax,
  height = 230,
}: {
  points: ObjectivePoint[];
  /** Fixed top of the y axis (largest J of the whole run), so the axis does not rescale while stepping. */
  yMax?: number;
  current?: { iteration: number; phase: "assign" | "update" } | null;
  maxIteration: number;
  height?: number;
}) {
  const pos = (p: { iteration: number; phase: string }) => (p.phase === "assign" ? p.iteration - 0.5 : p.iteration);
  const finite = points.filter((p) => Number.isFinite(p.J));
  const top = (yMax != null && yMax > 0 ? yMax : Math.max(0, ...finite.map((p) => p.J)) || 1) * 1.08;
  const legend: LegendItem[] = [
    { label: "after assignment (centroids not yet moved)", color: "var(--viz-model)", kind: "ring" },
    { label: "after update: iteration complete", color: "var(--viz-model)", kind: "dot" },
  ];
  const cur = current ? finite.find((p) => p.iteration === current.iteration && p.phase === current.phase) : undefined;
  return (
    <Chart
      x={{ domain: [0, Math.max(1, maxIteration)], label: "iteration t", format: (v) => (Number.isInteger(v) ? String(v) : "") }}
      y={{ domain: [0, top], label: "J (within-cluster SS)" }}
      height={height}
      legend={legend}
      ariaLabel="Objective J after each assignment and update step"
      hover={{
        points: finite.map((p) => ({
          x: pos(p),
          y: p.J,
          content: (
            <div className="font-mono tabular-nums">
              <div className="font-sans font-medium">
                iteration {p.iteration} · {p.phase === "assign" ? "after assignment" : "after update"}
              </div>
              J = {fmt(p.J, 5)}
            </div>
          ),
        })),
      }}
      table={{ columns: ["iteration", "phase", "J"], rows: finite.map((p) => [p.iteration, p.phase === "assign" ? "after assignment" : "after update", fmt(p.J, 6)]) }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          <path d={linePath(finite.map((p) => [pos(p), p.J]), s)} stroke="var(--viz-model)" strokeWidth={2} fill="none" />
          {finite.map((p) =>
            p.phase === "assign" ? (
              <circle key={`a${p.iteration}`} cx={s.x(pos(p))} cy={s.y(p.J)} r={4} fill="var(--surface)" stroke="var(--viz-model)" strokeWidth={1.75} />
            ) : (
              <circle key={`u${p.iteration}`} cx={s.x(pos(p))} cy={s.y(p.J)} r={4} fill="var(--viz-model)" stroke="var(--surface)" strokeWidth={1.5} />
            ),
          )}
          {cur && <circle cx={s.x(pos(cur))} cy={s.y(cur.J)} r={8} fill="none" stroke="var(--fg)" strokeWidth={1.5} />}
        </g>
      )}
    </Chart>
  );
}

/* ------------------------------------------------------------------ comparing initializations */

export interface InitTrial {
  method: string;
  seed: number;
  inertia: number;
  iterations: number;
}

/** Final J of many initializations, one row per method. Click a run to load it. */
export function InitComparisonPlot({
  trials,
  methods,
  best,
  selected,
  onSelect,
  height = 200,
}: {
  trials: InitTrial[];
  methods: { id: string; label: string }[];
  /** Lowest J found over all trials. */
  best: number;
  selected?: { method: string; seed: number } | null;
  onSelect?: (t: InitTrial) => void;
  height?: number;
}) {
  const row = (m: string) => methods.length - 1 - methods.findIndex((x) => x.id === m);
  const yOf = (t: InitTrial) => row(t.method) + (((t.seed * 7) % 9) - 4) * 0.055;
  const Js = trials.map((t) => t.inertia);
  const lo = Math.min(...Js);
  const hi = Math.max(...Js);
  const pad = (hi - lo || hi || 1) * 0.08;
  const reached = (t: InitTrial) => t.inertia <= best * (1 + 1e-9);
  return (
    <Chart
      x={{ domain: [Math.max(0, lo - pad), hi + pad], label: "final J (lower is better)" }}
      y={{ domain: [-0.6, methods.length - 0.4], label: "", ticks: methods.length, format: () => "" }}
      height={height}
      legend={[
        { label: "reached the lowest J found", color: "var(--viz-model)", kind: "dot" },
        { label: "stuck in a worse local minimum", color: "var(--viz-residual)", kind: "dot" },
      ]}
      ariaLabel="Final objective for each initialization seed, by method"
      onSelect={onSelect ? (i) => onSelect(trials[i]) : undefined}
      hover={{
        points: trials.map((t) => ({
          x: t.inertia,
          y: yOf(t),
          content: (
            <div className="font-mono tabular-nums">
              <div className="font-sans font-medium">
                {methods.find((m) => m.id === t.method)?.label} · seed {t.seed}
              </div>
              J = {fmt(t.inertia, 4)}
              <br />
              {t.iterations} iterations
              {onSelect && <div className="font-sans text-fg-3">click to load this run</div>}
            </div>
          ),
        })),
      }}
      table={{ columns: ["method", "seed", "final J", "iterations"], rows: trials.map((t) => [t.method, t.seed, fmt(t.inertia, 5), t.iterations]) }}
    >
      {(s) => (
        <g clipPath={`url(#${s.clipId})`}>
          {methods.map((m, r) => (
            <text key={m.id} x={6} y={s.y(methods.length - 1 - r + 0.42)} dy="0.35em" fontSize={11.5} fill="var(--fg-2)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">
              {m.label}
            </text>
          ))}
          <line x1={s.x(best)} x2={s.x(best)} y1={0} y2={s.height} stroke="var(--viz-model)" strokeDasharray="4 3" />
          {trials.map((t, i) => (
            <circle
              key={i}
              cx={s.x(t.inertia)}
              cy={s.y(yOf(t))}
              r={4.5}
              fill={reached(t) ? "var(--viz-model)" : "var(--viz-residual)"}
              stroke="var(--surface)"
              strokeWidth={1.5}
            />
          ))}
          {selected &&
            trials
              .filter((t) => t.method === selected.method && t.seed === selected.seed)
              .map((t) => <circle key="sel" cx={s.x(t.inertia)} cy={s.y(yOf(t))} r={8.5} fill="none" stroke="var(--fg)" strokeWidth={1.75} />)}
        </g>
      )}
    </Chart>
  );
}
