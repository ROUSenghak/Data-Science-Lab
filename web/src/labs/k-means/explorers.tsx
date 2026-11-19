"use client";

import { useEffect, useMemo, useState } from "react";
import { TexBlock } from "@/components/math/Tex";
import { SeriesPlot } from "@/components/plot/SeriesPlot";
import { Button, Icon, Segmented } from "@/components/ui/controls";
import { Callout } from "@/components/ui/primitives";
import { fmt, texNum } from "@/lib/format";
import { assign, elbow, runKMeans, sqDist, type InitMethod, type KMeansRun, type Phase, type Vec2 } from "@/lib/ml/kmeans";
import { ClusterPlot, ClusterTag, InitComparisonPlot, ObjectivePlot, plotDomain, type InitTrial, type ObjectivePoint } from "./plots";

export const INIT_METHODS: { id: InitMethod; label: string; hint: string }[] = [
  { id: "random", label: "random points", hint: "K distinct data points, uniformly at random" },
  { id: "kmeans++", label: "k-means++", hint: "each next centroid drawn with probability ∝ D(x)²" },
  { id: "uniform", label: "uniform box", hint: "uniform positions in the data's bounding box (can leave clusters empty)" },
];

export function objectivePoints(phases: Phase[], upTo = phases.length - 1): ObjectivePoint[] {
  return phases.slice(0, upTo + 1).flatMap((p) => (p.kind === "init" || p.J == null ? [] : [{ iteration: p.iteration, phase: p.kind, J: p.J }]));
}

const PHASE_SPEED = { slow: 1100, normal: 550, fast: 180 } as const;

/* ------------------------------------------------------------------ step through Lloyd's algorithm */

export function LloydExplorer({
  X,
  truth,
  k,
  method,
  seed,
  onAddPoint,
  compact = false,
  maxIter = 100,
}: {
  X: Vec2[];
  /** Generating group of each point, for comparison (optional). */
  truth?: number[] | null;
  k: number;
  method: InitMethod;
  seed: number;
  onAddPoint?: (p: Vec2) => void;
  compact?: boolean;
  maxIter?: number;
}) {
  const run = useMemo<KMeansRun | null>(() => (X.length >= k ? runKMeans(X, k, { method, seed, maxIter }) : null), [X, k, method, seed, maxIter]);
  const phases = run?.phases ?? [];
  const last = Math.max(0, phases.length - 1);
  const [p, setP] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<keyof typeof PHASE_SPEED>("normal");
  const [selected, setSelected] = useState<number | null>(null);
  const [colorBy, setColorBy] = useState<"kmeans" | "truth">("kmeans");
  const [showCells, setShowCells] = useState(true);

  // Restart the replay when the experiment changes (state adjustment during render).
  const [prevRun, setPrevRun] = useState(run);
  if (prevRun !== run) {
    setPrevRun(run);
    setP(0);
    setPlaying(false);
    if (selected != null && selected >= X.length) setSelected(null);
  }

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setP((v) => {
        if (v >= last) {
          setPlaying(false);
          return v;
        }
        return v + 1;
      });
    }, PHASE_SPEED[speed]);
    return () => window.clearInterval(id);
  }, [playing, speed, last]);

  const domain = useMemo(() => plotDomain(X), [X]);
  const pi = Math.min(p, last);
  const phase = phases[pi];

  if (!run || !phase) {
    return (
      <div className="space-y-3">
        <Callout>
          Add at least K = {k} points{onAddPoint ? " by clicking in the plot" : ""}. There are {X.length} now.
        </Callout>
        <ClusterPlot X={X} labels={null} k={k} domain={domain} onAddPoint={onAddPoint} ariaLabel="Data points" height={compact ? 300 : 380} />
      </div>
    );
  }

  const C = phase.centroids;
  // Points that lie in another centroid's region after an update: they switch at the next assignment.
  const willMove = phase.kind === "update" && !phase.converged ? assign(X, C).flatMap((l, i) => (l !== phase.labels![i] ? [i] : [])) : [];
  const shownLabels = colorBy === "truth" && truth ? truth : phase.labels;
  const nextAssign = phases[pi + 1]?.kind === "assign" ? phases[pi + 1] : null;
  const stepLabel =
    phase.kind === "init" ? "Initialization" : `Iteration ${phase.iteration} · ${phase.kind === "assign" ? "assignment step" : "update step"}`;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" onClick={() => (pi >= last ? (setP(0), setPlaying(true)) : setPlaying((v) => !v))} ariaLabel={playing ? "Pause" : "Play"}>
          {playing ? Icon.pause : Icon.play} {playing ? "Pause" : pi >= last ? "Replay" : "Play"}
        </Button>
        <Button onClick={() => (setPlaying(false), setP((v) => Math.max(0, v - 1)))} disabled={pi === 0} ariaLabel="Previous half-step">
          {Icon.back}
        </Button>
        <Button onClick={() => (setPlaying(false), setP((v) => Math.min(last, v + 1)))} disabled={pi >= last} ariaLabel="Next half-step">
          {Icon.step} Step
        </Button>
        <Button variant="ghost" onClick={() => (setPlaying(false), setP(0))} ariaLabel="Reset to initialization">
          {Icon.reset} Reset
        </Button>
        <Segmented ariaLabel="Playback speed" value={speed} onChange={setSpeed} options={[{ value: "slow", label: "slow" }, { value: "normal", label: "normal" }, { value: "fast", label: "fast" }]} />
        <input type="range" aria-label="Half-step" className="min-w-32 flex-1" min={0} max={last} value={pi} onChange={(e) => (setPlaying(false), setP(Number(e.target.value)))} />
        <span className="whitespace-nowrap font-mono text-[12px] tabular-nums text-fg-2">
          {stepLabel}
          <span className="text-fg-3"> / {run.iterations}</span>
        </span>
      </div>

      <div className={`grid gap-4 ${compact ? "" : "lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]"}`}>
        <div className="min-w-0">
          <ClusterPlot
            X={X}
            labels={shownLabels}
            k={colorBy === "truth" && truth ? Math.max(...truth) + 1 : k}
            centroids={C}
            prevCentroids={phase.kind === "update" ? phase.prevCentroids : null}
            cells={showCells ? C : null}
            changed={colorBy === "kmeans" ? (phase.kind === "assign" && phase.iteration > 1 ? phase.changed : willMove) : []}
            changedLabel={phase.kind === "update" ? "will switch at the next assignment" : "changed cluster"}
            selected={selected}
            onSelect={setSelected}
            onAddPoint={onAddPoint}
            distancesFrom={selected}
            domain={domain}
            height={compact ? 320 : 400}
            ariaLabel={`K-Means with K = ${k}: ${stepLabel}`}
          />
          <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[12px] text-fg-2">
            <label className="inline-flex items-center gap-1.5">
              <input type="checkbox" className="accent-[var(--viz-model)]" checked={showCells} onChange={(e) => setShowCells(e.target.checked)} /> nearest-centroid regions
            </label>
            {truth && (
              <Segmented
                ariaLabel="Colour points by"
                value={colorBy}
                onChange={setColorBy}
                options={[
                  { value: "kmeans", label: "colour: K-Means labels" },
                  { value: "truth", label: "colour: generating groups" },
                ]}
              />
            )}
            <span className="text-fg-3">{onAddPoint ? "Click empty space to add a point." : "Click a point to inspect its distances."}</span>
          </div>
        </div>
        <div className="min-w-0 space-y-3">
          <PhaseExplanation phase={phase} X={X} willMove={willMove.length} nextAssign={nextAssign} />
          {selected != null && X[selected] && <PointInspector X={X} i={selected} phase={phase} />}
        </div>
      </div>

      {!compact && (
        <div className="min-w-0">
          <ObjectivePlot
            points={objectivePoints(phases, pi)}
            yMax={Math.max(...objectivePoints(phases).map((q) => q.J))}
            current={phase.kind === "init" ? null : { iteration: phase.iteration, phase: phase.kind }}
            maxIteration={run.iterations}
          />
          <p className="mt-1 text-[11.5px] text-fg-3">
            Every half-step can only lower J (or leave it unchanged). Final J = {fmt(run.inertia, 4)} after {run.iterations} iterations{run.converged ? " (converged)" : " (stopped at the iteration limit)"}.
          </p>
        </div>
      )}
    </div>
  );
}

function PhaseExplanation({ phase, X, willMove, nextAssign }: { phase: Phase; X: Vec2[]; willMove: number; nextAssign: Phase | null }) {
  const box = "rounded-md border border-line bg-surface-2 px-3 py-2 text-[14px]";
  if (phase.kind === "init") {
    return (
      <div className={box}>
        <div className="text-[12px] font-medium text-fg-2">Initial centroids μ⁽⁰⁾</div>
        <TexBlock>{`\\mu^{(0)}_j,\\; j = 1,\\dots,${phase.centroids.length}`}</TexBlock>
        <p className="text-[13px] text-fg-2">
          The dashed lines are the perpendicular bisectors between centroids: each region holds the points nearest to its centroid. The first assignment colours every point by its region.
        </p>
      </div>
    );
  }
  if (phase.kind === "assign") {
    return (
      <div className={box}>
        <div className="text-[12px] font-medium text-fg-2">Assignment step · intermediate state</div>
        <TexBlock>{`c_i^{(${phase.iteration})} = \\arg\\min_j \\lVert x_i - \\mu_j^{(${phase.iteration - 1})} \\rVert^2`}</TexBlock>
        <TexBlock>{`J\\big(c^{(${phase.iteration})}, \\mu^{(${phase.iteration - 1})}\\big) = ${texNum(phase.J, 4)}`}</TexBlock>
        <p className="text-[13px] text-fg-2">
          {phase.iteration === 1 ? (
            <>Every point joins its nearest centroid.</>
          ) : phase.changed.length ? (
            <>
              <b className="font-medium text-fg">{phase.changed.length}</b> point{phase.changed.length > 1 ? "s" : ""} changed cluster (ringed): each now lies in another centroid&apos;s region because the centroids moved in the last update.
            </>
          ) : (
            <>No point changed cluster, so the next update cannot move any centroid: the algorithm has converged.</>
          )}{" "}
          The centroids have not moved yet.
        </p>
      </div>
    );
  }
  const moved = phase.shift ?? [];
  return (
    <div className={box}>
      <div className="text-[12px] font-medium text-fg-2">Update step · iteration {phase.iteration} complete</div>
      <TexBlock>{`\\mu_j^{(${phase.iteration})} = \\frac{1}{|C_j|}\\sum_{i \\in C_j} x_i, \\qquad J\\big(c^{(${phase.iteration})}, \\mu^{(${phase.iteration})}\\big) = ${texNum(phase.J, 4)}`}</TexBlock>
      <div className="max-h-44 overflow-auto">
        <table className="w-full text-left font-mono text-[12px] tabular-nums">
          <thead className="text-[11px] text-fg-3">
            <tr>
              <th className="py-0.5 pr-2 font-sans font-medium">cluster</th>
              <th className="py-0.5 pr-2 font-sans font-medium">|Cⱼ|</th>
              <th className="py-0.5 pr-2 font-sans font-medium">new μⱼ</th>
              <th className="py-0.5 font-sans font-medium">moved</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line text-fg">
            {phase.centroids.map((c, j) => (
              <tr key={j}>
                <td className="py-0.5 pr-2 font-sans">
                  <ClusterTag j={j} />
                </td>
                <td className="py-0.5 pr-2">{phase.counts?.[j]}</td>
                <td className="py-0.5 pr-2">
                  ({fmt(c[0], 3)}, {fmt(c[1], 3)})
                </td>
                <td className="py-0.5">{phase.counts?.[j] === 0 ? <span className="font-sans text-warn">empty: kept</span> : fmt(moved[j], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1.5 text-[13px] text-fg-2">
        {phase.converged ? (
          <>No label changed in this iteration, so the means are the same as before: a fixed point. Lloyd&apos;s algorithm stops.</>
        ) : willMove ? (
          <>
            Moving the centroids moved the boundaries: <b className="font-medium text-fg">{willMove}</b> point{willMove > 1 ? "s" : ""} (ringed) now lie in another region and will switch at the next assignment
            {nextAssign?.J != null && <>, lowering J to {fmt(nextAssign.J, 4)}</>}.
          </>
        ) : (
          <>Every point is still in its own centroid&apos;s region; the next assignment changes nothing.</>
        )}
        {phase.empty && phase.empty.length > 0 && <> An empty cluster keeps its previous centroid (deterministic policy).</>}
      </p>
      <p className="mt-1 text-[11.5px] text-fg-3">{X.length} points · each mean minimizes its cluster&apos;s sum of squares, so J cannot increase.</p>
    </div>
  );
}

function PointInspector({ X, i, phase }: { X: Vec2[]; i: number; phase: Phase }) {
  const x = X[i];
  const C = phase.centroids;
  const d = C.map((c) => sqDist(x, c));
  const best = d.indexOf(Math.min(...d));
  const label = phase.labels?.[i];
  const usedFor = phase.kind === "update" ? "the next assignment" : phase.kind === "init" ? "the first assignment" : "this assignment";
  return (
    <div className="rounded-md border border-line px-3 py-2 text-[14px]">
      <div className="text-[12px] font-medium text-fg-2">
        Point {i + 1} · distances used by {usedFor}
      </div>
      <TexBlock>{distanceTex(x, C[best], i, best, d[best])}</TexBlock>
      <table className="w-full text-left font-mono text-[12px] tabular-nums">
        <thead className="text-[11px] text-fg-3">
          <tr>
            <th className="py-0.5 pr-2 font-sans font-medium">centroid</th>
            <th className="py-0.5 font-sans font-medium">‖xᵢ − μⱼ‖²</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {d.map((v, j) => (
            <tr key={j} className={j === best ? "font-semibold text-fg" : "text-fg-2"}>
              <td className="py-0.5 pr-2 font-sans">
                <ClusterTag j={j} />
              </td>
              <td className="py-0.5">
                {fmt(v, 4)} {j === best && <span className="font-sans font-normal text-fg-3">← nearest</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {label != null && phase.kind === "update" && label !== best && (
        <p className="mt-1 text-[12.5px] text-fg-2">
          Currently in cluster {label + 1}, but μ{best + 1} is now nearer: it will switch at the next assignment.
        </p>
      )}
    </div>
  );
}

/** x_i − μ_j as a difference vector, then its squared length: short enough for narrow panels. */
export function distanceTex(x: Vec2, c: Vec2, i: number, j: number, d: number): string {
  const dx = x[0] - c[0];
  const dy = x[1] - c[1];
  return `\\begin{aligned} x_{${i + 1}} - \\mu_{${j + 1}} &= (${texNum(dx, 3)},\\ ${texNum(dy, 3)}) \\\\ \\lVert x_{${i + 1}} - \\mu_{${j + 1}}\\rVert^2 &= ${texNum(dx, 3, true)}^2 + ${texNum(dy, 3, true)}^2 = ${texNum(d, 4)} \\end{aligned}`;
}

/* ------------------------------------------------------------------ comparing initializations */

export function InitComparison({
  X,
  k,
  seeds = 30,
  selected,
  onPick,
}: {
  X: Vec2[];
  k: number;
  seeds?: number;
  selected?: { method: InitMethod; seed: number } | null;
  onPick?: (method: InitMethod, seed: number) => void;
}) {
  const trials = useMemo<InitTrial[]>(() => {
    if (X.length < k) return [];
    return INIT_METHODS.flatMap((m) =>
      Array.from({ length: seeds }, (_, seed) => {
        const r = runKMeans(X, k, { method: m.id, seed });
        return { method: m.id, seed, inertia: r.inertia, iterations: r.iterations };
      }),
    );
  }, [X, k, seeds]);
  if (!trials.length) return <Callout>Not enough points for K = {k}.</Callout>;
  const best = Math.min(...trials.map((t) => t.inertia));
  const stats = INIT_METHODS.map((m) => {
    const ts = trials.filter((t) => t.method === m.id).sort((a, b) => a.inertia - b.inertia);
    return {
      m,
      hits: ts.filter((t) => t.inertia <= best * (1 + 1e-9)).length,
      median: ts[Math.floor(ts.length / 2)].inertia,
      worst: ts[ts.length - 1].inertia,
      iters: ts.reduce((s, t) => s + t.iterations, 0) / ts.length,
    };
  });
  return (
    <div className="space-y-3">
      <InitComparisonPlot
        trials={trials}
        methods={INIT_METHODS.map((m) => ({ id: m.id, label: m.label }))}
        best={best}
        selected={selected}
        onSelect={onPick ? (t) => onPick(t.method as InitMethod, t.seed) : undefined}
      />
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[12.5px] tabular-nums">
          <thead className="text-[11px] text-fg-3">
            <tr>
              <th className="py-1 pr-3 font-medium">initialization</th>
              <th className="py-1 pr-3 font-medium">reached lowest J</th>
              <th className="py-1 pr-3 font-medium">median J</th>
              <th className="py-1 pr-3 font-medium">worst J</th>
              <th className="py-1 font-medium">mean iterations</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line font-mono text-fg-2">
            {stats.map((s) => (
              <tr key={s.m.id}>
                <td className="py-1 pr-3 font-sans text-fg">{s.m.label}</td>
                <td className="py-1 pr-3">
                  {s.hits}/{seeds}
                </td>
                <td className="py-1 pr-3">{fmt(s.median, 3)}</td>
                <td className="py-1 pr-3">{fmt(s.worst, 3)}</td>
                <td className="py-1">{fmt(s.iters, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11.5px] text-fg-3">
        {seeds} seeds per method on the same data and K = {k}. &ldquo;Lowest J&rdquo; is the best value any run found ({fmt(best, 4)}); it need not be the global minimum. These counts are empirical for this dataset, not guarantees.
        {onPick && " Click a dot to step through that run above."}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ choosing K */

export function ElbowExplorer({ X, currentK, kMax = 8 }: { X: Vec2[]; currentK?: number; kMax?: number }) {
  const rows = useMemo(() => (X.length >= 2 ? elbow(X, kMax, 8, 0) : []), [X, kMax]);
  if (rows.length < 2) return <Callout>Add more points to compare values of K.</Callout>;
  const sil = rows.filter((r) => Number.isFinite(r.silhouette));
  const bestSil = sil.reduce((a, b) => (b.silhouette > a.silhouette ? b : a), sil[0]);
  return (
    <div className="space-y-2">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-[12px] font-medium text-fg-2">Within-cluster sum of squares J*(K)</div>
          <SeriesPlot
            series={[{ label: "J*(K)", color: "var(--viz-model)", points: rows.map((r) => [r.k, r.inertia]) }]}
            xLabel="K"
            yLabel="J"
            yFormat={(v) => (Math.abs(v) >= 1000 ? `${v / 1000}k` : String(v))}
            includeZero
            current={currentK}
            xDomain={[1, rows.length]}
            height={220}
            ariaLabel="Elbow plot: best within-cluster sum of squares for each K"
          />
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-[12px] font-medium text-fg-2">Mean silhouette (higher is better)</div>
          <SeriesPlot
            series={[{ label: "silhouette", color: "var(--viz-alt)", points: rows.map((r) => [r.k, Number.isFinite(r.silhouette) ? r.silhouette : null]) }]}
            xLabel="K"
            yLabel="silhouette"
            includeZero
            current={currentK}
            xDomain={[1, rows.length]}
            height={220}
            digits={3}
            ariaLabel="Mean silhouette coefficient for each K"
          />
        </div>
      </div>
      <p className="text-[11.5px] text-fg-3">
        Each K uses the best of 8 k-means++ runs. J* always shrinks as K grows (with K = n it is 0), so look for the bend, not the minimum. The silhouette peaks at K = {bestSil?.k} here. Silhouette is undefined for K = 1.
      </p>
    </div>
  );
}
