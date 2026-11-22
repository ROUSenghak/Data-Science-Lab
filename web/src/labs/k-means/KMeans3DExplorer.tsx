"use client";

/**
 * Lloyd's algorithm on three features, step by step. Same playback, same half-steps and the same
 * TypeScript implementation as the 2D explorer; only the number of features changes. The point of
 * the figure: the distance uses every feature, and a two-feature projection can hide a cluster.
 */
import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { usePlayback, useTween } from "@/components/anim/hooks";
import { PlaybackBar } from "@/components/anim/PlaybackBar";
import { TexBlock } from "@/components/math/Tex";
import { Plot3DBoundary } from "@/components/plot3d/Boundary";
import { useWebGLSupport } from "@/components/plot3d/support";
import { Button, Segmented, Toggle } from "@/components/ui/controls";
import type { SpeedDef } from "@/lib/anim/playback";
import { tweenDuration } from "@/lib/anim/tween";
import { fmt, texNum, texSq } from "@/lib/format";
import { agreement, generateClusters3D, runKMeans, sqDist, type Vec2, type Vec3 } from "@/lib/ml/kmeans";
import { objectivePoints } from "./explorers";
import { ClusterPlot, ClusterTag, ObjectivePlot, plotDomain } from "./plots";

const KMeans3D = dynamic(() => import("./KMeans3D"), {
  ssr: false,
  loading: () => <div className="flex h-full min-h-40 items-center justify-center rounded-md bg-surface-2 text-[12px] text-fg-3">Loading the 3D view…</div>,
});

const SPEEDS: Record<"slow" | "normal" | "fast", SpeedDef> = {
  slow: { label: "slow", interval: 1400, stride: 1 },
  normal: { label: "normal", interval: 800, stride: 1 },
  fast: { label: "fast", interval: 260, stride: 1 },
};

const triples = (v: number[]): Vec3[] => Array.from({ length: v.length / 3 }, (_, j) => [v[3 * j], v[3 * j + 1], v[3 * j + 2]]);

export function KMeans3DExplorer({ n = 120, dataSeed = 4 }: { n?: number; dataSeed?: number }) {
  const { X, truth } = useMemo(() => generateClusters3D(n, dataSeed), [n, dataSeed]);
  const [k, setK] = useState(3);
  const [seed, setSeed] = useState(3);
  const [spokes, setSpokes] = useState(true);
  const [shadows, setShadows] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const webgl = useWebGLSupport();

  const run = useMemo(() => runKMeans(X, k, { method: "random", seed }), [X, k, seed]);
  const pb = usePlayback(run.phases.length, { speeds: SPEEDS, initialSpeed: "normal", resetKey: run });
  const phase = run.phases[pb.index];
  const tween = useTween(phase.centroids.flat(), { singleStep: pb.singleStep, duration: tweenDuration(pb.speedDef.interval) });
  const shownC = tween.moving ? triples(tween.value) : phase.centroids;
  const allJ = objectivePoints(run.phases);
  const atEnd = pb.index >= pb.last;
  const step = phase.kind === "init" ? "Initialization" : `Iteration ${phase.iteration} · ${phase.kind === "assign" ? "assignment" : "update"}`;
  const muIdx = phase.kind === "assign" ? phase.iteration - 1 : phase.iteration;

  const view3d =
    webgl === false ? (
      <Projections X={X} labels={phase.labels} centroids={shownC} k={k} />
    ) : (
      <Plot3DBoundary fallback={<Projections X={X} labels={phase.labels} centroids={shownC} k={k} />}>
        <KMeans3D
          X={X}
          labels={phase.labels}
          centroids={shownC}
          prevCentroids={phase.kind === "update" ? phase.prevCentroids : null}
          spokes={spokes && phase.kind !== "init" && !tween.moving}
          shadows={shadows}
          hovered={hovered}
          onHover={setHovered}
          height={380}
          ariaLabel={`K-Means with K = ${k} on three features: ${step}. ${phase.J != null ? `J = ${fmt(phase.J, 3)}.` : ""}`}
        />
      </Plot3DBoundary>
    );

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[13px] text-fg-2">
        <span className="inline-flex items-center gap-2">
          K
          <Segmented ariaLabel="Number of clusters" value={String(k)} onChange={(v) => setK(Number(v))} options={["2", "3", "4"].map((v) => ({ value: v, label: v }))} />
        </span>
        <Button variant="ghost" onClick={() => setSeed((s) => s + 1)}>
          New random start
        </Button>
        <span className="text-[12px] text-fg-3">start: K random data points (seed {seed})</span>
      </div>
      <PlaybackBar pb={pb} speeds={(["slow", "normal", "fast"] as const).map((v) => ({ value: v, label: v }))} scrubLabel="Half-step" status={<>{step}<span className="text-fg-3"> / {run.iterations}</span></>} />

      <div className="grid gap-x-6 gap-y-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          {view3d}
          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-fg-3">
            <Toggle label="assignment lines" checked={spokes} onChange={setSpokes} />
            <Toggle label="shadows on the (x₁, x₂) floor" checked={shadows} onChange={setShadows} />
            <span>Hover a point to see its distance to every centroid. Press Top view to see only x₁ and x₂.</span>
          </div>
          {phase.kind === "update" && <p className="mt-1 text-[11.5px] text-fg-3">The update is one jump from μ⁽ᵗ⁻¹⁾ (faint) to μ⁽ᵗ⁾; the glide only shows where each centroid goes.</p>}
        </div>

        <div className="min-w-0 space-y-3 text-[14px]">
          <div className="rounded-md border border-line bg-surface-2 px-3 py-2">
            <div className="text-[12px] font-medium text-fg-2">{step}</div>
            {phase.kind === "init" && <TexBlock>{`\\mu^{(0)}_j = x_{i_j},\\quad j = 1,\\dots,${k}\\quad\\text{(random data points)}`}</TexBlock>}
            {phase.kind === "assign" && (
              <>
                <TexBlock>{`\\begin{aligned} c_i &= \\arg\\min_j \\lVert x_i - \\mu^{(${muIdx})}_j\\rVert^2 \\\\ \\lVert x - \\mu\\rVert^2 &= (x_1 - \\mu_1)^2 + (x_2 - \\mu_2)^2 + (x_3 - \\mu_3)^2 \\end{aligned}`}</TexBlock>
                <p className="text-[13px] text-fg-2">{phase.iteration === 1 ? "Every point joins its nearest centroid, measured with all three features." : phase.changed.length ? `${phase.changed.length} point${phase.changed.length > 1 ? "s" : ""} changed cluster.` : "No point changed cluster: the next update leaves every centroid in place."}</p>
              </>
            )}
            {phase.kind === "update" && <TexBlock>{`\\mu^{(${phase.iteration})}_j = \\frac{1}{|C_j|}\\sum_{i\\in C_j} x_i \\quad\\text{(each feature averaged)}`}</TexBlock>}
            {phase.J != null && <TexBlock>{`J\\big(c^{(${phase.iteration})}, \\mu^{(${muIdx})}\\big) = ${texNum(phase.J, 3)}`}</TexBlock>}
          </div>
          {hovered != null && X[hovered] ? <DistancePanel X={X} i={hovered} centroids={phase.centroids} label={phase.labels?.[hovered] ?? null} /> : <CentroidTable centroids={phase.centroids} counts={phase.labels ? countLabels(phase.labels, k) : null} />}
          {atEnd && (
            <p className="text-[13px] text-fg-2">
              Converged after {run.iterations} iterations with J = {fmt(run.inertia, 3)}. The clusters match the three generating groups for{" "}
              <b className="font-medium text-fg">{fmt(agreement(run.labels, truth, Math.max(k, 3)) * 100, 0)}%</b> of the points{k !== 3 && <> (K = {k}, but the data has 3 groups)</>}. Press <i>Top view</i>: in x₁ and x₂ alone, two of the groups sit on top of each other.
            </p>
          )}
        </div>
      </div>

      <div className="min-w-0">
        <ObjectivePlot points={objectivePoints(run.phases, pb.index)} yMax={Math.max(...allJ.map((q) => q.J))} current={phase.kind === "init" ? null : { iteration: phase.iteration, phase: phase.kind }} maxIteration={run.iterations} height={200} />
      </div>
    </div>
  );
}

const countLabels = (labels: number[], k: number) => labels.reduce((c, l) => (c[l]++, c), new Array<number>(k).fill(0));

function CentroidTable({ centroids, counts }: { centroids: Vec3[]; counts: number[] | null }) {
  return (
    <table className="w-full text-left font-mono text-[12px] tabular-nums">
      <thead className="font-sans text-[11px] text-fg-3">
        <tr>
          <th className="py-0.5 pr-2 font-medium">centroid</th>
          <th className="py-0.5 pr-2 font-medium">(x₁, x₂, x₃)</th>
          <th className="py-0.5 font-medium">|Cⱼ|</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line">
        {centroids.map((c, j) => (
          <tr key={j}>
            <td className="py-0.5 pr-2 font-sans">
              <ClusterTag j={j} />
            </td>
            <td className="py-0.5 pr-2">({c.map((v) => fmt(v, 2)).join(", ")})</td>
            <td className="py-0.5">{counts ? counts[j] : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** ‖xᵢ − μⱼ‖² written out over the three features, for the hovered point. */
function DistancePanel({ X, i, centroids, label }: { X: Vec3[]; i: number; centroids: Vec3[]; label: number | null }) {
  const x = X[i];
  const d = centroids.map((c) => sqDist(x, c));
  const best = d.indexOf(Math.min(...d));
  const c = centroids[best];
  return (
    <div className="rounded-md border border-line px-3 py-2">
      <div className="text-[12px] font-medium text-fg-2">Point {i + 1} · nearest centroid μ{best + 1}</div>
      <TexBlock>{`\\lVert x_{${i + 1}} - \\mu_{${best + 1}}\\rVert^2 = ${[0, 1, 2].map((k) => texSq(x[k] - c[k], 2)).join(" + ")} = ${texNum(d[best], 3)}`}</TexBlock>
      <table className="w-full text-left font-mono text-[12px] tabular-nums">
        <tbody className="divide-y divide-line">
          {d.map((v, j) => (
            <tr key={j} className={j === best ? "font-semibold text-fg" : "text-fg-2"}>
              <td className="py-0.5 pr-2 font-sans">
                <ClusterTag j={j} />
              </td>
              <td className="py-0.5">
                {fmt(v, 3)} {j === best && <span className="font-sans font-normal text-fg-3">← nearest</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {label != null && label !== best && <p className="mt-1 text-[12.5px] text-fg-2">Labelled cluster {label + 1} now; it switches at the next assignment.</p>}
    </div>
  );
}

/** Without WebGL: the same state as two exact projections, (x₁, x₂) and (x₁, x₃). */
function Projections({ X, labels, centroids, k }: { X: Vec3[]; labels: number[] | null; centroids: Vec3[]; k: number }) {
  const views: { name: string; d: [number, number] }[] = [
    { name: "x₁ and x₂", d: [0, 1] },
    { name: "x₁ and x₃", d: [0, 2] },
  ];
  return (
    <div className="space-y-2">
      <p className="text-[12px] text-warn">WebGL is unavailable, so the same run is shown as two projections.</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {views.map((v) => {
          const P = X.map((p): Vec2 => [p[v.d[0]], p[v.d[1]]]);
          return (
            <div key={v.name} className="min-w-0">
              <div className="text-[12px] text-fg-2">Projection onto {v.name}</div>
              <ClusterPlot X={P} labels={labels} k={k} centroids={centroids.map((c): Vec2 => [c[v.d[0]], c[v.d[1]]])} domain={plotDomain(P)} height={240} ariaLabel={`Projection onto ${v.name}`} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
