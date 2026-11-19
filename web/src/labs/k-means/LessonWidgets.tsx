"use client";

import { useMemo, useState } from "react";
import { Segmented } from "@/components/ui/controls";
import { fmt } from "@/lib/format";
import { DATASETS, generateClusters, runKMeans, type ClusterDataKind, type InitMethod, type KMeansRun } from "@/lib/ml/kmeans";
import { ElbowExplorer, InitComparison, INIT_METHODS, LloydExplorer } from "./explorers";
import { ClusterPlot, plotDomain } from "./plots";

const useBlobs = (seed = 3) => useMemo(() => generateClusters("blobs", 90, seed, 1), [seed]);

export function LessonLloyd() {
  const { X, truth } = useBlobs();
  const [method, setMethod] = useState<InitMethod>("random");
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-[13px] text-fg-2">
        Initialization
        <Segmented ariaLabel="Initialization" value={method} onChange={setMethod} options={INIT_METHODS.map((m) => ({ value: m.id, label: m.label }))} />
      </div>
      <LloydExplorer X={X} truth={truth} k={3} method={method} seed={2} />
    </div>
  );
}

/** The same data, two random starts: one reaches the best clustering, one is stuck. */
export function LessonLocalMinimum() {
  const { X } = useBlobs(5);
  const runs = useMemo(() => [0, 1].map((seed) => runKMeans(X, 3, { method: "random", seed })), [X]);
  const domain = useMemo(() => plotDomain(X), [X]);
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {runs.map((r, s) => (
        <div key={s} className="min-w-0">
          <div className="mb-1 text-[12.5px] text-fg-2">
            random start, seed {s}: <span className="font-mono tabular-nums text-fg">J = {fmt(r.inertia, 1)}</span> after {r.iterations} iterations
          </div>
          <ClusterPlot X={X} labels={r.labels} k={3} centroids={r.centroids} cells={r.centroids} domain={domain} height={260} ariaLabel={`Converged clustering from random seed ${s}`} />
        </div>
      ))}
    </div>
  );
}

export function LessonInit() {
  const { X } = useBlobs();
  return <InitComparison X={X} k={3} />;
}

export function LessonElbow() {
  const { X } = useBlobs();
  return <ElbowExplorer X={X} currentK={3} />;
}

const GALLERY: ClusterDataKind[] = ["elongated", "unequal", "rings", "moons"];

/** Best of 10 k-means++ runs on four datasets that violate K-Means' assumptions. */
export function LessonFailures() {
  const cases = useMemo(
    () =>
      GALLERY.map((kind) => {
        const d = DATASETS.find((x) => x.kind === kind)!;
        const { X } = generateClusters(kind, 160, 3, 1);
        let best: KMeansRun | null = null;
        for (let s = 0; s < 10; s++) {
          const r = runKMeans(X, d.k, { method: "kmeans++", seed: s });
          if (!best || r.inertia < best.inertia) best = r;
        }
        return { d, X, run: best!, domain: plotDomain(X) };
      }),
    [],
  );
  return (
    <div className="grid gap-x-4 gap-y-5 md:grid-cols-2">
      {cases.map(({ d, X, run, domain }) => (
        <div key={d.kind} className="min-w-0">
          <div className="text-[13px] font-medium text-fg">
            {d.label} <span className="font-normal text-fg-3">· K = {d.k}</span>
          </div>
          <p className="mb-1 text-[12.5px] leading-snug text-fg-3">{d.note}</p>
          <ClusterPlot X={X} labels={run.labels} k={d.k} centroids={run.centroids} cells={run.centroids} domain={domain} height={240} ariaLabel={`K-Means on ${d.label}`} />
        </div>
      ))}
    </div>
  );
}
