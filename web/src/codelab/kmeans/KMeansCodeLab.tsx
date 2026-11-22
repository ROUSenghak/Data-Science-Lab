"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/controls";
import { Panel } from "@/components/ui/primitives";
import { fmt } from "@/lib/format";
import { assign, updateCentroids, type Vec2 } from "@/lib/ml/kmeans";
import { ClusterPlot, ClusterTag, ObjectivePlot, plotDomain } from "@/labs/k-means/plots";
import { findStepBy } from "../fold";
import { CodePanel, OfflineNotice, OutputPanel, ReplayToolbar, RunCallouts, VariablesPanel } from "../shell";
import type { KMeansState, RunResponse } from "../types";
import { useCodeLab } from "../useCodeLab";
import { KMeansInspector } from "./KMeansInspector";
import { allPositions, changedPoints, foldKMeans, iterationOf, objectiveHistory, shownLabels, type KFrame } from "./trace";

const FRAMEWORKS = [
  { id: "kmeans-numpy", key: "numpy", label: "NumPy", file: "kmeans_numpy.py" },
  { id: "kmeans-sklearn", key: "sklearn", label: "scikit-learn", file: "kmeans_sklearn.py" },
] as const;
type FrameworkKey = (typeof FRAMEWORKS)[number]["key"];

const INIT_NAMES: Record<string, string[]> = { numpy: ["random", "k-means++", "uniform"], sklearn: ["random", "k-means++"] };

const PRESETS: Record<FrameworkKey, { label: string; title: string; params: Record<string, number> }[]> = {
  numpy: [
    { label: "random points", title: "init_method = 0 with the current seeds", params: { init_method: 0 } },
    { label: "k-means++", title: "init_method = 1 with the current seeds", params: { init_method: 1 } },
    { label: "uniform box", title: "init_method = 2 with the current seeds", params: { init_method: 2 } },
    { label: "poor local minimum", title: "seed 28, random init seed 3: two blobs share a centroid", params: { seed: 28, n_samples: 60, n_blobs: 3, cluster_std: 1, k: 3, init_method: 0, init_seed: 3 } },
    { label: "empty cluster", title: "uniform init with K = 5: a centroid with no points keeps its position", params: { seed: 0, n_samples: 60, n_blobs: 3, cluster_std: 1, k: 5, init_method: 2, init_seed: 0 } },
  ],
  sklearn: [
    { label: "1 random init", title: 'init="random", n_init = 1', params: { init_method: 0, n_init: 1 } },
    { label: "10 random inits", title: 'init="random", n_init = 10: best of 10', params: { init_method: 0, n_init: 10 } },
    { label: "k-means++, n_init = 1", title: 'init="k-means++", n_init = 1', params: { init_method: 1, n_init: 1 } },
    { label: "poor local minimum data", title: "seed 28 (try random init with several random_state values)", params: { seed: 28, n_samples: 60, n_blobs: 3, cluster_std: 1, k: 3, init_method: 0, n_init: 1 } },
  ],
};

interface RunRecord {
  serial: number;
  framework: FrameworkKey;
  params: Record<string, number>;
  inertia: number | null;
  iterations: number | null;
  converged: boolean | null;
}

function summarize(fw: FrameworkKey, res: RunResponse<KMeansState>, serial: number): RunRecord {
  const frames = foldKMeans(res.snapshots);
  const st = frames.at(-1)?.state ?? {};
  const loops = frames.filter((f) => f.snap.step === "objective");
  return {
    serial,
    framework: fw,
    params: res.params,
    inertia: st.inertia ?? null,
    iterations: fw === "numpy" ? loops.length : (st.n_iter ?? null),
    converged: fw === "numpy" ? (loops.length ? st.n_changed === 0 : null) : null,
  };
}

export function KMeansCodeLab() {
  const [point, setPoint] = useState(0);
  const [records, setRecords] = useState<RunRecord[]>([]);
  const lab = useCodeLab<FrameworkKey, KMeansState, KFrame>({
    frameworks: FRAMEWORKS,
    fold: foldKMeans,
    onResult: (id, res) => {
      const fw = FRAMEWORKS.find((f) => f.id === id)!.key;
      setRecords((r) => [...r, summarize(fw, res, (r.at(-1)?.serial ?? -1) + 1)].slice(-8));
    },
  });
  const { fw, run, frames, frame, i, last } = lab;
  const domain = useMemo(() => plotDomain(allPositions(frames)), [frames]);

  if (lab.loadError?.offline) return <OfflineNotice onRetry={lab.retry} />;

  const st = frame?.state;
  const step = frame?.snap.step;
  const X = (st?.X ?? []) as Vec2[];
  const k = run?.params.k ?? st?.k ?? 1;
  const t = frame ? iterationOf(frame) : null;
  const totalIters = frames.filter((f) => f.snap.step === "objective").length;
  const stepInfo = lab.meta?.steps.find((s) => s.id === step);
  const sel = Math.min(point, Math.max(0, X.length - 1));

  const jumpToStep = (s: string) => {
    const k2 = findStepBy(frames, i, s, (x) => x.iteration);
    if (k2 >= 0) lab.goTo(k2);
  };
  const nextIteration = () => {
    for (let k2 = i + 1; k2 < frames.length; k2++) if (frames[k2].snap.step === "distances" || frames[k2].snap.step === "evaluate") return lab.goTo(k2);
    lab.goTo(last);
  };

  // What the plot shows at this snapshot.
  const centroids = (st?.centroids ?? null) as Vec2[] | null;
  const showCentroids = step !== "data" && step !== "model" ? centroids : null;
  const labels = st ? shownLabels(st) : null;
  const isUpdate = step === "update";
  let ringed: number[] = [];
  let ringLabel = "changed cluster";
  if (step === "assign" && st) ringed = changedPoints(st);
  else if ((isUpdate || step === "objective") && st?.labels && centroids && st.n_changed !== 0) {
    const next = assign(X, centroids);
    ringed = st.labels.flatMap((l, j) => (l !== next[j] ? [j] : []));
    ringLabel = "will switch at the next assignment";
  }
  const showDistances = step === "distances" || step === "assign" || step === "transform";

  return (
    <div className="space-y-4">
      <ReplayToolbar
        lab={lab}
        onSelectStep={jumpToStep}
        loopLabel="each iteration"
        extraButtons={
          fw === "numpy" && (
            <Button variant="ghost" onClick={nextIteration} disabled={!run || i >= last} title="Jump to the next iteration">
              Next iteration
            </Button>
          )
        }
        status={
          fw === "numpy" &&
          t != null && (
            <span>
              iteration {t}
              <span className="text-fg-3">/{totalIters}</span>
            </span>
          )
        }
      />

      <RunCallouts lab={lab} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,11fr)_minmax(0,13fr)]">
        <div className="min-w-0 space-y-3 xl:sticky xl:top-[9.5rem] xl:self-start">
          <CodePanel lab={lab} />
          {run && lab.vals && (
            <Panel title="Initialization experiments">
              <div className="flex flex-wrap gap-1.5">
                {PRESETS[fw].map((p) => (
                  <Button key={p.label} onClick={() => lab.runWith(p.params)} disabled={lab.running} title={p.title}>
                    {p.label}
                  </Button>
                ))}
              </div>
              <p className="mt-2 text-[11.5px] text-fg-3">Each button sets the boxed parameters and runs the script. Recent runs:</p>
              <div className="mt-1 overflow-x-auto">
                <table className="w-full text-left text-[12px] tabular-nums">
                  <thead className="text-[11px] text-fg-3">
                    <tr>
                      <th className="py-1 pr-2 font-medium">run</th>
                      <th className="py-1 pr-2 font-medium">version</th>
                      <th className="py-1 pr-2 font-medium">data seed</th>
                      <th className="py-1 pr-2 font-medium">K</th>
                      <th className="py-1 pr-2 font-medium">init</th>
                      <th className="py-1 pr-2 font-medium">iterations</th>
                      <th className="py-1 font-medium">final J</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line font-mono text-fg-2">
                    {[...records].reverse().map((r, j) => (
                      <tr key={r.serial} className={j === 0 ? "text-fg" : ""}>
                        <td className="py-1 pr-2">#{r.serial + 1}</td>
                        <td className="py-1 pr-2 font-sans">{r.framework === "numpy" ? "NumPy" : "scikit-learn"}</td>
                        <td className="py-1 pr-2">{r.params.seed}</td>
                        <td className="py-1 pr-2">{r.params.k}</td>
                        <td className="py-1 pr-2 font-sans">
                          {INIT_NAMES[r.framework][r.params.init_method]}
                          <span className="font-mono text-fg-3">
                            {r.framework === "numpy" ? ` · seed ${r.params.init_seed}` : ` · rs ${r.params.random_state} · n_init ${r.params.n_init}`}
                          </span>
                        </td>
                        <td className="py-1 pr-2">
                          {r.iterations ?? "—"}
                          {r.converged === false && <span className="font-sans text-warn"> limit</span>}
                        </td>
                        <td className="py-1">{fmt(r.inertia, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-1.5 text-[11.5px] text-fg-3">J values are traced from Python. Compare runs on the same data seed: different initializations can stop at different local minima.</p>
            </Panel>
          )}
          {run && <OutputPanel run={run} framework={fw} />}
        </div>

        <div className="min-w-0 space-y-4">
          <Panel title={<>Mathematics{stepInfo ? <> · <span className="text-fg">{stepInfo.title}</span></> : null}</>}>
            {frame && run ? (
              <>
                {stepInfo?.summary && <p className="mb-1 text-[13px] text-fg-2">{stepInfo.summary}</p>}
                <KMeansInspector frame={frame} framework={fw} point={sel} params={run.params} />
              </>
            ) : (
              <div className="py-6 text-center text-sm text-fg-3">{lab.running ? "Executing Python…" : "Run the script to record snapshots."}</div>
            )}
          </Panel>

          {X.length > 0 && (
            <Panel title="Clusters">
              <ClusterPlot
                X={X}
                labels={step === "data" || step === "model" ? null : labels}
                k={k}
                centroids={showCentroids}
                prevCentroids={isUpdate ? ((frame?.prev.centroids ?? null) as Vec2[] | null) : null}
                cells={showCentroids}
                changed={ringed}
                changedLabel={ringLabel}
                selected={sel}
                onSelect={setPoint}
                distancesFrom={showDistances ? sel : null}
                spokes={step === "assign" || step === "update" || step === "objective"}
                domain={domain}
                height={380}
                ariaLabel={`Data and clusters at snapshot ${i + 1}`}
              />
              <p className="mt-1 text-[11.5px] text-fg-3">
                Positions, labels and centroids are traced from Python. The dashed regions (nearest-centroid cells) are computed exactly in the browser from the traced centroids. Click a point to inspect it.
              </p>
            </Panel>
          )}

          {fw === "numpy" && run && (
            <Panel title="Objective J per half-step">
              <ObjectivePlot
                points={objectiveHistory(frames, i)}
                yMax={Math.max(0, ...objectiveHistory(frames, last).map((q) => q.J))}
                current={t != null && (step === "assign" || step === "objective") ? { iteration: t, phase: step === "assign" ? "assign" : "update" } : null}
                maxIteration={Math.max(1, totalIters)}
              />
              <p className="mt-1 text-[11.5px] text-fg-3">Hollow points: J after an assignment (intermediate). Filled points: J after the update, i.e. after a complete iteration. The sequence never increases.</p>
            </Panel>
          )}

          {fw === "sklearn" && run && st?.centroids && (step === "fit" || step === "transform" || step === "check") && (
            <Panel title="Fitted attributes">
              <SklearnResult st={st} X={X} />
            </Panel>
          )}

          {frame && <VariablesPanel frame={frame} />}
        </div>
      </div>
    </div>
  );
}

function SklearnResult({ st, X }: { st: KMeansState; X: Vec2[] }) {
  const C = st.centroids as Vec2[];
  const sizes = updateCentroids(X, st.labels ?? [], C).counts;
  return (
    <>
      <table className="w-full text-left font-mono text-[12.5px] tabular-nums">
        <thead className="text-[11px] text-fg-3">
          <tr>
            <th className="py-1 pr-3 font-sans font-medium">cluster</th>
            <th className="py-1 pr-3 font-sans font-medium">cluster_centers_</th>
            <th className="py-1 font-sans font-medium">points in labels_</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line text-fg">
          {C.map((c, j) => (
            <tr key={j}>
              <td className="py-1 pr-3 font-sans">
                <ClusterTag j={j} />
              </td>
              <td className="py-1 pr-3">
                ({fmt(c[0], 5)}, {fmt(c[1], 5)})
              </td>
              <td className="py-1">{sizes[j]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 font-mono text-[12.5px] tabular-nums text-fg-2">
        inertia_ = {fmt(st.inertia, 6)} · n_iter_ = {st.n_iter ?? "—"}
        {st.check_J != null && <> · our J = {fmt(st.check_J, 6)}</>}
      </p>
      <p className="mt-1 text-[11.5px] text-fg-3">All values come from the fitted estimator in the Python run; the iterations behind them are not observable.</p>
    </>
  );
}
