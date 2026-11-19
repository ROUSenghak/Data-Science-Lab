"use client";

import { useMemo, useState } from "react";
import { Tex } from "@/components/math/Tex";
import { Button, Segmented, Slider } from "@/components/ui/controls";
import { Callout, ExecutionBadge, SectionHeading } from "@/components/ui/primitives";
import { fmt } from "@/lib/format";
import { DATASETS, generateClusters, MAX_K, runKMeans, type ClusterDataKind, type InitMethod, type Vec2 } from "@/lib/ml/kmeans";
import { ElbowExplorer, InitComparison, INIT_METHODS, LloydExplorer } from "./explorers";

type Kind = ClusterDataKind | "custom";

const FAILURES: ClusterDataKind[] = ["elongated", "unequal", "rings", "moons", "overlapping", "uniform"];

export function KMeansLab() {
  const [kind, setKind] = useState<Kind>("blobs");
  const [n, setN] = useState(150);
  const [spread, setSpread] = useState(1);
  const [dataSeed, setDataSeed] = useState(3);
  const [added, setAdded] = useState<Vec2[]>([]);
  const [addMode, setAddMode] = useState(false);
  const [k, setK] = useState(3);
  const [method, setMethod] = useState<InitMethod>("random");
  const [seed, setSeed] = useState(1);

  const base = useMemo(() => (kind === "custom" ? { X: [] as Vec2[], truth: [] as number[] } : generateClusters(kind, n, dataSeed, spread)), [kind, n, dataSeed, spread]);
  const X = useMemo(() => [...base.X, ...added], [base, added]);
  const truth = useMemo(() => (kind === "custom" || kind === "uniform" ? null : [...base.truth, ...added.map(() => Math.max(...base.truth) + 1)]), [base, added, kind]);
  const summary = useMemo(() => (X.length >= k ? runKMeans(X, k, { method, seed }) : null), [X, k, method, seed]);
  const note = DATASETS.find((d) => d.kind === kind)?.note;

  const loadDataset = (next: Kind) => {
    setKind(next);
    setAdded([]);
    const d = DATASETS.find((x) => x.kind === next);
    if (d) setK(d.k);
    if (next === "custom") setAddMode(true);
  };

  return (
    <div className="grid gap-8 lg:grid-cols-[250px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-20 lg:self-start">
        <div className="space-y-4 rounded-lg border border-line bg-surface p-4">
          <div>
            <label htmlFor="km-dataset" className="text-sm font-semibold">
              Dataset
            </label>
            <select
              id="km-dataset"
              value={kind}
              onChange={(e) => loadDataset(e.target.value as Kind)}
              className="mt-1.5 w-full rounded-md border border-line-strong bg-surface px-2 py-1.5 text-[13px]"
            >
              {DATASETS.map((d) => (
                <option key={d.kind} value={d.kind}>
                  {d.label}
                </option>
              ))}
              <option value="custom">Empty canvas (click to add)</option>
            </select>
          </div>
          {kind !== "custom" && (
            <>
              <Slider label="points n" value={n} min={20} max={300} step={10} onChange={setN} />
              <Slider label="spread" value={spread} min={0.3} max={2} step={0.1} onChange={setSpread} format={(v) => `${fmt(v, 1)}×`} />
              <Slider label="data seed" value={dataSeed} min={0} max={99} step={1} onChange={setDataSeed} />
            </>
          )}
          <div className="space-y-2 border-t border-line pt-3">
            <label className="flex items-center gap-2 text-[13px] text-fg-2">
              <input type="checkbox" className="accent-[var(--viz-model)]" checked={addMode} onChange={(e) => setAddMode(e.target.checked)} />
              Click in the plot to add points
            </label>
            <div className="flex items-center justify-between text-[12px] text-fg-3">
              <span>{added.length} added</span>
              <Button variant="ghost" onClick={() => setAdded([])} disabled={!added.length}>
                Clear added
              </Button>
            </div>
          </div>
          <div className="space-y-3 border-t border-line pt-3">
            <Slider label={<Tex>{"\\text{clusters } K"}</Tex>} value={k} min={1} max={MAX_K} step={1} onChange={setK} />
            <div>
              <div className="mb-1 text-[13px] text-fg-2">Initialization</div>
              <Segmented ariaLabel="Initialization method" value={method} onChange={setMethod} options={INIT_METHODS.map((m) => ({ value: m.id, label: m.label }))} />
              <div className="mt-1 text-[11px] leading-snug text-fg-3">{INIT_METHODS.find((m) => m.id === method)?.hint}</div>
            </div>
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <Slider label="init seed" value={seed} min={0} max={29} step={1} onChange={setSeed} />
              </div>
              <Button onClick={() => setSeed((s) => (s + 1) % 30)} title="Next initialization seed">
                next
              </Button>
            </div>
          </div>
          <dl className="grid grid-cols-2 gap-x-2 gap-y-1 border-t border-line pt-3 font-mono text-[11.5px] tabular-nums text-fg-2">
            <dt className="text-fg-3">points</dt>
            <dd>{X.length}</dd>
            <dt className="text-fg-3">final J</dt>
            <dd>{summary ? fmt(summary.inertia, 4) : "—"}</dd>
            <dt className="text-fg-3">iterations</dt>
            <dd>{summary ? `${summary.iterations}${summary.converged ? "" : " (limit)"}` : "—"}</dd>
          </dl>
          <ExecutionBadge kind="browser" />
        </div>
      </aside>

      <div className="min-w-0">
        <SectionHeading id="algorithm" kicker="1" first>
          Lloyd&apos;s algorithm, one half-step at a time
        </SectionHeading>
        <p className="mb-3 max-w-2xl text-[15px] text-fg-2">
          Each iteration has two half-steps. <b className="font-medium text-fg">Assign</b>: every point joins its nearest centroid. <b className="font-medium text-fg">Update</b>: every centroid moves to the mean of its points. Step through them and watch which points change cluster, and why.
        </p>
        {note && kind !== "blobs" && (
          <div className="mb-3">
            <Callout tone={FAILURES.includes(kind as ClusterDataKind) ? "warn" : "note"}>{note}</Callout>
          </div>
        )}
        <LloydExplorer X={X} truth={truth} k={k} method={method} seed={seed} onAddPoint={addMode ? (p) => setAdded((a) => [...a, p]) : undefined} />

        <SectionHeading id="initialization" kicker="2">
          Initialization and local minima
        </SectionHeading>
        <p className="mb-3 max-w-2xl text-[15px] text-fg-2">
          Lloyd&apos;s algorithm only ever goes downhill, so where it starts decides where it stops. Below, 30 initializations of each method on the current data and K.
        </p>
        <InitComparison
          X={X}
          k={k}
          selected={{ method, seed }}
          onPick={(m, s) => {
            setMethod(m);
            setSeed(s);
            document.getElementById("algorithm")?.scrollIntoView({ behavior: "smooth" });
          }}
        />

        <SectionHeading id="choosing-k" kicker="3">
          Choosing K
        </SectionHeading>
        <p className="mb-3 max-w-2xl text-[15px] text-fg-2">
          The objective cannot choose K: more centroids always fit at least as well. Read the bend of J*(K) and the silhouette together, and check the result against what the clusters are for.
        </p>
        <ElbowExplorer X={X} currentK={k} />

        <SectionHeading id="failure-cases" kicker="4">
          Failure cases
        </SectionHeading>
        <p className="mb-3 max-w-2xl text-[15px] text-fg-2">
          K-Means assumes compact, roughly round clusters of similar size. Load a dataset, then switch the plot to <i>colour: generating groups</i> to compare.
        </p>
        <ul className="divide-y divide-line border-y border-line">
          {FAILURES.map((f) => {
            const d = DATASETS.find((x) => x.kind === f)!;
            return (
              <li key={f} className="flex flex-col gap-2 py-3 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
                <div className="min-w-0">
                  <div className="font-medium">
                    {d.label} <span className="font-normal text-fg-3">· K = {d.k}</span>
                  </div>
                  <div className="text-[14px] text-fg-3">{d.note}</div>
                </div>
                <Button
                  onClick={() => {
                    loadDataset(f);
                    document.getElementById("algorithm")?.scrollIntoView({ behavior: "smooth" });
                  }}
                  variant={kind === f ? "primary" : "default"}
                >
                  {kind === f ? "Loaded" : "Load"}
                </Button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
