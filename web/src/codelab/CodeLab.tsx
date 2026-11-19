"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CodeView, isParamValid } from "@/components/code/CodeView";
import { Button, Icon, Segmented } from "@/components/ui/controls";
import { Badge, Callout, ExecutionBadge, Panel } from "@/components/ui/primitives";
import { fmt } from "@/lib/format";
import { leastSquares } from "@/lib/ml/linreg";
import { DataFitPlot, LossCurvePlot, LossSurfacePlot, type FitLine } from "@/labs/linear-regression/plots";
import { BackendUnavailable, getExperiment, runExperiment } from "./api";
import { foldSnapshots, lossHistory, paramPath } from "./fold";
import { Inspector } from "./Inspector";
import type { ExperimentMeta, RunResponse } from "./types";

const FRAMEWORKS = [
  { id: "linreg-numpy", key: "numpy", label: "NumPy", file: "linreg_numpy.py" },
  { id: "linreg-pytorch", key: "pytorch", label: "PyTorch", file: "linreg_pytorch.py" },
  { id: "linreg-sklearn", key: "sklearn", label: "scikit-learn", file: "linreg_sklearn.py" },
] as const;
type FrameworkKey = (typeof FRAMEWORKS)[number]["key"];

const SPEEDS = { slow: 600, normal: 160, fast: 30 } as const;

export function CodeLab() {
  const search = useSearchParams();
  const initial = (FRAMEWORKS.find((f) => f.key === search.get("framework"))?.key ?? "numpy") as FrameworkKey;
  const [fw, setFw] = useState<FrameworkKey>(initial);
  const exp = FRAMEWORKS.find((f) => f.key === fw)!;

  const [metas, setMetas] = useState<Record<string, ExperimentMeta>>({});
  const [values, setValues] = useState<Record<string, Record<string, number>>>({});
  const [runs, setRuns] = useState<Record<string, RunResponse>>({});
  const [loadError, setLoadError] = useState<{ offline: boolean; message: string } | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<keyof typeof SPEEDS>("normal");
  const [point, setPoint] = useState(0);
  const [logLoss, setLogLoss] = useState(false);
  const [attempt, setAttempt] = useState(0);

  const meta = metas[exp.id];
  const run = runs[exp.id];
  const vals = values[exp.id];

  const execute = useCallback(async (id: string, params: Record<string, number>) => {
    setRunning(true);
    setRunError(null);
    setPlaying(false);
    try {
      const res = await runExperiment(id, params);
      setRuns((r) => ({ ...r, [id]: res }));
      setIdx(0);
    } catch (e) {
      if (e instanceof BackendUnavailable) setLoadError({ offline: true, message: e.message });
      else setRunError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }, []);

  // Load experiment metadata (source + parameter specs), then execute once with defaults.
  useEffect(() => {
    if (metas[exp.id]) return;
    let cancelled = false;
    getExperiment(exp.id)
      .then((m) => {
        if (cancelled) return;
        setMetas((x) => ({ ...x, [m.id]: m }));
        const defaults = Object.fromEntries(m.params.map((p) => [p.name, p.default]));
        setValues((v) => ({ ...v, [m.id]: v[m.id] ?? defaults }));
        if (!runs[m.id]) execute(m.id, defaults);
      })
      .catch((e: Error) => !cancelled && setLoadError({ offline: e instanceof BackendUnavailable, message: e.message }));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exp.id, attempt]);

  const frames = useMemo(() => (run ? foldSnapshots(run.snapshots) : []), [run]);
  const last = Math.max(0, frames.length - 1);
  const i = Math.min(idx, last);
  const frame = frames[i];

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setIdx((v) => {
        if (v >= last) {
          setPlaying(false);
          return v;
        }
        return v + 1;
      });
    }, SPEEDS[speed]);
    return () => window.clearInterval(id);
  }, [playing, speed, last]);

  // keyboard: ← / → step through snapshots
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [role=dialog]")) return;
      if (e.key === "ArrowRight") setIdx((v) => Math.min(last, v + 1));
      else if (e.key === "ArrowLeft") setIdx((v) => Math.max(0, v - 1));
      else return;
      setPlaying(false);
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  const switchFramework = (key: FrameworkKey) => {
    setFw(key);
    setIdx(0);
    setPlaying(false);
    setRunError(null);
    setLoadError(null);
    const url = new URL(window.location.href);
    url.searchParams.set("framework", key);
    window.history.replaceState(null, "", url);
  };

  const nextEpoch = () => {
    for (let k = i + 1; k < frames.length; k++) {
      if (frames[k].state.epoch !== frame?.state.epoch && frames[k].state.epoch != null) return setIdx(k);
    }
    setIdx(last);
  };

  const allValid = !!meta && !!vals && meta.params.every((p) => isParamValid(p, vals[p.name]));
  const stale = !!run && !!vals && Object.entries(vals).some(([k, v]) => run.params[k] !== v);

  /* ---------------------------------------------------------------- render */

  if (loadError?.offline) {
    return (
      <div className="mx-auto max-w-2xl py-10">
        <Callout tone="bad" title="The Python execution service is not running">
          <p>The Code Lab only shows results from real Python execution, so nothing is displayed without the service. Start it in a second terminal:</p>
          <pre className="mt-2 overflow-x-auto rounded bg-surface px-3 py-2 font-mono text-[12.5px]">cd api && .venv/bin/uvicorn app.main:app --port 8000</pre>
          <div className="mt-3">
            <Button
              onClick={() => {
                setLoadError(null);
                setAttempt((a) => a + 1);
              }}
            >
              Retry
            </Button>
          </div>
        </Callout>
      </div>
    );
  }

  const stepTitle = meta?.steps.find((s) => s.id === frame?.snap.step)?.title ?? frame?.snap.step;
  const st = frame?.state;
  const data = st?.x && st?.y ? { x: st.x, y: st.y } : null;
  const ls = data ? leastSquares(data) : null;
  const isGD = fw !== "sklearn";
  const epochs = run?.params.epochs ?? 1;

  const fitLines: FitLine[] = [];
  if (frame && st?.w != null && st?.b != null) {
    const fp = frame.forwardParams;
    if (fp && (fp.w !== st.w || fp.b !== st.b)) {
      fitLines.push({ ...fp, label: "model used for ŷ", color: "var(--viz-model)" });
      fitLines.push({ w: st.w, b: st.b, label: "after update", color: "var(--viz-model)", dashed: true });
    } else fitLines.push({ w: st.w, b: st.b, label: `model ŷ = ${fmt(st.w, 3)}x + ${fmt(st.b, 3)}`, color: "var(--viz-model)" });
  }
  const yHat = st?.y_hat && st.y_hat.every((v) => v != null) ? (st.y_hat as number[]) : null;
  const showNext = (frame?.snap.step === "gradient" || frame?.snap.step === "backward") && st?.w != null && st?.b != null && st?.dw != null && st?.db != null && st?.lr != null;

  return (
    <div className="space-y-4">
      {/* toolbar */}
      <div className="z-30 -mx-4 md:sticky md:top-14 border-b border-line bg-bg/95 px-4 py-2.5 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="flex flex-wrap items-center gap-2">
          <Segmented ariaLabel="Implementation" value={fw} onChange={switchFramework} options={FRAMEWORKS.map((f) => ({ value: f.key, label: f.label }))} />
          <Button variant="primary" onClick={() => vals && execute(exp.id, vals)} disabled={running || !allValid} title="Execute the script on the Python service">
            {Icon.run} {running ? "Running…" : "Run"}
          </Button>
          <span className="mx-1 hidden h-5 w-px bg-line sm:block" aria-hidden />
          <Button onClick={() => (setPlaying(false), setIdx(0))} disabled={!run} ariaLabel="Reset to first snapshot" title="Reset to first snapshot">
            {Icon.reset}
          </Button>
          <Button onClick={() => (setPlaying(false), setIdx((v) => Math.max(0, v - 1)))} disabled={!run || i === 0} ariaLabel="Previous snapshot" title="Previous snapshot (←)">
            {Icon.back}
          </Button>
          <Button onClick={() => (i >= last ? (setIdx(0), setPlaying(true)) : setPlaying((p) => !p))} disabled={!run} ariaLabel={playing ? "Pause replay" : "Play replay"}>
            {playing ? Icon.pause : Icon.play} {playing ? "Pause" : "Play"}
          </Button>
          <Button onClick={() => (setPlaying(false), setIdx((v) => Math.min(last, v + 1)))} disabled={!run || i >= last} ariaLabel="Next snapshot" title="Next snapshot (→)">
            {Icon.step} Step
          </Button>
          {isGD && (
            <Button variant="ghost" onClick={() => (setPlaying(false), nextEpoch())} disabled={!run || i >= last} title="Jump to the next epoch">
              Next epoch
            </Button>
          )}
          <Segmented ariaLabel="Replay speed" value={speed} onChange={setSpeed} options={[{ value: "slow", label: "slow" }, { value: "normal", label: "normal" }, { value: "fast", label: "fast" }]} />
          <div className="flex min-w-48 flex-1 items-center gap-2">
            <input type="range" aria-label="Snapshot" className="flex-1" min={0} max={last} value={i} disabled={!run} onChange={(e) => (setPlaying(false), setIdx(Number(e.target.value)))} />
            <span className="whitespace-nowrap font-mono text-[11.5px] tabular-nums text-fg-2">
              {run ? `${i + 1}/${frames.length}` : "—"}
            </span>
          </div>
        </div>
        {frame && (
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12.5px] text-fg-2">
            <span>
              Step: <span className="font-medium text-fg">{stepTitle}</span>
            </span>
            {st?.epoch != null && <span className="font-mono tabular-nums">epoch {st.epoch}</span>}
            <span className="font-mono tabular-nums">
              line {frame.snap.lines[0]}
              {frame.snap.lines[1] !== frame.snap.lines[0] ? `–${frame.snap.lines[1]}` : ""}
            </span>
            <span className="hidden text-fg-3 sm:inline">Replaying a recorded trace: Python ran once; each snapshot is the program state after the highlighted lines.</span>
          </div>
        )}
      </div>

      {stale && (
        <Callout tone="warn">
          Parameters changed since the last run. Press <b>Run</b> to execute with the new values; the panels still show the previous run.
        </Callout>
      )}
      {runError && <Callout tone="bad" title="Run rejected">{runError}</Callout>}
      {run?.error && <Callout tone="bad" title="Python raised an exception">{run.error}</Callout>}
      {loadError && !loadError.offline && <Callout tone="bad" title="Could not load experiment">{loadError.message}</Callout>}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,11fr)_minmax(0,13fr)]">
        {/* A. code */}
        <div className="min-w-0 space-y-3 xl:sticky xl:top-[9.5rem] xl:self-start">
          <Panel
            title={
              <span className="flex items-center gap-2">
                <span className="font-mono">{exp.file}</span>
                {meta && (meta.execution === "library" ? <Badge tone="accent">Library computation</Badge> : <Badge>Educational implementation</Badge>)}
              </span>
            }
            actions={<ExecutionBadge kind="python" />}
          >
            {meta && vals ? (
              <>
                <CodeView
                  source={run && !stale ? run.source : meta.source}
                  active={frame ? frame.snap.lines : null}
                  groups={meta.step_groups}
                  params={meta.params}
                  paramLines={meta.param_lines}
                  values={vals}
                  onParamChange={(name, v) => setValues((all) => ({ ...all, [exp.id]: { ...all[exp.id], [name]: v } }))}
                  onLineClick={(line) => {
                    // jump to the next snapshot produced by the clicked step group
                    const k = frames.findIndex((f, j) => j > i && line >= f.snap.lines[0] && line <= f.snap.lines[1]);
                    const k2 = k >= 0 ? k : frames.findIndex((f) => line >= f.snap.lines[0] && line <= f.snap.lines[1]);
                    if (k2 >= 0) {
                      setPlaying(false);
                      setIdx(k2);
                    }
                  }}
                  maxHeight={620}
                />
                <ul className="mt-2 space-y-0.5 text-[12px] leading-snug text-fg-3">
                  {meta.notes.map((n) => (
                    <li key={n}>· {n}</li>
                  ))}
                  <li>· Edit the highlighted numbers, then Run. Click a marked line (grey gutter) to jump to its next snapshot.</li>
                </ul>
              </>
            ) : (
              <div className="py-10 text-center text-sm text-fg-3">Loading source from the execution service…</div>
            )}
          </Panel>
          {run && (
            <Panel title="Program output">
              <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[12px] text-fg">{run.stdout || "(no output)"}</pre>
              {run.warnings.map((w) => (
                <div key={w} className="mt-1 font-mono text-[12px] text-warn">
                  {w}
                </div>
              ))}
              <div className="mt-2 text-[11.5px] text-fg-3">
                {run.snapshots.length} snapshots · executed in {fmt(run.elapsed_ms, 1)} ms · Python {run.environment.python} · NumPy {run.environment.numpy}
                {fw === "pytorch" && <> · PyTorch {run.environment.torch}</>}
                {fw === "sklearn" && <> · scikit-learn {run.environment.sklearn}</>}
              </div>
            </Panel>
          )}
        </div>

        {/* B + C. inspector and visualizations */}
        <div className="min-w-0 space-y-4">
          <Panel title={<>Mathematical inspector{stepTitle ? <> · <span className="text-fg">{stepTitle}</span></> : null}</>}>
            {frame ? (
              <Inspector frame={frame} framework={fw} point={point} />
            ) : (
              <div className="py-6 text-center text-sm text-fg-3">{running ? "Executing Python…" : "Run the script to record snapshots."}</div>
            )}
          </Panel>

          {data && (
            <>
              <Panel title="Data and model">
                <DataFitPlot data={data} lines={fitLines} residualFrom={yHat} selected={point} onSelect={setPoint} height={280} />
              </Panel>
              <div className="grid gap-4 lg:grid-cols-2">
                <Panel title="Parameter space (w, b)">
                  <LossSurfacePlot
                    data={data}
                    trajectory={paramPath(frames, i)}
                    current={st?.w != null && st?.b != null ? [st.w, st.b] : null}
                    next={showNext ? [st!.w! - st!.lr! * st!.dw!, st!.b! - st!.lr! * st!.db!] : null}
                    anchor={run?.params.w_init != null ? [run.params.w_init, run.params.b_init] : undefined}
                    height={260}
                  />
                  <p className="mt-1 text-[11.5px] text-fg-3">Level sets are computed exactly in the browser from the traced x, y; the path is the traced (w, b).</p>
                </Panel>
                {isGD ? (
                  <Panel
                    title="Training loss"
                    actions={<Segmented ariaLabel="Loss axis scale" value={logLoss ? "log" : "lin"} onChange={(v) => setLogLoss(v === "log")} options={[{ value: "lin", label: "linear" }, { value: "log", label: "log" }]} />}
                  >
                    <LossCurvePlot losses={lossHistory(frames, i)} totalEpochs={epochs} minimum={ls?.loss} current={st?.epoch ?? null} log={logLoss} height={260} />
                  </Panel>
                ) : (
                  <Panel title="Result">
                    <table className="w-full font-mono text-[12.5px] tabular-nums">
                      <thead className="text-left text-fg-3">
                        <tr>
                          <th className="py-1 font-normal" />
                          <th className="py-1 font-normal">w</th>
                          <th className="py-1 font-normal">b</th>
                        </tr>
                      </thead>
                      <tbody className="text-fg">
                        <tr className="border-t border-line">
                          <td className="py-1 font-sans text-fg-2">scikit-learn</td>
                          <td>{fmt(st?.w, 6)}</td>
                          <td>{fmt(st?.b, 6)}</td>
                        </tr>
                        <tr className="border-t border-line">
                          <td className="py-1 font-sans text-fg-2">normal equations</td>
                          <td>{st?.theta ? fmt(st.theta[0], 6) : "—"}</td>
                          <td>{st?.theta ? fmt(st.theta[1], 6) : "—"}</td>
                        </tr>
                      </tbody>
                    </table>
                    <p className="mt-2 text-[12px] text-fg-3">Both rows come from the Python run. The normal-equations row appears at the final step.</p>
                  </Panel>
                )}
              </div>
            </>
          )}

          {frame && (
            <Panel title="Variables (Python namespace at this snapshot)">
              <div className="max-h-80 overflow-auto">
                <table className="w-full text-left text-[11.5px]">
                  <thead className="sticky top-0 bg-surface text-[10.5px] uppercase tracking-wide text-fg-3">
                    <tr>
                      <th className="py-1 pr-3 font-medium">name</th>
                      <th className="py-1 pr-3 font-medium">type</th>
                      <th className="py-1 font-medium">value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line font-mono">
                    {Object.entries(frame.locals).map(([name, v]) => {
                      const changed = name in frame.snap.locals && frame.snap.index > 0;
                      return (
                        <tr key={name} className={changed ? "bg-accent-soft" : ""}>
                          <td className={`py-1 pr-3 align-top ${changed ? "font-semibold text-fg" : "text-fg-2"}`}>
                            {name}
                            {changed && <span className="sr-only"> (changed)</span>}
                          </td>
                          <td className="whitespace-nowrap py-1 pr-3 align-top text-fg-3">
                            {v.type}
                            {v.shape ? ` ${JSON.stringify(v.shape)}` : ""}
                            {v.dtype ? ` ${v.dtype}` : ""}
                          </td>
                          <td className="py-1 align-top">
                            <pre className="whitespace-pre text-fg">{v.value}</pre>
                            {v.grad_fn && <div className="text-[10.5px] text-fg-3">grad_fn={v.grad_fn}</div>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="mt-1.5 text-[11px] text-fg-3">Highlighted rows changed in this snapshot.</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
