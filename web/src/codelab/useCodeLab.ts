"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { isParamValid } from "@/components/code/CodeView";
import { BackendUnavailable, getExperiment, runExperiment } from "./api";
import type { TraceFrame } from "./fold";
import type { ExperimentMeta, RunResponse, Snapshot } from "./types";

/**
 * Lab-independent Code Lab state: experiment metadata, parameter values, execution on the
 * Python service, and replay of the recorded snapshots. Each lab supplies its experiments,
 * a fold function (snapshots → frames) and renders its own inspector and plots.
 */

export const SPEEDS = { slow: 600, normal: 160, fast: 30 } as const;
export type Speed = keyof typeof SPEEDS;

export interface FrameworkDef<K extends string = string> {
  id: string;
  key: K;
  label: string;
  file: string;
}

export function useCodeLab<K extends string, S, F extends TraceFrame<S>>(opts: {
  frameworks: readonly FrameworkDef<K>[];
  fold: (snapshots: Snapshot<S>[]) => F[];
  onResult?: (experimentId: string, res: RunResponse<S>) => void;
}) {
  const { frameworks, fold } = opts;
  const search = useSearchParams();
  const initial = frameworks.find((f) => f.key === search.get("framework"))?.key ?? frameworks[0].key;
  const [fw, setFw] = useState<K>(initial);
  const exp = frameworks.find((f) => f.key === fw)!;

  const [metas, setMetas] = useState<Record<string, ExperimentMeta>>({});
  const [values, setValues] = useState<Record<string, Record<string, number>>>({});
  const [runs, setRuns] = useState<Record<string, RunResponse<S>>>({});
  const [loadError, setLoadError] = useState<{ offline: boolean; message: string } | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>("normal");
  const [attempt, setAttempt] = useState(0);
  const onResult = useRef(opts.onResult);
  useEffect(() => {
    onResult.current = opts.onResult;
  });

  const meta = metas[exp.id];
  const run = runs[exp.id];
  const vals = values[exp.id];

  const execute = useCallback(async (id: string, params: Record<string, number>) => {
    setRunning(true);
    setRunError(null);
    setPlaying(false);
    try {
      const res = await runExperiment<S>(id, params);
      setRuns((r) => ({ ...r, [id]: res }));
      setIdx(0);
      onResult.current?.(id, res);
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

  const frames = useMemo(() => (run ? fold(run.snapshots) : []), [run, fold]);
  const last = Math.max(0, frames.length - 1);
  const i = Math.min(idx, last);
  const frame: F | undefined = frames[i];

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
      const t = e.target;
      if (t instanceof Element && t.closest("input, textarea, select, [role=dialog]")) return;
      if (e.key === "ArrowRight") setIdx((v) => Math.min(last, v + 1));
      else if (e.key === "ArrowLeft") setIdx((v) => Math.max(0, v - 1));
      else return;
      setPlaying(false);
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [last]);

  const switchFramework = (key: K) => {
    setFw(key);
    setIdx(0);
    setPlaying(false);
    setRunError(null);
    setLoadError(null);
    const url = new URL(window.location.href);
    url.searchParams.set("framework", key);
    window.history.replaceState(null, "", url);
  };

  const stepCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const f of frames) c[f.snap.step] = (c[f.snap.step] ?? 0) + 1;
    return c;
  }, [frames]);

  /** Go to snapshot k and stop playback. */
  const goTo = (k: number) => {
    setPlaying(false);
    setIdx(Math.max(0, Math.min(last, k)));
  };

  const setParam = (name: string, v: number) => setValues((all) => ({ ...all, [exp.id]: { ...all[exp.id], [name]: v } }));
  /** Set several parameters and execute immediately. */
  const runWith = (patch: Record<string, number>) => {
    if (!vals) return;
    const next = { ...vals, ...patch };
    setValues((all) => ({ ...all, [exp.id]: next }));
    execute(exp.id, next);
  };

  const allValid = !!meta && !!vals && meta.params.every((p) => isParamValid(p, vals[p.name]));
  const stale = !!run && !!vals && Object.entries(vals).some(([k, v]) => run.params[k] !== v);

  return {
    fw,
    exp,
    frameworks,
    meta,
    run,
    vals,
    frames,
    frame,
    i,
    last,
    playing,
    setPlaying,
    setIdx,
    goTo,
    speed,
    setSpeed,
    running,
    runError,
    loadError,
    retry: () => {
      setLoadError(null);
      setAttempt((a) => a + 1);
    },
    execute,
    runWith,
    setParam,
    switchFramework,
    stepCounts,
    allValid,
    stale,
  };
}

export type CodeLabState<K extends string, S, F extends TraceFrame<S>> = ReturnType<typeof useCodeLab<K, S, F>>;
