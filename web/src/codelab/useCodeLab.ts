"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePlayback } from "@/components/anim/hooks";
import { isParamValid } from "@/components/code/CodeView";
import type { SpeedDef } from "@/lib/anim/playback";
import { BackendUnavailable, getExperiment, runExperiment } from "./api";
import type { TraceFrame } from "./fold";
import type { ExperimentMeta, RunResponse, Snapshot } from "./types";

/**
 * Lab-independent Code Lab state: experiment metadata, parameter values, execution on the
 * Python service, and replay of the recorded snapshots. Each lab supplies its experiments,
 * a fold function (snapshots → frames) and renders its own inspector and plots.
 */

/** Replay speeds (ms per snapshot). Playback itself is the shared usePlayback hook. */
export const SPEEDS: Record<"slow" | "normal" | "fast", SpeedDef> = {
  slow: { label: "slow", interval: 600, stride: 1 },
  normal: { label: "normal", interval: 160, stride: 1 },
  fast: { label: "fast", interval: 30, stride: 1 },
};
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
  const [attempt, setAttempt] = useState(0);
  const onResult = useRef(opts.onResult);
  useEffect(() => {
    onResult.current = opts.onResult;
  });

  const meta = metas[exp.id];
  const run = runs[exp.id];
  const vals = values[exp.id];

  // A new run (or another framework's run) restarts the replay at the first snapshot (resetKey below).
  const frames = useMemo(() => (run ? fold(run.snapshots) : []), [run, fold]);
  const pb = usePlayback(frames.length, { speeds: SPEEDS, initialSpeed: "normal", resetKey: run, keyboard: true });
  const pause = useRef(pb.pause);
  useEffect(() => {
    pause.current = pb.pause;
  });

  const execute = useCallback(async (id: string, params: Record<string, number>) => {
    setRunning(true);
    setRunError(null);
    pause.current();
    try {
      const res = await runExperiment<S>(id, params);
      setRuns((r) => ({ ...r, [id]: res }));
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

  const last = pb.last;
  const i = pb.index;
  const frame: F | undefined = frames[i];

  const switchFramework = (key: K) => {
    setFw(key);
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
  const goTo = (k: number) => pb.seek(k);

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
    playing: pb.playing,
    setPlaying: pb.setPlaying,
    /** Play/pause; playing from the last snapshot replays from the first. */
    togglePlay: pb.toggle,
    setIdx: (k: number) => pb.seek(k),
    goTo,
    speed: pb.speed,
    setSpeed: pb.setSpeed,
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
