"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { clampIndex, initialPlayback, isSingleStep, playbackReducer, type PlaybackAction, type SpeedDef } from "@/lib/anim/playback";
import { easeInOutCubic, lerpArray, sameValues, shouldTween } from "@/lib/anim/tween";

/* ------------------------------------------------------------------ reduced motion */

const QUERY = "(prefers-reduced-motion: reduce)";
const subscribe = (cb: () => void) => {
  const mq = window.matchMedia(QUERY);
  mq.addEventListener("change", cb);
  return () => mq.removeEventListener("change", cb);
};

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribe, () => window.matchMedia(QUERY).matches, () => false);
}

/* ------------------------------------------------------------------ playback over computed states */

/**
 * Play / pause / step / seek over `count` computed states. The index always points at a real state.
 * `resetKey` restarts playback when the underlying computation changes (new data, new parameters).
 */
export function usePlayback<K extends string>(
  count: number,
  opts: { speeds: Record<K, SpeedDef>; initialSpeed: K; resetKey?: unknown; keyboard?: boolean },
) {
  const [state, setState] = useState(initialPlayback);
  const [speed, setSpeed] = useState<K>(opts.initialSpeed);
  const def = opts.speeds[speed];

  const [prevKey, setPrevKey] = useState(opts.resetKey);
  if (prevKey !== opts.resetKey) {
    setPrevKey(opts.resetKey);
    setState(initialPlayback);
  }

  const dispatch = useCallback((a: PlaybackAction, n = count) => setState((s) => playbackReducer(s, a, n)), [count]);

  useEffect(() => {
    if (!state.playing) return;
    const id = window.setInterval(() => setState((s) => playbackReducer(s, { type: "tick", stride: def.stride }, count)), def.interval);
    return () => window.clearInterval(id);
  }, [state.playing, def.interval, def.stride, count]);

  useEffect(() => {
    if (!opts.keyboard) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof Element && t.closest("input, textarea, select, [role=dialog]")) return;
      if (e.key === "ArrowRight") dispatch({ type: "step", delta: 1 });
      else if (e.key === "ArrowLeft") dispatch({ type: "step", delta: -1 });
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [opts.keyboard, dispatch]);

  const index = clampIndex(state.index, count);
  return {
    index,
    last: Math.max(0, count - 1),
    count,
    playing: state.playing,
    /** The last change moved exactly one state (the only kind of change that is animated). */
    singleStep: isSingleStep(state),
    speed,
    /** Ignores names that are not one of the configured speeds. */
    setSpeed: (v: string) => {
      if (Object.hasOwn(opts.speeds, v)) setSpeed(v as K);
    },
    speedDef: def,
    play: () => dispatch({ type: "play" }),
    pause: () => dispatch({ type: "pause" }),
    toggle: () => dispatch({ type: "toggle" }),
    next: () => dispatch({ type: "step", delta: 1 }),
    prev: () => dispatch({ type: "step", delta: -1 }),
    reset: () => dispatch({ type: "reset" }),
    /** Seek to a state. Pass `inCount` when the sequence length changes in the same event. */
    seek: (i: number, inCount?: number) => dispatch({ type: "seek", index: i }, inCount ?? count),
    setPlaying: (v: boolean | ((p: boolean) => boolean)) => setState((s) => playbackReducer(s, { type: (typeof v === "function" ? v(s.playing) : v) ? "play" : "pause" }, count)),
  };
}

export type Playback = ReturnType<typeof usePlayback<string>>;

/* ------------------------------------------------------------------ tween between two computed states */

/**
 * Displayed values that glide to `target` when it changes by a single step (see lib/anim/tween.ts
 * for when that is allowed). A new tween always starts from the previous *computed* state, never
 * from a half-way position, so the drawn path is always a real update segment.
 */
export function useTween(target: readonly number[], opts: { singleStep: boolean; duration: number }) {
  const reduced = useReducedMotion();
  const [prev, setPrev] = useState<readonly number[]>(target);
  const [anim, setAnim] = useState<{ from: number[]; to: number[] } | null>(null);
  const [progress, setProgress] = useState(1);

  if (!sameValues(prev, target)) {
    setPrev(target);
    if (shouldTween(prev, target, { singleStep: opts.singleStep, reducedMotion: reduced || opts.duration <= 0 })) {
      setAnim({ from: [...prev], to: [...target] });
      setProgress(0);
    } else setAnim(null);
  }

  useEffect(() => {
    if (!anim) return;
    let raf = 0;
    let start: number | null = null;
    const tick = (now: number) => {
      start ??= now;
      const p = Math.min(1, (now - start) / opts.duration);
      setProgress(p);
      if (p < 1) raf = requestAnimationFrame(tick);
      else setAnim(null);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [anim, opts.duration]);

  const moving = anim !== null && progress < 1;
  return { value: moving ? lerpArray(anim.from, anim.to, easeInOutCubic(progress)) : (target as number[]), moving };
}
