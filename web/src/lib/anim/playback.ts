/**
 * Discrete playback over a sequence of computed states (epochs, half-steps, snapshots).
 * Pure: no timers, no React. The index always points at a real computed state; animation
 * between two states is a separate, purely visual concern (see tween.ts).
 */

export interface PlaybackState {
  /** Index of the computed state being shown. */
  index: number;
  /** Index shown before the last change; |index − from| = 1 means a single step. */
  from: number;
  playing: boolean;
}

export interface SpeedDef {
  label: string;
  /** Milliseconds between ticks while playing. */
  interval: number;
  /** States advanced per tick (> 1 skips states and therefore never animates them). */
  stride: number;
}

export type PlaybackAction =
  | { type: "seek"; index: number }
  | { type: "step"; delta: number }
  | { type: "tick"; stride: number }
  | { type: "play" }
  | { type: "pause" }
  | { type: "toggle" }
  | { type: "reset" };

export const initialPlayback: PlaybackState = { index: 0, from: 0, playing: false };

export function clampIndex(i: number, count: number): number {
  if (!Number.isFinite(i)) return 0;
  return Math.max(0, Math.min(Math.max(0, count - 1), Math.round(i)));
}

const move = (s: PlaybackState, index: number, playing: boolean): PlaybackState =>
  index === s.index && playing === s.playing ? s : { index, from: s.index, playing };

export function playbackReducer(s: PlaybackState, a: PlaybackAction, count: number): PlaybackState {
  const last = Math.max(0, count - 1);
  const cur = clampIndex(s.index, count);
  const base = cur === s.index ? s : { ...s, index: cur };
  switch (a.type) {
    case "seek":
      return move(base, clampIndex(a.index, count), false);
    case "step":
      return move(base, clampIndex(cur + a.delta, count), false);
    case "reset":
      return move(base, 0, false);
    case "pause":
      return base.playing ? { ...base, playing: false } : base;
    case "play":
      // Playing from the end replays from the start.
      if (count <= 1) return base;
      return cur >= last ? move(base, 0, true) : { ...base, playing: true };
    case "toggle":
      return playbackReducer(base, { type: base.playing ? "pause" : "play" }, count);
    case "tick": {
      if (!base.playing) return base;
      const next = Math.min(last, cur + Math.max(1, a.stride));
      return move(base, next, next < last);
    }
  }
}

/** True when the last change moved exactly one computed state (the only case worth animating). */
export const isSingleStep = (s: PlaybackState) => Math.abs(s.index - s.from) === 1;
