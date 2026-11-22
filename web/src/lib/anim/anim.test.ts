import katex from "katex";
import { describe, expect, it } from "vitest";
import { decodeFocus, encodeFocus, hl, sameFocus, type Focus } from "./focus";
import { clampIndex, initialPlayback, isSingleStep, playbackReducer, type PlaybackAction, type PlaybackState } from "./playback";
import { easeInOutCubic, lerpArray, shouldTween } from "./tween";

const run = (actions: PlaybackAction[], count: number, s: PlaybackState = initialPlayback) => actions.reduce((acc, a) => playbackReducer(acc, a, count), s);

describe("playback reducer", () => {
  it("always points at an existing state", () => {
    for (const count of [0, 1, 2, 7]) {
      for (const a of [{ type: "seek", index: 99 }, { type: "seek", index: -3 }, { type: "step", delta: 5 }, { type: "tick", stride: 4 }] as PlaybackAction[]) {
        const s = run([{ type: "play" }, a], count);
        expect(s.index).toBeGreaterThanOrEqual(0);
        expect(s.index).toBeLessThanOrEqual(Math.max(0, count - 1));
      }
    }
    expect(clampIndex(NaN, 5)).toBe(0);
  });

  it("plays to the end and stops there; playing again replays from the start", () => {
    let s = run([{ type: "play" }], 4);
    const seen = [s.index];
    while (s.playing) {
      s = playbackReducer(s, { type: "tick", stride: 1 }, 4);
      seen.push(s.index);
    }
    expect(seen).toEqual([0, 1, 2, 3]);
    s = playbackReducer(s, { type: "toggle" }, 4);
    expect(s).toMatchObject({ index: 0, playing: true });
  });

  it("stride > 1 skips states and lands exactly on the last one", () => {
    let s = run([{ type: "play" }], 10);
    const seen: number[] = [];
    while (s.playing) seen.push((s = playbackReducer(s, { type: "tick", stride: 4 }, 10)).index);
    expect(seen).toEqual([4, 8, 9]);
  });

  it("manual navigation pauses and records where it came from", () => {
    const s = run([{ type: "play" }, { type: "tick", stride: 1 }, { type: "step", delta: 1 }], 10);
    expect(s).toMatchObject({ index: 2, from: 1, playing: false });
    expect(isSingleStep(s)).toBe(true);
    const j = playbackReducer(s, { type: "seek", index: 8 }, 10);
    expect(isSingleStep(j)).toBe(false);
  });

  it("a shrinking sequence clamps the index", () => {
    const s = run([{ type: "seek", index: 9 }], 10);
    expect(playbackReducer(s, { type: "pause" }, 3).index).toBe(2);
  });
});

describe("tweening", () => {
  it("easing is monotone with fixed end points", () => {
    expect(easeInOutCubic(0)).toBe(0);
    expect(easeInOutCubic(1)).toBe(1);
    let prev = -1;
    for (let t = 0; t <= 1; t += 0.01) {
      expect(easeInOutCubic(t)).toBeGreaterThanOrEqual(prev);
      prev = easeInOutCubic(t);
    }
  });

  it("interpolation ends exactly on the computed state", () => {
    const a = [0.1, -3, 7];
    const b = [2.5, 4, -1];
    expect(lerpArray(a, b, 0)).toEqual(a);
    expect(lerpArray(a, b, 1)).toEqual(b);
  });

  it("only animates single steps between finite states", () => {
    const opts = { singleStep: true, reducedMotion: false };
    expect(shouldTween([0, 0], [1, 1], opts)).toBe(true);
    expect(shouldTween([0, 0], [1, 1], { ...opts, singleStep: false })).toBe(false);
    expect(shouldTween([0, 0], [1, 1], { ...opts, reducedMotion: true })).toBe(false);
    expect(shouldTween([0, 0], [Infinity, 1], opts)).toBe(false);
    expect(shouldTween([NaN, 0], [1, 1], opts)).toBe(false);
    expect(shouldTween(null, [1, 1], opts)).toBe(false);
  });
});

describe("focus encoding (equation ↔ plot)", () => {
  const all: Focus[] = [
    { kind: "residual", i: 0 },
    { kind: "residual", i: 41 },
    { kind: "loss" },
    { kind: "grad", axis: "w" },
    { kind: "grad", axis: "b" },
    { kind: "step", axis: "w" },
    { kind: "cluster", j: 7 },
    { kind: "point", i: 3 },
  ];
  it("round-trips", () => {
    for (const f of all) expect(decodeFocus(encodeFocus(f))).toEqual(f);
    for (const f of all) expect(sameFocus(f, decodeFocus(encodeFocus(f)))).toBe(true);
  });
  it("rejects malformed keys", () => {
    for (const s of ["", "residual", "residual.x", "grad.z", "cluster.-1", "loss.2", "nope.1"]) expect(decodeFocus(s)).toBeNull();
  });
  it("KaTeX renders a marked term with the key the plots listen for", () => {
    const f: Focus = { kind: "grad", axis: "w" };
    const html = katex.renderToString(`\\frac{\\partial L}{\\partial w} = ${hl(f, "1.23")}`, { trust: (c) => c.command === "\\htmlData", strict: "ignore" });
    const key = html.match(/data-hl="([^"]+)"/)?.[1];
    expect(decodeFocus(key)).toEqual(f);
  });
});
