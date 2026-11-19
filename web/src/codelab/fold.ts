import type { LinRegState, Snapshot, VarView } from "./types";

/** Full execution state at one snapshot, reconstructed from the diff-encoded trace. */
export interface Frame {
  snap: Snapshot;
  state: LinRegState;
  /** State at the previous snapshot (for "before → after" equations). */
  prev: LinRegState;
  locals: Record<string, VarView>;
  /** (w, b) that produced the current y_hat, i.e. the parameters at the latest forward pass. */
  forwardParams: { w: number; b: number } | null;
}

export function foldSnapshots(snapshots: Snapshot[]): Frame[] {
  const frames: Frame[] = [];
  let state: LinRegState = {};
  let locals: Record<string, VarView> = {};
  let forwardParams: Frame["forwardParams"] = null;
  for (const snap of snapshots) {
    const prev = state;
    state = { ...state, ...snap.state };
    locals = { ...locals, ...snap.locals };
    if (snap.removed.length) {
      locals = { ...locals };
      for (const k of snap.removed) delete locals[k];
    }
    if ("y_hat" in snap.state && state.w != null && state.b != null) forwardParams = { w: state.w, b: state.b };
    frames.push({ snap, state, prev, locals, forwardParams });
  }
  return frames;
}

export function lossHistory(frames: Frame[], upTo: number) {
  const out: { epoch: number; loss: number | null }[] = [];
  for (let i = 0; i <= upTo && i < frames.length; i++) {
    const f = frames[i];
    if (f.snap.step === "loss" && f.state.epoch != null) out.push({ epoch: f.state.epoch, loss: f.state.loss ?? null });
  }
  return out;
}

export function paramPath(frames: Frame[], upTo: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i <= upTo && i < frames.length; i++) {
    const f = frames[i];
    if ((f.snap.step === "init" || f.snap.step === "update" || f.snap.step === "fit") && f.state.w != null && f.state.b != null) out.push([f.state.w, f.state.b]);
  }
  return out;
}
