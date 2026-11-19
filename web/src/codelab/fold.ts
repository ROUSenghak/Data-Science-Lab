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
    // y_hat is recomputed at these steps even when its value did not change (and so is absent from the diff).
    const forward = snap.step === "forward" || snap.step === "evaluate" || snap.step === "predict" || "y_hat" in snap.state;
    if (forward) forwardParams = state.w != null && state.b != null ? { w: state.w, b: state.b } : null;
    frames.push({ snap, state, prev, locals, forwardParams });
  }
  return frames;
}

/** Quantities of one epoch t, all evaluated at the same parameters θ_t. */
export interface EpochRecord {
  epoch: number;
  /** θ_t: parameters used by the forward pass of epoch t. */
  w: number | null;
  b: number | null;
  /** L(θ_t). */
  loss: number | null;
  /** ∇L(θ_t); null for the final evaluation, which has no backward pass. */
  dw: number | null;
  db: number | null;
}

/**
 * Per-epoch training record from the snapshots up to frame `upTo`.
 * The `evaluate` step runs after the loop on θ_T, so it is recorded as epoch T = last epoch + 1.
 */
export function trainingHistory(frames: Frame[], upTo: number): EpochRecord[] {
  const byEpoch = new Map<number, EpochRecord>();
  for (let i = 0; i <= upTo && i < frames.length; i++) {
    const { snap, state: s } = frames[i];
    const t = snap.step === "evaluate" ? (s.epoch ?? -1) + 1 : s.epoch;
    if (t == null) continue;
    let rec = byEpoch.get(t);
    if (!rec) byEpoch.set(t, (rec = { epoch: t, w: null, b: null, loss: null, dw: null, db: null }));
    if (snap.step === "forward" || snap.step === "evaluate") {
      rec.w = s.w ?? null;
      rec.b = s.b ?? null;
    }
    if (snap.step === "loss" || snap.step === "evaluate") rec.loss = s.loss ?? null;
    if (snap.step === "gradient" || snap.step === "backward") {
      rec.dw = s.dw ?? null;
      rec.db = s.db ?? null;
    }
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
}

export function lossHistory(frames: Frame[], upTo: number) {
  return trainingHistory(frames, upTo)
    .filter((r) => r.loss != null || r.w != null)
    .map((r) => ({ epoch: r.epoch, loss: r.loss }));
}

/**
 * Index of the snapshot for `step` that belongs to the same epoch as frame `from`
 * (falling back to the first snapshot of that step). -1 if the step never ran.
 */
export function findStep(frames: Frame[], from: number, step: string): number {
  const epoch = frames[from]?.state.epoch;
  let first = -1;
  for (let k = 0; k < frames.length; k++) {
    if (frames[k].snap.step !== step) continue;
    if (first < 0) first = k;
    if (epoch != null && frames[k].state.epoch === epoch) return k;
  }
  return first;
}

export function paramPath(frames: Frame[], upTo: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i <= upTo && i < frames.length; i++) {
    const f = frames[i];
    if ((f.snap.step === "init" || f.snap.step === "update" || f.snap.step === "fit") && f.state.w != null && f.state.b != null) out.push([f.state.w, f.state.b]);
  }
  return out;
}
