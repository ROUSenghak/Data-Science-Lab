import type { Vec2 } from "@/lib/ml/kmeans";
import type { ObjectivePoint } from "@/labs/k-means/plots";
import { foldTrace, type TraceFrame } from "../fold";
import type { KMeansState, Snapshot } from "../types";

export type KFrame = TraceFrame<KMeansState>;

export const foldKMeans = (snapshots: Snapshot<KMeansState>[]): KFrame[] => foldTrace(snapshots);

const LOOP_STEPS = new Set(["distances", "assign", "update", "objective"]);

/** Iteration t = Python's loop index + 1 for snapshots inside the loop; null outside it. */
export function iterationOf(f: KFrame): number | null {
  return LOOP_STEPS.has(f.snap.step) && f.state.iteration != null ? f.state.iteration + 1 : null;
}

/** J after each assignment (labels new, centroids old) and each update (iteration complete), up to frame `upTo`. */
export function objectiveHistory(frames: KFrame[], upTo: number): ObjectivePoint[] {
  const out: ObjectivePoint[] = [];
  for (let i = 0; i <= upTo && i < frames.length; i++) {
    const f = frames[i];
    const t = iterationOf(f);
    if (t == null || f.state.J == null) continue;
    if (f.snap.step === "assign") out.push({ iteration: t, phase: "assign", J: f.state.J });
    else if (f.snap.step === "objective") out.push({ iteration: t, phase: "update", J: f.state.J });
  }
  return out;
}

/** Labels to colour points with; null while no point is assigned (all −1 or absent). */
export function shownLabels(s: KMeansState): number[] | null {
  return s.labels && s.labels.some((l) => l >= 0) ? s.labels : null;
}

/** Indices of points whose label changed at an assignment step (none in the first iteration). */
export function changedPoints(s: KMeansState): number[] {
  if (!s.labels || !s.prev_labels || s.prev_labels.every((l) => l < 0)) return [];
  return s.labels.flatMap((l, i) => (l !== s.prev_labels![i] ? [i] : []));
}

/** Every position a centroid takes in the run, plus the data (to fix the plot domain). */
export function allPositions(frames: KFrame[]): Vec2[] {
  const pts: Vec2[] = [...(frames.at(-1)?.state.X ?? [])];
  for (const f of frames) if (f.snap.state.centroids) pts.push(...f.snap.state.centroids);
  return pts;
}
