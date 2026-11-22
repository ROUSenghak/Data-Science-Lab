/**
 * Interpolation between two computed states, for display only.
 *
 * Rule: we only animate between two *consecutive* computed states, and only when the straight
 * path between them means something (a gradient-descent update is exactly the segment
 * θ_t → θ_t − η∇L; a centroid update is a jump we draw as a glide). Jumps over several states
 * snap, because a straight line between non-adjacent states would be a path the algorithm
 * never took.
 */

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export function lerpArray(a: readonly number[], b: readonly number[], t: number): number[] {
  return b.map((v, i) => lerp(a[i] ?? v, v, t));
}

/** Smooth start and stop; f(0) = 0, f(1) = 1, monotone. */
export function easeInOutCubic(t: number): number {
  const u = Math.min(1, Math.max(0, t));
  return u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
}

export const allFinite = (v: readonly number[]) => v.every(Number.isFinite);

export function sameValues(a: readonly number[] | null, b: readonly number[] | null): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false;
  return true;
}

export function shouldTween(from: readonly number[] | null, to: readonly number[], opts: { singleStep: boolean; reducedMotion: boolean }): boolean {
  return !!from && opts.singleStep && !opts.reducedMotion && from.length === to.length && allFinite(from) && allFinite(to) && !sameValues(from, to);
}

/** Tween duration for a playback tick: leave the state visible for a moment before the next tick. */
export const tweenDuration = (interval: number, max = 480) => Math.max(0, Math.min(max, interval * 0.7));
