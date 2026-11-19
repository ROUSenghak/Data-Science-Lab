/** Seeded pseudo-random numbers for reproducible in-browser experiments. */

/** mulberry32: small, fast 32-bit PRNG returning floats in [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Rng {
  uniform(lo: number, hi: number): number;
  normal(mean: number, std: number): number;
}

export function createRng(seed: number): Rng {
  const next = mulberry32(seed);
  let spare: number | null = null;
  return {
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    normal(mean, std) {
      // Box–Muller transform; caches the second variate.
      if (spare !== null) {
        const z = spare;
        spare = null;
        return mean + std * z;
      }
      let u = 0;
      while (u === 0) u = next();
      const v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return mean + std * r * Math.cos(2 * Math.PI * v);
    },
  };
}
