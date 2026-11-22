/** Number formatting shared by plots, inspectors and equations. */

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined) return "—";
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "NaN" : v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(Math.max(1, digits - 2)).replace("-", "−");
  const s = v.toFixed(digits);
  return s.replace("-", "−");
}

/** Number for LaTeX: wraps negatives in parentheses when used as an operand. */
export function texNum(v: number | null | undefined, digits = 4, paren = false): string {
  if (v === null || v === undefined) return "\\text{—}";
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "\\text{NaN}" : v > 0 ? "\\infty" : "-\\infty";
  const a = Math.abs(v);
  let s: string;
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) {
    const [m, e] = v.toExponential(Math.max(1, digits - 2)).split("e");
    s = `${m}\\times10^{${Number(e)}}`;
  } else s = v.toFixed(digits);
  return paren && v < 0 ? `(${s})` : s;
}

/** v² for LaTeX. Parenthesizes negatives and scientific notation (a bare "1.2\\times10^{5}^2" is a double superscript). */
export function texSq(v: number | null | undefined, digits = 4): string {
  const s = texNum(v, digits);
  return v != null && (v < 0 || s.includes("\\times")) ? `(${s})^2` : `${s}^2`;
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
