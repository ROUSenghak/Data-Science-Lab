/**
 * A "focus" names one mathematical object that several views can show at once: a residual,
 * a gradient component, a cluster. Equations mark their terms with it (KaTeX \htmlData) and plots
 * emphasise the matching geometry, so hovering either side highlights both.
 */

export type Focus =
  | { kind: "residual"; i: number }
  | { kind: "loss" }
  | { kind: "grad"; axis: "w" | "b" }
  | { kind: "step"; axis: "w" | "b" }
  | { kind: "cluster"; j: number }
  | { kind: "point"; i: number };

export function encodeFocus(f: Focus): string {
  switch (f.kind) {
    case "residual":
    case "point":
      return `${f.kind}.${f.i}`;
    case "cluster":
      return `cluster.${f.j}`;
    case "grad":
    case "step":
      return `${f.kind}.${f.axis}`;
    case "loss":
      return "loss";
  }
}

export function decodeFocus(s: string | null | undefined): Focus | null {
  if (!s) return null;
  const [kind, arg] = s.split(".");
  const idx = arg != null && /^\d+$/.test(arg) ? Number(arg) : null;
  switch (kind) {
    case "loss":
      return arg == null ? { kind: "loss" } : null;
    case "residual":
    case "point":
      return idx != null ? { kind, i: idx } : null;
    case "cluster":
      return idx != null ? { kind, j: idx } : null;
    case "grad":
    case "step":
      return arg === "w" || arg === "b" ? { kind, axis: arg } : null;
    default:
      return null;
  }
}

export const sameFocus = (a: Focus | null, b: Focus | null) => (a && b ? encodeFocus(a) === encodeFocus(b) : a === b);

/** Wrap a TeX fragment so that hovering it sets `f` (rendered by <Tex> as a data-hl attribute). */
export const hl = (f: Focus, tex: string) => `\\htmlData{hl=${encodeFocus(f)}}{${tex}}`;

/* Convenience predicates for plots. */
export const focusedResidual = (f: Focus | null) => (f?.kind === "residual" ? f.i : f?.kind === "point" ? f.i : null);
export const focusedCluster = (f: Focus | null) => (f?.kind === "cluster" ? f.j : null);
export const focusedAxis = (f: Focus | null) => (f?.kind === "grad" || f?.kind === "step" ? f.axis : null);
