"use client";

import { useSyncExternalStore } from "react";

/* ------------------------------------------------------------------ WebGL availability */

let cached: boolean | null = null;

/** True when this browser can create a WebGL context (checked once, then cached). */
export function detectWebGL(): boolean {
  if (cached != null) return cached;
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2") ?? c.getContext("webgl");
    cached = !!gl;
    (gl as WebGLRenderingContext | null)?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    cached = false;
  }
  return cached;
}

const noop = () => () => {};

/** null while prerendering (unknown), then whether 3D views can be shown. */
export function useWebGLSupport(): boolean | null {
  return useSyncExternalStore(noop, detectWebGL, () => null);
}

/* ------------------------------------------------------------------ theme colours for WebGL */

/** CSS custom properties a 3D scene needs; WebGL cannot read var(--…) itself. */
export const THEME_TOKENS = [
  "--surface",
  "--surface-2",
  "--fg",
  "--fg-2",
  "--fg-3",
  "--viz-grid",
  "--viz-axis",
  "--viz-model",
  "--viz-residual",
  "--viz-alt",
  "--viz-data",
  "--viz-run-1",
  "--viz-run-2",
  "--viz-run-3",
  "--viz-cluster-1",
  "--viz-cluster-2",
  "--viz-cluster-3",
  "--viz-cluster-4",
] as const;
export type ThemeToken = (typeof THEME_TOKENS)[number];
export type ThemeColors = Record<ThemeToken, string> & { dark: boolean };

function readTokens(): ThemeColors {
  const cs = getComputedStyle(document.documentElement);
  const out = { dark: document.documentElement.dataset.theme === "dark" } as ThemeColors;
  for (const t of THEME_TOKENS) out[t] = cs.getPropertyValue(t).trim() || "#888888";
  return out;
}

let snap: ThemeColors | null = null;
let snapKey: string | undefined;
const getColors = () => {
  const key = document.documentElement.dataset.theme;
  if (!snap || key !== snapKey) {
    snap = readTokens();
    snapKey = key;
  }
  return snap;
};
const subscribeTheme = (cb: () => void) => {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => mo.disconnect();
};

/** Current theme colours (null while prerendering), re-read when html[data-theme] changes. */
export function useThemeColors(): ThemeColors | null {
  return useSyncExternalStore(subscribeTheme, getColors, () => null);
}
