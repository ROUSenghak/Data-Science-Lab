"use client";

import { useEffect, useSyncExternalStore } from "react";

type Mode = "light" | "dark" | "system";
const KEY = "dsl-theme";

/** Script that runs before paint so the page never flashes the wrong theme. */
export const themeInitScript = `(function(){try{var m=localStorage.getItem("${KEY}")||"system";var d=m==="dark"||(m==="system"&&matchMedia("(prefers-color-scheme: dark)").matches);document.documentElement.dataset.theme=d?"dark":"light"}catch(e){}})();`;

const listeners = new Set<() => void>();
const subscribe = (cb: () => void) => {
  listeners.add(cb);
  return () => listeners.delete(cb);
};
function readMode(): Mode {
  try {
    return (localStorage.getItem(KEY) as Mode) || "system";
  } catch {
    return "system";
  }
}
function apply(mode: Mode) {
  const dark = mode === "dark" || (mode === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}
function writeMode(mode: Mode) {
  try {
    localStorage.setItem(KEY, mode);
  } catch {}
  apply(mode);
  listeners.forEach((l) => l());
}

export function ThemeToggle() {
  const mode = useSyncExternalStore(subscribe, readMode, () => "system" as Mode);

  // follow OS changes while in system mode
  useEffect(() => {
    if (mode !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => apply("system");
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [mode]);

  const next: Mode = mode === "system" ? "light" : mode === "light" ? "dark" : "system";
  const label = { light: "Light theme", dark: "Dark theme", system: "System theme" }[mode];
  return (
    <button
      type="button"
      onClick={() => writeMode(next)}
      className="inline-flex h-8 w-8 items-center justify-center rounded-md text-fg-2 hover:bg-surface-2 hover:text-fg"
      aria-label={`${label}; switch to ${next}`}
      title={label}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
        {mode === "light" && (
          <>
            <circle cx="12" cy="12" r="4.5" />
            <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
          </>
        )}
        {mode === "dark" && <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />}
        {mode === "system" && (
          <>
            <rect x="3" y="4" width="18" height="12" rx="1.5" />
            <path d="M8 20h8M12 16v4" />
          </>
        )}
      </svg>
    </button>
  );
}
