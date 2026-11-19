"use client";

import { useId, type ReactNode } from "react";

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => String(v),
  hint,
  scale = "linear",
  disabled,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: ReactNode;
  /** "log" maps the slider position logarithmically (for learning rates). */
  scale?: "linear" | "log";
  disabled?: boolean;
}) {
  const id = useId();
  const toPos = (v: number) => (scale === "log" ? Math.log10(v) : v);
  const fromPos = (p: number) => {
    if (scale !== "log") return p;
    const v = 10 ** p;
    const mag = 10 ** Math.floor(Math.log10(v) - 2);
    return Math.round(v / mag) * mag; // three significant digits
  };
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-2 text-[13px]">
        <label htmlFor={id} className="text-fg-2">
          {label}
        </label>
        <output htmlFor={id} className="font-mono text-[12px] tabular-nums text-fg">
          {format(value)}
        </output>
      </div>
      <input
        id={id}
        type="range"
        className="mt-1 w-full"
        min={toPos(min)}
        max={toPos(max)}
        step={scale === "log" ? 0.001 : step}
        value={toPos(value)}
        disabled={disabled}
        onChange={(e) => onChange(fromPos(Number(e.target.value)))}
      />
      {hint && <div className="text-[11px] leading-snug text-fg-3">{hint}</div>}
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; hint?: ReactNode }) {
  const id = useId();
  return (
    <div className="flex items-start gap-2 text-[13px]">
      <input id={id} type="checkbox" className="mt-0.5 accent-[var(--viz-model)]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <label htmlFor={id} className="text-fg-2">
        {label}
        {hint && <span className="block text-[11px] text-fg-3">{hint}</span>}
      </label>
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "default",
  disabled,
  title,
  ariaLabel,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "ghost";
  disabled?: boolean;
  title?: string;
  ariaLabel?: string;
  type?: "button" | "submit";
}) {
  const v = {
    default: "border border-line-strong bg-surface text-fg hover:bg-surface-2",
    primary: "border border-transparent bg-[var(--viz-model)] text-white hover:brightness-110",
    ghost: "border border-transparent text-fg-2 hover:bg-surface-2 hover:text-fg",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      className={`inline-flex h-8 items-center justify-center gap-1.5 rounded-md px-2.5 text-[13px] font-medium disabled:cursor-not-allowed disabled:opacity-40 ${v}`}
    >
      {children}
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, ariaLabel }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; ariaLabel: string }) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="inline-flex rounded-md border border-line-strong p-0.5 text-[12px]">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`rounded px-2 py-1 ${value === o.value ? "bg-surface-2 font-medium text-fg" : "text-fg-3 hover:text-fg"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export const Icon = {
  play: <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M3 1.5v9l7.5-4.5z" fill="currentColor" /></svg>,
  pause: <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z" fill="currentColor" /></svg>,
  step: <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M2 1.5v9l6-4.5zM8.5 1.5h1.8v9H8.5z" fill="currentColor" /></svg>,
  back: <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M10 1.5v9l-6-4.5zM1.7 1.5h1.8v9H1.7z" fill="currentColor" /></svg>,
  reset: <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden><path d="M2.5 6a3.5 3.5 0 1 0 1-2.5" /><path d="M2 1.5v2.5h2.5" /></svg>,
  run: <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M2.5 1.5v9l8-4.5z" fill="currentColor" /></svg>,
};
