"use client";

import { Fragment } from "react";

export interface StepDef {
  id: string;
  title: string;
  summary: string;
}

/**
 * The experiment's computational steps in execution order. Steps that ran more than once
 * form the loop body and are grouped. Clicking a step jumps to it (within the current epoch).
 * Generic: knows nothing about the algorithm beyond step ids and how often each ran.
 */
export function StepPipeline({
  steps,
  counts,
  current,
  loopLabel = "each epoch",
  onSelect,
}: {
  steps: StepDef[];
  /** How many snapshots each step produced in this run. */
  counts: Record<string, number>;
  current: string | null;
  loopLabel?: string;
  onSelect: (id: string) => void;
}) {
  const inLoop = steps.map((s) => (counts[s.id] ?? 0) > 1);
  const first = inLoop.indexOf(true);
  const last = inLoop.lastIndexOf(true);
  const before = first < 0 ? steps : steps.slice(0, first);
  const loop = first < 0 ? [] : steps.slice(first, last + 1);
  const after = first < 0 ? [] : steps.slice(last + 1);

  const chip = (s: StepDef) => {
    const active = s.id === current;
    return (
      <button
        type="button"
        onClick={() => onSelect(s.id)}
        disabled={!counts[s.id]}
        aria-current={active ? "step" : undefined}
        title={s.summary || s.title}
        className={`whitespace-nowrap rounded px-1.5 py-0.5 font-medium disabled:cursor-not-allowed disabled:opacity-40 ${
          active ? "bg-[var(--viz-model)] text-white" : "text-fg-2 hover:bg-surface-2 hover:text-fg"
        }`}
      >
        {s.title}
      </button>
    );
  };
  const arrow = <span className="px-0.5 text-fg-3" aria-hidden>→</span>;
  const run = (list: StepDef[]) =>
    list.map((s, k) => (
      <Fragment key={s.id}>
        {k > 0 && arrow}
        {chip(s)}
      </Fragment>
    ));

  return (
    <div className="flex flex-wrap items-center gap-y-1.5 text-[12px]" role="group" aria-label="Computational steps">
      {run(before)}
      {loop.length > 0 && (
        <>
          {before.length > 0 && arrow}
          <span className="inline-flex flex-wrap items-center rounded-md border border-dashed border-line-strong px-1 py-0.5" role="group" aria-label={`Loop: ${loopLabel}`}>
            <span className="mr-1 text-[10.5px] uppercase tracking-wide text-fg-3">↻ {loopLabel}</span>
            {run(loop)}
          </span>
        </>
      )}
      {after.length > 0 && (
        <>
          {arrow}
          {run(after)}
        </>
      )}
    </div>
  );
}
