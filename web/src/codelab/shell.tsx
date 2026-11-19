"use client";

import type { ReactNode } from "react";
import { CodeView } from "@/components/code/CodeView";
import { Button, Icon, Segmented } from "@/components/ui/controls";
import { Badge, Callout, ExecutionBadge, Panel } from "@/components/ui/primitives";
import { fmt } from "@/lib/format";
import type { TraceFrame } from "./fold";
import { StepPipeline } from "./StepPipeline";
import type { RunResponse } from "./types";
import type { CodeLabState } from "./useCodeLab";

/* Lab-independent pieces of the Code Lab page. Each takes the state returned by useCodeLab. */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyLab = CodeLabState<any, any, TraceFrame<any>>;

export function OfflineNotice({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="mx-auto max-w-2xl py-10">
      <Callout tone="bad" title="The Python execution service is not running">
        <p>The Code Lab only shows results from real Python execution, so nothing is displayed without the service. Start it in a second terminal:</p>
        <pre className="mt-2 overflow-x-auto rounded bg-surface px-3 py-2 font-mono text-[12.5px]">cd api && .venv/bin/uvicorn app.main:app --port 8000</pre>
        <div className="mt-3">
          <Button onClick={onRetry}>Retry</Button>
        </div>
      </Callout>
    </div>
  );
}

/** Sticky toolbar: implementation, Run, replay controls, step pipeline. */
export function ReplayToolbar({
  lab,
  extraButtons,
  status,
  loopLabel,
  onSelectStep,
}: {
  lab: AnyLab;
  /** Jump to a step (within the current loop iteration). */
  onSelectStep: (step: string) => void;
  /** Lab-specific navigation (e.g. "Next epoch"). */
  extraButtons?: ReactNode;
  /** Lab-specific position (e.g. "epoch 3/100"), shown before the line number. */
  status?: ReactNode;
  loopLabel?: string;
}) {
  const { run, frame, meta, i, last, playing, frames } = lab;
  return (
    <div className="z-30 -mx-4 md:sticky md:top-14 border-b border-line bg-bg/95 px-4 py-2.5 backdrop-blur sm:-mx-6 sm:px-6">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented ariaLabel="Implementation" value={lab.fw} onChange={lab.switchFramework} options={lab.frameworks.map((f) => ({ value: f.key, label: f.label }))} />
        <Button variant="primary" onClick={() => lab.vals && lab.execute(lab.exp.id, lab.vals)} disabled={lab.running || !lab.allValid} title="Execute the script on the Python service">
          {Icon.run} {lab.running ? "Running…" : "Run"}
        </Button>
        <span className="mx-1 hidden h-5 w-px bg-line sm:block" aria-hidden />
        <Button onClick={() => lab.goTo(0)} disabled={!run} ariaLabel="Reset to first snapshot" title="Reset to first snapshot">
          {Icon.reset}
        </Button>
        <Button onClick={() => lab.goTo(i - 1)} disabled={!run || i === 0} ariaLabel="Previous snapshot" title="Previous snapshot (←)">
          {Icon.back}
        </Button>
        <Button onClick={() => (i >= last ? (lab.setIdx(0), lab.setPlaying(true)) : lab.setPlaying((p) => !p))} disabled={!run} ariaLabel={playing ? "Pause replay" : "Play replay"}>
          {playing ? Icon.pause : Icon.play} {playing ? "Pause" : "Play"}
        </Button>
        <Button onClick={() => lab.goTo(i + 1)} disabled={!run || i >= last} ariaLabel="Next snapshot" title="Next snapshot (→)">
          {Icon.step} Step
        </Button>
        {extraButtons}
        <Segmented ariaLabel="Replay speed" value={lab.speed} onChange={lab.setSpeed} options={[{ value: "slow", label: "slow" }, { value: "normal", label: "normal" }, { value: "fast", label: "fast" }]} />
        <div className="flex min-w-48 flex-1 items-center gap-2">
          <input type="range" aria-label="Snapshot" className="flex-1" min={0} max={last} value={i} disabled={!run} onChange={(e) => lab.goTo(Number(e.target.value))} />
          <span className="whitespace-nowrap font-mono text-[11.5px] tabular-nums text-fg-2">{run ? `${i + 1}/${frames.length}` : "—"}</span>
        </div>
      </div>
      {frame && meta && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <StepPipeline steps={meta.steps} counts={lab.stepCounts} current={frame.snap.step} onSelect={onSelectStep} loopLabel={loopLabel} />
          <span className="flex items-center gap-3 font-mono text-[12px] tabular-nums text-fg-2">
            {status}
            <span>
              line {frame.snap.lines[0]}
              {frame.snap.lines[1] !== frame.snap.lines[0] ? `–${frame.snap.lines[1]}` : ""}
            </span>
          </span>
          <span className="hidden text-[11.5px] text-fg-3 xl:inline" title="Python ran once. Each snapshot is the program state right after the highlighted lines executed.">
            Replay of one recorded Python run
          </span>
        </div>
      )}
    </div>
  );
}

export function RunCallouts({ lab }: { lab: AnyLab }) {
  return (
    <>
      {lab.stale && (
        <Callout tone="warn">
          Parameters changed since the last run. Press <b>Run</b> to execute with the new values; the panels still show the previous run.
        </Callout>
      )}
      {lab.runError && <Callout tone="bad" title="Run rejected">{lab.runError}</Callout>}
      {lab.run?.error && <Callout tone="bad" title="Python raised an exception">{lab.run.error}</Callout>}
      {lab.loadError && !lab.loadError.offline && <Callout tone="bad" title="Could not load experiment">{lab.loadError.message}</Callout>}
    </>
  );
}

/** Source with editable parameters, highlighted active lines and clickable step groups. */
export function CodePanel({ lab, maxHeight = 620 }: { lab: AnyLab; maxHeight?: number }) {
  const { meta, vals, run, frame, frames, i } = lab;
  return (
    <Panel
      title={
        <span className="flex flex-wrap items-center gap-2">
          <span className="font-mono">{lab.exp.file}</span>
          {meta && (meta.execution === "library" ? <Badge tone="accent">Library computation</Badge> : <Badge>Educational implementation</Badge>)}
        </span>
      }
      actions={<ExecutionBadge kind="python" />}
    >
      {meta && vals ? (
        <>
          <CodeView
            source={run && !lab.stale ? run.source : meta.source}
            active={frame ? frame.snap.lines : null}
            groups={meta.step_groups}
            params={meta.params}
            paramLines={meta.param_lines}
            values={vals}
            onParamChange={lab.setParam}
            onLineClick={(line) => {
              // jump to the next snapshot produced by the clicked step group
              const k = frames.findIndex((f, j) => j > i && line >= f.snap.lines[0] && line <= f.snap.lines[1]);
              const k2 = k >= 0 ? k : frames.findIndex((f) => line >= f.snap.lines[0] && line <= f.snap.lines[1]);
              if (k2 >= 0) lab.goTo(k2);
            }}
            maxHeight={maxHeight}
          />
          <ul className="mt-2 space-y-0.5 text-[12px] leading-snug text-fg-3">
            {meta.notes.map((n) => (
              <li key={n}>· {n}</li>
            ))}
            <li>· Edit the boxed numbers, then Run. Click a marked line (grey gutter) or a step above to jump to it.</li>
          </ul>
        </>
      ) : (
        <div className="py-10 text-center text-sm text-fg-3">Loading source from the execution service…</div>
      )}
    </Panel>
  );
}

export function OutputPanel({ run, framework }: { run: RunResponse<unknown>; framework: string }) {
  return (
    <Panel title="Program output">
      <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[12px] text-fg">{run.stdout || "(no output)"}</pre>
      {run.warnings.map((w) => (
        <div key={w} className="mt-1 font-mono text-[12px] text-warn">
          {w}
        </div>
      ))}
      <div className="mt-2 text-[11.5px] text-fg-3">
        {run.snapshots.length} snapshots · executed in {fmt(run.elapsed_ms, 1)} ms · Python {run.environment.python} · NumPy {run.environment.numpy}
        {framework === "pytorch" && <> · PyTorch {run.environment.torch}</>}
        {framework === "sklearn" && <> · scikit-learn {run.environment.sklearn}</>}
      </div>
    </Panel>
  );
}

export function VariablesPanel({ frame }: { frame: TraceFrame<unknown> }) {
  return (
    <Panel title="Variables (Python namespace at this snapshot)">
      <div className="max-h-80 overflow-auto">
        <table className="w-full text-left text-[11.5px]">
          <thead className="sticky top-0 bg-surface text-[10.5px] uppercase tracking-wide text-fg-3">
            <tr>
              <th className="py-1 pr-3 font-medium">name</th>
              <th className="py-1 pr-3 font-medium">type</th>
              <th className="py-1 font-medium">value</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line font-mono">
            {Object.entries(frame.locals).map(([name, v]) => {
              const changed = name in frame.snap.locals && frame.snap.index > 0;
              return (
                <tr key={name} className={changed ? "bg-accent-soft" : ""}>
                  <td className={`py-1 pr-3 align-top ${changed ? "font-semibold text-fg" : "text-fg-2"}`}>
                    {name}
                    {changed && <span className="sr-only"> (changed)</span>}
                  </td>
                  <td className="whitespace-nowrap py-1 pr-3 align-top text-fg-3">
                    {v.type}
                    {v.shape ? ` ${JSON.stringify(v.shape)}` : ""}
                    {v.dtype ? ` ${v.dtype}` : ""}
                  </td>
                  <td className="py-1 align-top">
                    <pre className="whitespace-pre text-fg">{v.value}</pre>
                    {v.grad_fn && <div className="text-[10.5px] text-fg-3">grad_fn={v.grad_fn}</div>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1.5 text-[11px] text-fg-3">Highlighted rows changed in this snapshot.</p>
    </Panel>
  );
}
