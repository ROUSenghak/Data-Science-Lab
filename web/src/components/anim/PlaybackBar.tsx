"use client";

import type { ReactNode } from "react";
import { Button, Icon, Segmented } from "@/components/ui/controls";
import type { Playback } from "./hooks";

/**
 * Shared transport for every replay: play/pause, step back/forward, reset, speed and a scrubber.
 * `status` names the computed state being shown (e.g. "epoch 12 · gradient").
 */
export function PlaybackBar({
  pb,
  speeds,
  status,
  scrubLabel,
  extra,
}: {
  pb: Playback;
  speeds: { value: string; label: ReactNode }[];
  status: ReactNode;
  scrubLabel: string;
  /** Extra controls placed after the speed selector. */
  extra?: ReactNode;
}) {
  const atEnd = pb.index >= pb.last;
  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Playback">
      <Button variant="primary" onClick={pb.toggle} disabled={pb.count <= 1} ariaLabel={pb.playing ? "Pause" : atEnd ? "Replay" : "Play"}>
        {pb.playing ? Icon.pause : Icon.play} <span className="w-11 text-left">{pb.playing ? "Pause" : atEnd ? "Replay" : "Play"}</span>
      </Button>
      <Button onClick={pb.prev} disabled={pb.index === 0} ariaLabel="Previous step" title="Previous step">
        {Icon.back}
      </Button>
      <Button onClick={pb.next} disabled={atEnd} ariaLabel="Next step" title="Next step">
        {Icon.step} Step
      </Button>
      <Button variant="ghost" onClick={pb.reset} disabled={pb.index === 0 && !pb.playing} ariaLabel="Reset to the first step">
        {Icon.reset} Reset
      </Button>
      <Segmented ariaLabel="Playback speed" value={pb.speed} onChange={(v) => pb.setSpeed(v)} options={speeds} />
      {extra}
      <input
        type="range"
        aria-label={scrubLabel}
        aria-valuetext={typeof status === "string" ? status : undefined}
        className="min-w-32 flex-1"
        min={0}
        max={pb.last}
        value={pb.index}
        onChange={(e) => pb.seek(Number(e.target.value))}
      />
      <span className="whitespace-nowrap font-mono text-[12px] tabular-nums text-fg-2" aria-live="polite">
        {status}
      </span>
    </div>
  );
}
