"use client";

import { useState, type ReactNode } from "react";
import { Button } from "@/components/ui/controls";

/**
 * "Predict, then run": the learner commits to an outcome before seeing it. The answer is computed
 * by the caller from the same data the animation uses. Resets when `resetKey` changes.
 */
export function PredictPrompt<T extends string>({
  question,
  options,
  answer,
  revealed: autoReveal,
  explanation,
  resetKey,
  onDismiss,
}: {
  question: ReactNode;
  options: { value: T; label: string }[];
  answer: T;
  /** Reveal automatically (e.g. when the replay reaches the end). */
  revealed: boolean;
  explanation: ReactNode;
  resetKey: unknown;
  onDismiss?: () => void;
}) {
  const [choice, setChoice] = useState<T | null>(null);
  const [forced, setForced] = useState(false);
  const [prevKey, setPrevKey] = useState(resetKey);
  if (prevKey !== resetKey) {
    setPrevKey(resetKey);
    setChoice(null);
    setForced(false);
  }
  const revealed = choice !== null && (forced || autoReveal);
  const correct = choice === answer;
  const label = (v: T) => options.find((o) => o.value === v)?.label;

  return (
    <div className="rounded-md border border-dashed border-line-strong px-3 py-2 text-[13.5px]" aria-live="polite">
      {choice === null ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 font-medium text-fg">Predict first:</span>
          <span className="text-fg-2">{question}</span>
          <span className="flex flex-wrap gap-1.5">
            {options.map((o) => (
              <Button key={o.value} onClick={() => setChoice(o.value)}>
                {o.label}
              </Button>
            ))}
          </span>
          {onDismiss && (
            <Button variant="ghost" onClick={onDismiss}>
              Skip predictions
            </Button>
          )}
        </div>
      ) : !revealed ? (
        <div className="flex flex-wrap items-center gap-2 text-fg-2">
          <span>
            Your prediction: <b className="font-medium text-fg">{label(choice)}</b>. Press play and watch.
          </span>
          <Button variant="ghost" onClick={() => setForced(true)}>
            Reveal now
          </Button>
        </div>
      ) : (
        <div>
          <span className={`font-medium ${correct ? "text-ok" : "text-warn"}`}>{correct ? "✓ Correct" : "✗ Not quite"}</span>
          <span className="text-fg-2">
            {correct ? (
              <> · it was <b className="font-medium text-fg">{label(answer)}</b>.</>
            ) : (
              <>
                {" "}· you said <b className="font-medium text-fg">{label(choice)}</b>, but it was <b className="font-medium text-fg">{label(answer)}</b>.
              </>
            )}
          </span>
          <div className="mt-1 text-[13px] text-fg-2">{explanation}</div>
        </div>
      )}
    </div>
  );
}
