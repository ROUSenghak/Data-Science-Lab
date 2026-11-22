import type { ReactNode } from "react";
import { ExecutionBadge, Expand } from "./primitives";

/** Shared building blocks of the long-form lesson pages. */

export const Lead = ({ children }: { children: ReactNode }) => <p className="max-w-[68ch] text-[16.5px] leading-relaxed text-fg-2">{children}</p>;

/**
 * An interactive figure in the text. Rules above and below instead of a card, so the controls and
 * plots inside are not boxed twice; the caption names the question the figure answers.
 */
export function Figure({ title, children, kind = "browser" }: { title: ReactNode; children: ReactNode; kind?: "browser" | "python" }) {
  return (
    <figure className="my-10 border-y border-line py-5">
      <figcaption className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-serif text-[17px] font-semibold tracking-tight text-fg">{title}</span>
        <ExecutionBadge kind={kind} />
      </figcaption>
      {children}
    </figure>
  );
}

export function Exercise({ n, children, solution }: { n: number; children: ReactNode; solution: ReactNode }) {
  return (
    <li className="border-t border-line py-4">
      <div className="flex gap-3">
        <span className="font-mono text-sm text-fg-3">{n}.</span>
        <div className="prose-lab min-w-0 flex-1">
          {children}
          <Expand title="Solution">{solution}</Expand>
        </div>
      </div>
    </li>
  );
}
