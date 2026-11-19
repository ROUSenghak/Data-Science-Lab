import type { ReactNode } from "react";

export function Badge({ children, tone = "neutral", title }: { children: ReactNode; tone?: "neutral" | "accent" | "ok" | "warn" | "muted"; title?: string }) {
  const tones = {
    neutral: "border-line-strong text-fg-2",
    accent: "border-accent/40 text-accent bg-accent-soft",
    ok: "border-ok/40 text-ok",
    warn: "border-warn/40 text-warn bg-warn-soft",
    muted: "border-line text-fg-3",
  };
  return (
    <span title={title} className={`inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-0.5 text-[11px] font-medium leading-none ${tones[tone]}`}>
      {children}
    </span>
  );
}

/** Explicit label distinguishing in-browser simulation from real Python execution. */
export function ExecutionBadge({ kind }: { kind: "browser" | "python" }) {
  return kind === "browser" ? (
    <Badge tone="neutral" title="Computed in your browser with a TypeScript implementation, tested against NumPy.">
      <span aria-hidden>◇</span> Runs in browser (TypeScript)
    </Badge>
  ) : (
    <Badge tone="accent" title="Executed by a Python process on the lab server.">
      <span aria-hidden>▶</span> Real Python execution
    </Badge>
  );
}

/** Collapsible deeper explanation. Native <details> keeps it accessible without JS. */
export function Expand({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <details className="group my-4 rounded-md border border-line bg-surface">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-2.5 text-sm font-medium text-fg-2 hover:text-fg [&::-webkit-details-marker]:hidden">
        <svg width="12" height="12" viewBox="0 0 12 12" className="transition-transform group-open:rotate-90" aria-hidden>
          <path d="M4 2l4 4-4 4" stroke="currentColor" strokeWidth="1.6" fill="none" />
        </svg>
        {title}
      </summary>
      <div className="prose-lab border-t border-line px-4 py-3">{children}</div>
    </details>
  );
}

export function Callout({ tone = "note", title, children }: { tone?: "note" | "warn" | "bad"; title?: ReactNode; children: ReactNode }) {
  const tones = { note: "border-line bg-surface-2", warn: "border-warn/40 bg-warn-soft", bad: "border-bad/40 bg-bad-soft" };
  return (
    <div className={`rounded-md border px-3.5 py-2.5 text-sm leading-relaxed ${tones[tone]}`} role={tone === "note" ? undefined : "status"}>
      {title && <div className="mb-0.5 font-medium">{title}</div>}
      {children}
    </div>
  );
}

export function SectionHeading({ id, children, kicker, first }: { id: string; children: ReactNode; kicker?: string; first?: boolean }) {
  return (
    <h2 id={id} className={`group ${first ? "mt-0" : "mt-14"} mb-3 scroll-mt-20 font-serif text-2xl font-semibold tracking-tight`}>
      {kicker && <span className="mr-2 font-sans text-sm font-normal text-fg-3">{kicker}</span>}
      {children}
      <a href={`#${id}`} className="ml-2 text-base text-fg-3 opacity-0 group-hover:opacity-100" aria-label="Link to section">
        #
      </a>
    </h2>
  );
}

export function Panel({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`min-w-0 rounded-lg border border-line bg-surface ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-3 py-2">
          <h3 className="text-[13px] font-medium text-fg-2">{title}</h3>
          {actions}
        </header>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}
