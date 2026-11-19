import type { Metadata } from "next";
import Link from "next/link";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Badge, ExecutionBadge } from "@/components/ui/primitives";
import { LABS } from "@/content/curriculum";

export const metadata: Metadata = { title: "Labs" };

export default function LabsPage() {
  const topics = [...new Set(LABS.map((l) => l.topic))];
  return (
    <div className="mx-auto max-w-5xl px-4 pb-20 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Labs" }]} />
      <h1 className="mt-3 font-serif text-3xl font-semibold tracking-tight">Labs</h1>
      <p className="mt-1 max-w-2xl text-[15px] text-fg-2">
        Interactive experiments by topic. Each lab states where it computes: in your browser, or by executing real Python.
      </p>
      {topics.map((t) => (
        <section key={t} className="mt-10">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-fg-3">{t}</h2>
          <ul className="divide-y divide-line border-y border-line">
            {LABS.filter((l) => l.topic === t).map((l) => {
              const inner = (
                <div className="flex flex-col gap-1 py-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
                  <div className="min-w-0">
                    <div className={`font-medium ${l.status === "available" ? "text-fg group-hover:text-accent" : "text-fg-3"}`}>{l.title}</div>
                    <div className="text-[14px] text-fg-3">{l.summary}</div>
                  </div>
                  <div className="flex shrink-0 flex-wrap items-center gap-1.5">
                    <Badge tone="muted">{l.level}</Badge>
                    {l.status === "available" ? l.execution.map((e) => <ExecutionBadge key={e} kind={e} />) : <Badge tone="muted">Planned</Badge>}
                  </div>
                </div>
              );
              return (
                <li key={l.slug}>
                  {l.status === "available" && l.href ? (
                    <Link href={l.href} className="group block">
                      {inner}
                    </Link>
                  ) : (
                    <div aria-disabled="true">{inner}</div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
