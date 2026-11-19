import type { Metadata } from "next";
import Link from "next/link";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Badge } from "@/components/ui/primitives";
import { MODULES } from "@/content/curriculum";

export const metadata: Metadata = { title: "Learn" };

export default function LearnPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 pb-20 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Learn" }]} />
      <h1 className="mt-3 font-serif text-3xl font-semibold tracking-tight">Learn</h1>
      <p className="mt-1 max-w-2xl text-[15px] text-fg-2">
        Seven modules, from mathematical foundations to applied work. Every lesson covers intuition, formulation, interactive examples, implementation, limitations and exercises.
      </p>
      <ol className="mt-10 space-y-10">
        {MODULES.map((m, k) => (
          <li key={m.slug} id={m.slug} className="scroll-mt-20">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-sm text-fg-3">{String(k + 1).padStart(2, "0")}</span>
              <div>
                <h2 className="font-serif text-xl font-semibold">{m.title}</h2>
                <p className="text-[14px] text-fg-3">{m.blurb}</p>
              </div>
            </div>
            <ul className="mt-3 ml-8 divide-y divide-line border-y border-line">
              {m.topics.map((t) => (
                <li key={t.slug}>
                  {t.status === "available" && t.href ? (
                    <Link href={t.href} className="group flex items-center justify-between gap-4 py-3">
                      <span>
                        <span className="font-medium group-hover:text-accent">{t.title}</span>
                        <span className="block text-[14px] text-fg-3">{t.summary}</span>
                      </span>
                      <Badge tone="ok">Available</Badge>
                    </Link>
                  ) : (
                    <div className="flex items-center justify-between gap-4 py-3" aria-disabled="true">
                      <span>
                        <span className="text-fg-3">{t.title}</span>
                        <span className="block text-[14px] text-fg-3">{t.summary}</span>
                      </span>
                      <Badge tone="muted">Planned</Badge>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  );
}
