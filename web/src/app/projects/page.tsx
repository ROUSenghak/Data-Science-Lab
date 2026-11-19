import type { Metadata } from "next";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Badge } from "@/components/ui/primitives";
import { PROJECTS } from "@/content/curriculum";

export const metadata: Metadata = { title: "Projects" };

export default function ProjectsPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 pb-20 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Projects" }]} />
      <h1 className="mt-3 font-serif text-3xl font-semibold tracking-tight">Projects</h1>
      <p className="mt-1 max-w-2xl text-[15px] text-fg-2">
        Guided projects that combine several lessons on realistic data. None are published yet. They will be added once the lessons they depend on are complete.
      </p>
      <ul className="mt-8 divide-y divide-line border-y border-line">
        {PROJECTS.map((p) => (
          <li key={p.title} className="flex flex-col gap-2 py-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="font-medium text-fg-2">{p.title}</div>
              <div className="text-[14px] text-fg-3">{p.summary}</div>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {p.concepts.map((c) => (
                  <Badge key={c} tone="muted">
                    {c}
                  </Badge>
                ))}
              </div>
            </div>
            <Badge tone="muted">Planned</Badge>
          </li>
        ))}
      </ul>
    </div>
  );
}
