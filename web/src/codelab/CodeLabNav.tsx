import Link from "next/link";

const EXPERIMENTS = [
  { href: "/code-lab", label: "Linear regression" },
  { href: "/code-lab/k-means", label: "K-Means" },
];

/** Switch between Code Lab experiments (each has its own page). */
export function CodeLabNav({ current }: { current: string }) {
  return (
    <nav aria-label="Code Lab experiments" className="inline-flex rounded-md border border-line-strong p-0.5 text-[12.5px]">
      {EXPERIMENTS.map((e) => (
        <Link
          key={e.href}
          href={e.href}
          aria-current={e.href === current ? "page" : undefined}
          className={`rounded px-2.5 py-1 ${e.href === current ? "bg-surface-2 font-medium text-fg" : "text-fg-3 hover:text-fg"}`}
        >
          {e.label}
        </Link>
      ))}
    </nav>
  );
}
