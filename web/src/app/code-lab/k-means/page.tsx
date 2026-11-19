import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { CodeLabNav } from "@/codelab/CodeLabNav";
import { KMeansCodeLab } from "@/codelab/kmeans/KMeansCodeLab";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Tex } from "@/components/math/Tex";
import { Expand } from "@/components/ui/primitives";

const ABOUT: [string, React.ReactNode][] = [
  ["What", <>Partition points into K clusters by minimizing the within-cluster sum of squares <Tex>{"J = \\sum_i \\lVert x_i - \\mu_{c_i}\\rVert^2"}</Tex>.</>],
  ["Why", <>The standard clustering baseline; its two alternating steps are the simplest example of coordinate descent (and of EM).</>],
  [
    "How",
    <>
      NumPy runs Lloyd&apos;s algorithm line by line: distances, nearest-centroid assignment, centroid means, objective. scikit-learn&apos;s <code>KMeans.fit</code> runs the same iteration internally; only its results are visible.
    </>,
  ],
  ["Experiment", <>Change <code>init_method</code> and <code>init_seed</code> on the same data and compare the final J in the run table. Try the presets for a local minimum and an empty cluster.</>],
  [
    "Limits",
    <>
      Finds a local minimum only; assumes compact, roughly round clusters (see the{" "}
      <Link href="/labs/k-means#failure-cases" className="text-accent underline underline-offset-2">
        failure cases
      </Link>
      ). K must be chosen by you.
    </>,
  ],
];

export const metadata: Metadata = { title: "Code Lab — K-Means" };

export default function KMeansCodeLabPage() {
  return (
    <div className="mx-auto max-w-[1440px] px-4 pb-16 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Code Lab", href: "/code-lab" }, { label: "K-Means" }]} />
      <div className="mt-3 mb-2 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-serif text-3xl font-semibold tracking-tight">K-Means, executed</h1>
          <p className="mt-1 max-w-3xl text-[15px] text-fg-2">
            Lloyd&apos;s algorithm written in NumPy, and scikit-learn&apos;s KMeans on the same data. Code, equations and plots all read from one recorded Python execution.{" "}
            <Link href="/learn/machine-learning/k-means" className="text-accent underline underline-offset-2">
              Read the lesson
            </Link>{" "}
            or{" "}
            <Link href="/labs/k-means" className="text-accent underline underline-offset-2">
              open the playground
            </Link>
            .
          </p>
        </div>
        <CodeLabNav current="/code-lab/k-means" />
      </div>
      <p className="mb-2 text-[13px] text-fg-2">
        <b className="font-medium text-fg">Use it:</b> pick an implementation → edit the boxed numbers → <b className="font-medium text-fg">Run</b> → step with ← / →, click a step or a code line, click a point to inspect its distances.
      </p>
      <Expand title="About this experiment: what, why, how, limits">
        <dl className="grid gap-x-4 gap-y-1.5 sm:grid-cols-[110px_minmax(0,1fr)]">
          {ABOUT.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="font-medium text-fg">{k}</dt>
              <dd className="m-0 text-fg-2">{v}</dd>
            </div>
          ))}
        </dl>
      </Expand>
      <Suspense fallback={<div className="py-16 text-center text-sm text-fg-3">Loading Code Lab…</div>}>
        <KMeansCodeLab />
      </Suspense>
    </div>
  );
}
