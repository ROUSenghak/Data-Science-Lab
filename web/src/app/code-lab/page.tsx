import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { CodeLab } from "@/codelab/CodeLab";
import { CodeLabNav } from "@/codelab/CodeLabNav";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Tex } from "@/components/math/Tex";
import { Expand } from "@/components/ui/primitives";

const ABOUT: [string, React.ReactNode][] = [
  ["What", <>Fit <Tex>{"\\hat y = w x + b"}</Tex> to noisy points by minimizing the mean squared error <Tex>{"L(w,b)"}</Tex>.</>],
  ["Why", <>The smallest model trained by loss minimization: forward pass, loss, gradient and update are exactly those of a neural network.</>],
  [
    "How",
    <>
      NumPy and PyTorch run gradient descent <Tex>{"\\theta \\leftarrow \\theta - \\eta\\nabla L"}</Tex> (hand-written gradient vs autograd). scikit-learn solves least squares directly, with no iterations.
    </>,
  ],
  [
    "Experiment",
    <>
      Vary η: up to <Tex>{"1/\\lambda_{\\max}"}</Tex> no overshoot; between <Tex>{"1/\\lambda_{\\max}"}</Tex> and <Tex>{"2/\\lambda_{\\max}"}</Tex> the path zig-zags but converges; above <Tex>{"2/\\lambda_{\\max}"}</Tex> it diverges. Compare runs in the history panel.
    </>,
  ],
  [
    "Limits",
    <>
      Only a straight line: curved trends and outliers distort it (see the{" "}
      <Link href="/labs/linear-regression#failure-cases" className="text-accent underline underline-offset-2">
        failure cases
      </Link>
      ). Gradient descent slows down when the loss surface is elongated (large κ).
    </>,
  ],
];

export const metadata: Metadata = { title: "Code Lab — Linear regression" };

export default function CodeLabPage() {
  return (
    <div className="mx-auto max-w-[1440px] px-4 pb-16 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Code Lab", href: "/code-lab" }, { label: "Linear regression" }]} />
      <div className="mt-3 mb-2 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-serif text-3xl font-semibold tracking-tight">Linear regression, executed</h1>
          <p className="mt-1 max-w-3xl text-[15px] text-fg-2">
            The same model in NumPy, PyTorch and scikit-learn. Code, equations and plots all read from one recorded Python execution.{" "}
            <Link href="/learn/machine-learning/linear-regression" className="text-accent underline underline-offset-2">
              Read the lesson
            </Link>
            .
          </p>
        </div>
        <CodeLabNav current="/code-lab" />
      </div>
      <p className="mb-2 text-[13px] text-fg-2">
        <b className="font-medium text-fg">Use it:</b> pick an implementation → edit the boxed numbers in the code → <b className="font-medium text-fg">Run</b> → step through the recorded trace with ← / →, or click a step or a code line.
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
        <CodeLab />
      </Suspense>
    </div>
  );
}
