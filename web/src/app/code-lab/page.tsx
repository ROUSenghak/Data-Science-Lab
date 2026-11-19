import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { CodeLab } from "@/codelab/CodeLab";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";

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
      </div>
      <Suspense fallback={<div className="py-16 text-center text-sm text-fg-3">Loading Code Lab…</div>}>
        <CodeLab />
      </Suspense>
    </div>
  );
}
