import type { Metadata } from "next";
import Link from "next/link";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { LinearRegressionLab } from "@/labs/linear-regression/LinearRegressionLab";

export const metadata: Metadata = { title: "Linear regression playground" };

export default function LinearRegressionLabPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 pb-20 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Labs", href: "/labs" }, { label: "Machine Learning" }, { label: "Linear regression" }]} />
      <h1 className="mt-3 font-serif text-3xl font-semibold tracking-tight">Linear regression playground</h1>
      <p className="mt-1 mb-6 max-w-3xl text-[15px] text-fg-2">
        Explore the loss surface and the gradient-descent dynamics behind least squares. Theory in the{" "}
        <Link href="/learn/machine-learning/linear-regression" className="text-accent underline underline-offset-2">
          lesson
        </Link>
        ; real Python execution in the{" "}
        <Link href="/code-lab" className="text-accent underline underline-offset-2">
          Code Lab
        </Link>
        .
      </p>
      <LinearRegressionLab />
    </div>
  );
}
