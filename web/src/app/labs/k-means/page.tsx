import type { Metadata } from "next";
import Link from "next/link";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { KMeansLab } from "@/labs/k-means/KMeansLab";

export const metadata: Metadata = { title: "K-Means playground" };

export default function KMeansLabPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 pb-20 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Labs", href: "/labs" }, { label: "Machine Learning" }, { label: "K-Means" }]} />
      <h1 className="mt-3 font-serif text-3xl font-semibold tracking-tight">K-Means playground</h1>
      <p className="mt-1 mb-6 max-w-3xl text-[15px] text-fg-2">
        Step through Lloyd&apos;s algorithm on 2-D data, compare initializations, choose K and find where K-Means fails. Theory in the{" "}
        <Link href="/learn/machine-learning/k-means" className="text-accent underline underline-offset-2">
          lesson
        </Link>
        ; real NumPy and scikit-learn in the{" "}
        <Link href="/code-lab/k-means" className="text-accent underline underline-offset-2">
          Code Lab
        </Link>
        .
      </p>
      <KMeansLab />
    </div>
  );
}
