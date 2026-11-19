import Link from "next/link";
import { TexBlock } from "@/components/math/Tex";
import { MODULES } from "@/content/curriculum";

const LOOP = [
  { verb: "Read", what: "the mathematics", href: "/learn/machine-learning/linear-regression#loss" },
  { verb: "Explore", what: "the code", href: "/code-lab" },
  { verb: "Execute", what: "the algorithm", href: "/code-lab" },
  { verb: "Watch", what: "the model learn", href: "/labs/linear-regression#gradient-descent" },
  { verb: "Inspect", what: "every intermediate value", href: "/code-lab?framework=pytorch" },
  { verb: "Understand", what: "why it works", href: "/learn/machine-learning/linear-regression#gradient-descent" },
];

export default function Home() {
  const available = MODULES.flatMap((m) => m.topics).filter((t) => t.status === "available").length;
  const total = MODULES.flatMap((m) => m.topics).length;
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <section className="grid gap-10 pt-16 pb-14 md:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)] md:items-center">
        <div>
          <h1 className="font-serif text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">Data science you can see, run and take apart.</h1>
          <p className="mt-4 max-w-xl text-[17px] leading-relaxed text-fg-2">
            Each topic connects the mathematics to an interactive experiment and to real Python, with every intermediate value open to inspection.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/learn/machine-learning/linear-regression" className="rounded-md bg-[var(--viz-model)] px-4 py-2 text-sm font-medium text-white hover:brightness-110">
              Start with linear regression
            </Link>
            <Link href="/code-lab" className="rounded-md border border-line-strong px-4 py-2 text-sm font-medium hover:bg-surface-2">
              Open the Code Lab
            </Link>
          </div>
        </div>
        <div className="rounded-lg border border-line bg-surface px-5 py-4 text-[15px]">
          <div className="text-[12px] font-medium uppercase tracking-wide text-fg-3">One gradient step, three views</div>
          <TexBlock>{"\\boldsymbol\\theta_{t+1} = \\boldsymbol\\theta_t - \\eta\\,\\nabla_{\\boldsymbol\\theta} L(\\boldsymbol\\theta_t)"}</TexBlock>
          <pre className="overflow-x-auto rounded-md bg-code px-3 py-2 font-mono text-[12.5px] leading-relaxed text-fg">
            {`loss.backward()      # autograd fills .grad
optimizer.step()     # θ ← θ − η·θ.grad`}
          </pre>
          <p className="mt-3 text-[13px] text-fg-3">In the Code Lab these lines run in real PyTorch, and the inspector shows the numbers each one produced.</p>
        </div>
      </section>

      <section aria-labelledby="loop" className="border-t border-line py-12">
        <h2 id="loop" className="text-sm font-semibold uppercase tracking-wide text-fg-3">
          How every topic works
        </h2>
        <ol className="mt-5 grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3 lg:grid-cols-6">
          {LOOP.map((s, k) => (
            <li key={s.verb}>
              <Link href={s.href} className="group block">
                <span className="font-mono text-xs text-fg-3">{k + 1}</span>
                <div className="font-serif text-lg font-semibold group-hover:text-accent">{s.verb}</div>
                <div className="text-[14px] text-fg-2">{s.what}</div>
              </Link>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="sections" className="grid gap-8 border-t border-line py-12 md:grid-cols-4">
        <h2 id="sections" className="sr-only">
          Sections
        </h2>
        {[
          { href: "/learn", title: "Learn", text: "Lessons: intuition, derivations and exercises." },
          { href: "/labs", title: "Labs", text: "Change parameters and watch algorithms respond." },
          { href: "/code-lab", title: "Code Lab", text: "Step through real NumPy, PyTorch and scikit-learn." },
          { href: "/projects", title: "Projects", text: "Guided end-to-end work (planned)." },
        ].map((c) => (
          <Link key={c.href} href={c.href} className="group">
            <div className="font-medium group-hover:text-accent">{c.title} →</div>
            <p className="mt-1 text-[14px] text-fg-3">{c.text}</p>
          </Link>
        ))}
      </section>

      <section aria-labelledby="curriculum" className="border-t border-line py-12">
        <div className="flex items-baseline justify-between">
          <h2 id="curriculum" className="text-sm font-semibold uppercase tracking-wide text-fg-3">
            Curriculum
          </h2>
          <span className="text-[13px] text-fg-3">
            {available} of {total} topics published
          </span>
        </div>
        <ul className="mt-4 grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          {MODULES.map((m) => (
            <li key={m.slug}>
              <Link href={`/learn#${m.slug}`} className="group block">
                <div className="font-medium group-hover:text-accent">{m.title}</div>
                <div className="text-[13.5px] text-fg-3">{m.blurb}</div>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
