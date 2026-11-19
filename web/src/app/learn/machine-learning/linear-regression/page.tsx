import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import { Breadcrumbs } from "@/components/layout/Breadcrumbs";
import { Tex, TexBlock } from "@/components/math/Tex";
import { Callout, ExecutionBadge, Expand, SectionHeading } from "@/components/ui/primitives";
import { LINEAR_REGRESSION_SECTIONS } from "@/content/curriculum";
import { LessonFit, LessonGD } from "@/labs/linear-regression/LessonWidgets";

export const metadata: Metadata = {
  title: "Linear Regression",
  description: "Least squares, gradient descent, and the stability limit of the learning rate.",
};

const Lead = ({ children }: { children: ReactNode }) => <p className="text-[16.5px] leading-relaxed text-fg-2">{children}</p>;

function Widget({ title, children, kind = "browser" }: { title: string; children: ReactNode; kind?: "browser" | "python" }) {
  return (
    <figure className="my-6 rounded-lg border border-line bg-surface p-4">
      <figcaption className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[13px] font-medium text-fg-2">
        {title}
        <ExecutionBadge kind={kind} />
      </figcaption>
      {children}
    </figure>
  );
}

function Exercise({ n, children, solution }: { n: number; children: ReactNode; solution: ReactNode }) {
  return (
    <li className="border-t border-line py-4">
      <div className="flex gap-3">
        <span className="font-mono text-sm text-fg-3">{n}.</span>
        <div className="prose-lab min-w-0 flex-1">
          {children}
          <Expand title="Solution">{solution}</Expand>
        </div>
      </div>
    </li>
  );
}

export default function LinearRegressionLesson() {
  return (
    <div className="mx-auto max-w-7xl px-4 pb-24 pt-6 sm:px-6">
      <Breadcrumbs items={[{ label: "Learn", href: "/learn" }, { label: "Machine Learning", href: "/learn#machine-learning" }, { label: "Linear Regression" }]} />

      <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_200px]">
        <article className="min-w-0 max-w-4xl">
          <header className="mt-4">
            <h1 className="font-serif text-4xl font-semibold tracking-tight">Linear Regression</h1>
            <p className="mt-2 text-[15px] text-fg-3">Prerequisites: derivatives, sums and basic matrix notation · about 30 minutes</p>
            <div className="mt-4 flex flex-wrap gap-2 text-[13px]">
              <Link href="/labs/linear-regression" className="rounded-md border border-line-strong px-3 py-1.5 hover:bg-surface-2">
                Open the playground
              </Link>
              <Link href="/code-lab" className="rounded-md border border-line-strong px-3 py-1.5 hover:bg-surface-2">
                Open in Code Lab
              </Link>
            </div>
          </header>

          {/* 1 */}
          <SectionHeading id="motivation">Motivation</SectionHeading>
          <Lead>
            Predict a number <Tex>y</Tex> from a feature <Tex>x</Tex> with a straight line. It is the simplest model trained by minimizing a loss, and the loss, gradient and learning rate here reappear unchanged in deep learning.
          </Lead>
          <Widget title="Try it: choose a line, read its error">
            <LessonFit />
          </Widget>

          {/* 2 */}
          <SectionHeading id="model">The model</SectionHeading>
          <TexBlock>{"\\hat y_i = w\\,x_i + b \\qquad\\Longleftrightarrow\\qquad \\hat{\\mathbf y} = A\\boldsymbol\\theta,\\quad A = \\begin{bmatrix} x_1 & 1\\\\ \\vdots & \\vdots \\\\ x_n & 1\\end{bmatrix},\\; \\boldsymbol\\theta = \\begin{bmatrix} w \\\\ b\\end{bmatrix}"}</TexBlock>
          <ul className="prose-lab">
            <li>
              <Tex>w</Tex> (slope): change in prediction per unit of <Tex>x</Tex>.
            </li>
            <li>
              <Tex>b</Tex> (intercept): prediction at <Tex>x = 0</Tex>.
            </li>
            <li>
              The residual <Tex>{"r_i = \\hat y_i - y_i"}</Tex> measures the error on point <Tex>i</Tex>.
            </li>
          </ul>

          {/* 3 */}
          <SectionHeading id="loss">Mean squared error</SectionHeading>
          <TexBlock>{"L(w, b) = \\frac{1}{n}\\sum_{i=1}^{n} \\big(w x_i + b - y_i\\big)^2"}</TexBlock>
          <div className="prose-lab">
            <p>Why squares?</p>
            <ul>
              <li>
                <Tex>L</Tex> is a convex quadratic in <Tex>(w, b)</Tex>: one minimum, found in closed form or by gradient descent.
              </li>
              <li>Under Gaussian noise, minimizing it is maximum likelihood (expand below).</li>
              <li>Cost: large errors dominate, so outliers have outsized influence.</li>
            </ul>
          </div>
          <Expand title="Derivation: least squares = maximum likelihood under Gaussian noise">
            <p>
              Assume <Tex>{"y_i = w x_i + b + \\varepsilon_i"}</Tex> with independent <Tex>{"\\varepsilon_i \\sim \\mathcal N(0, \\sigma^2)"}</Tex>. The log-likelihood is
            </p>
            <TexBlock>{"\\log p(\\mathbf y\\mid w,b) = -\\frac{n}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^n (y_i - w x_i - b)^2."}</TexBlock>
            <p>
              The first term does not depend on <Tex>(w,b)</Tex>, so maximizing the likelihood is the same as minimizing the sum of squared residuals, and therefore <Tex>L</Tex>.
            </p>
          </Expand>

          {/* 4 */}
          <SectionHeading id="closed-form">Closed-form solution</SectionHeading>
          <Lead>Set both partial derivatives to zero and solve:</Lead>
          <TexBlock>{"w^* = \\frac{\\sum_i (x_i - \\bar x)(y_i - \\bar y)}{\\sum_i (x_i - \\bar x)^2} = \\frac{\\operatorname{Cov}(x, y)}{\\operatorname{Var}(x)}, \\qquad b^* = \\bar y - w^*\\bar x"}</TexBlock>
          <TexBlock>{"\\text{matrix form:}\\qquad A^\\top A\\,\\boldsymbol\\theta^* = A^\\top \\mathbf y \\quad\\Rightarrow\\quad \\boldsymbol\\theta^* = (A^\\top A)^{-1}A^\\top \\mathbf y"}</TexBlock>
          <Expand title="Derivation of the normal equations">
            <p>
              Write <Tex>{"L(\\boldsymbol\\theta) = \\tfrac1n\\lVert A\\boldsymbol\\theta - \\mathbf y\\rVert^2"}</Tex>. Its gradient is
            </p>
            <TexBlock>{"\\nabla L(\\boldsymbol\\theta) = \\tfrac{2}{n} A^\\top (A\\boldsymbol\\theta - \\mathbf y)."}</TexBlock>
            <p>
              Setting it to zero gives <Tex>{"A^\\top A\\boldsymbol\\theta = A^\\top\\mathbf y"}</Tex>. The second row of this system reads <Tex>{"\\sum_i (w x_i + b - y_i) = 0"}</Tex>, i.e. <Tex>{"b = \\bar y - w\\bar x"}</Tex>. Substituting into the first row gives the covariance formula for <Tex>w</Tex>.
            </p>
            <p>
              <Tex>{"A^\\top A"}</Tex> is invertible iff the <Tex>x_i</Tex> are not all equal (<Tex>{"\\operatorname{Var}(x) > 0"}</Tex>). Libraries solve the system with QR or SVD (e.g. <code>scipy.linalg.lstsq</code>) rather than forming the inverse, which is numerically safer.
            </p>
          </Expand>

          {/* 5 */}
          <SectionHeading id="gradient-descent">Gradient descent</SectionHeading>
          <Lead>
            Most models have no closed form. Gradient descent instead takes small steps against the gradient, using only <Tex>{"\\nabla L"}</Tex>:
          </Lead>
          <TexBlock>{"\\frac{\\partial L}{\\partial w} = \\frac{2}{n}\\sum_i r_i x_i,\\qquad \\frac{\\partial L}{\\partial b} = \\frac{2}{n}\\sum_i r_i, \\qquad \\boldsymbol\\theta_{t+1} = \\boldsymbol\\theta_t - \\eta\\,\\nabla L(\\boldsymbol\\theta_t)"}</TexBlock>
          <Callout title="The learning rate has a hard limit">
            Gradient descent on this loss converges if and only if <Tex>{"0 < \\eta < 2/\\lambda_{\\max}(H)"}</Tex>, where <Tex>{"H = \\tfrac{2}{n}A^\\top A"}</Tex> is the Hessian. The convergence speed depends on the condition number <Tex>{"\\kappa = \\lambda_{\\max}/\\lambda_{\\min}"}</Tex>.
          </Callout>
          <Expand title="Why: the error evolves linearly">
            <p>
              Because <Tex>L</Tex> is quadratic, <Tex>{"\\nabla L(\\boldsymbol\\theta) = H(\\boldsymbol\\theta - \\boldsymbol\\theta^*)"}</Tex>. With error <Tex>{"\\mathbf e_t = \\boldsymbol\\theta_t - \\boldsymbol\\theta^*"}</Tex>:
            </p>
            <TexBlock>{"\\mathbf e_{t+1} = (I - \\eta H)\\,\\mathbf e_t \\quad\\Rightarrow\\quad \\mathbf e_t = (I-\\eta H)^t\\,\\mathbf e_0."}</TexBlock>
            <p>
              In the eigenbasis of <Tex>H</Tex>, each component is multiplied by <Tex>{"1 - \\eta\\lambda_k"}</Tex> per step. All components shrink iff <Tex>{"|1-\\eta\\lambda_k| < 1"}</Tex> for every <Tex>k</Tex>, i.e. <Tex>{"\\eta < 2/\\lambda_{\\max}"}</Tex>. The slowest component shrinks by <Tex>{"1-\\eta\\lambda_{\\min}"}</Tex>; with the largest stable η this is about <Tex>{"1 - 2/\\kappa"}</Tex>, so elongated loss surfaces converge slowly.
            </p>
            <p>
              Here <Tex>{"H = 2\\begin{bmatrix}\\overline{x^2} & \\bar x \\\\ \\bar x & 1\\end{bmatrix}"}</Tex>. When <Tex>{"\\bar x \\neq 0"}</Tex>, <Tex>w</Tex> and <Tex>b</Tex> are coupled and the ellipses are tilted. Centering <Tex>x</Tex> removes the coupling.
            </p>
          </Expand>
          <Widget title="Watch gradient descent on the exact loss surface">
            <LessonGD />
          </Widget>

          {/* 6 */}
          <SectionHeading id="implementation">Implementation</SectionHeading>
          <Lead>The Code Lab runs three implementations of the same model on the same data, and records each step for inspection.</Lead>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-[14px]">
              <thead className="border-b border-line text-[12px] uppercase tracking-wide text-fg-3">
                <tr>
                  <th className="py-2 pr-4 font-medium">Version</th>
                  <th className="py-2 pr-4 font-medium">Gradient computed by</th>
                  <th className="py-2 pr-4 font-medium">What you can inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                <tr>
                  <td className="py-2.5 pr-4">
                    <Link className="text-accent underline underline-offset-2" href="/code-lab?framework=numpy">
                      NumPy
                    </Link>
                  </td>
                  <td className="py-2.5 pr-4">Hand-derived formula</td>
                  <td className="py-2.5 pr-4">Every operation: forward, loss, gradient, update</td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-4">
                    <Link className="text-accent underline underline-offset-2" href="/code-lab?framework=pytorch">
                      PyTorch
                    </Link>
                  </td>
                  <td className="py-2.5 pr-4">Autograd (<code className="font-mono text-[13px]">loss.backward()</code>)</td>
                  <td className="py-2.5 pr-4">
                    <code className="font-mono text-[13px]">.grad</code> after backward, parameters after <code className="font-mono text-[13px]">optimizer.step()</code>
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-4">
                    <Link className="text-accent underline underline-offset-2" href="/code-lab?framework=sklearn">
                      scikit-learn
                    </Link>
                  </td>
                  <td className="py-2.5 pr-4">No gradient: direct least-squares solve</td>
                  <td className="py-2.5 pr-4">
                    Only the fitted <code className="font-mono text-[13px]">coef_</code>, <code className="font-mono text-[13px]">intercept_</code>; <code className="font-mono text-[13px]">.fit()</code> internals are not exposed
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* 7 */}
          <SectionHeading id="limitations">Assumptions and failure cases</SectionHeading>
          <div className="prose-lab">
            <ul>
              <li>
                <b>Linearity.</b> If <Tex>{"\\mathbb E[y\\mid x]"}</Tex> is curved, the fit is still the best line but residuals show structure.
              </li>
              <li>
                <b>Outliers.</b> Squared loss lets one extreme point move the whole line.
              </li>
              <li>
                <b>Independent, constant-variance noise.</b> Needed for standard errors and confidence intervals, not for computing the fit.
              </li>
              <li>
                <b>Extrapolation.</b> Nothing guarantees the line holds outside the observed range of <Tex>x</Tex>.
              </li>
              <li>
                <b>Association, not causation.</b> <Tex>w</Tex> describes how <Tex>y</Tex> varies with <Tex>x</Tex> in this data; a confounder can produce any slope.
              </li>
            </ul>
            <p>
              Counterexample: Anscombe&rsquo;s quartet consists of four datasets with nearly identical <Tex>w^*</Tex>, <Tex>b^*</Tex> and correlation, yet one is curved, one has an outlier and one is driven by a single point. Always plot residuals. Try the{" "}
              <Link href="/labs/linear-regression#failure-cases">failure cases in the playground</Link>.
            </p>
          </div>

          {/* 8 */}
          <SectionHeading id="exercises">Exercises</SectionHeading>
          <ol className="mt-2">
            <Exercise
              n={1}
              solution={
                <>
                  <p>
                    <Tex>{"\\bar x = 1,\\ \\bar y = 3"}</Tex>. <Tex>{"\\sum (x_i-\\bar x)(y_i-\\bar y) = (-1)(-2) + 0 + (1)(2) = 4"}</Tex>, <Tex>{"\\sum (x_i - \\bar x)^2 = 2"}</Tex>.
                  </p>
                  <p>
                    So <Tex>{"w^* = 2,\\ b^* = 3 - 2\\cdot 1 = 1"}</Tex>, and <Tex>{"L^* = 0"}</Tex>: the points lie exactly on <Tex>{"y = 2x + 1"}</Tex>.
                  </p>
                </>
              }
            >
              <p>
                Compute <Tex>w^*</Tex>, <Tex>b^*</Tex> and <Tex>L^*</Tex> by hand for <Tex>{"x = (0, 1, 2)"}</Tex>, <Tex>{"y = (1, 3, 5)"}</Tex>.
              </p>
            </Exercise>
            <Exercise
              n={2}
              solution={
                <p>
                  At the optimum <Tex>{"\\partial L/\\partial b = \\tfrac{2}{n}\\sum_i r_i = 0"}</Tex>, so <Tex>{"\\sum_i r_i = 0"}</Tex>. Without an intercept there is no such equation, and the residuals need not sum to zero.
                </p>
              }
            >
              <p>Show that the least-squares residuals sum to zero when the model has an intercept. What changes without one?</p>
            </Exercise>
            <Exercise
              n={3}
              solution={
                <>
                  <p>
                    With <Tex>{"\\bar x \\approx 2,\\ \\overline{x^2} \\approx 16/3"}</Tex>: <Tex>{"H \\approx 2\\begin{bmatrix}16/3 & 2\\\\ 2 & 1\\end{bmatrix}"}</Tex>, trace <Tex>{"\\approx 12.67"}</Tex>, determinant <Tex>{"\\approx 5.33"}</Tex>.
                  </p>
                  <p>
                    <Tex>{"\\lambda_{\\max} \\approx 12.23,\\ \\lambda_{\\min} \\approx 0.44"}</Tex>, so <Tex>{"\\eta_{\\text{crit}} = 2/\\lambda_{\\max} \\approx 0.164"}</Tex> and <Tex>{"\\kappa \\approx 28"}</Tex>. Compare with the value the playground reports for a large <Tex>n</Tex>.
                  </p>
                </>
              }
            >
              <p>
                For <Tex>{"x_i \\sim \\mathcal U(0, 4)"}</Tex> and large <Tex>n</Tex>, estimate the largest stable learning rate and the condition number.
              </p>
            </Exercise>
            <Exercise
              n={4}
              solution={
                <p>
                  After centering, <Tex>{"\\bar x = 0"}</Tex> so <Tex>{"H = \\operatorname{diag}(2\\operatorname{Var}(x),\\, 2)"}</Tex>. For <Tex>{"\\mathcal U(0,4)"}</Tex>, <Tex>{"\\operatorname{Var}(x) = 4/3"}</Tex>, so <Tex>{"\\kappa = 4/3"}</Tex> instead of <Tex>{"\\approx 28"}</Tex>, and <Tex>{"\\eta_{\\text{crit}} = 0.75"}</Tex>. Each parameter now converges independently and quickly.
                </p>
              }
            >
              <p>Explain why centering the feature speeds up gradient descent. Compute the new condition number for the data in exercise 3.</p>
            </Exercise>
          </ol>

          <nav className="mt-14 flex flex-wrap justify-between gap-3 border-t border-line pt-5 text-sm" aria-label="Lesson navigation">
            <Link href="/learn" className="text-fg-2 hover:text-fg">
              ← All modules
            </Link>
            <span className="text-fg-3">Next: Logistic regression (planned)</span>
          </nav>
        </article>

        <aside className="hidden lg:block">
          <nav aria-label="On this page" className="sticky top-20 mt-6 text-[13px]">
            <div className="mb-2 font-medium text-fg-2">On this page</div>
            <ul className="space-y-1.5 border-l border-line">
              {LINEAR_REGRESSION_SECTIONS.map((s) => (
                <li key={s.id}>
                  <a href={`#${s.id}`} className="-ml-px block border-l border-transparent pl-3 text-fg-3 hover:border-fg-2 hover:text-fg">
                    {s.title}
                  </a>
                </li>
              ))}
            </ul>
            <div className="mt-6 mb-2 font-medium text-fg-2">Related</div>
            <ul className="space-y-1.5 text-fg-3">
              <li>
                <Link href="/labs/linear-regression" className="hover:text-fg">
                  Playground
                </Link>
              </li>
              <li>
                <Link href="/code-lab" className="hover:text-fg">
                  Code Lab
                </Link>
              </li>
            </ul>
          </nav>
        </aside>
      </div>
    </div>
  );
}
