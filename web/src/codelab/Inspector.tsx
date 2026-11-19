"use client";

import type { ReactNode } from "react";
import { Tex, TexBlock } from "@/components/math/Tex";
import { fmt, texNum } from "@/lib/format";
import { gradient, leastSquares, mean } from "@/lib/ml/linreg";
import type { Frame } from "./fold";
import type { ExperimentMeta } from "./types";

type Framework = ExperimentMeta["framework"];

interface Props {
  frame: Frame;
  framework: Framework;
  point: number;
}

/** Relative-or-absolute agreement check between a traced value and a browser recomputation. */
export function Check({ label, traced, recomputed, tol, refLabel = "browser recomputation" }: { label: ReactNode; traced: number | null | undefined; recomputed: number; tol: number; refLabel?: string }) {
  if (traced == null || !Number.isFinite(traced) || !Number.isFinite(recomputed)) {
    return <div className="text-[12px] text-fg-3">Cross-check unavailable (non-finite values).</div>;
  }
  const diff = Math.abs(traced - recomputed);
  const ok = diff <= tol * Math.max(1, Math.abs(recomputed));
  return (
    <div
      className="flex items-start gap-1.5 text-[12px] text-fg-2"
      title={`Traced Python value ${traced} vs ${refLabel} ${recomputed}. Passes if |Δ| ≤ ${tol} · max(1, |value|).`}
    >
      <span className={ok ? "text-ok" : "text-bad"} aria-hidden>
        {ok ? "✓" : "✗"}
      </span>
      <span>
        {label} = {fmt(recomputed, 6)} by {refLabel}
        <span className="text-fg-3">
          {" "}· |Δ| = {diff === 0 ? "0" : diff.toExponential(1)} {ok ? "≤" : ">"} tol {tol.toExponential(0)}
        </span>
        {!ok && <b className="text-bad"> mismatch</b>}
      </span>
    </div>
  );
}

export const Explain = ({ children }: { children: ReactNode }) => <p className="text-[13.5px] leading-relaxed text-fg-2">{children}</p>;

export function Inspector({ frame, framework, point }: Props) {
  const { state: s, prev, snap, locals } = frame;
  const x = s.x ?? [];
  const y = s.y ?? [];
  const n = x.length;
  const i = Math.min(point, Math.max(0, n - 1));
  const tol = framework === "pytorch" ? 1e-4 : 1e-9; // float32 vs float64
  const sub = `_{${i + 1}}`;
  const torch = framework === "pytorch";

  let body: ReactNode = null;

  switch (snap.step) {
    case "data":
      body = (
        <>
          <TexBlock>{`x_i \\sim \\mathcal U(0, 4),\\qquad y_i = 2x_i + 1 + \\varepsilon_i,\\quad \\varepsilon_i \\sim \\mathcal N(0, \\sigma^2)`}</TexBlock>
          {n > 0 && <TexBlock>{`n = ${n},\\qquad x${sub} = ${texNum(x[i])},\\quad y${sub} = ${texNum(y[i])}`}</TexBlock>}
          <Explain>
            {torch ? (
              <>NumPy generates the data, then it is copied into float32 tensors of shape (n, 1): PyTorch layers expect a batch dimension and a feature dimension.</>
            ) : framework === "sklearn" ? (
              <>scikit-learn expects a 2-D feature matrix <Tex>{"X\\in\\mathbb R^{n\\times 1}"}</Tex>, hence <code>reshape(-1, 1)</code>.</>
            ) : (
              <>A fixed seed makes the data reproducible. The same seed gives the same points in all three implementations.</>
            )}
          </Explain>
        </>
      );
      break;

    case "init":
      body = (
        <>
          <TexBlock>{`w_0 = ${texNum(s.w)},\\qquad b_0 = ${texNum(s.b)}`}</TexBlock>
          <Explain>
            {torch ? (
              <>
                <code>nn.Linear(1, 1)</code> holds a 1×1 weight and a bias. Its random initialization is overwritten inside <code>torch.no_grad()</code>, so the assignment is not recorded by autograd.
              </>
            ) : (
              <>Starting point of gradient descent. Any start converges for this convex loss if η is small enough.</>
            )}
          </Explain>
        </>
      );
      break;

    case "zero_grad":
      body = (
        <>
          <TexBlock>{`\\texttt{w.grad} \\leftarrow \\texttt{None},\\qquad \\texttt{b.grad} \\leftarrow \\texttt{None}`}</TexBlock>
          <Explain>
            <code>backward()</code> adds into <code>.grad</code>. Without clearing, gradients from previous epochs would accumulate. Epoch {s.epoch} starts here.
          </Explain>
        </>
      );
      break;

    case "forward": {
      const yh = s.y_hat?.[i];
      body = (
        <>
          <TexBlock>{`\\hat y_i = w\\,x_i + b`}</TexBlock>
          <TexBlock>{`\\hat y${sub} = ${texNum(s.w)}\\cdot${texNum(x[i], 4, true)} + ${texNum(s.b, 4, true)} = ${texNum(yh)}`}</TexBlock>
          <Explain>
            {torch ? (
              <>
                <code>model(X)</code> computes <Tex>{"XW^\\top + b"}</Tex> for all {n} points at once and records the operation for autograd
                {locals.y_hat?.grad_fn ? (
                  <>
                    {" "}(<code>grad_fn = {locals.y_hat.grad_fn}</code>)
                  </>
                ) : null}
                .
              </>
            ) : (
              <>Vectorized: one multiply-add per point. Showing point {i + 1}; click a point in the data plot to inspect another.</>
            )}
          </Explain>
          {yh != null && s.w != null && s.b != null && <Check label={`ŷ${i + 1}`} traced={yh} recomputed={s.w * x[i] + s.b} tol={tol} />}
        </>
      );
      break;
    }

    case "loss": {
      const yh = s.y_hat ?? [];
      const r = yh.map((v, k) => (v ?? NaN) - y[k]);
      const recomputed = mean(r.map((v) => v * v));
      body = (
        <>
          <TexBlock>{`L = \\frac{1}{n}\\sum_{i=1}^{n} (\\hat y_i - y_i)^2`}</TexBlock>
          {r.length >= 2 && (
            <TexBlock>{`L = \\tfrac{1}{${n}}\\big(${texNum(r[0], 3, true)}^2 + ${texNum(r[1], 3, true)}^2 + \\cdots\\big) = ${texNum(s.loss, 5)}`}</TexBlock>
          )}
          <Explain>
            {torch ? (
              <>
                <code>nn.MSELoss()</code> averages the squared residuals
                {locals.loss?.grad_fn ? <> and adds <code>{locals.loss.grad_fn}</code> to the graph</> : null}.
              </>
            ) : framework === "sklearn" ? (
              <>The training error of the fitted line. No line through these points has a lower MSE.</>
            ) : (
              <>Residual for point {i + 1}: <Tex>{`r${sub} = ${texNum(r[i], 4)}`}</Tex>. Squaring penalizes large errors heavily.</>
            )}
          </Explain>
          {r.length > 0 && <Check label="L" traced={s.loss} recomputed={recomputed} tol={tol} />}
        </>
      );
      break;
    }

    case "gradient":
    case "backward": {
      // gradient is evaluated at the parameters of the latest forward pass
      const w = frame.forwardParams?.w ?? s.w ?? 0;
      const b = frame.forwardParams?.b ?? s.b ?? 0;
      const g = n ? gradient(w, b, { x, y }) : { dw: NaN, db: NaN };
      const ri = n ? w * x[i] + b - y[i] : NaN;
      body = (
        <>
          <TexBlock>{`\\frac{\\partial L}{\\partial w} = \\frac{2}{n}\\sum_{i} (\\hat y_i - y_i)\\,x_i = ${texNum(s.dw, 5)}`}</TexBlock>
          <TexBlock>{`\\frac{\\partial L}{\\partial b} = \\frac{2}{n}\\sum_{i} (\\hat y_i - y_i) = ${texNum(s.db, 5)}`}</TexBlock>
          {n > 0 && (
            <TexBlock>{`\\text{point } ${i + 1}\\text{ contributes}\\quad \\tfrac{2}{${n}}\\,r${sub}\\,x${sub} = ${texNum((2 / n) * ri * x[i], 4)},\\qquad \\tfrac{2}{${n}}\\,r${sub} = ${texNum((2 / n) * ri, 4)}`}</TexBlock>
          )}
          <Explain>
            {torch ? (
              <>
                Autograd walked the recorded graph backwards ({locals.loss?.grad_fn ?? "loss"} → {locals.y_hat?.grad_fn ?? "linear"}) and stored the results in <code>model.weight.grad</code> and <code>model.bias.grad</code>. The formulas above are the hand derivation.
              </>
            ) : (
              <>Chain rule on the MSE. A negative ∂L/∂w means increasing w lowers the loss.</>
            )}
          </Explain>
          <Check label="∂L/∂w" traced={s.dw} recomputed={g.dw} tol={tol} />
          <Check label="∂L/∂b" traced={s.db} recomputed={g.db} tol={tol} />
        </>
      );
      break;
    }

    case "update": {
      const lr = s.lr ?? NaN;
      body = (
        <>
          <TexBlock>{`\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta L`}</TexBlock>
          <TexBlock>{`w = ${texNum(prev.w)} - ${texNum(lr, 4)}\\cdot${texNum(s.dw, 4, true)} = ${texNum(s.w)}`}</TexBlock>
          <TexBlock>{`b = ${texNum(prev.b)} - ${texNum(lr, 4)}\\cdot${texNum(s.db, 4, true)} = ${texNum(s.b)}`}</TexBlock>
          <Explain>
            {torch ? <><code>optimizer.step()</code> applies plain SGD (no momentum, no weight decay) to every parameter in place.</> : <>One full-batch step against the gradient. The dashed line in the data plot is the updated model.</>}
          </Explain>
          {prev.w != null && s.dw != null && <Check label="w" traced={s.w} recomputed={prev.w - lr * s.dw} tol={tol} />}
          {prev.b != null && s.db != null && <Check label="b" traced={s.b} recomputed={prev.b - lr * s.db} tol={tol} />}
        </>
      );
      break;
    }

    case "evaluate": {
      const yh = s.y_hat ?? [];
      const recomputed = n && s.w != null && s.b != null ? mean(x.map((xi, k) => (s.w! * xi + s.b! - y[k]) ** 2)) : NaN;
      body = (
        <>
          <TexBlock>{`L(\\theta_T) = \\frac{1}{n}\\sum_{i} (w_T x_i + b_T - y_i)^2 = ${texNum(s.loss, 6)}`}</TexBlock>
          <TexBlock>{`w_T = ${texNum(s.w, 6)},\\qquad b_T = ${texNum(s.b, 6)}`}</TexBlock>
          <Explain>
            Inside the loop, <code>loss</code> is computed <i>before</i> each update, so its last value belongs to θ<sub>T−1</sub>. This step evaluates the parameters the script actually returns
            {torch ? <>, under <code>torch.no_grad()</code> so no graph is built</> : null}.
          </Explain>
          {yh.length > 0 && <Check label="L(θ_T)" traced={s.loss} recomputed={recomputed} tol={tol} />}
        </>
      );
      break;
    }

    case "fit": {
      const ls = n ? leastSquares({ x, y }) : null;
      body = (
        <>
          <TexBlock>{`(\\hat w, \\hat b) = \\arg\\min_{w,b} \\sum_i (w x_i + b - y_i)^2`}</TexBlock>
          <TexBlock>{`\\hat w = \\frac{\\sum_i (x_i-\\bar x)(y_i-\\bar y)}{\\sum_i (x_i-\\bar x)^2},\\qquad \\hat b = \\bar y - \\hat w\\,\\bar x`}</TexBlock>
          <TexBlock>{`\\texttt{coef\\_} = ${texNum(s.w, 6)},\\qquad \\texttt{intercept\\_} = ${texNum(s.b, 6)}`}</TexBlock>
          <Explain>
            <code>.fit()</code> runs as one library call; its internals are not traced. For dense input, LinearRegression centres X and y, then calls <code>scipy.linalg.lstsq</code> on the centred X. It exposes the solution (<code>coef_</code>, <code>intercept_</code>) and the solver&apos;s <code>rank_</code> and <code>singular_</code>.
          </Explain>
          {s.singular && (
            <TexBlock>{`\\texttt{rank\\_} = ${s.rank ?? "\\text{—}"},\\qquad \\texttt{singular\\_} = ${texNum(s.singular[0], 6)} \\;\\big(= \\lVert x - \\bar x\\rVert\\big)`}</TexBlock>
          )}
          {ls && (
            <>
              <Check label="ŵ" refLabel="centred formula" traced={s.w} recomputed={ls.w} tol={1e-9} />
              <Check label="b̂" refLabel="centred formula" traced={s.b} recomputed={ls.b} tol={1e-9} />
              {s.singular && <Check label="singular_" refLabel="‖x − x̄‖" traced={s.singular[0]} recomputed={Math.sqrt(x.reduce((a, v) => a + (v - mean(x)) ** 2, 0))} tol={1e-9} />}
            </>
          )}
        </>
      );
      break;
    }

    case "predict": {
      const yh = s.y_hat?.[i];
      body = (
        <>
          <TexBlock>{`\\hat y${sub} = ${texNum(s.w)}\\cdot${texNum(x[i], 4, true)} + ${texNum(s.b, 4, true)} = ${texNum(yh)}`}</TexBlock>
          <Explain>
            <code>predict</code> evaluates <Tex>{"X\\hat w + \\hat b"}</Tex>. Click a point in the data plot to inspect another one.
          </Explain>
          {yh != null && s.w != null && s.b != null && <Check label={`ŷ${i + 1}`} traced={yh} recomputed={s.w * x[i] + s.b} tol={1e-9} />}
        </>
      );
      break;
    }

    case "check": {
      const sx = x.reduce((a, v) => a + v, 0);
      const sxx = x.reduce((a, v) => a + v * v, 0);
      const sy = y.reduce((a, v) => a + v, 0);
      const sxy = x.reduce((a, v, k) => a + v * y[k], 0);
      body = (
        <>
          <TexBlock>{`A^\\top A\\,\\theta = A^\\top y,\\qquad A = [\\,x\\;\\; \\mathbf 1\\,]`}</TexBlock>
          <TexBlock>{`\\begin{bmatrix} ${texNum(sxx, 3)} & ${texNum(sx, 3)} \\\\ ${texNum(sx, 3)} & ${n} \\end{bmatrix}\\theta = \\begin{bmatrix} ${texNum(sxy, 3)} \\\\ ${texNum(sy, 3)} \\end{bmatrix} \\;\\Rightarrow\\; \\theta = \\begin{bmatrix} ${texNum(s.theta?.[0], 6)} \\\\ ${texNum(s.theta?.[1], 6)} \\end{bmatrix}`}</TexBlock>
          <Explain>Our own NumPy solve of the normal equations, independent of scikit-learn. It agrees with coef_ and intercept_ to floating-point precision.</Explain>
          <Check label="θ₀" refLabel="scikit-learn coef_" traced={s.theta?.[0]} recomputed={s.w ?? NaN} tol={1e-9} />
          <Check label="θ₁" refLabel="scikit-learn intercept_" traced={s.theta?.[1]} recomputed={s.b ?? NaN} tol={1e-9} />
        </>
      );
      break;
    }

    default:
      body = <Explain>No equation is associated with this step.</Explain>;
  }

  return <div className="space-y-2 text-[15px]">{body}</div>;
}
