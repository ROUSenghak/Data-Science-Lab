"use client";

import type { ReactNode } from "react";
import { Tex, TexBlock } from "@/components/math/Tex";
import { fmt, texNum, texSq } from "@/lib/format";
import { argmin, distanceMatrix, objective, updateCentroids, type Vec2 } from "@/lib/ml/kmeans";
import { distanceTex } from "@/labs/k-means/explorers";
import { ClusterTag } from "@/labs/k-means/plots";
import { Check, Explain } from "../Inspector";
import type { KFrame } from "./trace";
import { iterationOf } from "./trace";

const TOL = 1e-9; // float64

/** Agreement of a whole traced array with a browser recomputation (max absolute difference). */
function ArrayCheck({ label, traced, recomputed, tol = TOL, refLabel = "browser recomputation" }: { label: ReactNode; traced: number[] | null | undefined; recomputed: number[]; tol?: number; refLabel?: string }) {
  if (!traced || traced.length !== recomputed.length || !traced.every(Number.isFinite)) {
    return <div className="text-[12px] text-fg-3">Cross-check unavailable.</div>;
  }
  let diff = 0;
  let scale = 1;
  traced.forEach((v, i) => {
    diff = Math.max(diff, Math.abs(v - recomputed[i]));
    scale = Math.max(scale, Math.abs(recomputed[i]));
  });
  const ok = diff <= tol * scale;
  return (
    <div className="flex items-start gap-1.5 text-[12px] text-fg-2" title={`Passes if max |Δ| ≤ ${tol} · max(1, |value|) over ${traced.length} values.`}>
      <span className={ok ? "text-ok" : "text-bad"} aria-hidden>
        {ok ? "✓" : "✗"}
      </span>
      <span>
        {label}: all {traced.length} values match the {refLabel}
        <span className="text-fg-3">
          {" "}· max |Δ| = {diff === 0 ? "0" : diff.toExponential(1)} {ok ? "≤" : ">"} tol {tol.toExponential(0)}
        </span>
        {!ok && <b className="text-bad"> mismatch</b>}
      </span>
    </div>
  );
}

/** Exact agreement of two integer arrays (labels). */
function LabelCheck({ label, a, b }: { label: ReactNode; a: number[] | null | undefined; b: number[] }) {
  if (!a) return null;
  const bad = a.filter((v, i) => v !== b[i]).length;
  return (
    <div className="flex items-start gap-1.5 text-[12px] text-fg-2">
      <span className={bad ? "text-bad" : "text-ok"} aria-hidden>
        {bad ? "✗" : "✓"}
      </span>
      <span>
        {label}: {bad ? <b className="text-bad">{bad} of {a.length} differ</b> : <>identical for all {a.length} points</>}
      </span>
    </div>
  );
}

function CentroidTable({ C, prev, counts, sums, shift, highlight }: { C: Vec2[]; prev?: Vec2[] | null; counts?: number[] | null; sums?: Vec2[] | null; shift?: number[] | null; highlight?: number | null }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left font-mono text-[12px] tabular-nums">
        <thead className="text-[11px] text-fg-3">
          <tr>
            <th className="py-0.5 pr-3 font-sans font-medium">j</th>
            {counts && <th className="py-0.5 pr-3 font-sans font-medium">|Cⱼ|</th>}
            {sums && <th className="py-0.5 pr-3 font-sans font-medium">Σ xᵢ</th>}
            {prev && <th className="py-0.5 pr-3 font-sans font-medium">before</th>}
            <th className="py-0.5 pr-3 font-sans font-medium">μⱼ</th>
            {shift && <th className="py-0.5 font-sans font-medium">moved</th>}
          </tr>
        </thead>
        <tbody className="divide-y divide-line text-fg-2">
          {C.map((c, j) => (
            <tr key={j} className={j === highlight ? "font-semibold text-fg" : ""}>
              <td className="py-0.5 pr-3 font-sans">
                <ClusterTag j={j} />
              </td>
              {counts && <td className="py-0.5 pr-3">{counts[j]}</td>}
              {sums && (
                <td className="py-0.5 pr-3">
                  ({fmt(sums[j][0], 3)}, {fmt(sums[j][1], 3)})
                </td>
              )}
              {prev && (
                <td className="py-0.5 pr-3">
                  ({fmt(prev[j]?.[0], 3)}, {fmt(prev[j]?.[1], 3)})
                </td>
              )}
              <td className="py-0.5 pr-3">
                ({fmt(c[0], 4)}, {fmt(c[1], 4)})
              </td>
              {shift && <td className="py-0.5">{counts?.[j] === 0 ? <span className="font-sans text-warn">empty: kept</span> : fmt(shift[j], 4)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DistanceRow({ x, i, C, d, title }: { x: Vec2; i: number; C: Vec2[]; d: number[]; title: string }) {
  const best = argmin(d);
  return (
    <>
      <TexBlock>{distanceTex(x, C[best], i, best, d[best])}</TexBlock>
      <table className="w-full text-left font-mono text-[12px] tabular-nums">
        <thead className="text-[11px] text-fg-3">
          <tr>
            <th className="py-0.5 pr-3 font-sans font-medium">centroid</th>
            <th className="py-0.5 pr-3 font-sans font-medium">μⱼ</th>
            <th className="py-0.5 font-sans font-medium">{title}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {d.map((v, j) => (
            <tr key={j} className={j === best ? "font-semibold text-fg" : "text-fg-2"}>
              <td className="py-0.5 pr-3 font-sans">
                <ClusterTag j={j} />
              </td>
              <td className="py-0.5 pr-3">
                ({fmt(C[j][0], 3)}, {fmt(C[j][1], 3)})
              </td>
              <td className="py-0.5">
                {fmt(v, 5)} {j === best && <span className="font-sans font-normal text-fg-3">← nearest (argmin)</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

const INIT_TEXT = [
  <>
    <code>rng_init.choice(n_samples, size=k, replace=False)</code> picks K distinct rows of X uniformly at random (Forgy initialization).
  </>,
  <>
    k-means++: the first centroid is a uniformly random point; each next one is point <Tex>x</Tex> with probability <Tex>{"D(x)^2/\\sum_{x'} D(x')^2"}</Tex>, where <Tex>D</Tex> is the distance to the nearest centroid chosen so far. Far-away points are likely, so centroids spread out.
  </>,
  <>Uniform positions in the bounding box of X. Some may land far from all data; their clusters can end up empty.</>,
];

export function KMeansInspector({ frame, framework, point, params }: { frame: KFrame; framework: string; point: number; params: Record<string, number> }) {
  const { state: s, prev, snap } = frame;
  const X = s.X ?? [];
  const n = X.length;
  const i = Math.min(point, Math.max(0, n - 1));
  const t = iterationOf(frame);
  const C = s.centroids ?? [];
  let body: ReactNode;

  switch (snap.step) {
    case "data":
      body = (
        <>
          <TexBlock>{`x_i = m_{g_i} + \\varepsilon_i,\\qquad g_i \\sim \\mathcal U\\{1,\\dots,${params.n_blobs}\\},\\quad \\varepsilon_i \\sim \\mathcal N(0, ${texSq(params.cluster_std, 1)} I)`}</TexBlock>
          {n > 0 && <TexBlock>{`X \\in \\mathbb R^{${n}\\times 2},\\qquad x_{${i + 1}} = (${texNum(X[i][0], 3)},\\ ${texNum(X[i][1], 3)})`}</TexBlock>}
          <Explain>
            Blob centres <Tex>{"m_b"}</Tex> are drawn uniformly in <Tex>{"[-5,5]^2"}</Tex>. K-Means never sees <Tex>g_i</Tex>: clustering is unsupervised. The same seed gives the same points in the NumPy and scikit-learn versions.
          </Explain>
        </>
      );
      break;

    case "init": {
      const C0 = s.centroids_init ?? C;
      const isRow = (c: Vec2) => X.some((x) => x[0] === c[0] && x[1] === c[1]);
      body = (
        <>
          <TexBlock>{`\\mu^{(0)}_1,\\dots,\\mu^{(0)}_{${C0.length}} \\in \\mathbb R^2`}</TexBlock>
          <CentroidTable C={C0} />
          <Explain>{INIT_TEXT[s.init_method ?? 0]}</Explain>
          {s.init_method !== 2 && (
            <LabelCheck label="every μ⁽⁰⁾ⱼ is a row of X" a={C0.map((c) => (isRow(c) ? 1 : 0))} b={C0.map(() => 1)} />
          )}
        </>
      );
      break;
    }

    case "distances": {
      const D = s.sq_dist ?? [];
      body = (
        <>
          <TexBlock>{`d_{ij} = \\lVert x_i - \\mu^{(${(t ?? 1) - 1})}_j\\rVert^2 = (x_{i1}-\\mu_{j1})^2 + (x_{i2}-\\mu_{j2})^2,\\qquad D \\in \\mathbb R^{${n}\\times ${C.length}}`}</TexBlock>
          {D[i] && <DistanceRow x={X[i]} i={i} C={C} d={D[i]} title={`d_${i + 1}ⱼ (traced)`} />}
          <Explain>
            Broadcasting: <code>X[:, None, :] - centroids[None, :, :]</code> has shape (n, K, 2); squaring and summing over the last axis gives all n·K squared distances at once. Click a point in the plot to inspect another row.
          </Explain>
          <ArrayCheck label="D" traced={D.flat()} recomputed={distanceMatrix(X, C).flat()} />
        </>
      );
      break;
    }

    case "assign": {
      const D = s.sq_dist ?? [];
      const labels = s.labels ?? [];
      const before = s.prev_labels?.[i];
      const now = labels[i];
      const recomputedJ = D.reduce((acc, row) => acc + Math.min(...row), 0);
      body = (
        <>
          <TexBlock>{`c^{(${t})}_i = \\arg\\min_j d_{ij},\\qquad c^{(${t})}_{${i + 1}} = ${now != null ? now + 1 : "\\text{—}"}`}</TexBlock>
          {before != null && before >= 0 && before !== now && D[i] && (
            <Explain>
              Point {i + 1} moves from cluster {before + 1} to {now + 1}: <Tex>{`d_{${i + 1},${now + 1}} = ${texNum(D[i][now], 4)} < d_{${i + 1},${before + 1}} = ${texNum(D[i][before], 4)}`}</Tex>. Its old centroid moved away in the last update.
            </Explain>
          )}
          <TexBlock>{`J\\big(c^{(${t})}, \\mu^{(${(t ?? 1) - 1})}\\big) = \\sum_{i=1}^{${n}} d_{i\\,c_i} = ${texNum(s.J, 5)}`}</TexBlock>
          <Explain>
            <b className="font-medium text-fg">Intermediate state:</b> labels are new, centroids are still μ<sup>({(t ?? 1) - 1})</sup>.{" "}
            {t === 1 ? (
              <>In the first iteration every point is newly assigned.</>
            ) : (
              <>
                <code>n_changed = {s.n_changed}</code> point{s.n_changed === 1 ? "" : "s"} changed cluster{s.n_changed === 0 ? ": the loop will stop after this iteration" : " (ringed in the plot)"}.
              </>
            )}{" "}
            Ties go to the lowest index (<code>np.argmin</code>).
          </Explain>
          <LabelCheck label="c = argmin of each row of the traced D" a={labels} b={D.map(argmin)} />
          <Check label="J" traced={s.J} recomputed={recomputedJ} tol={TOL} />
        </>
      );
      break;
    }

    case "update": {
      const labels = s.labels ?? [];
      const old = prev.centroids ?? [];
      const re = updateCentroids(X, labels, old);
      const c = labels[i];
      body = (
        <>
          <TexBlock>{`\\mu^{(${t})}_j = \\frac{1}{|C_j|}\\sum_{i\\in C_j} x_i = \\frac{\\texttt{sums}_j}{\\texttt{counts}_j},\\qquad C_j = \\{\\, i : c_i = j \\,\\}`}</TexBlock>
          {c != null && s.counts && s.sums && s.counts[c] > 0 && (
            <TexBlock>{`\\mu^{(${t})}_{${c + 1}} = \\tfrac{1}{${s.counts[c]}}\\,(${texNum(s.sums[c][0], 3)},\\ ${texNum(s.sums[c][1], 3)}) = (${texNum(C[c]?.[0], 4)},\\ ${texNum(C[c]?.[1], 4)})`}</TexBlock>
          )}
          <CentroidTable C={C} prev={old} counts={s.counts} sums={s.sums} shift={s.shift} highlight={c} />
          <Explain>
            <code>R[i, j] = (labels[i] == j)</code> is the membership matrix <Tex>{"r_{ij}"}</Tex>; <code>R.T @ X</code> sums the points of each cluster. The mean is the point with the smallest sum of squared distances to the cluster, so J cannot increase.
            {s.counts?.includes(0) && (
              <>
                {" "}
                <b className="font-medium text-warn">Empty cluster:</b> with no points the mean is undefined; this implementation keeps the previous centroid (scikit-learn would relocate it).
              </>
            )}
          </Explain>
          <ArrayCheck label="μ⁽ᵗ⁾" traced={C.flat()} recomputed={re.centroids.flat()} />
          <ArrayCheck label="cluster sizes" traced={s.counts} recomputed={re.counts} tol={0} />
        </>
      );
      break;
    }

    case "objective": {
      const labels = s.labels ?? [];
      const Jassign = prev.J;
      body = (
        <>
          <TexBlock>{`J\\big(c^{(${t})}, \\mu^{(${t})}\\big) = \\sum_{i=1}^{${n}} \\lVert x_i - \\mu^{(${t})}_{c_i}\\rVert^2 = ${texNum(s.J, 5)}`}</TexBlock>
          {Jassign != null && s.J != null && <TexBlock>{`\\Delta J_{\\text{update}} = ${texNum(s.J, 4)} - ${texNum(Jassign, 4)} = ${texNum(s.J - Jassign, 4)} \\;\\le 0`}</TexBlock>}
          <Explain>
            <b className="font-medium text-fg">Iteration {t} complete.</b> This is the within-cluster sum of squares (scikit-learn&apos;s <code>inertia_</code>) of the current labels and centroids.{" "}
            {s.n_changed === 0 ? <>No label changed in this iteration, so <code>break</code> runs next.</> : <>{s.n_changed} label{s.n_changed === 1 ? "" : "s"} changed, so the loop continues.</>}
          </Explain>
          <Check label="J" traced={s.J} recomputed={objective(X, C, labels)} tol={TOL} />
          {Jassign != null && s.J != null && (
            <div className="flex items-start gap-1.5 text-[12px] text-fg-2">
              <span className={s.J <= Jassign * (1 + 1e-12) ? "text-ok" : "text-bad"} aria-hidden>
                {s.J <= Jassign * (1 + 1e-12) ? "✓" : "✗"}
              </span>
              <span>J did not increase in this half-step (mathematical guarantee)</span>
            </div>
          )}
        </>
      );
      break;
    }

    case "evaluate": {
      const labels = s.labels ?? [];
      const iters = s.iteration != null ? s.iteration + 1 : null;
      const converged = s.n_changed === 0;
      body = (
        <>
          <TexBlock>{`c_i = \\arg\\min_j \\lVert x_i - \\mu_j\\rVert^2,\\qquad \\text{inertia} = \\sum_i \\lVert x_i - \\mu_{c_i}\\rVert^2 = ${texNum(s.inertia, 6)}`}</TexBlock>
          <Explain>
            {converged ? (
              <>Converged after {iters} iterations: the labels are unchanged and the inertia equals the last J.</>
            ) : (
              <>Stopped at <code>max_iter = {iters}</code> without convergence: the labels are recomputed for the returned centroids, which can only lower J.</>
            )}
          </Explain>
          <LabelCheck label="labels = nearest returned centroid" a={labels} b={distanceMatrix(X, C).map(argmin)} />
          <Check label="inertia" traced={s.inertia} recomputed={objective(X, C, labels)} tol={TOL} />
        </>
      );
      break;
    }

    /* ---------------- scikit-learn */

    case "model":
      body = (
        <>
          <TexBlock>{`\\texttt{KMeans}(\\texttt{n\\_clusters}=${params.k},\\ \\texttt{init}=\\text{"${params.init_method === 0 ? "random" : "k-means++"}"},\\ \\texttt{n\\_init}=${params.n_init},\\ \\texttt{max\\_iter}=${params.max_iter},\\ \\texttt{tol}=${params.tol})`}</TexBlock>
          <Explain>
            Constructing the estimator stores hyper-parameters only. <code>n_init</code> independent initializations are run inside <code>fit</code>, and the one with the lowest inertia is kept. With <code>tol</code> &gt; 0, a run also stops when the summed squared centroid shift falls below <Tex>{"\\text{tol}\\cdot\\overline{\\operatorname{Var}}(X)"}</Tex>.
          </Explain>
        </>
      );
      break;

    case "fit": {
      const labels = s.labels ?? [];
      const means = updateCentroids(X, labels, C);
      let gap = 0;
      means.centroids.forEach((m, j) => (gap = Math.max(gap, Math.hypot(m[0] - C[j][0], m[1] - C[j][1]))));
      body = (
        <>
          <TexBlock>{`\\texttt{inertia\\_} = ${texNum(s.inertia, 6)},\\qquad \\texttt{n\\_iter\\_} = ${s.n_iter ?? "\\text{—}"}`}</TexBlock>
          <CentroidTable C={C} counts={means.counts} />
          <Explain>
            <code>.fit()</code> runs as one library call; its iterations are not traced. It exposes <code>cluster_centers_</code>, <code>labels_</code>, <code>inertia_</code> and <code>n_iter_</code>
            {params.n_init > 1 && <> of the best of {params.n_init} runs</>}. Compare with the NumPy version for every intermediate step.
          </Explain>
          <Explain>
            Largest distance between a <code>cluster_centers_</code> row and the mean of its <code>labels_</code>: <span className="font-mono">{fmt(gap, 3)}</span>.{" "}
            {gap < 1e-9 ? "Zero up to rounding: the run stopped because no label changed." : "Not zero: the run stopped on the tol rule or max_iter, and labels_ were re-assigned to the final centres."}
          </Explain>
        </>
      );
      break;
    }

    case "transform": {
      const dist = s.dist ?? [];
      body = (
        <>
          <TexBlock>{`\\texttt{transform}(X)_{ij} = \\lVert x_i - \\mu_j\\rVert \\quad\\text{(Euclidean, not squared)}`}</TexBlock>
          {dist[i] && <DistanceRow x={X[i]} i={i} C={C} d={dist[i].map((v) => v * v)} title={`dist² for point ${i + 1}`} />}
          <Explain>
            Squaring is monotone, so the nearest centre is the same for <Tex>{"\\lVert\\cdot\\rVert"}</Tex> and <Tex>{"\\lVert\\cdot\\rVert^2"}</Tex>. <code>labels_</code>[{i}] = {s.labels?.[i] != null ? s.labels[i] : "—"} (cluster {s.labels?.[i] != null ? s.labels[i] + 1 : "—"}).
          </Explain>
          <ArrayCheck label="transform(X)²" traced={dist.flat().map((v) => v * v)} recomputed={distanceMatrix(X, C).flat()} tol={1e-9} />
        </>
      );
      break;
    }

    case "check": {
      const D = s.sq_dist ?? [];
      body = (
        <>
          <TexBlock>{`J = \\sum_i \\min_j \\lVert x_i - \\mu_j \\rVert^2 = ${texNum(s.check_J, 6)} \\quad\\text{vs}\\quad \\texttt{inertia\\_} = ${texNum(s.inertia, 6)}`}</TexBlock>
          <Explain>Our own NumPy computation from the fitted <code>cluster_centers_</code>, independent of scikit-learn&apos;s internals.</Explain>
          <LabelCheck label="our argmin vs labels_" a={s.check_labels} b={s.labels ?? []} />
          <Check label="our J" refLabel="scikit-learn inertia_" traced={s.check_J} recomputed={s.inertia ?? NaN} tol={1e-9} />
          <ArrayCheck label="our D" refLabel="squared transform(X)" traced={D.flat()} recomputed={(s.dist ?? []).flat().map((v) => v * v)} tol={1e-9} />
        </>
      );
      break;
    }

    default:
      body = <Explain>No equation is associated with this step.</Explain>;
  }

  return (
    <div className="space-y-2 text-[15px]">
      {body}
      {framework === "numpy" && t != null && <div className="text-[11px] text-fg-3">Superscript (t) is the iteration; Python&apos;s loop variable is iteration = t − 1 = {t - 1}.</div>}
    </div>
  );
}
