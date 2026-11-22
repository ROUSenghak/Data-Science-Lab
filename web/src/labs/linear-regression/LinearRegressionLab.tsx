"use client";

import { useMemo, useState } from "react";
import { Tex } from "@/components/math/Tex";
import { Slider, Toggle } from "@/components/ui/controls";
import { Callout, ExecutionBadge, SectionHeading } from "@/components/ui/primitives";
import { fmt } from "@/lib/format";
import { centerX, generateDataset, leastSquares, stability, type DataConfig } from "@/lib/ml/linreg";
import { FitExplorer, GradientDescentExplorer, ResidualPlot } from "./explorers";
import { GDComparison } from "./GDComparison";
import { DataFitPlot } from "./plots";

export function LinearRegressionLab() {
  const [cfg, setCfg] = useState<DataConfig>({ n: 30, noise: 1, seed: 7 });
  const [centered, setCentered] = useState(false);
  const data = useMemo(() => {
    const d = generateDataset(cfg);
    return centered ? centerX(d) : d;
  }, [cfg, centered]);
  const ls = useMemo(() => leastSquares(data), [data]);
  const clean = useMemo(() => leastSquares(generateDataset({ ...cfg, outlier: false })), [cfg]);
  const stab = useMemo(() => stability(data), [data]);
  const set = (patch: Partial<DataConfig>) => setCfg((c) => ({ ...c, ...patch }));

  return (
    <div className="grid gap-8 lg:grid-cols-[240px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-20 lg:self-start">
        <div className="space-y-4 rounded-lg border border-line bg-surface p-4">
          <div>
            <h2 className="text-sm font-semibold">Dataset</h2>
            <p className="mt-0.5 text-[12px] leading-snug text-fg-3">
              <Tex>{"x_i\\sim U(0,4)"}</Tex>
              <br />
              <Tex>{"y_i = 2x_i + 1 + \\varepsilon_i,\\; \\varepsilon_i\\sim\\mathcal N(0,\\sigma^2)"}</Tex>
            </p>
          </div>
          <Slider label="samples n" value={cfg.n} min={5} max={120} step={1} onChange={(n) => set({ n })} />
          <Slider label={<Tex>{"\\text{noise } \\sigma"}</Tex>} value={cfg.noise} min={0} max={4} step={0.1} onChange={(noise) => set({ noise })} format={(v) => fmt(v, 1)} />
          <Slider label="random seed" value={cfg.seed} min={0} max={99} step={1} onChange={(seed) => set({ seed })} />
          <div className="space-y-2 border-t border-line pt-3">
            <div className="text-[12px] font-medium text-fg-2">Failure cases</div>
            <Toggle label="Add one outlier" checked={!!cfg.outlier} onChange={(outlier) => set({ outlier })} />
            <Toggle label="Curved ground truth" hint="y = 0.4x² + 0.4x + 1 + ε" checked={!!cfg.nonlinear} onChange={(nonlinear) => set({ nonlinear })} />
            <Toggle label="Center the feature" hint="x ← x − x̄" checked={centered} onChange={setCentered} />
          </div>
          <dl className="grid grid-cols-2 gap-x-2 gap-y-1 border-t border-line pt-3 font-mono text-[11.5px] tabular-nums text-fg-2">
            <dt className="text-fg-3">w*</dt>
            <dd>{fmt(ls.w, 4)}</dd>
            <dt className="text-fg-3">b*</dt>
            <dd>{fmt(ls.b, 4)}</dd>
            <dt className="text-fg-3">L*</dt>
            <dd>{fmt(ls.loss, 4)}</dd>
            <dt className="text-fg-3">2/λmax</dt>
            <dd>{fmt(stab.criticalLR, 4)}</dd>
            <dt className="text-fg-3">κ</dt>
            <dd>{fmt(stab.condition, 1)}</dd>
          </dl>
          <ExecutionBadge kind="browser" />
        </div>
      </aside>

      <div className="min-w-0">
        <SectionHeading id="fit-by-hand" kicker="1" first>
          Fit a line by hand
        </SectionHeading>
        <p className="mb-4 max-w-2xl text-[15px] text-fg-2">
          Move <Tex>w</Tex> and <Tex>b</Tex>. The orange segments are residuals; the loss is their mean square. The right panel shows where your guess sits on the loss surface.
        </p>
        <FitExplorer data={data} />

        <SectionHeading id="gradient-descent" kicker="2">
          Gradient descent
        </SectionHeading>
        <p className="mb-4 max-w-2xl text-[15px] text-fg-2">
          Each epoch computes the exact full-batch gradient and moves against it. Switch to <i>inside an epoch</i> to see the four computations one at a time. Try η just below and just above <span className="font-mono">{fmt(stab.criticalLR, 3)}</span>, drag θ₀ on the loss surface, then center the feature and compare the path.
        </p>
        <GradientDescentExplorer data={data} />

        <SectionHeading id="compare" kicker="3">
          Compare two learning rates
        </SectionHeading>
        <p className="mb-4 max-w-2xl text-[15px] text-fg-2">
          Same data, same starting point, two values of η, replayed in sync. Predict which one arrives first, then watch the paths and the excess loss.
        </p>
        <GDComparison data={data} />

        <SectionHeading id="failure-cases" kicker="4">
          Failure cases
        </SectionHeading>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="min-w-0">
            <DataFitPlot
              data={data}
              lines={[
                { w: ls.w, b: ls.b, label: "least-squares fit", color: "var(--viz-model)" },
                ...(cfg.outlier && !centered ? [{ w: clean.w, b: clean.b, label: "fit without the outlier", color: "var(--viz-alt)", dashed: true }] : []),
              ]}
              height={260}
            />
          </div>
          <div className="min-w-0">
            <ResidualPlot data={data} height={260} />
          </div>
        </div>
        <div className="mt-4 space-y-3">
          {cfg.outlier ? (
            <Callout tone="warn" title="One outlier moves the whole line">
              Squaring makes a residual of 10 cost 100× a residual of 1. The slope changed from {fmt(clean.w, 3)} to {fmt(ls.w, 3)} because of a single point. Robust losses (Huber, absolute error) grow linearly instead.
            </Callout>
          ) : null}
          {cfg.nonlinear ? (
            <Callout tone="warn" title="Model misspecification">
              The fit is still the best straight line, but the residuals form a U-shape instead of random scatter. Least squares cannot detect this for you; the residual plot can.
            </Callout>
          ) : null}
          {centered ? (
            <Callout tone="note" title="Centering decouples w and b">
              With <Tex>{"\\bar x = 0"}</Tex> the Hessian is diagonal, so the loss ellipses align with the axes and κ falls to {fmt(stab.condition, 1)}. Gradient descent converges in far fewer steps. The intercept now means &ldquo;prediction at the mean x&rdquo;.
            </Callout>
          ) : null}
          {!cfg.outlier && !cfg.nonlinear && !centered && (
            <p className="text-[14px] text-fg-3">Turn on a failure case in the Dataset panel to see how the fit and the residuals react.</p>
          )}
        </div>
      </div>
    </div>
  );
}
