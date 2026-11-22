"use client";

/**
 * The loss over parameter space, as 2D contours or as a 3D surface. Both views receive the same
 * props from the same parent state: data, window, trajectory, current point and pending update, so
 * switching never changes the experiment or the playback position. three.js is only downloaded when
 * the 3D view is first opened, and the 2D view is used when WebGL is unavailable.
 */
import dynamic from "next/dynamic";
import { useMemo, useState, type ComponentProps, type ReactNode } from "react";
import { useFocus } from "@/components/anim/Focus";
import { Plot3DBoundary } from "@/components/plot3d/Boundary";
import { useWebGLSupport } from "@/components/plot3d/support";
import { Segmented } from "@/components/ui/controls";
import { autoDomain, plotTop, type Pt } from "@/lib/ml/lossLandscape";
import { LossSurfacePlot } from "./plots";

const LossLandscape3D = dynamic(() => import("./LossLandscape3D"), {
  ssr: false,
  loading: () => <div className="flex h-full min-h-40 items-center justify-center rounded-md bg-surface-2 text-[12px] text-fg-3">Loading the 3D view…</div>,
});

export type LandscapeView = "2d" | "3d";

type SurfaceProps = ComponentProps<typeof LossSurfacePlot>;

export function LossLandscapeView({
  title = "Loss over the parameters (w, b)",
  view: controlled,
  onViewChange,
  defaultView = "2d",
  currentLabel,
  emphasizeHeight,
  note2d,
  ...plot
}: SurfaceProps & {
  title?: ReactNode;
  view?: LandscapeView;
  onViewChange?: (v: LandscapeView) => void;
  defaultView?: LandscapeView;
  /** Name of the current state in the 3D labels, e.g. "θ₁₂". */
  currentLabel?: string;
  /** Emphasise the height of the current point in 3D (it is the loss). */
  emphasizeHeight?: boolean;
  /** Caption shown under the 2D view only. */
  note2d?: ReactNode;
}) {
  const webgl = useWebGLSupport();
  const [own, setOwn] = useState<LandscapeView>(defaultView);
  const view = controlled ?? own;
  const setView = (v: LandscapeView) => {
    setOwn(v);
    onViewChange?.(v);
  };
  const { focus } = useFocus();
  const first = plot.trajectory[0];
  // One window for both views. Without a fixed domain, the 2D view's automatic window is used.
  const domain = useMemo(
    () => plot.domain ?? autoDomain(plot.data, plot.anchor, first),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [plot.domain, plot.data, plot.anchor?.[0], plot.anchor?.[1], first?.[0], first?.[1]],
  );
  // Top of the 3D box: 30% above the highest starting loss (the whole window if nothing starts).
  const starts = [first, plot.start, ...(plot.paths ?? []).map((p) => p.points[0])] as (Pt | undefined)[];
  const startKey = starts.map((p) => (p ? `${p[0]},${p[1]}` : "")).join("|");
  const zTop = useMemo(
    () => plotTop(plot.data, domain, starts),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [plot.data, domain, startKey],
  );
  const show3d = view === "3d" && webgl === true;
  const plot2d = (
    <>
      <LossSurfacePlot {...plot} domain={domain} />
      {note2d}
    </>
  );

  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[12px] font-medium text-fg-2">{title}</span>
        <Segmented
          ariaLabel="Loss landscape view"
          value={show3d ? "3d" : "2d"}
          onChange={setView}
          options={[
            { value: "2d", label: "2D contours" },
            { value: "3d", label: "3D surface", disabled: webgl === false, title: webgl === false ? "WebGL is not available in this browser" : undefined },
          ]}
        />
      </div>
      {view === "3d" && webgl === false && <p className="mb-1 text-[12px] text-warn">This browser cannot create a WebGL context, so the 2D view is shown. It displays the same data, path and values.</p>}
      {show3d ? (
        <Plot3DBoundary
          fallback={
            <>
              <p className="mb-1 text-[12px] text-warn">The 3D view could not start (WebGL error). Showing the 2D view.</p>
              {plot2d}
            </>
          }
        >
          <LossLandscape3D
            data={plot.data}
            domain={domain}
            zTop={zTop}
            equalAspect={!!plot.equalAspect}
            trajectory={plot.trajectory as Pt[]}
            current={plot.current as Pt | null | undefined}
            currentLabel={currentLabel}
            next={plot.next as Pt | null | undefined}
            components={plot.components}
            levelThrough={plot.levelThrough}
            paths={plot.paths}
            start={plot.start as Pt | undefined}
            onStartChange={plot.onStartChange}
            focus={focus}
            emphasizeHeight={emphasizeHeight}
            height={plot.height ?? 300}
          />
        </Plot3DBoundary>
      ) : (
        plot2d
      )}
    </div>
  );
}
