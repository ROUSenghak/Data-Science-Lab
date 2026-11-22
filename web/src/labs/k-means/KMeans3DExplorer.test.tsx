// @vitest-environment jsdom
/** The 3D K-Means figure shows the computed half-steps, and falls back to exact 2D projections without WebGL. */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { generateClusters3D, runKMeans } from "@/lib/ml/kmeans";
import type { KMeans3DProps } from "./KMeans3D";

let webgl: boolean | null = true;
let last: KMeans3DProps | null = null;

vi.mock("@/components/plot3d/support", () => ({ useWebGLSupport: () => webgl }));
vi.mock("./KMeans3D", () => ({
  default: (p: KMeans3DProps) => {
    last = p;
    return <div data-testid="kmeans-3d" />;
  },
}));
vi.mock("next/dynamic", async () => {
  const mod = await import("./KMeans3D");
  return { default: () => mod.default };
});

beforeAll(() => {
  window.matchMedia = ((q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  globalThis.ResizeObserver = class {
    constructor(private cb: ResizeObserverCallback) {}
    observe() {
      this.cb([{ contentRect: { width: 480 } } as ResizeObserverEntry], this as unknown as ResizeObserver);
    }
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
});
beforeEach(() => {
  webgl = true;
  last = null;
});
afterEach(cleanup);

describe("K-Means with three features", () => {
  it("each playback step hands the 3D scene the computed centroids and labels", async () => {
    const { KMeans3DExplorer } = await import("./KMeans3DExplorer");
    render(<KMeans3DExplorer />);
    const { X } = generateClusters3D(120, 4);
    const run = runKMeans(X, 3, { method: "random", seed: 3 });
    expect(last!.labels).toBeNull(); // initialization: nothing assigned yet
    expect(last!.centroids).toEqual(run.phases[0].centroids);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(last!.labels).toEqual(run.phases[1].labels);
    fireEvent.click(screen.getByRole("button", { name: "Next step" }));
    expect(last!.centroids).toEqual(run.phases[2].centroids); // reduced motion: no glide
    expect(last!.prevCentroids).toEqual(run.phases[1].centroids);
  });

  it("without WebGL it shows two exact projections of the same state", async () => {
    webgl = false;
    const { KMeans3DExplorer } = await import("./KMeans3DExplorer");
    render(<KMeans3DExplorer />);
    expect(screen.queryByTestId("kmeans-3d")).toBeNull();
    expect(screen.getByText("Projection onto x₁ and x₂")).toBeTruthy();
    expect(screen.getByText("Projection onto x₁ and x₃")).toBeTruthy();
  });
});
