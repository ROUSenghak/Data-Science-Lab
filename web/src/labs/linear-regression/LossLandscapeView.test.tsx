// @vitest-environment jsdom
/**
 * Integration tests: the 3D view shows exactly the computed state of the gradient-descent explorer,
 * switching views never changes the experiment or the playback position, equation focus reaches the
 * 3D view, and the 2D view is used when WebGL is unavailable.
 *
 * jsdom has no WebGL, so the three.js renderer is replaced by a stub that records the props it gets;
 * everything else (explorer, playback, equations, the 2D chart, the view switch) is the real code.
 */
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { generateDataset, gradientDescent, mse } from "@/lib/ml/linreg";
import { plotTop } from "@/lib/ml/lossLandscape";
import type { LossLandscape3DProps } from "./LossLandscape3D";

let webgl: boolean | null = true;
let lastProps: LossLandscape3DProps | null = null;

vi.mock("@/components/plot3d/support", () => ({ useWebGLSupport: () => webgl }));
vi.mock("./LossLandscape3D", () => ({
  default: (p: LossLandscape3DProps) => {
    lastProps = p;
    return <div data-testid="surface-3d">3D</div>;
  },
}));
// next/dynamic would load the chunk asynchronously; load the (mocked) module synchronously instead.
vi.mock("next/dynamic", async () => {
  const mod = await import("./LossLandscape3D");
  return { default: () => mod.default };
});

beforeAll(() => {
  // jsdom lacks these browser APIs. Reduced motion is on, so values snap (no glides mid-test).
  window.matchMedia = ((q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  // Charts draw once they know their width.
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
  lastProps = null;
});
afterEach(cleanup);

const data = generateDataset({ n: 30, noise: 1, seed: 7 });

async function renderExplorer(detailed = false) {
  const { GradientDescentExplorer } = await import("./GradientDescentExplorer");
  render(<GradientDescentExplorer data={data} defaultDetailed={detailed} predictions={false} />);
}
const playback = () => screen.getByRole("group", { name: "Playback" });
const status = () => playback().querySelector('[aria-live="polite"]')!;
const step = (k: number) => {
  for (let i = 0; i < k; i++) fireEvent.click(screen.getByRole("button", { name: "Next step" }));
};
const choose = (label: RegExp) => fireEvent.click(within(screen.getByRole("radiogroup", { name: "Loss landscape view" })).getByRole("radio", { name: label }));

describe("2D / 3D loss landscape in the gradient-descent explorer", () => {
  it("the 3D view receives the computed frame, and switching keeps the playback position", async () => {
    await renderExplorer();
    step(4);
    expect(status().textContent).toBe("epoch 4 / 59");
    expect(screen.queryByTestId("surface-3d")).toBeNull();

    choose(/3D/);
    expect(screen.getByTestId("surface-3d")).toBeTruthy();
    expect(status().textContent).toBe("epoch 4 / 59");

    // Same run as the explorer's defaults: η = 0.08, θ₀ = (0, −1), 60 epochs.
    const steps = gradientDescent(data, { lr: 0.08, epochs: 60, w0: 0, b0: -1 });
    const p = lastProps!;
    expect(p.current).toEqual([steps[4].w, steps[4].b]);
    expect(p.trajectory).toEqual(steps.slice(0, 5).map((s) => [s.w, s.b]));
    expect(p.next).toEqual([steps[4].wNext, steps[4].bNext]);
    expect(p.currentLabel).toBe("θ₄");
    // The box ends 30% above the starting loss, so θ₀ and every later iterate are inside it.
    expect(p.zTop).toBeCloseTo(plotTop(data, p.domain, [[0, -1]]), 12);
    expect(mse(0, -1, data)).toBeLessThan(p.zTop);

    // Stepping in 3D moves the 3D state, and the equations show the same epoch.
    step(1);
    expect(lastProps!.current).toEqual([steps[5].w, steps[5].b]);
    expect(screen.getByText(/Epoch 5: the numbers of this run/)).toBeTruthy();

    choose(/2D/);
    expect(screen.queryByTestId("surface-3d")).toBeNull();
    expect(status().textContent).toBe("epoch 5 / 59");
  });

  it("changing the learning rate in 3D recomputes the same run the 2D view would show", async () => {
    await renderExplorer();
    choose(/3D/);
    const slider = screen.getAllByRole("slider")[0]; // η, on a log scale
    act(() => {
      fireEvent.change(slider, { target: { value: String(Math.log10(0.17)) } });
    });
    step(3);
    const steps = gradientDescent(data, { lr: 0.17, epochs: 60, w0: 0, b0: -1 });
    expect(lastProps!.current).toEqual([steps[3].w, steps[3].b]);
  });

  it("inside an epoch, the loss stage emphasises the height of θ (the loss is the height)", async () => {
    await renderExplorer(true);
    choose(/3D/);
    expect(lastProps!.emphasizeHeight).toBe(false); // stage 1: predict
    step(1);
    expect(status().textContent).toBe("epoch 0 · 2 loss");
    expect(lastProps!.emphasizeHeight).toBe(true);
    step(1); // gradient: the pending update is drawn
    expect(lastProps!.emphasizeHeight).toBe(false);
    expect(lastProps!.next).not.toBeNull();
    step(1); // update: θ₁ is shown, labelled as such
    expect(lastProps!.currentLabel).toBe("θ₁");
  });

  it("hovering a term in the equations sets the focus the 3D view draws", async () => {
    await renderExplorer();
    choose(/3D/);
    const term = document.querySelector('[data-hl="grad.w"]')!;
    fireEvent.pointerOver(term);
    expect(lastProps!.focus).toEqual({ kind: "grad", axis: "w" });
    fireEvent.pointerOver(document.querySelector('[data-hl="loss"]')!);
    expect(lastProps!.focus).toEqual({ kind: "loss" });
  });

  it("without WebGL the 3D option is disabled and the 2D view stays", async () => {
    webgl = false;
    await renderExplorer();
    const radio = within(screen.getByRole("radiogroup", { name: "Loss landscape view" })).getByRole("radio", { name: /3D/ }) as HTMLButtonElement;
    expect(radio.disabled).toBe(true);
    expect(radio.title).toMatch(/WebGL/);
    fireEvent.click(radio);
    expect(screen.queryByTestId("surface-3d")).toBeNull();
    expect(screen.getByLabelText(/Loss surface over slope and intercept/)).toBeTruthy();
  });
});
