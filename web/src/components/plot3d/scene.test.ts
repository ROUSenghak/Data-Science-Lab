import { describe, expect, it } from "vitest";
import { boxMap, clampElevation, DEFAULT_VIEW, ELEVATION_LIMITS, fitZoom, nearSides, niceTicks, orbitPosition, tickAnchor, TOP_VIEW, viewBasis, type V3 } from "./scene";

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

describe("box mapping", () => {
  const map = boxMap({ x: [-2, 6], y: [0, 4], z: [0, 50], equalXY: true, height: 1.1 });
  it("round-trips the floor and maps z to height", () => {
    const [X, Y, Z] = map.toScene(1.5, 3, 25);
    expect(Y).toBeCloseTo(0.55, 12);
    const [x, y] = map.toData(X, Z);
    expect(x).toBeCloseTo(1.5, 12);
    expect(y).toBeCloseTo(3, 12);
  });
  it("keeps equal units on both floor axes (true angles) and puts larger y further away", () => {
    expect(map.hx / map.hz).toBeCloseTo(8 / 4, 12);
    const a = map.toScene(0, 0, 0);
    const b = map.toScene(1, 1, 0);
    expect(b[0] - a[0]).toBeCloseTo(-(b[2] - a[2]), 12); // one unit of x = one unit of y, y runs along −Z
  });
  it("without equal units, both floor sides span [−1, 1]", () => {
    const m = boxMap({ x: [0, 100], y: [0, 1], z: [0, 1] });
    expect(m.hx).toBe(1);
    expect(m.hz).toBe(1);
  });
});

describe("camera", () => {
  it("the top view looks straight down with x to the right and y up, as the 2D chart", () => {
    const { right, up } = viewBasis(TOP_VIEW);
    const map = boxMap({ x: [0, 1], y: [0, 1], z: [0, 1] });
    const p0 = map.toScene(0, 0, 0);
    const px = map.toScene(1, 0, 0);
    const py = map.toScene(0, 1, 0);
    const d = (a: V3, b: V3): V3 => [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    expect(dot(d(p0, px), right)).toBeGreaterThan(0.99);
    expect(dot(d(p0, py), up)).toBeGreaterThan(0.99);
  });
  it("orbitPosition is at the requested distance and elevation", () => {
    const p = orbitPosition(DEFAULT_VIEW, 5, [0, 0.4, 0]);
    expect(Math.hypot(p[0], p[1] - 0.4, p[2])).toBeCloseTo(5, 12);
    expect(Math.asin((p[1] - 0.4) / 5)).toBeCloseTo(DEFAULT_VIEW.elevation, 12);
  });
  it("the basis is orthonormal and perpendicular to the viewing direction", () => {
    for (const v of [DEFAULT_VIEW, TOP_VIEW, { azimuth: 2, elevation: 0.3 }]) {
      const { right, up } = viewBasis(v);
      const p = orbitPosition(v, 1, [0, 0, 0]);
      expect(dot(right, right)).toBeCloseTo(1, 12);
      expect(dot(up, up)).toBeCloseTo(1, 12);
      expect(dot(right, up)).toBeCloseTo(0, 12);
      expect(dot(right, p)).toBeCloseTo(0, 12);
      expect(dot(up, p)).toBeCloseTo(0, 12);
    }
  });
  it("elevation never goes below the floor or over the top", () => {
    expect(clampElevation(-1)).toBe(ELEVATION_LIMITS[0]);
    expect(clampElevation(3)).toBe(ELEVATION_LIMITS[1]);
  });
  it("fitZoom makes every box corner land inside the canvas, padding included", () => {
    const map = boxMap({ x: [-3, 6], y: [-4, 6], z: [0, 50], equalXY: true });
    const target: V3 = [0, map.height / 2, 0];
    for (const [w, h] of [
      [468, 300],
      [340, 300],
      [900, 420],
    ]) {
      const z = fitZoom(map, DEFAULT_VIEW, target, w, h);
      const { right, up } = viewBasis(DEFAULT_VIEW);
      for (const X of [-map.hx, map.hx])
        for (const Y of [0, map.height])
          for (const Z of [-map.hz, map.hz]) {
            const c: V3 = [X - target[0], Y - target[1], Z - target[2]];
            expect(Math.abs(dot(c, right)) * z).toBeLessThanOrEqual(w / 2 - 34 + 1e-9);
            expect(Math.abs(dot(c, up)) * z).toBeLessThanOrEqual(h / 2 - 26 + 1e-9);
          }
    }
  });
});

describe("axes", () => {
  const map = boxMap({ x: [0, 10], y: [0, 10], z: [0, 1] });
  it("tick labels go on the floor edges nearest the camera", () => {
    expect(nearSides(3, -1)).toEqual({ x: 1, z: -1 });
    const s = nearSides(-2, 5);
    expect(tickAnchor(map, "x", 5, s)[2]).toBeGreaterThan(map.hz); // front edge, outside the box
    expect(tickAnchor(map, "y", 5, s)[0]).toBeLessThan(-map.hx); // left edge, outside the box
  });
  it("vertical ticks sit at their height", () => {
    expect(tickAnchor(map, "z", 0.5, { x: 1, z: 1 })[1]).toBeCloseTo(map.height / 2, 12);
  });
  it("ticks are nice numbers inside the range", () => {
    expect(niceTicks([0, 53.5], 4)).toEqual([0, 10, 20, 30, 40, 50]);
  });
});
