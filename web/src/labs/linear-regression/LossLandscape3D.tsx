"use client";

/**
 * The MSE loss L(w, b) as a surface over parameter space, with gradient descent on it.
 * Lazy-loaded: three.js is only fetched when a learner opens the 3D view.
 *
 * Every drawn number is exact: the surface nodes, contour heights, iterates, the current marker and
 * the hover read-out all call mse(). Two things are drawing choices, and both are stated on the page:
 *  - between grid nodes the surface is made of flat triangles (error bound shown under the plot);
 *  - θ moves along a straight segment in (w, b), drawn on the surface; points between iterates are
 *    drawing positions, not iterations.
 * The box ends at `zTop` (see plotTop): the surface and paths are cut there, never squashed.
 */
import { useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Legend, type LegendItem } from "@/components/plot/Chart";
import Plot3D, { type SceneLabel } from "@/components/plot3d/Plot3D";
import { Arrow3, Dots, FatLines, mixColor, toColor } from "@/components/plot3d/primitives";
import { boxMap, type BoxMap, type BoxSpec, type V3 } from "@/components/plot3d/scene";
import type { ThemeColors } from "@/components/plot3d/support";
import { focusedAxis, type Focus } from "@/lib/anim/focus";
import { fmt } from "@/lib/format";
import { lossEllipse, mse, stability, type Dataset } from "@/lib/ml/linreg";
import { clipAbove, clipPolyline, contourLevels, densify, inDomain, interpolationErrorBound, levelShade, LEVEL_COUNT, lift as liftExact, surfaceGrid, type ParamDomain, type Pt } from "@/lib/ml/lossLandscape";

const RES = 72;
/** Opacity of one level-set layer in the 2D view; the surface uses the same compositing. */
const LAYER = 0.07;
const HOVER_HINT = "hover the surface to read L(w, b)";

export interface PathOnSurface {
  points: Pt[];
  color: string;
  label: string;
  current?: Pt | null;
}

export interface LossLandscape3DProps {
  data: Dataset;
  domain: ParamDomain;
  /** Top of the plotted loss range. */
  zTop: number;
  equalAspect: boolean;
  trajectory: Pt[];
  current?: Pt | null;
  /** Name of the current state, e.g. "θ₁₂". */
  currentLabel?: string;
  next?: Pt | null;
  components?: boolean;
  levelThrough?: boolean;
  paths?: PathOnSurface[];
  start?: Pt;
  onStartChange?: (w: number, b: number) => void;
  focus: Focus | null;
  /** Emphasise the height of the current point (the loss stage, or the loss term is hovered). */
  emphasizeHeight?: boolean;
  height: number;
}

const ok = (p: readonly number[] | null | undefined): p is Pt => !!p && p.every(Number.isFinite);
const offsetUp = (p: V3, d: number): V3 => [p[0], p[1] + d, p[2]];

export default function LossLandscape3D(props: LossLandscape3DProps) {
  const { data, domain, zTop, equalAspect, height } = props;
  const grid = useMemo(() => surfaceGrid(data, domain, RES), [data, domain]);
  const box = useMemo<BoxSpec>(() => ({ x: domain.w, y: domain.b, z: [0, zTop], equalXY: equalAspect, height: 1.1 }), [domain, zTop, equalAspect]);
  // Plot3D builds the same mapping from the same box; labels need it outside the canvas.
  const toScene = useMemo(() => boxMap(box).toScene, [box]);
  const axisTitles = useMemo(() => ({ x: <>slope w</>, y: <>intercept b</>, z: <>L(w, b)</> }), []);
  const readout = useRef<HTMLDivElement>(null);
  const lambdaMax = useMemo(() => stability(data).lambdaMax, [data]);
  const errBound = interpolationErrorBound(domain, RES, lambdaMax);
  const opt = grid.opt;

  const inWindow = ok(props.current) && inDomain(props.current, domain);
  const curL = inWindow ? mse(props.current![0], props.current![1], data) : null;
  const cur = inWindow && curL != null && curL <= zTop ? props.current! : null;
  const name = props.currentLabel ?? "θ";

  const labels: SceneLabel[] = [{ id: "opt", at: offsetUp(toScene(opt.w, opt.b, opt.loss), 0.07), content: "θ*", className: "-ml-3 text-[11.5px] font-medium text-[var(--viz-alt)]" }];
  if (cur && curL != null)
    labels.push({
      id: "cur",
      at: offsetUp(toScene(cur[0], cur[1], curL), 0.075),
      content: props.emphasizeHeight ? `${name}: L = ${fmt(curL, 4)}` : name,
      className: `ml-4 text-[11.5px] font-medium ${props.emphasizeHeight ? "text-[var(--viz-residual)]" : "text-fg"}`,
    });

  const legend: LegendItem[] = [
    { label: "L(w, b)", color: "color-mix(in srgb, var(--viz-model) 45%, transparent)", kind: "area" },
    { label: "θ*", color: "var(--viz-alt)", kind: "dot" },
  ];
  if (props.trajectory.length) legend.push({ label: "path of (w, b)", color: "var(--fg-2)", kind: "line" });
  if (props.current) legend.push({ label: "current θ at height L(θ)", color: "var(--fg)", kind: "dot" });
  if (props.levelThrough && props.current) legend.push({ label: "level set through θ", color: "var(--viz-model)", kind: "line" });
  if (props.next) legend.push({ label: "next update −η∇L", color: "var(--viz-residual)", kind: "arrow" });
  for (const p of props.paths ?? []) legend.push({ label: p.label, color: p.color, kind: "line" });

  return (
    <div className="min-w-0">
      <Plot3D
        box={box}
        axisTitles={axisTitles}
        height={height}
        ariaLabel={`Loss surface over slope and intercept.${props.current && curL != null ? ` ${name}: w = ${fmt(props.current[0], 3)}, b = ${fmt(props.current[1], 3)}, loss ${fmt(curL, 4)}.` : ""} Minimum at w = ${fmt(opt.w, 3)}, b = ${fmt(opt.b, 3)}, loss ${fmt(opt.loss, 4)}.`}
        labels={labels}
        overlay={
          <div ref={readout} className="pointer-events-none absolute left-2 top-1.5 font-mono text-[11px] tabular-nums text-fg-3" aria-hidden>
            {HOVER_HINT}
          </div>
        }
      >
        {(map, colors) => <Scene {...props} map={map} colors={colors} grid={grid} readoutRef={readout} cur={cur} curL={curL} />}
      </Plot3D>
      <div className="mt-1.5">
        <Legend items={legend} />
      </div>
      <p className="mt-1 text-[11.5px] leading-snug text-fg-3" aria-live="polite">
        {props.current && ok(props.current) && curL != null ? (
          <span className="font-mono tabular-nums text-fg-2">
            {name} = ({fmt(props.current[0], 4)}, {fmt(props.current[1], 4)}), L = {fmt(curL, 4)}
            {!cur && <span className="font-sans text-warn"> (above the plotted range)</span>}
            {" · "}
          </span>
        ) : props.current ? (
          <span className="text-bad">θ has left the plotted window. </span>
        ) : null}
        <span className="font-mono tabular-nums">
          θ* = ({fmt(opt.w, 3)}, {fmt(opt.b, 3)}), L* = {fmt(opt.loss, 4)}
        </span>
        <br />
        Exact heights{zTop < grid.max * (1 - 1e-9) && <>, cut at L = {fmt(zTop, 1)} (30% above the start)</>}; the mesh between grid nodes is within {fmt(errBound, 2)} of L.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ scene */

type SceneProps = LossLandscape3DProps & {
  map: BoxMap;
  colors: ThemeColors;
  grid: ReturnType<typeof surfaceGrid>;
  readoutRef: React.RefObject<HTMLDivElement | null>;
  cur: Pt | null;
  curL: number | null;
};

function Scene({ data, domain, zTop, map, colors: C, grid, readoutRef, trajectory, next, components, levelThrough, paths, start, onStartChange, focus, emphasizeHeight, cur, curL }: SceneProps) {
  const invalidate = useThree((st) => st.invalidate);
  const lift = (p: Pt, z = mse(p[0], p[1], data)): V3 => map.toScene(p[0], p[1], z);
  const floor = (p: Pt): V3 => {
    const s = map.toScene(p[0], p[1], map.spec.z[0]);
    return [s[0], 0.002, s[2]];
  };
  /** A path in (w, b) drawn on the surface: inside the window, below the box top. */
  const onSurface = (pts: Pt[], perSegment: number): V3[][] =>
    clipPolyline(densify(pts, perSegment), domain)
      .flatMap((part) => clipAbove(liftExact(part, data), zTop))
      .map((part) => part.map(([w, b, L]) => map.toScene(w, b, L)));
  const below = (p: Pt) => inDomain(p, domain) && mse(p[0], p[1], data) <= zTop;
  const axisFocus = focusedAxis(focus);
  const heightFocus = emphasizeHeight || focus?.kind === "loss";

  /* surface, cut at the top of the box by a clipping plane */
  const surface = useMemo(() => buildSurface(grid, map, C), [grid, map, C]);
  const material = useMemo(
    () =>
      new THREE.MeshLambertMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: 1.5,
        polygonOffsetUnits: 1.5,
        clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, -1, 0), map.height)],
      }),
    [map.height],
  );
  useEffect(() => () => surface.dispose(), [surface]);
  useEffect(() => () => material.dispose(), [material]);

  /* contours: the 2D view's levels, at their height (when inside the box) and all of them on the floor */
  const contours = useMemo(() => {
    const lifted: V3[][] = [];
    const flat: V3[][] = [];
    for (const lv of contourLevels(data, domain))
      for (const part of clipPolyline(lossEllipse(data, lv, 160), domain)) {
        if (lv <= zTop) lifted.push(part.map((p) => map.toScene(p[0], p[1], lv)));
        flat.push(part.map((p) => floorPoint(map, p)));
      }
    return { lifted, flat };
  }, [data, domain, map, zTop]);

  /* this run's path */
  const finiteTraj = trajectory.filter(ok);
  const pathLines = onSurface(finiteTraj, 10);
  const shadow = clipPolyline(finiteTraj, domain).map((part) => part.map(floor));
  const iterates = finiteTraj.length < 260 ? finiteTraj.filter(below).map((p) => lift(p)) : [];

  /* the pending update: on the floor (exact, as in 2D) and traced over the surface */
  const nextOk = ok(next) && ok(cur) ? next : null;
  const stepLines = cur && nextOk ? onSurface([cur, nextOk], 24) : [];
  const stepEnd = stepLines.length ? stepLines[stepLines.length - 1] : null;
  const nextInside = !!nextOk && inDomain(nextOk, domain);

  /* level set through the current point: a horizontal slice of the bowl at height L(θ) */
  const slice = levelThrough && cur && curL != null ? clipPolyline(lossEllipse(data, curL, 160), domain).map((part) => part.map((p) => map.toScene(p[0], p[1], curL))) : [];

  /* hover read-out and click-to-start. Raycasts ignore clipping, so skip hits above the cut. */
  const hoverDot = useRef<THREE.Mesh>(null);
  const visibleHit = (e: ThreeEvent<PointerEvent | MouseEvent>) => e.intersections.find((i) => i.object === e.eventObject && i.point.y <= map.height + 1e-6)?.point ?? null;
  const hide = () => {
    if (readoutRef.current) readoutRef.current.textContent = HOVER_HINT;
    if (hoverDot.current) hoverDot.current.visible = false;
    invalidate();
  };
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    const hit = visibleHit(e);
    if (!hit) return hide();
    const [w, b] = map.toData(hit.x, hit.z);
    const L = mse(w, b, data);
    if (readoutRef.current) readoutRef.current.textContent = `w = ${fmt(w, 3)}  b = ${fmt(b, 3)}  L(w, b) = ${fmt(L, 4)}${onStartChange ? "  · click: start here" : ""}`;
    if (hoverDot.current) {
      hoverDot.current.position.set(...lift([w, b], L));
      hoverDot.current.visible = true;
    }
    invalidate();
  };
  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (!onStartChange || e.delta > 4) return; // a drag rotates the view
    const hit = visibleHit(e);
    if (hit) onStartChange(...map.toData(hit.x, hit.z));
  };

  const optP = lift([grid.opt.w, grid.opt.b], grid.opt.loss);
  const ring = start && ok(start) && inDomain(start, domain) && onStartChange ? circleOnFloor(map, start, 0.045) : null;
  const fade = axisFocus ? 0.35 : 1;

  return (
    <group>
      <mesh geometry={surface} material={material} onPointerMove={onMove} onPointerOut={hide} onClick={onClick} />
      <FatLines lines={contours.lifted} color={C["--viz-model"]} width={1} opacity={0.5} />
      <FatLines lines={contours.flat} color={C["--viz-model"]} width={1} opacity={0.28} />

      {/* minimum */}
      <mesh position={optP}>
        <octahedronGeometry args={[0.032]} />
        <meshLambertMaterial color={toColor(C["--viz-alt"])} />
      </mesh>
      <FatLines lines={[[optP, [optP[0], 0, optP[2]]]]} color={C["--viz-alt"]} width={1} opacity={0.6} dashed dash={[0.02, 0.015]} />

      {/* comparison runs */}
      {paths?.map((p) => {
        const pts = p.points.filter(ok);
        const col = cssVar(C, p.color);
        const c = ok(p.current) && below(p.current) ? lift(p.current) : null;
        return (
          <group key={p.label}>
            <FatLines lines={onSurface(pts, 10)} color={col} width={2} />
            <FatLines lines={clipPolyline(pts, domain).map((part) => part.map(floor))} color={col} width={1.25} opacity={0.45} />
            {pts.length < 260 && <Dots points={pts.filter(below).map((q) => lift(q))} radius={0.009} color={col} />}
            {c && (
              <mesh position={c}>
                <sphereGeometry args={[0.028, 20, 14]} />
                <meshLambertMaterial color={toColor(col)} />
              </mesh>
            )}
          </group>
        );
      })}

      {/* this run */}
      <FatLines lines={pathLines} color={C["--fg-2"]} width={2} />
      <FatLines lines={shadow} color={C["--fg-3"]} width={1.25} opacity={0.6} />
      {iterates.length > 0 && <Dots points={iterates} radius={0.0095} color={C["--fg-2"]} />}
      {slice.length > 0 && <FatLines lines={slice} color={C["--viz-model"]} width={2} />}

      {cur && curL != null && (
        <group>
          <mesh position={lift(cur, curL)}>
            <sphereGeometry args={[0.03, 24, 16]} />
            <meshLambertMaterial color={toColor(C["--fg"])} />
          </mesh>
          {/* the height of the current point is its loss */}
          <FatLines lines={[[lift(cur, curL), floor(cur)]]} color={heightFocus ? C["--viz-residual"] : C["--fg-3"]} width={heightFocus ? 2.5 : 1} dashed={!heightFocus} dash={[0.02, 0.015]} />
          <mesh position={floor(cur)} rotation={[-Math.PI / 2, 0, 0]}>
            <circleGeometry args={[0.018, 20]} />
            <meshBasicMaterial color={toColor(C["--fg-2"])} />
          </mesh>
        </group>
      )}

      {cur && nextOk && (
        <group>
          {nextInside && <Arrow3 from={floor(cur)} to={floor(nextOk)} color={C["--viz-residual"]} width={2.25} opacity={fade} size={0.05} />}
          {components && nextInside && (
            <>
              <Arrow3 from={floor(cur)} to={floor([nextOk[0], cur[1]])} color={C["--viz-model"]} width={axisFocus === "w" ? 3 : 1.5} dashed={axisFocus !== "w"} opacity={axisFocus === "b" ? 0.3 : 1} size={0.04} />
              <Arrow3 from={floor([nextOk[0], cur[1]])} to={floor(nextOk)} color={C["--viz-model"]} width={axisFocus === "b" ? 3 : 1.5} dashed={axisFocus !== "b"} opacity={axisFocus === "w" ? 0.3 : 1} size={0.04} />
            </>
          )}
          <FatLines lines={stepLines} color={C["--viz-residual"]} width={2.25} opacity={0.95 * fade} />
          {stepEnd && stepEnd.length > 2 && below(nextOk) && <Arrow3 from={stepEnd[stepEnd.length - 3]} to={stepEnd[stepEnd.length - 1]} color={C["--viz-residual"]} width={2.25} size={0.05} opacity={fade} />}
        </group>
      )}

      {ring && <FatLines lines={[ring]} color={C["--fg"]} width={1.5} dashed dash={[0.012, 0.01]} />}

      <mesh ref={hoverDot} visible={false}>
        <sphereGeometry args={[0.016, 16, 12]} />
        <meshBasicMaterial color={toColor(C["--fg-2"])} />
      </mesh>
    </group>
  );
}

/** Resolve "var(--x)" to the theme colour WebGL needs. */
function cssVar(C: ThemeColors, v: string): string {
  const m = /^var\((--[\w-]+)\)$/.exec(v.trim());
  if (!m) return v;
  return (C as unknown as Record<string, string>)[m[1]] ?? getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim();
}

function floorPoint(map: BoxMap, p: Pt): V3 {
  const s = map.toScene(p[0], p[1], map.spec.z[0]);
  return [s[0], 0.001, s[2]];
}

function circleOnFloor(map: BoxMap, c: Pt, r: number): V3[] {
  const [X, , Z] = map.toScene(c[0], c[1], map.spec.z[0]);
  return Array.from({ length: 41 }, (_, k): V3 => [X + r * Math.cos((2 * Math.PI * k) / 40), 0.003, Z + r * Math.sin((2 * Math.PI * k) / 40)]);
}

/**
 * Triangle mesh through the exact grid values. Colour follows the 2D view: a point inside k of the
 * contour levels is covered by k translucent layers there, so here it gets the same composite colour.
 */
function buildSurface(grid: ReturnType<typeof surfaceGrid>, map: BoxMap, C: ThemeColors): THREE.BufferGeometry {
  const { res, w, b, L, max, opt } = grid;
  const maxEx = max - opt.loss;
  const pos = new Float32Array(res * res * 3);
  const col = new Float32Array(res * res * 3);
  const c = new THREE.Color();
  for (let j = 0; j < res; j++)
    for (let i = 0; i < res; i++) {
      const k = j * res + i;
      pos.set(map.toScene(w[i], b[j], L[k]), 3 * k);
      const layers = 1 + levelShade(L[k], opt.loss, maxEx) * (LEVEL_COUNT - 1);
      mixColor(C["--surface"], C["--viz-model"], 1 - (1 - LAYER) ** layers, c);
      col.set([c.r, c.g, c.b], 3 * k);
    }
  const idx: number[] = [];
  for (let j = 0; j + 1 < res; j++)
    for (let i = 0; i + 1 < res; i++) {
      const a = j * res + i;
      idx.push(a, a + 1, a + res + 1, a, a + res + 1, a + res);
    }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  g.computeBoundingSphere();
  return g;
}
