"use client";

/**
 * A box-axes 3D plot: the reusable base for 3D views (loss landscapes today; clusters in R³, PCA or
 * probability surfaces later). It owns the camera, axes, tick labels and view controls; the caller
 * draws its content in scene coordinates through the BoxMap it receives.
 *
 *  - Renders on demand only (frameloop "demand"): a static scene costs no GPU time.
 *  - Labels are HTML (crisp text, theme colours, KaTeX if needed), projected onto the canvas each
 *    rendered frame without React re-renders.
 *  - Orthographic projection: parallel lines stay parallel, and the top view is exactly the 2D chart.
 *  - The camera orbits a fixed target with limited elevation; there is no panning and no automatic
 *    rotation. The wheel scrolls the page unless Ctrl/⌘ is held, so the plot never traps scrolling.
 */
import { Canvas, useFrame, useThree, type RootState } from "@react-three/fiber";
import { useId, useLayoutEffect, useMemo, useRef, type ReactNode, type RefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FatLines } from "./primitives";
import { boxMap, clampElevation, DEFAULT_VIEW, fitZoom, nearSides, niceTicks, orbitPosition, tickAnchor, TOP_VIEW, verticalAxisCorner, type BoxMap, type BoxSpec, type V3, type ViewAngles } from "./scene";
import { useThemeColors, type ThemeColors } from "./support";

/** Camera distance from the target; with an orthographic camera it only matters for depth clipping. */
const DISTANCE = 8;
const ROTATE_STEP = (15 * Math.PI) / 180;
const TILT_STEP = (8 * Math.PI) / 180;
const ZOOM_RANGE: [number, number] = [0.5, 2.5];

export interface SceneLabel {
  id: string;
  /** Scene coordinates (use map.toScene for data coordinates). */
  at: V3;
  content: ReactNode;
  className?: string;
}

export interface ViewApi {
  rotate(dAzimuth: number): void;
  tilt(dElevation: number): void;
  zoom(factor: number): void;
  set(view: ViewAngles): void;
  reset(): void;
}

type Sides = { x: 1 | -1; z: 1 | -1 };
type Entry = { id: string; pos: (s: Sides) => V3; content: ReactNode; className: string };

export interface Plot3DProps {
  box: BoxSpec;
  axisTitles: { x: ReactNode; y: ReactNode; z: ReactNode };
  format?: Partial<Record<"x" | "y" | "z", (v: number) => string>>;
  height: number;
  ariaLabel: string;
  /** Extra labels anchored in the scene. */
  labels?: SceneLabel[];
  /** HTML drawn over the canvas (e.g. a hover read-out), positioned by the caller. */
  overlay?: ReactNode;
  initialView?: ViewAngles;
  /** Scene content, drawn in scene coordinates. Rendered once theme colours are known. */
  children: (map: BoxMap, colors: ThemeColors) => ReactNode;
}

const defaultFormat = (v: number) => {
  const s = Math.abs(v) >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(+v.toFixed(2));
  return s.replace("-", "−");
};

export default function Plot3D({ box, axisTitles, format, height, ariaLabel, labels = [], overlay, initialView = DEFAULT_VIEW, children }: Plot3DProps) {
  const map = useMemo(() => boxMap(box), [box]);
  const colors = useThemeColors();
  const api = useRef<ViewApi | null>(null);
  const store = useRef<RootState | null>(null);
  const nodes = useRef(new Map<string, HTMLElement>());
  const helpId = useId();

  const entries = useMemo<Entry[]>(() => {
    const f = { x: format?.x ?? defaultFormat, y: format?.y ?? defaultFormat, z: format?.z ?? defaultFormat };
    const tick = "text-[10.5px] tabular-nums text-fg-3";
    const out: Entry[] = [];
    for (const axis of ["x", "y", "z"] as const) {
      const r = map.spec[axis];
      const ticks = niceTicks(r, axis === "z" ? 4 : 5).filter((v) => v >= r[0] && v <= r[1]);
      for (const v of ticks) out.push({ id: `t${axis}${v}`, pos: (s) => tickAnchor(map, axis, v, s), content: f[axis](v), className: tick });
    }
    const title = "text-[12px] font-medium text-fg-2";
    out.push({ id: "ax-x", pos: (s) => tickAnchor(map, "x", (map.spec.x[0] + map.spec.x[1]) / 2, s, 0.26), content: axisTitles.x, className: title });
    out.push({ id: "ax-y", pos: (s) => tickAnchor(map, "y", (map.spec.y[0] + map.spec.y[1]) / 2, s, 0.28), content: axisTitles.y, className: title });
    out.push({
      id: "ax-z",
      pos: (s) => {
        const [X, Z] = verticalAxisCorner(map, s);
        return [X, map.height + 0.12, Z];
      },
      content: axisTitles.z,
      className: title,
    });
    for (const l of labels) out.push({ id: `u-${l.id}`, pos: () => l.at, content: l.content, className: l.className ?? "text-[11.5px] text-fg" });
    return out;
  }, [map, format, axisTitles, labels]);

  const entriesRef = useRef(entries);
  useLayoutEffect(() => {
    entriesRef.current = entries;
    store.current?.invalidate();
  }, [entries]);

  const onKey = (e: React.KeyboardEvent) => {
    const a = api.current;
    if (!a) return;
    const k = e.key;
    if (k === "ArrowLeft") a.rotate(-ROTATE_STEP);
    else if (k === "ArrowRight") a.rotate(ROTATE_STEP);
    else if (k === "ArrowUp") a.tilt(TILT_STEP);
    else if (k === "ArrowDown") a.tilt(-TILT_STEP);
    else if (k === "+" || k === "=") a.zoom(1 / 1.15);
    else if (k === "-" || k === "_") a.zoom(1.15);
    else if (k === "0") a.reset();
    else if (k === "t" || k === "T") a.set(TOP_VIEW);
    else return;
    e.preventDefault();
    e.stopPropagation(); // keep arrow keys from also stepping a playback
  };

  return (
    <div className="group/plot3d min-w-0">
      <div
        className="relative overflow-hidden rounded-md outline-offset-2"
        style={{ height }}
        tabIndex={0}
        role="group"
        aria-label={ariaLabel}
        aria-describedby={helpId}
        aria-roledescription="3D plot"
        onKeyDown={onKey}
      >
        {colors && (
          <Canvas
            frameloop="demand"
            flat
            dpr={[1, 2]}
            gl={{ antialias: true, powerPreference: "low-power" }}
            orthographic
            camera={{ near: 0.1, far: 2 * DISTANCE + 10, position: [0, 2, DISTANCE], zoom: 80 }}
            onCreated={(s) => {
              store.current = s;
              enableClipping(s.gl);
            }}
            style={{ touchAction: "pan-y" }}
          >
            <Rig map={map} apiRef={api} initialView={initialView} />
            <ambientLight intensity={Math.PI * (colors.dark ? 0.62 : 0.72)} />
            <directionalLight position={[1.2, 3, 2.2]} intensity={Math.PI * 0.42} />
            <Axes map={map} colors={colors} entries={entriesRef} nodes={nodes} />
            {children(map, colors)}
          </Canvas>
        )}
        <div className="pointer-events-none absolute inset-0 select-none" aria-hidden>
          {entries.map((e) => (
            <span
              key={e.id}
              ref={(el) => {
                if (el) nodes.current.set(e.id, el);
                else nodes.current.delete(e.id);
              }}
              className={`absolute left-0 top-0 whitespace-nowrap leading-none [text-shadow:0_0_3px_var(--surface),0_0_3px_var(--surface)] ${e.className}`}
              style={{ visibility: "hidden" }}
            >
              {e.content}
            </span>
          ))}
        </div>
        {overlay}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-1 gap-y-1 text-[11.5px] text-fg-3">
        <ViewButton label="Rotate left" onClick={() => api.current?.rotate(-ROTATE_STEP)}>
          <path d="M3.5 6a3.5 3.5 0 1 1 1 2.5" />
          <path d="M3 3.2v2.8h2.8" />
        </ViewButton>
        <ViewButton label="Rotate right" onClick={() => api.current?.rotate(ROTATE_STEP)}>
          <path d="M8.5 6a3.5 3.5 0 1 0-1 2.5" />
          <path d="M9 3.2v2.8H6.2" />
        </ViewButton>
        <button type="button" onClick={() => api.current?.set(TOP_VIEW)} className="rounded px-1.5 py-0.5 text-fg-2 hover:bg-surface-2 hover:text-fg">
          Top view
        </button>
        <button type="button" onClick={() => api.current?.reset()} className="rounded px-1.5 py-0.5 text-fg-2 hover:bg-surface-2 hover:text-fg">
          Reset view
        </button>
        <span id={helpId} className="ml-1">
          Drag to rotate · Ctrl/⌘ + scroll to zoom
          <span className="sr-only group-focus-within/plot3d:not-sr-only"> · keys: ← → rotate, ↑ ↓ tilt, + − zoom, T top view, 0 reset</span>
        </span>
      </div>
    </div>
  );
}

/** Lets materials use clipping planes (e.g. a surface cut at the top of the box). */
function enableClipping(gl: THREE.WebGLRenderer) {
  gl.localClippingEnabled = true;
}

function ViewButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick} className="inline-flex h-6 w-6 items-center justify-center rounded text-fg-2 hover:bg-surface-2 hover:text-fg">
      <svg width="13" height="13" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden>
        {children}
      </svg>
    </button>
  );
}

/* ------------------------------------------------------------------ camera */

function Rig({ map, apiRef, initialView }: { map: BoxMap; apiRef: RefObject<ViewApi | null>; initialView: ViewAngles }) {
  // Read the renderer state through get() inside effects: these are imperative three.js objects.
  const get = useThree((s) => s.get);
  const size = useThree((s) => s.size);
  const controlsRef = useRef<OrbitControls | null>(null);
  const zoomRef = useRef(1);
  const viewRef = useRef<ViewAngles>(initialView);
  const target = useMemo<V3>(() => [0, map.height / 2, 0], [map]);
  // Fitted to the initial view; free rotation keeps this zoom (Reset view returns to it).
  const fit = fitZoom(map, initialView, target, size.width, size.height);

  useLayoutEffect(() => {
    const { camera, gl, invalidate } = get();
    const controls = createControls(camera, gl.domElement);
    controlsRef.current = controls;
    const onChange = () => {
      viewRef.current = { azimuth: controls.getAzimuthalAngle(), elevation: Math.PI / 2 - controls.getPolarAngle() };
      invalidate();
    };
    controls.addEventListener("change", onChange);
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return; // let the page scroll
      e.preventDefault();
      apiRef.current?.zoom(Math.exp(e.deltaY * 0.0025));
    };
    gl.domElement.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      controls.removeEventListener("change", onChange);
      gl.domElement.removeEventListener("wheel", onWheel);
      controls.dispose();
      controlsRef.current = null;
    };
  }, [get, apiRef]);

  useLayoutEffect(() => {
    const place = (v: ViewAngles) => {
      const controls = controlsRef.current;
      if (!controls) return;
      const { camera, invalidate } = get();
      const next = { azimuth: v.azimuth, elevation: clampElevation(v.elevation) };
      viewRef.current = next;
      setOrtho(camera, orbitPosition(next, DISTANCE, target), fit * zoomRef.current);
      controls.target.set(...target);
      controls.update();
      invalidate();
    };
    apiRef.current = {
      rotate: (d) => place({ ...viewRef.current, azimuth: viewRef.current.azimuth + d }),
      tilt: (d) => place({ ...viewRef.current, elevation: viewRef.current.elevation + d }),
      zoom: (f) => {
        zoomRef.current = Math.min(ZOOM_RANGE[1], Math.max(ZOOM_RANGE[0], zoomRef.current / f));
        place(viewRef.current);
      },
      set: (v) => place(v),
      reset: () => {
        zoomRef.current = 1;
        place(initialView);
      },
    };
    // Keep the angles, refit the zoom when the canvas size changes.
    place(viewRef.current);
  }, [get, fit, target, apiRef, initialView]);

  return null;
}

function setOrtho(camera: THREE.Camera, position: V3, zoom: number) {
  camera.position.set(...position);
  if (camera instanceof THREE.OrthographicCamera) {
    camera.zoom = zoom;
    camera.updateProjectionMatrix();
  }
}

/** Orbit around a fixed target: no panning, no wheel zoom (handled separately), limited elevation. */
function createControls(camera: THREE.Camera, el: HTMLElement): OrbitControls {
  const c = new OrbitControls(camera, el);
  c.enablePan = false;
  c.enableZoom = false;
  c.enableDamping = false;
  c.rotateSpeed = 0.75;
  // OrbitControls measures the polar angle from the vertical.
  c.minPolarAngle = Math.PI / 2 - clampElevation(Infinity);
  c.maxPolarAngle = Math.PI / 2 - clampElevation(-Infinity);
  // Vertical swipes keep scrolling the page on touch screens; horizontal drags rotate.
  el.style.touchAction = "pan-y";
  return c;
}

/* ------------------------------------------------------------------ axes and labels */

function Axes({ map, colors, entries, nodes }: { map: BoxMap; colors: ThemeColors; entries: RefObject<Entry[]>; nodes: RefObject<Map<string, HTMLElement>> }) {
  const corners = useRef<(THREE.Group | null)[]>([]);
  const xt = useMemo(() => niceTicks(map.spec.x, 5).filter((v) => v >= map.spec.x[0] && v <= map.spec.x[1]), [map]);
  const yt = useMemo(() => niceTicks(map.spec.y, 5).filter((v) => v >= map.spec.y[0] && v <= map.spec.y[1]), [map]);
  const zt = useMemo(() => niceTicks(map.spec.z, 4).filter((v) => v >= map.spec.z[0] && v <= map.spec.z[1]), [map]);

  const grid = useMemo<V3[][]>(() => {
    const { hx, hz } = map;
    return [...xt.map((v): V3[] => [[map.toScene(v, 0, map.spec.z[0])[0], 0, -hz], [map.toScene(v, 0, map.spec.z[0])[0], 0, hz]]), ...yt.map((v): V3[] => [[-hx, 0, map.toScene(0, v, map.spec.z[0])[2]], [hx, 0, map.toScene(0, v, map.spec.z[0])[2]]])];
  }, [map, xt, yt]);
  const outline = useMemo<V3[][]>(() => {
    const { hx, hz } = map;
    return [
      [
        [-hx, 0, -hz],
        [hx, 0, -hz],
        [hx, 0, hz],
        [-hx, 0, hz],
        [-hx, 0, -hz],
      ],
    ];
  }, [map]);
  const cornerList = useMemo(
    () =>
      ([[1, 1], [1, -1], [-1, 1], [-1, -1]] as const).map(([sx, sz]) => {
        const [X, Z] = verticalAxisCorner(map, { x: sx, z: sz });
        const axis: V3[] = [
          [X, 0, Z],
          [X, map.height, Z],
        ];
        const ticks = zt.map((v): V3[] => {
          const Y = map.zToY(v);
          return [
            [X, Y, Z],
            [X - sx * 0.035, Y, Z + sz * 0.035],
          ];
        });
        return { key: `${sx}${sz}`, sx, sz, lines: [axis, ...ticks] };
      }),
    [map, zt],
  );

  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    // useFrame runs before the render that updates the camera's matrices; update them now so the
    // labels are projected with this frame's camera, not the previous one.
    camera.updateMatrixWorld();
    const sides = nearSides(camera.position.x, camera.position.z);
    const dy = camera.position.y - map.height / 2;
    const steep = dy / Math.hypot(camera.position.x, dy, camera.position.z) > Math.sin((72 * Math.PI) / 180);
    cornerList.forEach((c, i) => {
      const g = corners.current[i];
      if (g) g.visible = c.sx === sides.x && c.sz === sides.z;
    });
    for (const e of entries.current) {
      const node = nodes.current.get(e.id);
      if (!node) continue;
      v.set(...e.pos(sides)).project(camera);
      const vertical = e.id.startsWith("tz") || e.id === "ax-z";
      const visible = v.z < 1 && v.z > -1 && !(vertical && steep);
      node.style.visibility = visible ? "visible" : "hidden";
      if (visible) node.style.transform = `translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px) translate(-50%, -50%)`;
    }
  });

  return (
    <group>
      <FatLines lines={grid} color={colors["--viz-grid"]} width={1} />
      <FatLines lines={outline} color={colors["--viz-axis"]} width={1} />
      {cornerList.map((c, i) => (
        <group
          key={c.key}
          ref={(g) => {
            corners.current[i] = g;
          }}
          visible={false}
        >
          <FatLines lines={c.lines} color={colors["--viz-axis"]} width={1} />
        </group>
      ))}
    </group>
  );
}
