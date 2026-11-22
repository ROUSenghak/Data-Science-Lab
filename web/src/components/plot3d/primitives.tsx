"use client";

/**
 * Small three.js building blocks for scientific 3D plots. Every geometry and material created here
 * is disposed when it is replaced or unmounted (R3F does not dispose `primitive` objects).
 */
import { useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import type { V3 } from "./scene";

/** Colour from a CSS hex string, converted from sRGB to the renderer's working space. */
export const toColor = (css: string) => new THREE.Color().setStyle(css, THREE.SRGBColorSpace);

/** Mix two CSS colours in sRGB (as CSS compositing does), returned as a THREE.Color. */
export function mixColor(a: string, b: string, t: number, out = new THREE.Color()): THREE.Color {
  const ca = new THREE.Color().setStyle(a, THREE.SRGBColorSpace).convertLinearToSRGB();
  const cb = new THREE.Color().setStyle(b, THREE.SRGBColorSpace).convertLinearToSRGB();
  return out.setRGB(ca.r + (cb.r - ca.r) * t, ca.g + (cb.g - ca.g) * t, ca.b + (cb.b - ca.b) * t, THREE.SRGBColorSpace);
}

/**
 * Polylines drawn with a constant width in pixels (WebGL's own lines are always 1px).
 * All polylines in one call share a single draw call.
 */
export function FatLines({
  lines,
  color,
  width = 1.5,
  opacity = 1,
  dashed = false,
  dash = [0.03, 0.02],
  depthTest = true,
  renderOrder = 0,
  visible = true,
}: {
  lines: V3[][];
  color: string;
  width?: number;
  opacity?: number;
  dashed?: boolean;
  /** Dash and gap length in scene units. */
  dash?: [number, number];
  depthTest?: boolean;
  renderOrder?: number;
  visible?: boolean;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const material = useMemo(() => new LineMaterial({ worldUnits: false, transparent: true }), []);
  const object = useMemo(() => new LineSegments2(undefined, material), [material]);
  useEffect(() => () => material.dispose(), [material]);

  const geometry = useMemo(() => {
    const pos: number[] = [];
    for (const l of lines)
      for (let k = 0; k + 1 < l.length; k++) {
        const a = l[k];
        const b = l[k + 1];
        if (a.every(Number.isFinite) && b.every(Number.isFinite)) pos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
    if (!pos.length) return null;
    const g = new LineSegmentsGeometry();
    g.setPositions(pos);
    return g;
  }, [lines]);
  useEffect(() => () => geometry?.dispose(), [geometry]);

  useLayoutEffect(() => {
    applyLine(object, material, geometry, { color, width, opacity, dashed, dash, depthTest, renderOrder, visible });
    invalidate();
  }, [material, object, geometry, color, width, opacity, dashed, dash, depthTest, renderOrder, visible, invalidate]);

  return <primitive object={object} />;
}

function applyLine(
  object: LineSegments2,
  material: LineMaterial,
  geometry: LineSegmentsGeometry | null,
  o: { color: string; width: number; opacity: number; dashed: boolean; dash: [number, number]; depthTest: boolean; renderOrder: number; visible: boolean },
) {
  material.color.copy(toColor(o.color));
  material.linewidth = o.width;
  material.opacity = o.opacity;
  material.dashed = o.dashed;
  material.dashSize = o.dash[0];
  material.gapSize = o.dash[1];
  material.depthTest = o.depthTest;
  material.depthWrite = o.depthTest && o.opacity >= 1;
  material.needsUpdate = true;
  if (geometry) {
    object.geometry = geometry;
    if (o.dashed) object.computeLineDistances();
  }
  object.renderOrder = o.renderOrder;
  object.visible = o.visible && !!geometry;
}

/** Many small spheres in one draw call. */
export function Dots({
  points,
  radius,
  color,
  opacity = 1,
  ids,
  onHover,
}: {
  points: V3[];
  radius: number;
  color: string;
  opacity?: number;
  /** Caller's id of each point (same order as `points`), reported by onHover. */
  ids?: number[];
  onHover?: (id: number | null) => void;
}) {
  const invalidate = useThree((s) => s.invalidate);
  const finite = useMemo(() => points.filter((p) => p.every(Number.isFinite)), [points]);
  // Capacity grows in powers of two so a longer run does not rebuild the mesh at every epoch.
  const capacity = Math.max(16, 2 ** Math.ceil(Math.log2(Math.max(1, finite.length))));
  const ref = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    finite.forEach((p, i) => mesh.setMatrixAt(i, m.makeTranslation(p[0], p[1], p[2])));
    mesh.count = finite.length;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
    invalidate();
  }, [finite, capacity, invalidate]);
  const c = useMemo(() => toColor(color), [color]);
  return (
    <instancedMesh
      key={capacity}
      ref={ref}
      args={[undefined, undefined, capacity]}
      frustumCulled={false}
      onPointerMove={
        onHover &&
        ((e) => {
          e.stopPropagation();
          const i = e.instanceId;
          onHover(i == null ? null : ids && finite.length === points.length ? (ids[i] ?? null) : i);
        })
      }
      onPointerOut={onHover && (() => onHover(null))}
    >
      <sphereGeometry args={[radius, 12, 8]} />
      <meshLambertMaterial color={c} transparent={opacity < 1} opacity={opacity} />
    </instancedMesh>
  );
}

const UP = new THREE.Vector3(0, 1, 0);

/** Cone whose tip is at `tip`, pointing along `dir`. */
export function ArrowHead({ tip, dir, color, size = 0.045, opacity = 1 }: { tip: V3; dir: V3; color: string; size?: number; opacity?: number }) {
  const d = new THREE.Vector3(...dir);
  if (!tip.every(Number.isFinite) || !(d.length() > 1e-9)) return null;
  d.normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(UP, d);
  const center: V3 = [tip[0] - (d.x * size) / 2, tip[1] - (d.y * size) / 2, tip[2] - (d.z * size) / 2];
  return (
    <mesh position={center} quaternion={q}>
      <coneGeometry args={[size * 0.38, size, 16]} />
      <meshBasicMaterial color={toColor(color)} transparent={opacity < 1} opacity={opacity} />
    </mesh>
  );
}

/** Arrow from `from` to `to`: a fat shaft and a cone head. Hidden when shorter than its head. */
export function Arrow3({ from, to, color, width = 2, size = 0.045, opacity = 1, dashed = false }: { from: V3; to: V3; color: string; width?: number; size?: number; opacity?: number; dashed?: boolean }) {
  const dir: V3 = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
  const len = Math.hypot(...dir);
  const lines = useMemo<V3[][]>(() => {
    if (!(len > size)) return [];
    const k = (len - size * 0.8) / len;
    return [[from, [from[0] + dir[0] * k, from[1] + dir[1] * k, from[2] + dir[2] * k]]];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [from[0], from[1], from[2], to[0], to[1], to[2], size]);
  if (!(len > size)) return null;
  return (
    <group>
      <FatLines lines={lines} color={color} width={width} opacity={opacity} dashed={dashed} dash={[0.025, 0.018]} />
      <ArrowHead tip={to} dir={dir} color={color} size={size} opacity={opacity} />
    </group>
  );
}
