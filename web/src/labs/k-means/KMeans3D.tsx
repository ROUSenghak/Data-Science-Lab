"use client";

/**
 * Points, centroids and assignments of Lloyd's algorithm in three features. Lazy-loaded (three.js).
 * Every position is a data value or a computed centroid; during an update the centroids glide from
 * μ⁽ᵗ⁻¹⁾ to μ⁽ᵗ⁾ (a drawn transition, labelled as such by the caller), never through iterations.
 * All three axes share one unit, so distances in the picture are the distances the algorithm uses.
 */
import { useMemo } from "react";
import Plot3D, { type SceneLabel } from "@/components/plot3d/Plot3D";
import { Arrow3, Dots, FatLines, toColor } from "@/components/plot3d/primitives";
import { boxMap, type BoxMap, type BoxSpec, type V3 } from "@/components/plot3d/scene";
import type { ThemeColors } from "@/components/plot3d/support";
import type { Vec3 } from "@/lib/ml/kmeans";

export interface KMeans3DProps {
  X: Vec3[];
  /** Cluster of each point; null before the first assignment (grey). */
  labels: number[] | null;
  /** Centroids as drawn. */
  centroids: Vec3[];
  /** Centroids before an update: ghosts with an arrow to the new position. */
  prevCentroids?: Vec3[] | null;
  /** Lines from each point to its own centroid. */
  spokes: boolean;
  /** Shadows of the points on the floor: the projection onto (x₁, x₂). */
  shadows: boolean;
  hovered: number | null;
  onHover: (i: number | null) => void;
  height: number;
  ariaLabel: string;
}

const clusterToken = (j: number) => `--viz-cluster-${(j % 4) + 1}` as `--viz-cluster-${1 | 2 | 3 | 4}`;
/** Literal class names, so Tailwind generates them. */
const CLUSTER_TEXT = ["text-[var(--viz-cluster-1)]", "text-[var(--viz-cluster-2)]", "text-[var(--viz-cluster-3)]", "text-[var(--viz-cluster-4)]"];

export default function KMeans3D(props: KMeans3DProps) {
  const { X, height, ariaLabel } = props;
  const box = useMemo<BoxSpec>(() => {
    const pad = (lo: number, hi: number): [number, number] => {
      const r = hi - lo || 1;
      return [Math.floor(lo - 0.08 * r), Math.ceil(hi + 0.08 * r)];
    };
    const ext = (d: number) => pad(Math.min(...X.map((p) => p[d])), Math.max(...X.map((p) => p[d])));
    return { x: ext(0), y: ext(1), z: ext(2), equalXYZ: true };
  }, [X]);
  const toScene = useMemo(() => boxMap(box).toScene, [box]);
  const titles = useMemo(() => ({ x: <>x₁</>, y: <>x₂</>, z: <>x₃</> }), []);
  const labels: SceneLabel[] = props.centroids.map((c, j) => {
    const p = toScene(c[0], c[1], c[2]);
    return { id: `mu${j}`, at: [p[0], p[1] + 0.1, p[2]], content: `μ${j + 1}`, className: `text-[12px] font-semibold ${CLUSTER_TEXT[j % 4]}` };
  });

  return (
    <Plot3D box={box} axisTitles={titles} height={height} ariaLabel={ariaLabel} labels={labels}>
      {(map, colors) => <Scene {...props} map={map} colors={colors} />}
    </Plot3D>
  );
}

function Scene({ X, labels, centroids, prevCentroids, spokes, shadows, hovered, onHover, map, colors: C }: KMeans3DProps & { map: BoxMap; colors: ThemeColors }) {
  const at = (p: readonly number[]): V3 => map.toScene(p[0], p[1], p[2]);
  const k = centroids.length;

  // One instanced mesh per cluster (one colour each); unassigned points are grey.
  const groups = useMemo(() => {
    const out = Array.from({ length: labels ? k : 1 }, () => ({ pts: [] as V3[], ids: [] as number[] }));
    X.forEach((p, i) => {
      const g = labels ? labels[i] : 0;
      out[g].pts.push(map.toScene(p[0], p[1], p[2]));
      out[g].ids.push(i);
    });
    return out;
  }, [X, labels, k, map]);
  const shadowPts = useMemo(() => (shadows ? X.map((p): V3 => [map.toScene(p[0], p[1], 0)[0], 0.003, map.toScene(p[0], p[1], 0)[2]]) : []), [X, shadows, map]);
  const spokeLines = labels && spokes ? Array.from({ length: k }, (_, j) => X.flatMap((p, i) => (labels[i] === j ? [[at(p), at(centroids[j])]] : []))) : [];

  const hov = hovered != null && X[hovered] ? X[hovered] : null;
  return (
    <group>
      {groups.map((g, j) => (
        <Dots key={labels ? `c${j}` : "grey"} points={g.pts} ids={g.ids} radius={0.026} color={labels ? C[clusterToken(j)] : C["--viz-data"]} onHover={onHover} />
      ))}
      {shadowPts.length > 0 && <Dots points={shadowPts} radius={0.014} color={C["--fg-3"]} opacity={0.55} />}
      {spokeLines.map((lines, j) => (lines.length ? <FatLines key={j} lines={lines} color={C[clusterToken(j)]} width={1} opacity={0.4} /> : null))}

      {/* hovered point: its distance to every centroid */}
      {hov &&
        centroids.map((c, j) => (
          <FatLines key={`d${j}`} lines={[[at(hov), at(c)]]} color={C[clusterToken(j)]} width={labels?.[hovered!] === j ? 2.5 : 1.25} dashed={labels?.[hovered!] !== j} dash={[0.03, 0.02]} />
        ))}
      {hov && (
        <mesh position={at(hov)}>
          <sphereGeometry args={[0.045, 20, 14]} />
          <meshBasicMaterial color={toColor(C["--fg"])} transparent opacity={0.25} />
        </mesh>
      )}

      {prevCentroids?.map((c, j) => (
        <group key={`p${j}`}>
          <mesh position={at(c)}>
            <sphereGeometry args={[0.04, 20, 14]} />
            <meshBasicMaterial color={toColor(C[clusterToken(j)])} transparent opacity={0.3} />
          </mesh>
          <Arrow3 from={at(c)} to={at(centroids[j])} color={C[clusterToken(j)]} width={2} size={0.06} />
        </group>
      ))}
      {centroids.map((c, j) => (
        <mesh key={`mu${j}`} position={at(c)}>
          <octahedronGeometry args={[0.065]} />
          <meshLambertMaterial color={toColor(C[clusterToken(j)])} />
        </mesh>
      ))}
    </group>
  );
}
