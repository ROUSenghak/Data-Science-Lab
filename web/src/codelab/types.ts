/** Wire types for the Python execution service (api/app/main.py). */

export interface ParamSpec {
  name: string;
  label: string;
  kind: "int" | "float";
  min: number;
  max: number;
  step: number;
  default: number;
  description: string;
}

export interface Environment {
  python: string;
  numpy: string;
  torch: string;
  sklearn: string;
}

export interface ExperimentMeta {
  id: string;
  lab: string;
  framework: "numpy" | "pytorch" | "sklearn";
  title: string;
  summary: string;
  execution: "educational-implementation" | "library";
  notes: string[];
  source: string;
  param_lines: Record<string, number>;
  params: ParamSpec[];
  steps: { id: string; title: string; summary: string }[];
  step_groups: { id: string; start: number; end: number }[];
  environment: Environment;
}

export interface VarView {
  type: string;
  value: string;
  shape?: number[];
  dtype?: string;
  grad_fn?: string;
}

/** Canonical linear-regression state, mapped from each script's own variables by the server's watch expressions. */
export interface LinRegState {
  x?: number[] | null;
  y?: number[] | null;
  w?: number | null;
  b?: number | null;
  y_hat?: (number | null)[] | null;
  loss?: number | null;
  dw?: number | null;
  db?: number | null;
  epoch?: number | null;
  lr?: number | null;
  theta?: number[] | null;
  /** scikit-learn lstsq outputs (rank_, singular_). */
  rank?: number | null;
  singular?: number[] | null;
}

/** Canonical K-Means state (kmeans-numpy and kmeans-sklearn watch expressions). */
export interface KMeansState {
  X?: [number, number][] | null;
  k?: number | null;
  centroids?: [number, number][] | null;
  centroids_init?: [number, number][] | null;
  labels?: number[] | null;
  prev_labels?: number[] | null;
  /** n × k squared Euclidean distances. */
  sq_dist?: number[][] | null;
  /** Objective: after the assignment step J(c⁽ᵗ⁾, μ⁽ᵗ⁻¹⁾), after the update J(c⁽ᵗ⁾, μ⁽ᵗ⁾). */
  J?: number | null;
  counts?: number[] | null;
  sums?: [number, number][] | null;
  shift?: number[] | null;
  n_changed?: number | null;
  /** Python loop index (0-based); iteration t = iteration + 1. */
  iteration?: number | null;
  inertia?: number | null;
  init_method?: number | null;
  /** scikit-learn only. */
  n_iter?: number | null;
  /** Euclidean distances from model.transform(X). */
  dist?: number[][] | null;
  check_labels?: number[] | null;
  check_J?: number | null;
  n_init?: number | null;
}

export interface Snapshot<S = LinRegState> {
  index: number;
  step: string;
  lines: [number, number];
  state: S;
  locals: Record<string, VarView>;
  removed: string[];
}

export interface RunResponse<S = LinRegState> {
  experiment_id: string;
  params: Record<string, number>;
  source: string;
  snapshots: Snapshot<S>[];
  stdout: string;
  warnings: string[];
  error: string | null;
  elapsed_ms: number;
  environment: Environment;
}
