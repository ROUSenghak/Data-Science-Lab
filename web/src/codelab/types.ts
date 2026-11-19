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
  steps: { id: string; title: string }[];
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
}

export interface Snapshot {
  index: number;
  step: string;
  lines: [number, number];
  state: LinRegState;
  locals: Record<string, VarView>;
  removed: string[];
}

export interface RunResponse {
  experiment_id: string;
  params: Record<string, number>;
  source: string;
  snapshots: Snapshot[];
  stdout: string;
  warnings: string[];
  error: string | null;
  elapsed_ms: number;
  environment: Environment;
}
