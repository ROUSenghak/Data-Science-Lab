import type { ExperimentMeta, LinRegState, RunResponse } from "./types";

/** Proxied to the FastAPI service by next.config.ts rewrites. */
const BASE = "/api/py";

export class BackendUnavailable extends Error {}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(BASE + path, { ...init, headers: { "content-type": "application/json" } });
  } catch {
    throw new BackendUnavailable("Could not reach the Python execution service.");
  }
  if (!res.ok) {
    let body: { detail?: unknown } | null = null;
    try {
      body = await res.json();
    } catch {}
    // A non-JSON 5xx comes from the dev proxy when the Python service is not running.
    if (!body && res.status >= 500) throw new BackendUnavailable("Could not reach the Python execution service.");
    const d = body?.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const getExperiment = (id: string) => call<ExperimentMeta>(`/experiments/${id}`);

export const runExperiment = <S = LinRegState>(id: string, params: Record<string, number>) =>
  call<RunResponse<S>>(`/experiments/${id}/run`, { method: "POST", body: JSON.stringify({ params }) });
