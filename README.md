# Data Science Lab

An interactive platform for learning data science: the mathematics, an in-browser playground, and real Python you can step through.

**Read the math → explore the code → execute it → watch the model learn → inspect each intermediate value.**

Two complete vertical slices: **linear regression** and **K-Means clustering**.

| Page | Route | Computes with |
|---|---|---|
| Home | `/` | — |
| Learning catalogue | `/learn` | — |
| Lesson: linear regression | `/learn/machine-learning/linear-regression` | Browser (TypeScript) |
| Lesson: K-Means | `/learn/machine-learning/k-means` | Browser (TypeScript) |
| Lab catalogue | `/labs` | — |
| Playground: linear regression | `/labs/linear-regression` | Browser (TypeScript) |
| Playground: K-Means | `/labs/k-means` | Browser (TypeScript) |
| Code Lab: linear regression | `/code-lab?framework=numpy\|pytorch\|sklearn` | **Real Python** (FastAPI) |
| Code Lab: K-Means | `/code-lab/k-means?framework=numpy\|sklearn` | **Real Python** (FastAPI) |
| Projects | `/projects` | Planned only; listed as such |

Topics, labs and projects that do not exist yet are shown as **Planned** and are never linked.

## Run it

Requirements: Node ≥ 20.9 and Python 3.12. NumPy, scikit-learn and PyTorch can come from the system Python, because the venv uses `--system-site-packages`.

```bash
make setup   # venv + pip install, npm install
make dev     # API on :8000 and web on :3000 (or run `make api` and `make web` in two terminals)
make test    # pytest + vitest + tsc + eslint
```

Open http://localhost:3000. The web app proxies `/api/py/*` to the Python service (`PY_API_URL`, default `http://127.0.0.1:8000`). If the service is down, the Code Lab says so and shows no results.

## Architecture

```
api/                         Python execution service (FastAPI)
  app/engine.py              template parsing, parameter substitution, line tracer, serialization
  app/experiments/__init__.py  experiment registry: params, steps, watch expressions, honesty notes
  app/experiments/templates/   runnable scripts: linreg_{numpy,pytorch,sklearn}.py, kmeans_{numpy,sklearn}.py
  app/main.py                GET /experiments, GET /experiments/{id}, POST /experiments/{id}/run
  tests/                     numerical correctness tests (test_linear_regression.py, test_kmeans.py)
  scripts/export_fixtures.py writes a real NumPy trajectory for the TypeScript tests
web/src/
  content/curriculum.ts      modules, topics, labs, projects, search index (single source of truth)
  lib/ml/                    pure TypeScript math (linreg, kmeans, seeded RNG) and its tests
  components/                layout (header, search, theme), math (KaTeX), plot (SVG Chart), code (CodeView), ui
  labs/linear-regression/    plots, browser explorers, playground, lesson widgets
  labs/k-means/              cluster plot (Voronoi cells), objective plot, Lloyd stepper, init comparison, elbow, lesson widgets
  codelab/                   API client, wire types, snapshot folding, linear-regression Inspector and CodeLab
  codelab/useCodeLab.ts      lab-independent state: load, run, replay, keyboard (shared by every Code Lab page)
  codelab/shell.tsx          lab-independent UI: toolbar, code panel, output, variables, callouts
  codelab/kmeans/            K-Means trace helpers, inspector and page component
  app/                       routes (Next.js App Router)
```

### How the Code Lab stays truthful

1. **There is no arbitrary code execution.** The client picks an experiment id and sends numbers. Each parameter is type-checked and range-checked (`Param.coerce`), then substituted into its `name = value` line. The executed source is exactly what the editor displays, and line numbers never change.
2. **Steps are traced, not animated.** Templates mark computational lines with trailing `# @step <id>` comments, which are stripped before execution. `sys.settrace`, restricted to the template's own frame, records a snapshot after each marked group executes. Each snapshot holds:
   - the line range that produced it, which is what the editor highlights;
   - *watch expressions* that map the script's variables to a canonical state (`w`, `b`, `y_hat`, `loss`, `dw`, `db`, …), which drives the plots and the inspector;
   - a debugger-style view of the namespace.

   Snapshots are diff-encoded; `codelab/fold.ts` rebuilds the full states.
3. **Library internals are not invented.** PyTorch gradients come from `loss.backward()` and are read from `.grad`; updates come from `torch.optim.SGD`. For scikit-learn, `.fit()` is a single untraced call, and the page says so. Only what the estimator exposes is shown (`coef_`, `intercept_`, and the `lstsq` outputs `rank_`, `singular_`), with a separate normal-equations solve as an independent check.
6. **Loss and parameters always match.** Inside the loop, `loss` is computed before each update, so the GD templates end with an `evaluate` step that computes L(θ_T) for the returned parameters. The training history plots θ_t, L(θ_t) and ∇L(θ_t) of the same epoch t; the final evaluation is epoch T.
4. **Browser checks are labelled.** The inspector substitutes traced Python values into each equation. Each "✓" line is a separate browser recomputation, compared at float64 tolerance (1e-9), or 1e-4 for PyTorch's float32.
5. **Reproducible.** Every template seeds `np.random.default_rng(seed)`, and PyTorch also calls `torch.manual_seed`. The same seed gives the same data in all three implementations, which is tested.

### Playground vs Code Lab

The Playground and the lesson widgets run a TypeScript implementation in the browser so they respond instantly. They are labelled "Runs in browser (TypeScript)". `web/src/lib/ml/linreg.test.ts` checks that implementation against a trajectory produced by the real NumPy experiment (to 1e-10) and against scikit-learn's closed-form solution.

Loss-surface contours are exact. The MSE is quadratic, so its level sets are ellipses computed from the Hessian's eigendecomposition; nothing is sampled or interpolated.

### K-Means specifics

- **One convention everywhere** (NumPy template, TypeScript, tests): squared Euclidean distance; `argmin` ties go to the lowest index; an **empty cluster keeps its previous centroid**; stop after the first iteration in which no label changes (scikit-learn's rule with `tol=0`). scikit-learn relocates empty clusters instead; the page says so.
- **Two kinds of objective value.** In the NumPy loop, `J` is recorded after the assignment (new labels, old centroids: an intermediate state) and after the update (a completed iteration, i.e. the inertia). Plots draw them as hollow and filled points.
- **scikit-learn is not traced inside `.fit()`.** Only `cluster_centers_`, `labels_`, `inertia_`, `n_iter_` and `transform()` are shown, plus our own NumPy re-computation as a check.
- **Cluster colours** are the 8 categorical slots in fixed order (`--viz-cluster-1…8`). Beyond three slots hue alone is not separable in a scatter (validated), so every cluster also has its own marker shape, centroids carry μⱼ labels, and every plot has a data table.
- Nearest-centroid regions are exact Voronoi cells (half-plane clipping), not a sampled raster.
- `MAX_ARRAY_LEN` is 2000 so that the n × K distance matrix (n ≤ 200, K ≤ 8) can be traced.

## Tests

- `api/tests/test_linear_regression.py` (24 tests):
  - every traced forward pass, loss, gradient and update equals the hand-derived formula;
  - gradients match finite differences;
  - gradient descent follows the exact linear error dynamics `θ_t − θ* = (I − ηH)^t (θ_0 − θ*)`;
  - divergence above `2/λ_max` is reported, not crashed;
  - PyTorch autograd matches the analytic gradient and the NumPy trajectory;
  - scikit-learn matches the normal equations;
  - snapshot lines point at the right code;
  - invalid parameters are rejected (422), including integers too large for a float;
  - the final loss belongs to the final parameters, and the printed loss is that value;
  - manifest steps match the template's `@step` markers;
  - run limits stop execution with an error instead of crashing.
- `api/tests/test_kmeans.py` (61 tests). Guarantees: every traced distance, assignment (including the tie rule), centroid mean, cluster sum and objective equals an independent recomputation; J never increases across half-steps; the stopping rule and final evaluation; the empty-cluster policy; duplicate points with coincident centroids; reproducible seeds; identical labels, centroids, inertia and `n_iter_` as `sklearn.cluster.KMeans` given the same initial centroids, `tol=0` and `max_iter`; consistency of the scikit-learn template's attributes; invalid parameters (422). One empirical, seed-specific test is labelled as such (restarts escaping a local minimum).
- `web/src/lib/ml/kmeans.test.ts`, `web/src/codelab/kmeans/trace.test.ts` (23 tests): the TypeScript Lloyd iteration reproduces three NumPy runs exactly (including empty clusters and a local minimum), silhouette equals scikit-learn's, J is non-increasing on 147 random cases, Voronoi cells tile the plane and contain exactly their nearest points, duplicates, invalid input, seeds.
- `web/src/lib/ml/linreg.test.ts`, `web/src/codelab/fold.test.ts` (16 tests): TypeScript math against NumPy and scikit-learn fixtures, the stability bound, ellipse level sets, PRNG determinism, snapshot folding, per-epoch history alignment and step lookup.

## Adding a lab

**A Python experiment** (K-Means is the worked example):
1. Write `api/app/experiments/templates/<lab>_numpy.py` as a normal runnable script. Put parameters on `name = <number>` lines and add `# @step assign` / `# @step update` markers.
2. Register an `Experiment` in `api/app/experiments/__init__.py` with its params, step titles, watch expressions (script variables → canonical state) and notes on what is library code versus our own.
3. Add tests that recompute each traced step independently.
4. In `web/src/codelab/<lab>/`, add the lab's state type (in `types.ts`), an inspector for its step ids, and a page component built on `useCodeLab` and `shell.tsx` (see `kmeans/KMeansCodeLab.tsx`). Generic and reusable: `CodeView`, `StepPipeline`, `foldTrace`/`findStepBy`, `Check`, `Chart` and `SeriesPlot`. Add a route under `app/code-lab/<lab>/` and an entry in `CodeLabNav`. Give each `StepInfo` a one-line `summary`; the UI shows it.

**A browser lab:** put pure math in `web/src/lib/ml/<topic>.ts` with tests, components in `web/src/labs/<topic>/`, and a route under `web/src/app/labs/<topic>/`. Then set its status to `available` in `web/src/content/curriculum.ts`, which also makes it searchable.

## Notes

- Next.js 16 with Cache Components; all pages prerender statically. Client components that read the URL (`useSearchParams`) sit inside `<Suspense>`.
- Plot colours use the roles defined in `globals.css` (`--viz-model`, `--viz-residual`, `--viz-alt`, …), with separate light and dark values. Every chart has axis labels, a legend, hover values and a data-table view.
- The execution service serializes runs with a lock, because tracing and stdout capture are process-global. Run time is bounded by parameter ranges (epochs ≤ 200, n ≤ 100) and by a 10 s / 5000-snapshot budget checked between template lines.

### Blockers for a public deployment

- Runs execute in the API process: there is no sandbox, memory limit or CPU limit. Use one subprocess (or container) per run, with resource limits and a hard kill.
- The time budget is checked between template lines, so it cannot interrupt a long library call.
- The global lock lets one slow run block every other user. There is no rate limiting or authentication.
- Templates run with full builtins. That is safe only because templates are trusted repository files; never accept source code from clients.
