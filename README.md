# Data Science Lab

An interactive platform for learning data science: the mathematics, an in-browser playground, and real Python you can step through.

**Read the math → explore the code → execute it → watch the model learn → inspect each intermediate value.**

The MVP has one complete vertical slice, **linear regression**:

| Page | Route | Computes with |
|---|---|---|
| Home | `/` | — |
| Learning catalogue | `/learn` | — |
| Lesson | `/learn/machine-learning/linear-regression` | Browser (TypeScript) |
| Lab catalogue | `/labs` | — |
| Playground | `/labs/linear-regression` | Browser (TypeScript) |
| Code Lab | `/code-lab?framework=numpy\|pytorch\|sklearn` | **Real Python** (FastAPI) |
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
  app/experiments/templates/   runnable scripts: linreg_numpy.py, linreg_pytorch.py, linreg_sklearn.py
  app/main.py                GET /experiments, GET /experiments/{id}, POST /experiments/{id}/run
  tests/                     numerical correctness tests
  scripts/export_fixtures.py writes a real NumPy trajectory for the TypeScript tests
web/src/
  content/curriculum.ts      modules, topics, labs, projects, search index (single source of truth)
  lib/ml/                    pure TypeScript math (linreg, seeded RNG) and its tests
  components/                layout (header, search, theme), math (KaTeX), plot (SVG Chart), code (CodeView), ui
  labs/linear-regression/    plots, browser explorers, playground, lesson widgets
  codelab/                   API client, wire types, snapshot folding, Inspector, CodeLab
  app/                       routes (Next.js App Router)
```

### How the Code Lab stays truthful

1. **There is no arbitrary code execution.** The client picks an experiment id and sends numbers. Each parameter is type-checked and range-checked (`Param.coerce`), then substituted into its `name = value` line. The executed source is exactly what the editor displays, and line numbers never change.
2. **Steps are traced, not animated.** Templates mark computational lines with trailing `# @step <id>` comments, which are stripped before execution. `sys.settrace`, restricted to the template's own frame, records a snapshot after each marked group executes. Each snapshot holds:
   - the line range that produced it, which is what the editor highlights;
   - *watch expressions* that map the script's variables to a canonical state (`w`, `b`, `y_hat`, `loss`, `dw`, `db`, …), which drives the plots and the inspector;
   - a debugger-style view of the namespace.

   Snapshots are diff-encoded; `codelab/fold.ts` rebuilds the full states.
3. **Library internals are not invented.** PyTorch gradients come from `loss.backward()` and are read from `.grad`; updates come from `torch.optim.SGD`. For scikit-learn, `.fit()` is a single untraced call, and the page says so. Only `coef_` and `intercept_` are shown, with a separate normal-equations solve as an independent check.
4. **Browser checks are labelled.** The inspector substitutes traced Python values into each equation. Each "✓" line is a separate browser recomputation, compared at float64 tolerance (1e-9), or 1e-4 for PyTorch's float32.
5. **Reproducible.** Every template seeds `np.random.default_rng(seed)`, and PyTorch also calls `torch.manual_seed`. The same seed gives the same data in all three implementations, which is tested.

### Playground vs Code Lab

The Playground and the lesson widgets run a TypeScript implementation in the browser so they respond instantly. They are labelled "Runs in browser (TypeScript)". `web/src/lib/ml/linreg.test.ts` checks that implementation against a trajectory produced by the real NumPy experiment (to 1e-10) and against scikit-learn's closed-form solution.

Loss-surface contours are exact. The MSE is quadratic, so its level sets are ellipses computed from the Hessian's eigendecomposition; nothing is sampled or interpolated.

## Tests

- `api/tests/test_linear_regression.py` (19 tests):
  - every traced forward pass, loss, gradient and update equals the hand-derived formula;
  - gradients match finite differences;
  - gradient descent follows the exact linear error dynamics `θ_t − θ* = (I − ηH)^t (θ_0 − θ*)`;
  - divergence above `2/λ_max` is reported, not crashed;
  - PyTorch autograd matches the analytic gradient and the NumPy trajectory;
  - scikit-learn matches the normal equations;
  - snapshot lines point at the right code;
  - invalid parameters are rejected (422).
- `web/src/lib/ml/linreg.test.ts`, `web/src/codelab/fold.test.ts` (13 tests): TypeScript math against NumPy and scikit-learn fixtures, the stability bound, ellipse level sets, PRNG determinism, and snapshot folding.

## Adding a lab

**A Python experiment** (e.g. K-Means):
1. Write `api/app/experiments/templates/kmeans_numpy.py` as a normal runnable script. Put parameters on `name = <number>` lines and add `# @step assign` / `# @step update` markers.
2. Register an `Experiment` in `api/app/experiments/__init__.py` with its params, step titles, watch expressions (script variables → canonical state) and notes on what is library code versus our own.
3. Add tests that recompute each traced step independently.
4. In `web/src/codelab/`, add the lab's state type, an inspector for its step ids, and plots. `CodeView`, the replay controls, `foldSnapshots` and `Chart` are generic.

**A browser lab:** put pure math in `web/src/lib/ml/<topic>.ts` with tests, components in `web/src/labs/<topic>/`, and a route under `web/src/app/labs/<topic>/`. Then set its status to `available` in `web/src/content/curriculum.ts`, which also makes it searchable.

## Notes

- Next.js 16 with Cache Components; all pages prerender statically. Client components that read the URL (`useSearchParams`) sit inside `<Suspense>`.
- Plot colours use the roles defined in `globals.css` (`--viz-model`, `--viz-residual`, `--viz-alt`, …), with separate light and dark values. Every chart has axis labels, a legend, hover values and a data-table view.
- The execution service serializes runs with a lock, because tracing and stdout capture are process-global. Run time is bounded by parameter ranges: epochs ≤ 200, n ≤ 100. Before any public deployment, add per-run process isolation and timeouts.
