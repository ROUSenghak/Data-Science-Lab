"""Export reference trajectories from real Python runs for the frontend tests.

Usage (from api/):  .venv/bin/python -m scripts.export_fixtures
"""

import json
from pathlib import Path

from app.experiments import EXPERIMENTS

FIXTURES = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "ml" / "__fixtures__"
OUT = FIXTURES / "linreg-numpy.json"

# K-Means: one ordinary run, one local minimum, one run with empty clusters (uniform init)
KMEANS_CASES = [
    {"seed": 0, "init_method": 0, "init_seed": 4},
    {"seed": 28, "init_method": 0, "init_seed": 3},
    {"seed": 0, "init_method": 2, "init_seed": 0, "k": 5},
]


def kmeans_case(params):
    _, result = EXPERIMENTS["kmeans-numpy"].run(params)
    assert result.error is None, result.error
    state, iterations = {}, []
    for s in result.snapshots:
        state = {**state, **s["state"]}
        if s["step"] == "assign":
            iterations.append({"labels": state["labels"], "J_assign": state["J"], "n_changed": state["n_changed"]})
        elif s["step"] == "update":
            iterations[-1].update(centroids=state["centroids"], counts=state["counts"])
        elif s["step"] == "objective":
            iterations[-1]["J_update"] = state["J"]
    from sklearn.metrics import silhouette_score

    sil = float(silhouette_score(state["X"], state["labels"])) if len(set(state["labels"])) > 1 else None
    return {"params": params, "silhouette": sil, "X": state["X"], "centroids_init": state["centroids_init"], "iterations": iterations,
            "final": {"labels": state["labels"], "centroids": state["centroids"], "inertia": state["inertia"]}}


def main() -> None:
    params = {"seed": 0, "n_samples": 30, "noise_std": 1.0, "learning_rate": 0.08, "epochs": 60}
    _, result = EXPERIMENTS["linreg-numpy"].run(params)
    state, steps = {}, []
    for s in result.snapshots:
        state = {**state, **s["state"]}
        if s["step"] == "update":
            steps.append({k: state[k] for k in ("epoch", "loss", "dw", "db", "w", "b")})
    _, sk = EXPERIMENTS["linreg-sklearn"].run({k: params[k] for k in ("seed", "n_samples", "noise_std")})
    sk_state = {}
    for s in sk.snapshots:
        sk_state = {**sk_state, **s["state"]}
    OUT.write_text(json.dumps({
        "params": params, "x": state["x"], "y": state["y"], "steps": steps,
        "sklearn": {"w": sk_state["w"], "b": sk_state["b"], "mse": sk_state["loss"]},
    }, indent=1))
    print(f"wrote {OUT}")
    km = FIXTURES / "kmeans-numpy.json"
    km.write_text(json.dumps([kmeans_case(p) for p in KMEANS_CASES]))
    print(f"wrote {km}")


if __name__ == "__main__":
    main()
