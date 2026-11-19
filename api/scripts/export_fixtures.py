"""Export reference trajectories from real Python runs for the frontend tests.

Usage (from api/):  .venv/bin/python -m scripts.export_fixtures
"""

import json
from pathlib import Path

from app.experiments import EXPERIMENTS

OUT = Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "ml" / "__fixtures__" / "linreg-numpy.json"


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


if __name__ == "__main__":
    main()
