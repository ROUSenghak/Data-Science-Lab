"""Numerical correctness of the linear-regression experiments.

Every check recomputes values independently from the recorded snapshots, so a
mismatch between code, trace and mathematics fails the test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.engine import ParamError, StepGroup, run_source
from app.experiments import EXPERIMENTS
from app.main import app


def fold(snapshots):
    """Reconstruct full states from diff-encoded snapshots (mirrors the frontend)."""
    state, out = {}, []
    for s in snapshots:
        state = {**state, **s["state"]}
        out.append((s, state))
    return out


def run(exp_id, **params):
    _, result = EXPERIMENTS[exp_id].run(params)
    assert result.error is None, result.error
    return result


def normal_equations(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    return np.linalg.solve(A.T @ A, A.T @ y)


# ------------------------------------------------------------------ NumPy GD


def test_numpy_gradient_descent_matches_hand_derived_math():
    result = run("linreg-numpy", epochs=25, learning_rate=0.05)
    states = fold(result.snapshots)
    x = np.array(states[0][1]["x"])
    y = np.array(states[0][1]["y"])
    w = b = None
    checked = 0
    for snap, st in states:
        if snap["step"] == "init":
            w, b = st["w"], st["b"]
        elif snap["step"] == "forward":
            assert np.allclose(st["y_hat"], w * x + b, rtol=0, atol=1e-12)
        elif snap["step"] == "loss":
            assert math.isclose(st["loss"], np.mean((w * x + b - y) ** 2), rel_tol=1e-12)
        elif snap["step"] == "gradient":
            r = w * x + b - y
            assert math.isclose(st["dw"], 2 * np.mean(r * x), rel_tol=1e-12)
            assert math.isclose(st["db"], 2 * np.mean(r), rel_tol=1e-12)
            # analytic gradient agrees with a central finite difference
            L = lambda w_, b_: np.mean((w_ * x + b_ - y) ** 2)  # noqa: E731
            h = 1e-6
            assert math.isclose(st["dw"], (L(w + h, b) - L(w - h, b)) / (2 * h), rel_tol=1e-6)
            assert math.isclose(st["db"], (L(w, b + h) - L(w, b - h)) / (2 * h), rel_tol=1e-6)
        elif snap["step"] == "update":
            assert math.isclose(st["w"], w - 0.05 * st["dw"], rel_tol=1e-12)
            assert math.isclose(st["b"], b - 0.05 * st["db"], rel_tol=1e-12)
            w, b = st["w"], st["b"]
            checked += 1
    assert checked == 25


def test_numpy_gradient_descent_follows_linear_error_dynamics():
    # For a quadratic loss, GD is linear: theta_t - theta* = (I - eta H)^t (theta_0 - theta*).
    eta, T = 0.07, 200
    final = fold(run("linreg-numpy", epochs=T, learning_rate=eta).snapshots)[-1][1]
    x, y = np.array(final["x"]), np.array(final["y"])
    theta_star = normal_equations(x, y)
    H = 2 * np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
    predicted = theta_star + np.linalg.matrix_power(np.eye(2) - eta * H, T) @ (np.zeros(2) - theta_star)
    assert [final["w"], final["b"]] == pytest.approx(list(predicted), abs=1e-10)
    # and it is close to the least-squares solution because eta < 2 / lambda_max
    assert eta < 2 / np.linalg.eigvalsh(H).max()
    assert [final["w"], final["b"]] == pytest.approx(list(theta_star), abs=1e-2)


def test_numpy_divergence_is_reported_not_crashed():
    # x ~ U(0, 4): lambda_max of the Hessian is ~12, so eta = 0.5 > 2 / lambda_max diverges.
    result = run("linreg-numpy", epochs=200, learning_rate=0.5)
    final = fold(result.snapshots)[-1][1]
    assert final["w"] is None or not math.isfinite(final["w"]) or abs(final["w"]) > 1e100
    losses = [st["loss"] for s, st in fold(result.snapshots) if s["step"] == "loss"]
    assert losses[5] > losses[0]  # loss grows


@pytest.mark.parametrize("exp_id", ["linreg-numpy", "linreg-pytorch"])
def test_final_loss_belongs_to_final_parameters(exp_id):
    # In the loop, loss is evaluated before each update. The final evaluate step must
    # report L(w_T, b_T), and the printed loss must be that value.
    result = run(exp_id, epochs=7, learning_rate=0.05)
    states = fold(result.snapshots)
    snap, final = states[-1]
    assert snap["step"] == "evaluate"
    x, y = np.array(final["x"]), np.array(final["y"])
    expected = np.mean((final["w"] * x + final["b"] - y) ** 2)
    tol = 1e-4 if exp_id == "linreg-pytorch" else 1e-12
    assert final["loss"] == pytest.approx(expected, rel=tol)
    in_loop = [st["loss"] for s, st in states if s["step"] == "loss"]
    assert final["loss"] < in_loop[-1]  # one more descent step on a convex loss
    assert f"loss = {final['loss']:.4f}" in result.stdout
    if exp_id == "linreg-pytorch":
        assert "grad_fn" not in snap["locals"].get("y_hat", {})  # evaluated under no_grad


def test_manifest_steps_match_template_markers():
    for exp in EXPERIMENTS.values():
        marked = {g.id for g in exp.tmpl.groups}
        declared = {s.id for s in exp.steps}
        assert marked == declared, exp.id


# ------------------------------------------------------------------ PyTorch


def test_pytorch_autograd_matches_analytic_gradient_and_numpy():
    pt = fold(run("linreg-pytorch", epochs=30, learning_rate=0.05).snapshots)
    npy = fold(run("linreg-numpy", epochs=30, learning_rate=0.05).snapshots)
    x = np.array(pt[0][1]["x"], dtype=np.float64)
    y = np.array(pt[0][1]["y"], dtype=np.float64)

    backward = [st for s, st in pt if s["step"] == "backward"]
    before = [st for s, st in pt if s["step"] == "zero_grad"]
    assert len(backward) == len(before) == 30
    for pre, st in zip(before, backward):
        r = pre["w"] * x + pre["b"] - y
        assert st["dw"] == pytest.approx(2 * np.mean(r * x), rel=1e-4, abs=1e-5)
        assert st["db"] == pytest.approx(2 * np.mean(r), rel=1e-4, abs=1e-5)

    # zero_grad() with set_to_none=True clears gradients
    assert all(st["dw"] is None for s, st in pt if s["step"] == "zero_grad")

    pt_updates = [(st["w"], st["b"]) for s, st in pt if s["step"] == "update"]
    np_updates = [(st["w"], st["b"]) for s, st in npy if s["step"] == "update"]
    assert np.allclose(pt_updates, np_updates, rtol=1e-4, atol=1e-4)  # float32 vs float64


# ------------------------------------------------------------------ scikit-learn


def test_sklearn_matches_normal_equations():
    final = fold(run("linreg-sklearn", seed=3, n_samples=50, noise_std=2.0).snapshots)[-1][1]
    x, y = np.array(final["x"]), np.array(final["y"])
    theta = normal_equations(x, y)
    assert final["w"] == pytest.approx(theta[0], rel=1e-10)
    assert final["b"] == pytest.approx(theta[1], rel=1e-10)
    assert final["theta"] == pytest.approx(list(theta), rel=1e-10)
    assert final["loss"] == pytest.approx(np.mean((theta[0] * x + theta[1] - y) ** 2), rel=1e-10)


def test_sklearn_solver_outputs_are_reported_truthfully():
    # LinearRegression runs lstsq on the centred X: one singular value, sqrt(sum (x - mean)^2).
    final = fold(run("linreg-sklearn", seed=5, n_samples=40).snapshots)[-1][1]
    x = np.array(final["x"])
    assert final["rank"] == 1
    assert final["singular"] == pytest.approx([np.sqrt(np.sum((x - x.mean()) ** 2))], rel=1e-10)


def test_same_seed_produces_same_data_across_frameworks():
    a = fold(run("linreg-numpy", seed=7, epochs=1).snapshots)[0][1]
    b = fold(run("linreg-sklearn", seed=7).snapshots)[0][1]
    c = fold(run("linreg-pytorch", seed=7, epochs=1).snapshots)[0][1]
    assert a["x"] == b["x"] and a["y"] == b["y"]
    assert np.allclose(a["x"], c["x"], atol=1e-6)


# ------------------------------------------------------------------ engine


def test_reproducible():
    r1, r2 = run("linreg-numpy", seed=11, epochs=5), run("linreg-numpy", seed=11, epochs=5)
    assert r1.snapshots == r2.snapshots


@pytest.mark.parametrize("exp_id", list(EXPERIMENTS))
def test_snapshot_lines_point_at_the_marked_code(exp_id):
    exp = EXPERIMENTS[exp_id]
    result = run(exp_id, **({"epochs": 2} if any(p.name == "epochs" for p in exp.params) else {}))
    lines = result.source.splitlines()
    assert "@step" not in result.source
    expected = {
        "forward": ("w * x + b", "model(X)"),
        "backward": ("loss.backward()",),
        "update": ("learning_rate * d", "optimizer.step()"),
        "fit": ("model.fit(X, y)",),
        "zero_grad": ("optimizer.zero_grad()",),
        "evaluate": ("w * x + b", "model(X)"),
    }
    for s in result.snapshots:
        start, end = s["lines"]
        code = "\n".join(lines[start - 1 : end])
        for needle in expected.get(s["step"], ()):
            if needle in code:
                break
        else:
            assert s["step"] not in expected, f"{s['step']} -> {code!r}"


def test_run_limits_stop_execution_with_an_error():
    src = "total = 0\nfor i in range(100):\n    total = total + i\n"
    tmpl_groups = [StepGroup("add", 3, 3)]
    result = run_source(src, "<limit-test>", tmpl_groups, {"total": "total"}, max_snapshots=10)
    assert result.error is not None and "RunLimitExceeded" in result.error
    assert len(result.snapshots) == 10


def test_rendered_source_contains_parameters():
    exp = EXPERIMENTS["linreg-numpy"]
    _, result = exp.run({"learning_rate": 0.02, "epochs": 3})
    assert "learning_rate = 0.02\n" in result.source
    assert "epochs = 3\n" in result.source
    assert len(result.source.splitlines()) == len(exp.tmpl.source.splitlines())


def test_template_defaults_match_manifest():
    for exp in EXPERIMENTS.values():
        defaults = exp.tmpl.default_values()
        for p in exp.params:
            assert float(defaults[p.name]) == p.default, (exp.id, p.name)


@pytest.mark.parametrize(
    "params",
    [
        {"learning_rate": 5.0},
        {"epochs": 2.5},
        {"epochs": True},
        {"seed": -1},
        {"noise_std": float("nan")},
        {"__import__": 1},
    ],
)
def test_invalid_parameters_are_rejected(params):
    with pytest.raises(ParamError):
        EXPERIMENTS["linreg-numpy"].run(params)


# ------------------------------------------------------------------ HTTP API


def test_api_roundtrip():
    client = TestClient(app)
    listed = client.get("/experiments").json()
    assert {e["id"] for e in listed} == set(EXPERIMENTS)
    meta = client.get("/experiments/linreg-numpy").json()
    assert meta["param_lines"]["learning_rate"] > 0
    r = client.post("/experiments/linreg-numpy/run", json={"params": {"epochs": 3}})
    assert r.status_code == 200
    body = r.json()
    assert body["params"]["epochs"] == 3 and len(body["snapshots"]) == 2 + 4 * 3 + 1
    assert client.post("/experiments/linreg-numpy/run", json={"params": {"epochs": 999}}).status_code == 422
    assert client.post("/experiments/nope/run", json={}).status_code == 404
    assert client.post("/experiments/linreg-numpy/run", json={"params": {"x": "__import__('os')"}}).status_code == 422
    # integers too large for float conversion are rejected, not a 500
    assert client.post("/experiments/linreg-numpy/run", json={"params": {"epochs": 10**400}}).status_code == 422
