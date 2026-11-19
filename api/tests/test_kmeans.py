"""Numerical correctness of the K-Means experiments.

Each check recomputes the traced quantities independently from the recorded
snapshots. Tests marked "guarantee" verify mathematical facts that hold for every
input; tests marked "empirical" document observed behaviour on fixed seeds.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans

from app.engine import ParamError, run_source
from app.experiments import EXPERIMENTS
from app.main import app

REL = 1e-12  # float64 recomputation tolerance


def fold(snapshots):
    state, out = {}, []
    for s in snapshots:
        prev = state
        state = {**state, **s["state"]}
        out.append((s, state, prev))
    return out


def run(exp_id, **params):
    _, result = EXPERIMENTS[exp_id].run(params)
    assert result.error is None, result.error
    return result


def sqdist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def lloyd_record(states):
    """Per-iteration record: centroids before, labels, J after assignment, centroids after, J after update."""
    it, out = None, []
    for snap, st, prev in states:
        if snap["step"] == "distances":
            it = {"C_before": np.array(st["centroids"]), "sq_dist": np.array(st["sq_dist"])}
        elif snap["step"] == "assign":
            it.update(labels=np.array(st["labels"]), J_assign=st["J"], n_changed=st["n_changed"], prev_labels=np.array(st["prev_labels"]))
        elif snap["step"] == "update":
            it.update(C_after=np.array(st["centroids"]), counts=np.array(st["counts"]), sums=np.array(st["sums"]), shift=np.array(st["shift"]))
        elif snap["step"] == "objective":
            it["J_update"] = st["J"]
            out.append(it)
    return out


CONFIGS = [
    dict(seed=s, init_method=m, init_seed=i, k=k, n_blobs=b, cluster_std=std)
    for (s, m, i, k, b, std) in [
        (0, 0, 4, 3, 3, 1.0),
        (28, 0, 3, 3, 3, 1.0),  # converges to a poor local minimum
        (5, 1, 0, 4, 3, 1.5),
        (7, 2, 1, 5, 4, 0.7),
        (0, 2, 0, 5, 3, 1.0),  # produces empty clusters
        (11, 0, 2, 8, 6, 2.5),
        (3, 1, 9, 1, 2, 1.0),
        (19, 0, 5, 6, 2, 3.0),
    ]
]


# ------------------------------------------------------------------ guarantees: each step is the formula


@pytest.mark.parametrize("cfg", CONFIGS)
def test_traced_steps_match_the_formulas(cfg):
    states = fold(run("kmeans-numpy", **cfg).snapshots)
    X = np.array(states[0][1]["X"])
    n, k = len(X), cfg["k"]
    iters = lloyd_record(states)
    assert iters, "no iteration recorded"
    for t, it in enumerate(iters):
        C, labels = it["C_before"], it["labels"]
        # distances
        assert np.allclose(it["sq_dist"], sqdist(X, C), rtol=REL, atol=0)
        # assignment: nearest centroid, ties to the lowest index
        D = sqdist(X, C)
        assert np.array_equal(labels, np.argmin(D, axis=1))
        assert np.all(D[np.arange(n), labels] <= D.min(axis=1))
        # J after assignment uses the old centroids
        assert it["J_assign"] == pytest.approx(D.min(axis=1).sum(), rel=REL)
        assert it["n_changed"] == int(np.sum(labels != it["prev_labels"]))
        if t == 0:
            assert np.all(it["prev_labels"] == -1)
        # update: mean of members, empty clusters keep their centroid
        for j in range(k):
            members = X[labels == j]
            assert it["counts"][j] == len(members)
            if len(members):
                assert np.allclose(it["sums"][j], members.sum(axis=0), rtol=REL, atol=1e-12)
                assert np.allclose(it["C_after"][j], members.mean(axis=0), rtol=1e-12, atol=1e-12)
            else:
                assert np.array_equal(it["C_after"][j], C[j])
        assert np.allclose(it["shift"], np.linalg.norm(it["C_after"] - C, axis=1), rtol=REL, atol=1e-12)
        # J after update: within-cluster sum of squares of the new centroids
        assert it["J_update"] == pytest.approx(((X - it["C_after"][labels]) ** 2).sum(), rel=REL)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_objective_never_increases(cfg):
    """Guarantee: argmin minimises J over labels, the mean minimises J over centroids."""
    iters = lloyd_record(fold(run("kmeans-numpy", **cfg).snapshots))
    seq = list(itertools.chain.from_iterable((it["J_assign"], it["J_update"]) for it in iters))
    slack = 1e-12 * max(seq)
    assert all(b <= a + slack for a, b in zip(seq, seq[1:])), seq


@pytest.mark.parametrize("cfg", CONFIGS)
def test_stopping_rule_and_final_state(cfg):
    result = run("kmeans-numpy", **cfg)
    states = fold(result.snapshots)
    iters = lloyd_record(states)
    final = states[-1][1]
    X = np.array(final["X"])
    C = np.array(final["centroids"])
    assert states[-1][0]["step"] == "evaluate"
    if len(iters) < 20:  # stopped before max_iter: the last iteration changed nothing
        assert iters[-1]["n_changed"] == 0
        assert np.all(iters[-1]["shift"] == 0)
        assert all(it["n_changed"] > 0 for it in iters[:-1])
        assert final["inertia"] == pytest.approx(iters[-1]["J_update"], rel=REL)
    assert np.array_equal(final["labels"], np.argmin(sqdist(X, C), axis=1))
    assert final["inertia"] == pytest.approx(sqdist(X, C).min(axis=1).sum(), rel=REL)
    assert f"stopped after {len(iters)} iterations" in result.stdout


def test_hitting_max_iter_reevaluates_labels():
    # One iteration from a poor start: the final evaluate step re-assigns to the returned centroids.
    states = fold(run("kmeans-numpy", seed=28, init_method=0, init_seed=3, max_iter=1).snapshots)
    it = lloyd_record(states)[0]
    final = states[-1][1]
    assert it["n_changed"] > 0
    assert final["inertia"] <= it["J_update"] + 1e-12


def test_empty_cluster_policy_is_applied():
    states = fold(run("kmeans-numpy", seed=0, init_method=2, init_seed=0, k=5).snapshots)
    iters = lloyd_record(states)
    empty = [(t, j) for t, it in enumerate(iters) for j in range(5) if it["counts"][j] == 0]
    assert empty, "this configuration is expected to produce an empty cluster"
    for t, j in empty:
        assert np.array_equal(iters[t]["C_after"][j], iters[t]["C_before"][j])
        assert iters[t]["shift"][j] == 0


def test_duplicate_points_and_coincident_centroids():
    """Duplicates make two initial centroids coincide; the tie goes to the lower index and the
    other cluster stays empty. Nothing becomes NaN and the objective still never increases."""
    exp = EXPERIMENTS["kmeans-numpy"]
    values = exp.resolve({"n_samples": 10, "k": 3, "init_method": 0, "init_seed": 0})
    source = exp.tmpl.render({p.name: p.literal(values[p.name]) for p in exp.params})
    data_line = exp.tmpl.lines[exp.tmpl.groups[0].end - 1]
    assert data_line.startswith("X = ")
    duplicate = "X = np.repeat(np.array([[0.0, 0.0], [4.0, 1.0]]), [6, 4], axis=0)"
    source = source.replace(data_line, duplicate)
    result = run_source(source, "<dup>", exp.tmpl.groups, exp.watches)
    assert result.error is None, result.error
    states = fold(result.snapshots)
    X = np.array(states[0][1]["X"])
    C0 = np.array(states[1][1]["centroids_init"])
    assert len({tuple(c) for c in C0}) < 3  # pigeonhole: 3 picks from 2 distinct locations
    first = lloyd_record(states)[0]
    D = sqdist(X, C0)
    # ties resolved to the lowest index
    for i in range(len(X)):
        assert first["labels"][i] == min(np.flatnonzero(D[i] == D[i].min()))
    assert 0 in first["counts"]
    final = states[-1][1]
    assert np.all(np.isfinite(final["centroids"]))
    seq = [v for it in lloyd_record(states) for v in (it["J_assign"], it["J_update"])]
    assert all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))


# ------------------------------------------------------------------ reproducibility


def test_fixed_seeds_are_reproducible():
    a = run("kmeans-numpy", seed=3, init_method=1, init_seed=8)
    b = run("kmeans-numpy", seed=3, init_method=1, init_seed=8)
    assert a.snapshots == b.snapshots
    c = run("kmeans-numpy", seed=3, init_method=1, init_seed=9)
    assert c.snapshots[0]["state"]["X"] == a.snapshots[0]["state"]["X"]  # init seed leaves the data alone
    assert c.snapshots[1]["state"]["centroids_init"] != a.snapshots[1]["state"]["centroids_init"]
    s1 = run("kmeans-sklearn", random_state=4, n_init=3)
    s2 = run("kmeans-sklearn", random_state=4, n_init=3)
    assert s1.snapshots == s2.snapshots


def test_same_seed_gives_same_data_in_both_implementations():
    a = fold(run("kmeans-numpy", seed=42, n_samples=80, n_blobs=4).snapshots)[0][1]
    b = fold(run("kmeans-sklearn", seed=42, n_samples=80, n_blobs=4).snapshots)[0][1]
    assert a["X"] == b["X"]


# ------------------------------------------------------------------ agreement with scikit-learn


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c["init_method"] != 2] + [dict(seed=s, init_method=1, init_seed=s, k=4, n_blobs=4, cluster_std=1.2) for s in range(6)])
def test_numpy_matches_sklearn_with_same_initialisation(cfg):
    """With the same initial centroids, tol=0 and the same max_iter, scikit-learn's Lloyd
    iterations give the same labels, centroids, inertia and iteration count."""
    states = fold(run("kmeans-numpy", **cfg).snapshots)
    iters = lloyd_record(states)
    if any(0 in it["counts"] for it in iters):
        pytest.skip("empty clusters: scikit-learn relocates them, our policy keeps them")
    X = np.array(states[0][1]["X"])
    C0 = np.array(states[1][1]["centroids_init"])
    final = states[-1][1]
    km = KMeans(n_clusters=cfg["k"], init=C0, n_init=1, max_iter=20, tol=0, algorithm="lloyd").fit(X)
    assert np.array_equal(km.labels_, final["labels"])
    assert np.allclose(km.cluster_centers_, final["centroids"], rtol=1e-10, atol=1e-10)
    assert km.inertia_ == pytest.approx(final["inertia"], rel=1e-10)
    assert km.n_iter_ == len(iters)


def test_sklearn_template_reports_consistent_attributes():
    states = fold(run("kmeans-sklearn", seed=9, k=4, n_blobs=4, n_init=5).snapshots)
    st = states[-1][1]
    X, C = np.array(st["X"]), np.array(st["centroids"])
    assert np.array_equal(st["check_labels"], st["labels"])
    assert st["check_J"] == pytest.approx(st["inertia"], rel=1e-10)
    assert np.allclose(np.array(st["dist"]) ** 2, sqdist(X, C), rtol=1e-9, atol=1e-9)
    assert np.allclose(st["sq_dist"], sqdist(X, C), rtol=REL)
    assert 1 <= st["n_iter"] <= 20
    # cluster_centers_ are the means of their members (strict convergence, no empty cluster)
    for j in range(4):
        assert np.allclose(C[j], X[np.array(st["labels"]) == j].mean(axis=0), atol=1e-9)


def test_empirical_restarts_escape_a_poor_local_minimum():
    """Empirical (seed-specific), not a guarantee: one random start is stuck; ten restarts are not."""
    poor = fold(run("kmeans-numpy", seed=28, init_method=0, init_seed=3).snapshots)[-1][1]["inertia"]
    best = fold(run("kmeans-sklearn", seed=28, init_method=1, n_init=10).snapshots)[-1][1]["inertia"]
    assert best < 0.5 * poor


# ------------------------------------------------------------------ engine integration and validation


@pytest.mark.parametrize("exp_id", ["kmeans-numpy", "kmeans-sklearn"])
def test_snapshot_lines_point_at_the_marked_code(exp_id):
    result = run(exp_id)
    lines = result.source.splitlines()
    expected = {
        "distances": "np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)",
        "assign": "np.argmin(sq_dist, axis=1)",
        "update": "sums[counts > 0] / counts[counts > 0, None]",
        "objective": "J = np.sum((X - centroids[labels]) ** 2)",
        "fit": "model.fit(X)",
        "transform": "model.transform(X)",
    }
    seen = set()
    for s in result.snapshots:
        start, end = s["lines"]
        code = "\n".join(lines[start - 1 : end])
        if s["step"] in expected:
            assert expected[s["step"]] in code, (s["step"], code)
            seen.add(s["step"])
    assert seen == {k for k in expected if k in {g.id for g in EXPERIMENTS[exp_id].tmpl.groups}}


def test_snapshot_sequence_per_iteration():
    snaps = run("kmeans-numpy").snapshots
    steps = [s["step"] for s in snaps]
    assert steps[:2] == ["data", "init"] and steps[-1] == "evaluate"
    body = steps[2:-1]
    assert len(body) % 4 == 0
    assert all(body[i : i + 4] == ["distances", "assign", "update", "objective"] for i in range(0, len(body), 4))


@pytest.mark.parametrize(
    "exp_id, params",
    [
        ("kmeans-numpy", {"k": 0}),
        ("kmeans-numpy", {"k": 9}),
        ("kmeans-numpy", {"k": 2.5}),
        ("kmeans-numpy", {"init_method": 3}),
        ("kmeans-numpy", {"n_samples": 5}),
        ("kmeans-numpy", {"cluster_std": 0.0}),
        ("kmeans-numpy", {"max_iter": 0}),
        ("kmeans-numpy", {"cluster_std": float("nan")}),
        ("kmeans-numpy", {"labels": 1}),
        ("kmeans-sklearn", {"init_method": 2}),
        ("kmeans-sklearn", {"n_init": 0}),
        ("kmeans-sklearn", {"tol": -1e-3}),
        ("kmeans-sklearn", {"k": True}),
    ],
)
def test_invalid_parameters_are_rejected(exp_id, params):
    with pytest.raises(ParamError):
        EXPERIMENTS[exp_id].run(params)


def test_largest_run_fits_the_serialization_limits():
    result = run("kmeans-numpy", n_samples=200, k=8, n_blobs=6, cluster_std=3.0, max_iter=50, init_method=2)
    st = fold(result.snapshots)[-1][1]
    assert len(st["X"]) == 200 and st["sq_dist"] is not None and len(st["sq_dist"][0]) == 8


def test_api_roundtrip_kmeans():
    client = TestClient(app)
    r = client.post("/experiments/kmeans-numpy/run", json={"params": {"k": 4, "init_method": 1}})
    assert r.status_code == 200 and r.json()["error"] is None
    assert client.post("/experiments/kmeans-numpy/run", json={"params": {"k": 20}}).status_code == 422
    meta = client.get("/experiments/kmeans-sklearn").json()
    assert meta["param_lines"]["n_init"] > 0 and meta["lab"] == "k-means"
