"""Registry of trusted, predefined experiments.

To add an experiment: write a runnable template in ``templates/``, mark its
computational steps with ``# @step <id>``, and register an ``Experiment`` here
with its parameters and the *watch* expressions that map the script's own
variables onto the lab's canonical state (used by visualizations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from ..engine import Param, ParamError, RunResult, Template, run_source

TEMPLATES = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class StepInfo:
    id: str
    title: str
    summary: str = ""  # one line: what this step computes


@dataclass
class Experiment:
    id: str
    lab: str
    framework: str  # "numpy" | "pytorch" | "sklearn"
    title: str
    summary: str
    execution: str  # "educational-implementation" | "library"
    template: str
    params: list[Param]
    steps: list[StepInfo]
    watches: dict[str, str]
    notes: list[str] = field(default_factory=list)

    @cached_property
    def tmpl(self) -> Template:
        return Template.load(TEMPLATES / self.template, [p.name for p in self.params])

    def resolve(self, raw: dict[str, Any]) -> dict[str, int | float]:
        known = {p.name: p for p in self.params}
        unknown = set(raw) - set(known)
        if unknown:
            raise ParamError(f"unknown parameters: {sorted(unknown)}")
        return {p.name: p.coerce(raw.get(p.name, p.default)) for p in self.params}

    def run(self, raw_params: dict[str, Any]) -> tuple[dict[str, int | float], RunResult]:
        values = self.resolve(raw_params)
        literals = {p.name: p.literal(values[p.name]) for p in self.params}
        source = self.tmpl.render(literals)
        result = run_source(source, f"<experiment:{self.id}>", self.tmpl.groups, self.watches)
        return values, result

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lab": self.lab,
            "framework": self.framework,
            "title": self.title,
            "summary": self.summary,
            "execution": self.execution,
            "notes": self.notes,
            "source": self.tmpl.source,
            "param_lines": self.tmpl.param_lines,
            "params": [p.__dict__ for p in self.params],
            "steps": [s.__dict__ for s in self.steps],
            "step_groups": [g.__dict__ for g in self.tmpl.groups],
        }


def _data_params(n_max: int = 100) -> list[Param]:
    return [
        Param("seed", "Random seed", "int", 0, 9999, 1, 0, "Seed for NumPy's default_rng; fixes the dataset."),
        Param("n_samples", "Samples n", "int", 5, n_max, 1, 30, "Number of (x, y) points."),
        Param("noise_std", "Noise σ", "float", 0.0, 5.0, 0.1, 1.0, "Standard deviation of Gaussian noise."),
    ]


def _gd_params() -> list[Param]:
    return [
        Param("learning_rate", "Learning rate η", "float", 0.0001, 0.5, 0.0001, 0.08, "Step size of gradient descent."),
        Param("epochs", "Epochs", "int", 1, 200, 1, 100, "Full-batch gradient steps."),
        Param("w_init", "Initial w", "float", -10.0, 10.0, 0.1, 0.0, ""),
        Param("b_init", "Initial b", "float", -10.0, 10.0, 0.1, 0.0, ""),
    ]


EXPERIMENTS: dict[str, Experiment] = {}


def _register(e: Experiment) -> None:
    EXPERIMENTS[e.id] = e


_register(
    Experiment(
        id="linreg-numpy",
        lab="linear-regression",
        framework="numpy",
        title="Gradient descent from scratch",
        summary="Full-batch gradient descent on the mean squared error, written directly in NumPy.",
        execution="educational-implementation",
        template="linreg_numpy.py",
        params=_data_params() + _gd_params(),
        steps=[
            StepInfo("data", "Generate data", "Draw x ~ U(0, 4) and y = 2x + 1 + noise with a fixed seed."),
            StepInfo("init", "Initialise parameters", "Set the starting point (w₀, b₀)."),
            StepInfo("forward", "Forward pass", "Predict ŷ = w·x + b for every point."),
            StepInfo("loss", "Loss", "Mean of squared residuals ŷ − y."),
            StepInfo("gradient", "Gradients", "∂L/∂w and ∂L/∂b from the hand-derived formulas."),
            StepInfo("update", "Parameter update", "Step against the gradient: θ ← θ − η∇L."),
            StepInfo("evaluate", "Final loss", "Loss of the trained parameters θ_T, after the last update."),
        ],
        watches={
            "x": "x",
            "y": "y",
            "w": "w",
            "b": "b",
            "y_hat": "y_hat",
            "loss": "loss",
            "dw": "dw",
            "db": "db",
            "epoch": "epoch",
            "lr": "learning_rate",
        },
        notes=[
            "Every operation is visible: this is our own implementation, not a library optimizer.",
            "Gradients are derived by hand: ∂L/∂w = 2·mean(r·x), ∂L/∂b = 2·mean(r).",
        ],
    )
)

_register(
    Experiment(
        id="linreg-pytorch",
        lab="linear-regression",
        framework="pytorch",
        title="nn.Linear + autograd + SGD",
        summary="The same model trained with PyTorch: nn.Linear, nn.MSELoss, loss.backward() and torch.optim.SGD.",
        execution="library",
        template="linreg_pytorch.py",
        params=_data_params() + _gd_params(),
        steps=[
            StepInfo("data", "Tensors", "Same NumPy data, copied to float32 tensors of shape (n, 1)."),
            StepInfo("init", "Initialise nn.Linear", "Overwrite the layer's random weight and bias with (w₀, b₀)."),
            StepInfo("zero_grad", "optimizer.zero_grad()", "Clear .grad; backward() would otherwise accumulate into it."),
            StepInfo("forward", "Forward pass", "model(X) computes X·Wᵀ + b and records the graph for autograd."),
            StepInfo("loss", "Loss", "nn.MSELoss(): mean of squared residuals, added to the graph."),
            StepInfo("backward", "loss.backward()", "Autograd fills weight.grad and bias.grad."),
            StepInfo("update", "optimizer.step()", "SGD updates the parameters in place: θ ← θ − η·grad."),
            StepInfo("evaluate", "Final loss (no_grad)", "Loss of the trained parameters θ_T, without recording a graph."),
        ],
        watches={
            "x": "X.squeeze(1)",
            "y": "y.squeeze(1)",
            "w": "model.weight.item()",
            "b": "model.bias.item()",
            "y_hat": "y_hat.squeeze(1)",
            "loss": "loss.item()",
            "dw": "model.weight.grad.item()",
            "db": "model.bias.grad.item()",
            "epoch": "epoch",
            "lr": "optimizer.param_groups[0]['lr']",
        },
        notes=[
            "Gradients are computed by PyTorch autograd (loss.backward()) and read from .grad.",
            "Parameters are updated by torch.optim.SGD; values are float32.",
        ],
    )
)

_register(
    Experiment(
        id="linreg-sklearn",
        lab="linear-regression",
        framework="sklearn",
        title="LinearRegression (closed form)",
        summary="scikit-learn's LinearRegression solves least squares directly, without gradient descent.",
        execution="library",
        template="linreg_sklearn.py",
        params=_data_params(),
        steps=[
            StepInfo("data", "Data and design matrix", "Same data; X reshaped to (n, 1)."),
            StepInfo("fit", "model.fit(X, y)", "One untraced library call: centred least squares via scipy.linalg.lstsq."),
            StepInfo("predict", "model.predict(X)", "Evaluate ŷ = X·coef_ + intercept_."),
            StepInfo("loss", "Mean squared error", "Training MSE of the fitted line."),
            StepInfo("check", "Normal-equations check", "Our own solve of AᵀAθ = Aᵀy, independent of scikit-learn."),
        ],
        watches={
            "x": "x",
            "y": "y",
            "w": "float(model.coef_[0])",
            "b": "float(model.intercept_)",
            "y_hat": "y_hat",
            "loss": "mse",
            "theta": "theta",
            "rank": "model.rank_",
            "singular": "model.singular_",
        },
        notes=[
            "model.fit() is a single library call: its internal computations are not traced here.",
            "LinearRegression centres X and y, then calls scipy.linalg.lstsq; it exposes coef_, intercept_, rank_ and singular_ (singular values of the centred X).",
            "The final step re-derives the solution with our own NumPy code as an independent check.",
        ],
    )
)


# ---------------------------------------------------------------- K-Means


def _cluster_data_params() -> list[Param]:
    return [
        Param("seed", "Random seed", "int", 0, 9999, 1, 0, "Seed for NumPy's default_rng; fixes the dataset."),
        Param("n_samples", "Samples n", "int", 10, 200, 1, 60, "Number of 2-D points."),
        Param("n_blobs", "True blobs", "int", 1, 6, 1, 3, "Number of Gaussian blobs that generate the data."),
        Param("cluster_std", "Blob σ", "float", 0.1, 3.0, 0.1, 1.0, "Standard deviation of each blob."),
        Param("k", "Clusters K", "int", 1, 8, 1, 3, "Number of centroids."),
    ]


_KMEANS_WATCHES_DATA = {"X": "X", "k": "k"}

_register(
    Experiment(
        id="kmeans-numpy",
        lab="k-means",
        framework="numpy",
        title="Lloyd's algorithm from scratch",
        summary="K-Means written directly in NumPy: explicit distances, assignments, centroid means and objective.",
        execution="educational-implementation",
        template="kmeans_numpy.py",
        params=_cluster_data_params()
        + [
            Param("init_method", "Initialization", "int", 0, 2, 1, 0, "0: random data points, 1: k-means++, 2: uniform in the bounding box."),
            Param("init_seed", "Init seed", "int", 0, 9999, 1, 4, "Seed of the initialization only; the data stay the same."),
            Param("max_iter", "Max iterations", "int", 1, 50, 1, 20, "Upper bound on Lloyd iterations."),
        ],
        steps=[
            StepInfo("data", "Generate data", "Gaussian blobs around random centres, fixed seed."),
            StepInfo("init", "Initialise centroids", "Pick the starting centroids μ⁽⁰⁾ (random points, k-means++ or uniform)."),
            StepInfo("distances", "Distances", "Squared Euclidean distance from every point to every centroid."),
            StepInfo("assign", "Assignment", "Each point joins its nearest centroid; J is evaluated with the old centroids."),
            StepInfo("update", "Centroid update", "Each centroid moves to the mean of its points; empty clusters keep theirs."),
            StepInfo("objective", "Objective J", "Within-cluster sum of squares after the update: the iteration is complete."),
            StepInfo("evaluate", "Final labels and inertia", "Labels and inertia of the returned centroids."),
        ],
        watches={
            **_KMEANS_WATCHES_DATA,
            "centroids": "centroids",
            "centroids_init": "centroids_init",
            "labels": "labels",
            "prev_labels": "prev_labels",
            "sq_dist": "sq_dist",
            "J": "J",
            "counts": "counts",
            "sums": "sums",
            "shift": "shift",
            "n_changed": "n_changed",
            "iteration": "iteration",
            "inertia": "inertia",
            "init_method": "init_method",
        },
        notes=[
            "Every operation is visible: this is our own implementation, not a library call.",
            "Ties in argmin go to the lowest centroid index. An empty cluster keeps its previous centroid (scikit-learn instead relocates it to a far-away point).",
            "Stopping rule as in scikit-learn with tol = 0: stop after the first iteration in which no point changes cluster.",
        ],
    )
)

_register(
    Experiment(
        id="kmeans-sklearn",
        lab="k-means",
        framework="sklearn",
        title="sklearn.cluster.KMeans",
        summary="The same data clustered by scikit-learn's KMeans; only its public inputs and fitted attributes are shown.",
        execution="library",
        template="kmeans_sklearn.py",
        params=_cluster_data_params()
        + [
            Param("init_method", "Initialization", "int", 0, 1, 1, 1, '0: "random" (k data points), 1: "k-means++" (greedy).'),
            Param("n_init", "Restarts n_init", "int", 1, 20, 1, 1, "Independent initializations; the run with the lowest inertia is kept."),
            Param("random_state", "random_state", "int", 0, 9999, 1, 0, "Seed of scikit-learn's initialization."),
            Param("max_iter", "Max iterations", "int", 1, 300, 1, 20, "Upper bound on Lloyd iterations per run."),
            Param("tol", "Tolerance", "float", 0.0, 0.01, 0.0001, 0.0001, "Stop when the squared centroid shift falls below tol × mean variance of X."),
        ],
        steps=[
            StepInfo("data", "Data", "Same blobs as the NumPy example (same seed, same values)."),
            StepInfo("model", "Configure KMeans", "Hyper-parameters only; nothing is computed yet."),
            StepInfo("fit", "model.fit(X)", "One untraced library call; read cluster_centers_, labels_, inertia_, n_iter_."),
            StepInfo("transform", "model.transform(X)", "Public API: Euclidean distance from every point to every centre."),
            StepInfo("check", "Independent check", "Our own NumPy argmin and objective, compared with the fitted attributes."),
        ],
        watches={
            **_KMEANS_WATCHES_DATA,
            "centroids": "model.cluster_centers_",
            "labels": "model.labels_",
            "inertia": "model.inertia_",
            "n_iter": "model.n_iter_",
            "dist": "dist",
            "sq_dist": "sq_dist",
            "check_labels": "check_labels",
            "check_J": "check_J",
            "init_method": "init_method",
            "n_init": "n_init",
        },
        notes=[
            "model.fit() is a single library call: its iterations are not traced here.",
            "KMeans exposes cluster_centers_, labels_, inertia_ and n_iter_ (iterations of the best run). transform() gives Euclidean distances to the centres.",
            'scikit-learn\'s "k-means++" is the greedy variant (several candidates per step); empty clusters are relocated to far-away points.',
        ],
    )
)
