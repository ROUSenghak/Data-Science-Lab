"""FastAPI service that executes trusted, predefined Python experiments.

There is no endpoint that accepts arbitrary code: clients choose an experiment
id and send numeric parameters, which are range-checked before substitution.
"""

from __future__ import annotations

import platform
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import sklearn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import ParamError
from .experiments import EXPERIMENTS

ENVIRONMENT = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "sklearn": sklearn.__version__,
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    for exp in EXPERIMENTS.values():  # validate templates at startup
        _ = exp.tmpl
    yield


app = FastAPI(title="Data Science Lab — execution service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


class RunRequest(BaseModel):
    params: dict[str, float | int] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "environment": ENVIRONMENT}


@app.get("/experiments")
def list_experiments() -> list[dict[str, Any]]:
    return [
        {k: e.describe()[k] for k in ("id", "lab", "framework", "title", "summary", "execution")}
        for e in EXPERIMENTS.values()
    ]


def _get(experiment_id: str):
    exp = EXPERIMENTS.get(experiment_id)
    if exp is None:
        raise HTTPException(404, f"unknown experiment '{experiment_id}'")
    return exp


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    return {**_get(experiment_id).describe(), "environment": ENVIRONMENT}


@app.post("/experiments/{experiment_id}/run")
def run_experiment(experiment_id: str, req: RunRequest) -> dict[str, Any]:
    exp = _get(experiment_id)
    try:
        params, result = exp.run(req.params)
    except ParamError as e:
        raise HTTPException(422, str(e)) from e
    return {
        "experiment_id": exp.id,
        "params": params,
        "source": result.source,
        "snapshots": result.snapshots,
        "stdout": result.stdout,
        "warnings": result.warnings,
        "error": result.error,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "environment": ENVIRONMENT,
    }
