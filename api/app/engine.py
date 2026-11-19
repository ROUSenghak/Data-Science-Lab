"""Controlled execution of predefined experiment templates.

A template is an ordinary, runnable Python script with two conventions:

* Parameter lines of the form ``name = <number>`` (declared in the experiment
  manifest). Only validated numbers are substituted into them, so the executed
  source is exactly the source shown to the user.
* Trailing ``# @step <id>`` markers. Consecutive lines with the same id form a
  step group. When the last line of a group has executed, the tracer records a
  snapshot. Markers are stripped before execution; line numbers do not change.

Execution is traced with ``sys.settrace`` restricted to the template's own
module frame, so library code (NumPy, PyTorch, scikit-learn) runs untraced.
"""

from __future__ import annotations

import io
import math
import re
import sys
import threading
import time
import traceback
import types
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STEP_MARKER = re.compile(r"\s*#\s*@step\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
PARAM_LINE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*) = (?P<value>[-+0-9.eE]+)(?P<rest>\s*(#.*)?)$")

MAX_ARRAY_LEN = 2000  # e.g. a K-Means distance matrix: n <= 200 points x k <= 8 centroids
MAX_SNAPSHOTS = 5000
MAX_SECONDS = 10.0
_EXEC_LOCK = threading.Lock()


class ParamError(ValueError):
    pass


class RunLimitExceeded(Exception):
    """Raised inside the traced script when a run exceeds its time or snapshot budget."""


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str  # "int" | "float"
    min: float
    max: float
    step: float
    default: float
    description: str = ""

    def coerce(self, raw: Any) -> int | float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ParamError(f"{self.name}: expected a number")
        if isinstance(raw, int) and not (self.min <= raw <= self.max):
            # exact int comparison: float(raw) would overflow for huge integers
            raise ParamError(f"{self.name}: must be in [{self.min}, {self.max}]")
        if not math.isfinite(raw):
            raise ParamError(f"{self.name}: must be finite")
        if self.kind == "int":
            if float(raw) != int(raw):
                raise ParamError(f"{self.name}: expected an integer")
            value: int | float = int(raw)
        else:
            value = float(raw)
        if not (self.min <= value <= self.max):
            raise ParamError(f"{self.name}: must be in [{self.min}, {self.max}]")
        return value

    def literal(self, value: int | float) -> str:
        return str(int(value)) if self.kind == "int" else repr(float(value))


@dataclass(frozen=True)
class StepGroup:
    id: str
    start: int  # 1-based, inclusive
    end: int


@dataclass
class Template:
    source: str  # markers stripped, default parameter values
    lines: list[str]
    groups: list[StepGroup]
    param_lines: dict[str, int]  # param name -> 1-based line

    @classmethod
    def load(cls, path: Path, param_names: list[str]) -> "Template":
        raw_lines = path.read_text().splitlines()
        lines: list[str] = []
        marks: list[str | None] = []
        for line in raw_lines:
            m = STEP_MARKER.search(line)
            marks.append(m.group(1) if m else None)
            lines.append(line[: m.start()] if m else line)

        groups: list[StepGroup] = []
        i = 0
        while i < len(marks):
            if marks[i] is None:
                i += 1
                continue
            j = i
            while j + 1 < len(marks) and marks[j + 1] == marks[i]:
                j += 1
            groups.append(StepGroup(marks[i], i + 1, j + 1))
            i = j + 1

        param_lines: dict[str, int] = {}
        for idx, line in enumerate(lines):
            m = PARAM_LINE.match(line)
            if m and m.group("name") in param_names and m.group("name") not in param_lines:
                param_lines[m.group("name")] = idx + 1
        missing = set(param_names) - set(param_lines)
        if missing:
            raise ValueError(f"{path.name}: parameter lines not found for {sorted(missing)}")
        return cls("\n".join(lines) + "\n", lines, groups, param_lines)

    def render(self, values: dict[str, str]) -> str:
        lines = list(self.lines)
        for name, literal in values.items():
            idx = self.param_lines[name] - 1
            m = PARAM_LINE.match(lines[idx])
            assert m is not None
            lines[idx] = f"{name} = {literal}{m.group('rest')}"
        return "\n".join(lines) + "\n"

    def default_values(self) -> dict[str, str]:
        out = {}
        for name, line_no in self.param_lines.items():
            m = PARAM_LINE.match(self.lines[line_no - 1])
            assert m is not None
            out[name] = m.group("value")
        return out


# ---------------------------------------------------------------- serialization


def _num(v: float) -> float | None:
    """JSON-safe float: non-finite values (overflow, NaN) become None."""
    v = float(v)
    return v if math.isfinite(v) else None


def to_json(value: Any) -> Any:
    """Convert NumPy / PyTorch values to plain JSON data."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore[assignment]

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _num(value)
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size > MAX_ARRAY_LEN:
            raise ValueError("array too large to serialize")
        if value.ndim == 0:
            return to_json(value.item())
        return [to_json(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _fmt_array(a: np.ndarray) -> str:
    return np.array2string(a, precision=4, threshold=8, edgeitems=3, separator=", ")


def describe(value: Any) -> dict[str, Any] | None:
    """Short, debugger-style description of a variable (or None to hide it)."""
    import torch
    from torch import nn

    if isinstance(value, (types.ModuleType, types.FunctionType, type)):
        return None
    if isinstance(value, bool):
        return {"type": "bool", "value": repr(value)}
    if isinstance(value, (int, np.integer)):
        return {"type": type(value).__name__, "value": str(int(value))}
    if isinstance(value, (float, np.floating)):
        return {"type": type(value).__name__, "value": f"{float(value):.6g}"}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "value": _fmt_array(value)}
    if isinstance(value, torch.Tensor):
        d = {
            "type": "Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "value": _fmt_array(value.detach().cpu().numpy()),
        }
        if value.grad_fn is not None:
            d["grad_fn"] = type(value.grad_fn).__name__
        return d
    if isinstance(value, nn.Module):
        parts = []
        for name, p in value.named_parameters():
            g = "None" if p.grad is None else _fmt_array(p.grad.detach().numpy())
            parts.append(f"{name} = {_fmt_array(p.detach().numpy())}  (grad = {g})")
        return {"type": type(value).__name__, "value": repr(value) + ("\n" + "\n".join(parts) if parts else "")}
    if isinstance(value, torch.optim.Optimizer):
        lr = value.param_groups[0].get("lr")
        return {"type": type(value).__name__, "value": f"{type(value).__name__}(lr={lr})"}
    if hasattr(value, "get_params") and hasattr(value, "fit"):  # scikit-learn estimator
        fitted = {k: _fmt_array(np.asarray(getattr(value, k))) for k in ("coef_", "intercept_", "cluster_centers_", "inertia_", "n_iter_") if hasattr(value, k)}
        extra = "".join(f"\n{k} = {v}" for k, v in fitted.items())
        return {"type": type(value).__name__, "value": repr(value) + extra}
    if isinstance(value, (list, tuple)) and len(value) <= 8:
        return {"type": type(value).__name__, "value": repr(value)}
    if isinstance(value, range):
        return {"type": "range", "value": repr(value)}
    return None


# ---------------------------------------------------------------- tracing


def _diff(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cur.items() if k not in prev or prev[k] != v}


@dataclass
class _Tracer:
    code: types.CodeType
    ns: dict[str, Any]
    groups_by_end: dict[int, StepGroup]
    watches: dict[str, str]
    deadline: float
    max_snapshots: int
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    _pending: StepGroup | None = None
    _prev_state: dict[str, Any] = field(default_factory=dict)
    _prev_locals: dict[str, Any] = field(default_factory=dict)

    def global_trace(self, frame: types.FrameType, event: str, arg: Any):
        # Trace only the template's own module frame.
        return self.local_trace if frame.f_code is self.code else None

    def local_trace(self, frame: types.FrameType, event: str, arg: Any):
        if event in ("line", "return"):
            if time.perf_counter() > self.deadline:
                raise RunLimitExceeded(f"run exceeded {MAX_SECONDS:g} s")
            if self._pending is not None:
                self._record(self._pending)
                self._pending = None
            if event == "line":
                group = self.groups_by_end.get(frame.f_lineno)
                if group is not None:
                    self._pending = group  # snapshot once this line has executed
        return self.local_trace

    def _record(self, group: StepGroup) -> None:
        if len(self.snapshots) >= self.max_snapshots:
            raise RunLimitExceeded(f"more than {self.max_snapshots} snapshots")
        state: dict[str, Any] = {}
        for key, expr in self.watches.items():
            try:
                state[key] = to_json(eval(expr, self.ns))  # trusted manifest expression
            except Exception:
                state[key] = None
        local_view = {}
        for name, value in self.ns.items():
            if name.startswith("_"):
                continue
            d = describe(value)
            if d is not None:
                local_view[name] = d
        self.snapshots.append(
            {
                "index": len(self.snapshots),
                "step": group.id,
                "lines": [group.start, group.end],
                "state": _diff(self._prev_state, state),
                "locals": _diff(self._prev_locals, local_view),
                "removed": sorted(set(self._prev_locals) - set(local_view)),
            }
        )
        self._prev_state, self._prev_locals = state, local_view


@dataclass
class RunResult:
    source: str
    snapshots: list[dict[str, Any]]
    stdout: str
    warnings: list[str]
    error: str | None
    elapsed_ms: float


def run_source(
    source: str,
    filename: str,
    groups: list[StepGroup],
    watches: dict[str, str],
    max_snapshots: int = MAX_SNAPSHOTS,
) -> RunResult:
    """Execute a trusted template under the tracer.

    Limits are enforced from the trace function, i.e. between lines of the
    template. A single long library call cannot be interrupted this way; real
    isolation (a subprocess with CPU/memory limits) is required before any
    public deployment.
    """
    code = compile(source, filename, "exec")
    ns: dict[str, Any] = {"__name__": "__experiment__", "__builtins__": __builtins__}
    out = io.StringIO()
    error: str | None = None
    caught: list[str] = []

    with _EXEC_LOCK:
        t0 = time.perf_counter()
        tracer = _Tracer(code, ns, {g.end: g for g in groups}, watches, t0 + MAX_SECONDS, max_snapshots)
        with warnings.catch_warnings(record=True) as w, redirect_stdout(out):
            warnings.simplefilter("always")
            prev = sys.gettrace()
            sys.settrace(tracer.global_trace)
            try:
                exec(code, ns)
            except Exception as exc:  # report, keep snapshots recorded so far
                tb = traceback.extract_tb(exc.__traceback__)
                frames = [f for f in tb if f.filename == filename]
                where = f" (line {frames[-1].lineno})" if frames else ""
                error = f"{type(exc).__name__}: {exc}{where}"
            finally:
                sys.settrace(prev)
            seen = set()
            for item in w:
                msg = f"{item.category.__name__}: {item.message}"
                if msg not in seen:
                    seen.add(msg)
                    caught.append(msg)
        elapsed = (time.perf_counter() - t0) * 1000

    return RunResult(source, tracer.snapshots, out.getvalue(), caught, error, elapsed)
