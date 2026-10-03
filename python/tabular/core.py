"""Shared contract for the `tabular` graph kind: static value types, runtime values, op base class."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from graph_core.registry import Operation
from graph_core.types import Diagnostic, Fix, OpError

REPO = Path(__file__).resolve().parents[2]

# Value kinds (= edge kinds). A table wire and a statistical-result wire never mean the same thing.
TABLE, FIT_STATE, MODEL, METRICS, DISTRIBUTION, NUMBER, TAIL, TEST_RESULT = (
    "table", "fit_state", "model", "metrics", "distribution", "number", "tail", "test_result")
VALUE_KINDS = (TABLE, FIT_STATE, MODEL, METRICS, DISTRIBUTION, NUMBER, TAIL, TEST_RESULT)


# ------------------------------------------------------------------------------------------------ static types
@dataclass
class VType:
    """Static type of a value on a wire: its kind plus facts known without running (schema, partition, ...)."""

    kind: str
    info: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.info}

    # table helpers -----------------------------------------------------------
    @property
    def columns(self) -> list[dict[str, str]]:
        return self.info.get("columns", [])

    @property
    def col_names(self) -> list[str]:
        return [c["name"] for c in self.columns]

    def dtype_of(self, name: str) -> str | None:
        for c in self.columns:
            if c["name"] == name:
                return c["dtype"]
        return None

    @property
    def complete(self) -> bool:
        return bool(self.info.get("columnsComplete", True))

    @property
    def partition(self) -> str:
        return self.info.get("partition", "full")


def table_type(columns: list[dict[str, str]], rows: int | None, partition: str = "full", complete: bool = True,
               pending: list[str] | None = None, **extra: Any) -> VType:
    return VType(TABLE, {"columns": columns, "rows": rows, "partition": partition, "columnsComplete": complete,
                         "pending": pending or [], **extra})


# ------------------------------------------------------------------------------------------------ runtime values
class ExecutionError(Exception):
    """A run-time failure with a stable code (data-dependent problems that static validation cannot see)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class Table:
    df: pd.DataFrame  # index = row_id of the source file (kept through every step)
    partition: str = "full"  # full | train | validation
    lineage: dict[str, Any] = field(default_factory=dict)

    def derive(self, df: pd.DataFrame, **lineage: Any) -> "Table":
        return Table(df, self.partition, {**self.lineage, **lineage})


@dataclass
class FitState:
    transform: str  # standardize | onehot | impute
    transformer: Any  # the fitted scikit-learn object
    columns: list[str]
    fitted_on: dict[str, Any]
    details: dict[str, Any]


@dataclass
class FittedModel:
    task: str  # regression | classification
    estimator: Any
    features: list[str]
    target: str
    fitted_on: dict[str, Any]
    details: dict[str, Any]


@dataclass
class Plain:
    """A JSON-able result value (distribution, tail probability, test result, metrics, number)."""

    kind: str
    data: dict[str, Any]
    obj: Any = None  # live object, e.g. a frozen scipy distribution


@dataclass
class ExecCtx:
    node_id: str
    run_id: str | None = None
    graph_hash: str | None = None


# ------------------------------------------------------------------------------------------------ op base
class TabularOperation(Operation):
    graph_kind = "tabular"
    backend = "pandas"
    in_kinds: ClassVar[dict[str, str]] = {}
    out_kinds: ClassVar[dict[str, str]] = {}
    summary_kind: ClassVar[str] = "step"  # which inspector view the node's summary feeds
    stores_output: ClassVar[bool] = True

    def infer(self, cfg: Any, ins: dict[str, VType], node_id: str) -> dict[str, VType]:
        raise NotImplementedError

    def warnings(self, cfg: Any, ins: dict[str, VType], node_id: str) -> list[tuple[str, str, str | None]]:
        return []

    def execute(self, cfg: Any, ins: dict[str, Any], ctx: ExecCtx) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    def explain(self, cfg: Any, inputs: dict[str, VType], outputs: dict[str, VType]) -> dict[str, Any]:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ helpers
def dtype_name(s: pd.Series) -> str:
    k = s.dtype.kind
    if k in "iu":
        return "int"
    if k == "f":
        return "float"
    if k == "b":
        return "bool"
    if k == "M":
        return "datetime"
    return "string"


def schema_of(df: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": str(c), "dtype": dtype_name(df[c])} for c in df.columns]


def resolve_path(path: str) -> Path:
    """Relative paths resolve against the server's working directory, then against the repository root."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    alt = REPO / p
    return alt.resolve() if alt.exists() else p.resolve()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def row_ids_sha256(index: pd.Index) -> str:
    return sha256_bytes(",".join(str(i) for i in index).encode())


def clean(v: Any) -> Any:
    """Make a value strictly JSON-safe (numpy scalars -> python, NaN/inf -> None)."""
    import numpy as np

    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def dumps(v: Any) -> str:
    return json.dumps(clean(v), sort_keys=True, allow_nan=False)


def need_columns(t: VType, names: list[str], port: str, what: str) -> None:
    """Static check that columns exist in the (known part of the) schema."""
    have = set(t.col_names)
    prefixes = tuple(t.info.get("pendingPrefixes", []))  # columns that will appear once a one-hot encoder's categories are fitted
    missing = [n for n in names if n not in have and not (prefixes and n.startswith(prefixes))]
    if missing:
        raise OpError("E_COLUMN_NOT_FOUND", f"{what}: column(s) {missing} not found; the input has {sorted(have)}.", port,
                      [Fix(f"Use one of the existing columns: {sorted(have)}")])


def need_numeric(t: VType, names: list[str], port: str, what: str) -> None:
    bad = [n for n in names if t.dtype_of(n) not in (None, "int", "float")]
    if bad:
        raise OpError("E_COLUMN_TYPE", f"{what}: column(s) {bad} are not numeric ({[t.dtype_of(b) for b in bad]}). "
                      "Cast them in a column selection, or encode them (one-hot) first.", port)


def leakage_check(t: VType, port: str, what: str) -> None:
    """A05: preprocessing and estimators may only be fitted on the training partition."""
    part = t.partition
    if part == "train":
        return
    if part == "validation":
        split = t.info.get("splitNode") or "the split node"
        raise OpError(
            "E_LEAKAGE_FIT_ON_HELDOUT",
            f"{what} would be fitted on the held-out validation partition (from '{split}'). Statistics learned from held-out data leak "
            "information into the model and make validation results optimistic.",
            port, [Fix(f"Connect the training output '{split}.train' (or a table derived from it) to this input; apply the fitted "
                       "state to the validation partition with an 'Apply fitted transform' node")])
    raise OpError(
        "E_LEAKAGE_FIT_BEFORE_SPLIT",
        f"{what} is fitted on a table that has not been partitioned, so its statistics would include rows that later become validation data.",
        port, [Fix("Insert a train/validation split before this node and connect its 'train' output here")])


def source_diag(code: str, message: str, node: str, port: str | None = None) -> Diagnostic:
    return Diagnostic(code, message, "error", node, port)
