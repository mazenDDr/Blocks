"""tabular.* operations: table source, profiling, cleaning, partitioning, fit/apply preprocessing, predictions export.

Every op declares ports (typed wires), a config schema, schema inference (no data is touched beyond the CSV header/dtypes of a
source), an executor on native pandas / scikit-learn objects and an explanation. Fit state is owned by the training partition:
fit nodes only accept a table whose partition is `train` (rule E_LEAKAGE_*), apply nodes take any partition."""
from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from graph_core.registry import register
from graph_core.types import Fix, OpError
from tabular.core import (FIT_STATE, TABLE, ExecCtx, ExecutionError, FitState, Table, TabularOperation, VType, clean, dtype_name,
                          leakage_check, need_columns, need_numeric, resolve_path, row_ids_sha256, schema_of, sha256_bytes, table_type)

from ._common import StrictConfig

NUMERIC = ("int", "float")
MAX_STATIC_BYTES = 50_000_000


def sample_rows(df: pd.DataFrame, n: int = 5) -> dict[str, Any]:
    head = df.head(n)
    return {"columns": [str(c) for c in head.columns], "rowIds": [int(i) for i in head.index],
            "rows": [[clean(v) for v in row] for row in head.itertuples(index=False, name=None)]}


def _same_table(t: VType, **upd: Any) -> VType:
    info = {**t.info, **upd}
    return VType(TABLE, info)


# ================================================================================================ csv source
class CsvSourceConfig(StrictConfig):
    path: str = Field("", description="Local CSV file. Relative paths are resolved against the server's working directory, then the repository root.")
    delimiter: str = Field(",", min_length=1, max_length=1)
    na_values: list[str] = Field(default_factory=list, title="extra missing-value strings")


@lru_cache(maxsize=8)
def _static_source(path: str, mtime: int, size: int, delimiter: str, na: tuple[str, ...]):
    raw = open(path, "rb").read()
    exact = size <= MAX_STATIC_BYTES
    df = pd.read_csv(io.BytesIO(raw), sep=delimiter, na_values=list(na) or None, nrows=None if exact else 20000)
    return {"schema": schema_of(df), "rows": len(df) if exact else None, "sha256": sha256_bytes(raw), "bytes": size}


def read_source(cfg: CsvSourceConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = resolve_path(cfg.path)
    if not p.is_file():
        raise ExecutionError("E_SOURCE_NOT_FOUND", f"CSV file '{cfg.path}' does not exist.")
    raw = p.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), sep=cfg.delimiter, na_values=list(cfg.na_values) or None)
    df.index.name = "row_id"
    return df, {"path": str(p), "configuredPath": cfg.path, "sha256": sha256_bytes(raw), "bytes": len(raw), "rows": len(df), "columns": len(df.columns)}


@register
class CsvSource(TabularOperation):
    type = "tabular.csv_source"
    backend = "pandas"
    inputs = ()
    outputs = ("table",)
    out_kinds = {"table": TABLE}
    Config = CsvSourceConfig
    summary_kind = "source"

    def infer(self, cfg, ins, node_id):
        if not cfg.path:
            raise OpError("E_SOURCE_NOT_SET", "No CSV file chosen.", None, [Fix("Set 'path' to a local CSV file")])
        p = resolve_path(cfg.path)
        if not p.is_file():
            raise OpError("E_SOURCE_NOT_FOUND", f"CSV file '{cfg.path}' was not found (looked for {p}).", None, [Fix("Check the path")])
        try:
            st = os.stat(p)
            s = _static_source(str(p), st.st_mtime_ns, st.st_size, cfg.delimiter, tuple(cfg.na_values))
        except Exception as e:  # noqa: BLE001
            raise OpError("E_SOURCE_UNREADABLE", f"Cannot parse '{cfg.path}' as CSV: {e}")
        return {"table": table_type(s["schema"], s["rows"], "full", True, sourceSha256=s["sha256"], sourceBytes=s["bytes"],
                                    rowsExact=s["rows"] is not None, sourceNode=node_id)}

    def execute(self, cfg, ins, ctx):
        df, prov = read_source(cfg)
        t = Table(df, "full", {"source": {"node": ctx.node_id, **prov}})
        return {"table": t}, {**prov, "schema": schema_of(df), "preview": sample_rows(df)}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "table = pandas.read_csv(path)", "note": "The source file is never modified. Its SHA-256 is recorded with every run; cleaning produces derived tables.",
                "rule": "column types follow pandas' CSV inference; row ids are the 0-based row positions in the file"}


# ================================================================================================ profile
class ProfileConfig(StrictConfig):
    top_k: int = Field(5, ge=1, le=50, title="top categories shown")


def profile_table(df: pd.DataFrame, top_k: int) -> dict[str, Any]:
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        dt = dtype_name(s)
        miss, uniq = int(s.isna().sum()), int(s.nunique(dropna=True))
        d: dict[str, Any] = {"name": str(c), "dtype": dt, "missing": miss, "missingFraction": miss / n if n else 0.0, "unique": uniq}
        flags = []
        if n and miss == n:
            flags.append("all_missing")
        elif uniq <= 1 and n > 1:
            flags.append("constant")
        if dt in ("int", "string") and n > 1 and uniq == n:
            flags.append("unique_per_row (possible identifier)")
        if dt in NUMERIC and s.notna().any():
            q = s.quantile([0.25, 0.5, 0.75])
            d["stats"] = {"count": int(s.count()), "mean": s.mean(), "std": s.std(), "min": s.min(), "q25": q.iloc[0], "median": q.iloc[1],
                          "q75": q.iloc[2], "max": s.max()}
        elif dt != "datetime":
            vc = s.value_counts(dropna=True).head(top_k)
            d["top"] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        d["flags"] = flags
        cols.append(d)
    dup_repeats = int(df.duplicated().sum())
    return clean({"rows": n, "columnCount": len(df.columns), "exact": True, "method": "exact over all rows", "duplicateRows": dup_repeats,
                  "rowsInDuplicateGroups": int(df.duplicated(keep=False).sum()), "columns": cols})


@register
class Profile(TabularOperation):
    type = "tabular.profile"
    inputs = ("table",)
    outputs = ("table",)
    in_kinds = {"table": TABLE}
    out_kinds = {"table": TABLE}
    Config = ProfileConfig
    summary_kind = "profile"

    def infer(self, cfg, ins, node_id):
        return {"table": _same_table(ins["table"])}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        return {"table": t}, profile_table(t.df, cfg.top_k)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "per column: missing = count(NaN); unique = nunique; numeric: mean, std (ddof=1), min, quartiles, max; else top-k values",
                "note": "Exact profile over every row of the input table (not sampled). A profile passes the table through unchanged."}


# ================================================================================================ duplicates
class DuplicatesConfig(StrictConfig):
    key_columns: list[str] = Field(default_factory=list, title="key columns (empty = all columns)")
    keep: Literal["first", "last", "none", "report_only"] = Field("first", title="keep policy")
    label_column: str | None = Field(None, title="label column (to report conflicting labels)")


@register
class Duplicates(TabularOperation):
    type = "tabular.duplicates"
    inputs = ("table",)
    outputs = ("table",)
    in_kinds = {"table": TABLE}
    out_kinds = {"table": TABLE}
    Config = DuplicatesConfig
    summary_kind = "step"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        need_columns(t, cfg.key_columns + ([cfg.label_column] if cfg.label_column else []), "table", f"Duplicate policy '{node_id}'")
        rows = t.info.get("rows") if cfg.keep == "report_only" else None
        return {"table": _same_table(t, rows=rows, rowsExact=rows is not None)}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        df = t.df
        subset = cfg.key_columns or None
        dup_mask = df.duplicated(subset=subset, keep=False)
        groups = df[dup_mask].groupby(subset or list(df.columns), dropna=False).apply(lambda g: [int(i) for i in g.index], include_groups=False)
        examples = [{"rowIds": ids[:6], "size": len(ids)} for ids in list(groups.values)[:5]]
        conflicts = None
        if cfg.label_column:
            sub = df[dup_mask]
            keys = subset or [c for c in df.columns if c != cfg.label_column]
            if keys:
                nun = sub.groupby(keys, dropna=False)[cfg.label_column].nunique(dropna=False)
                conflicts = int((nun > 1).sum())
        if cfg.keep == "report_only":
            out = df
        elif cfg.keep == "none":
            out = df[~dup_mask]
        else:
            out = df.drop_duplicates(subset=subset, keep=cfg.keep)
        rep = {"keyColumns": cfg.key_columns or "all columns (exact match)", "keep": cfg.keep, "rowsBefore": len(df), "rowsAfter": len(out),
               "removed": len(df) - len(out), "duplicateGroups": int(len(groups)), "rowsInDuplicateGroups": int(dup_mask.sum()),
               "conflictingLabelGroups": conflicts, "examples": examples, "before": sample_rows(df), "after": sample_rows(out)}
        return {"table": t.derive(out)}, clean(rep)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "group rows by key (exact equality); keep = first | last | none (drop every member of a duplicate group) | report_only",
                "rule": "Identity rule is fixed before splitting so duplicates cannot span partitions. Conflicting labels are reported, never silently resolved.",
                "note": "Run this node before the train/validation split."}


# ================================================================================================ typed column selection
class ColumnSpec(StrictConfig):
    name: str
    dtype: Literal["float", "int", "string", "bool"] = "float"


class SelectConfig(StrictConfig):
    columns: list[ColumnSpec] = Field(default_factory=list, title="columns and their types")
    on_cast_failure: Literal["error", "set_missing"] = Field("error", title="when a value cannot be converted")


def _cast(s: pd.Series, dtype: str, name: str, policy: str) -> tuple[pd.Series, dict[str, Any]]:
    """Convert one column to its declared type. Values that cannot be converted are counted (and listed) before anything is changed."""
    original = str(s.dtype)
    bad = pd.Series(False, index=s.index)
    if dtype in ("float", "int"):
        out = pd.to_numeric(s, errors="coerce").astype("float64")
        bad = out.isna() & s.notna()
        if dtype == "int":
            bad = bad | (out.notna() & (out != np.floor(out)))
    elif dtype == "bool":
        if s.dtype.kind == "b":
            out = s
        else:
            out = s.astype("str").str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
            bad = out.isna() & s.notna()
    else:
        out = s.astype("str")
    nbad = int(bad.sum())
    info = {"column": name, "requested": dtype, "originalDtype": original, "failures": nbad, "examples": [str(v) for v in s[bad].head(3).tolist()]}
    if nbad and policy == "error":
        raise ExecutionError("E_CAST_FAILED", f"Column '{name}': {nbad} value(s) cannot be converted to {dtype} (e.g. {info['examples']}). "
                             "Choose 'set_missing' to turn them into missing values, or fix the source.")
    if nbad:
        out = out.where(~bad)
    if dtype == "int":
        if out.isna().any():
            raise ExecutionError("E_CAST_FAILED", f"Column '{name}' has {int(out.isna().sum())} missing value(s) and cannot be an int column; declare it as float or remove the missing values first.")
        out = out.astype("int64")
    elif dtype == "bool":
        out = out.astype("boolean") if out.isna().any() else out.astype(bool)
    return out, info


@register
class SelectColumns(TabularOperation):
    type = "tabular.select_columns"
    inputs = ("table",)
    outputs = ("table",)
    in_kinds = {"table": TABLE}
    out_kinds = {"table": TABLE}
    Config = SelectConfig
    summary_kind = "step"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        if not cfg.columns:
            raise OpError("E_NO_COLUMNS", "No columns selected. Excluded columns (identifiers, unused fields) simply are not listed.", "table",
                          [Fix("Add at least one column")])
        names = [c.name for c in cfg.columns]
        if len(set(names)) != len(names):
            raise OpError("E_DUPLICATE_COLUMN", "A column is selected more than once.", "table")
        need_columns(t, names, "table", f"Column selection '{node_id}'")
        return {"table": _same_table(t, columns=[{"name": c.name, "dtype": c.dtype} for c in cfg.columns], columnsComplete=True, pending=[])}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        cols, reports = {}, []
        for c in cfg.columns:
            if c.name not in t.df.columns:
                raise ExecutionError("E_COLUMN_NOT_FOUND", f"Column '{c.name}' not found; the table has {list(t.df.columns)}.")
            cols[c.name], rep = _cast(t.df[c.name], c.dtype, c.name, cfg.on_cast_failure)
            reports.append(rep)
        out = pd.DataFrame(cols, index=t.df.index)
        excluded = [c for c in t.df.columns if c not in cols]
        return {"table": t.derive(out)}, clean({"selected": list(cols), "excluded": excluded, "casts": reports, "rows": len(out),
                                                "before": sample_rows(t.df), "after": sample_rows(out)})

    def explain(self, cfg, inputs, outputs):
        return {"equation": "keep the listed columns in order, converting each to its declared type",
                "rule": "float/int via pandas.to_numeric; bool accepts true/false/yes/no/1/0; string via str. Conversion failures are an error unless set_missing is chosen.",
                "note": "Excluded identifiers never reach the estimator. Target and feature assignment happen on the estimator block."}


# ================================================================================================ missing rows
class DropMissingConfig(StrictConfig):
    columns: list[str] = Field(default_factory=list, title="columns to check (empty = all)")
    how: Literal["any", "all"] = "any"


@register
class DropMissing(TabularOperation):
    type = "tabular.drop_missing"
    inputs = ("table",)
    outputs = ("table",)
    in_kinds = {"table": TABLE}
    out_kinds = {"table": TABLE}
    Config = DropMissingConfig
    summary_kind = "step"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        need_columns(t, cfg.columns, "table", f"Drop missing '{node_id}'")
        return {"table": _same_table(t, rows=None, rowsExact=False)}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        subset = cfg.columns or None
        out = t.df.dropna(subset=subset, how=cfg.how)
        miss = {str(c): int(t.df[c].isna().sum()) for c in (cfg.columns or t.df.columns)}
        return {"table": t.derive(out)}, clean({"how": cfg.how, "rowsBefore": len(t.df), "rowsAfter": len(out), "dropped": len(t.df) - len(out),
                                                "missingPerColumn": miss, "droppedRowIds": [int(i) for i in t.df.index.difference(out.index)[:20]],
                                                "before": sample_rows(t.df), "after": sample_rows(out)})

    def explain(self, cfg, inputs, outputs):
        return {"equation": "DataFrame.dropna(subset, how)", "rule": "Row removal by a fixed rule uses no fitted statistics, so it does not leak. "
                "To fill values instead, use a fitted imputer (train-fitted)."}


# ================================================================================================ train / validation split
class SplitConfig(StrictConfig):
    seed: int = 0
    validation_fraction: float = Field(0.25, gt=0, lt=1)
    stratify_by: str | None = Field(None, title="stratify by column")
    group_by: str | None = Field(None, title="group by column (groups never span partitions)")


@register
class TrainValidationSplit(TabularOperation):
    type = "tabular.train_validation_split"
    inputs = ("table",)
    outputs = ("train", "validation")
    in_kinds = {"table": TABLE}
    out_kinds = {"train": TABLE, "validation": TABLE}
    Config = SplitConfig
    summary_kind = "split"

    def infer(self, cfg, ins, node_id):
        t = ins["table"]
        if cfg.stratify_by and cfg.group_by:
            raise OpError("E_SPLIT_CONFLICT", "Stratified and grouped splitting cannot be combined here.", "table",
                          [Fix("Clear 'stratify_by'", node_id, "stratify_by", None), Fix("Clear 'group_by'", node_id, "group_by", None)])
        need_columns(t, [c for c in (cfg.stratify_by, cfg.group_by) if c], "table", f"Split '{node_id}'")
        n = t.info.get("rows")
        n_val = n_tr = None
        if n is not None and not cfg.group_by and t.info.get("rowsExact", True):
            n_val = math.ceil(cfg.validation_fraction * n)
            n_tr = n - n_val
        base = dict(t.info)
        base.update(splitNode=node_id, splitSeed=cfg.seed)
        return {"train": VType(TABLE, {**base, "partition": "train", "rows": n_tr, "rowsExact": n_tr is not None}),
                "validation": VType(TABLE, {**base, "partition": "validation", "rows": n_val, "rowsExact": n_val is not None})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        df = t.df
        pos = np.arange(len(df))
        if len(df) < 2:
            raise ExecutionError("E_SPLIT_TOO_SMALL", "Need at least 2 rows to split.")
        if cfg.group_by:
            g = df[cfg.group_by]
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=cfg.validation_fraction, random_state=cfg.seed).split(pos, groups=g))
        else:
            strat = df[cfg.stratify_by] if cfg.stratify_by else None
            try:
                tr, va = train_test_split(pos, test_size=cfg.validation_fraction, random_state=cfg.seed, shuffle=True, stratify=strat)
            except ValueError as e:
                raise ExecutionError("E_SPLIT_FAILED", str(e))
        tr, va = np.sort(tr), np.sort(va)
        train, val = df.iloc[tr], df.iloc[va]
        assert not set(train.index) & set(val.index)
        split = {"node": ctx.node_id, "seed": cfg.seed, "validationFraction": cfg.validation_fraction, "stratifyBy": cfg.stratify_by,
                 "groupBy": cfg.group_by, "nTrain": len(train), "nValidation": len(val),
                 "trainRowIdsSha256": row_ids_sha256(train.index), "validationRowIdsSha256": row_ids_sha256(val.index)}
        summary = {**split, "method": "GroupShuffleSplit" if cfg.group_by else ("train_test_split(stratify)" if cfg.stratify_by else "train_test_split"),
                   "overlapRows": 0, "trainRowIds": [int(i) for i in train.index], "validationRowIds": [int(i) for i in val.index]}
        if cfg.group_by:
            summary["groupOverlap"] = len(set(train[cfg.group_by]) & set(val[cfg.group_by]))
        return ({"train": Table(train, "train", {**t.lineage, "split": split}), "validation": Table(val, "validation", {**t.lineage, "split": split})},
                clean(summary))

    def explain(self, cfg, inputs, outputs):
        return {"equation": "scikit-learn train_test_split(random_state=seed, test_size=validation_fraction)",
                "rule": "Seeded and recorded; the row ids of both partitions are hashed into the run. Fit nodes may only read the 'train' output.",
                "note": "n_validation = ceil(fraction x n); n_train = n - n_validation. Grouped splits keep every group in one partition."}


# ================================================================================================ fit ops
def _pick_columns(t: VType, names: list[str], kinds: tuple[str, ...], port: str, what: str) -> list[str] | None:
    if names:
        need_columns(t, names, port, what)
        return names
    if not t.complete:
        return None
    return [c["name"] for c in t.columns if c["dtype"] in kinds]


def _fit_info(t: VType) -> dict[str, Any]:
    return {"node": t.info.get("splitNode"), "port": "train", "partition": "train", "seed": t.info.get("splitSeed")}


def _owner(t: Table, port_node: str) -> dict[str, Any]:
    sp = t.lineage.get("split", {})
    return {"node": sp.get("node"), "port": "train", "partition": t.partition, "seed": sp.get("seed"), "rows": len(t.df),
            "rowIdsSha256": row_ids_sha256(t.df.index), "fittedBy": port_node}


def _require_train(t: Table, what: str) -> None:
    if t.partition != "train":
        raise ExecutionError("E_LEAKAGE_FIT_ON_HELDOUT" if t.partition == "validation" else "E_LEAKAGE_FIT_BEFORE_SPLIT",
                             f"{what} refuses to fit on the '{t.partition}' partition; fit state is owned by the training partition.")


class FitStandardizeConfig(StrictConfig):
    columns: list[str] = Field(default_factory=list, title="columns (empty = all numeric)")
    with_mean: bool = True
    with_std: bool = True


@register
class FitStandardize(TabularOperation):
    type = "tabular.fit_standardize"
    backend = "scikit-learn"
    inputs = ("train",)
    outputs = ("fit",)
    in_kinds = {"train": TABLE}
    out_kinds = {"fit": FIT_STATE}
    Config = FitStandardizeConfig
    summary_kind = "fit_state"

    def infer(self, cfg, ins, node_id):
        t = ins["train"]
        leakage_check(t, "train", f"Fit standardization '{node_id}'")
        cols = _pick_columns(t, cfg.columns, NUMERIC, "train", f"Fit standardization '{node_id}'")
        if cols is not None:
            need_numeric(t, cols, "train", f"Fit standardization '{node_id}'")
            if not cols:
                raise OpError("E_NO_COLUMNS", "There are no numeric columns to standardize.", "train")
        return {"fit": VType(FIT_STATE, {"transform": "standardize", "columns": cols, "fittedOn": _fit_info(t)})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["train"]
        _require_train(t, f"Fit standardization '{ctx.node_id}'")
        cols = cfg.columns or [str(c) for c in t.df.columns if dtype_name(t.df[c]) in NUMERIC]
        sc = StandardScaler(with_mean=cfg.with_mean, with_std=cfg.with_std).fit(t.df[cols].to_numpy(dtype="float64"))
        owner = _owner(t, ctx.node_id)
        details = {"transform": "standardize", "columns": cols, "fittedOn": owner, "params": sc.get_params(),
                   "mean": None if sc.mean_ is None else dict(zip(cols, sc.mean_.tolist())),
                   "scale": None if sc.scale_ is None else dict(zip(cols, sc.scale_.tolist())),
                   "variance": None if sc.var_ is None else dict(zip(cols, sc.var_.tolist())),
                   "nSamplesSeen": np.asarray(sc.n_samples_seen_).tolist(), "ddof": 0}
        fs = FitState("standardize", sc, cols, owner, clean(details))
        return {"fit": fs}, fs.details

    def explain(self, cfg, inputs, outputs):
        return {"equation": "z = (x - mean_train) / std_train   (population std, ddof = 0)", "rule": "mean and std are learned from the training partition only; the same numbers are applied to validation.",
                "note": "sklearn.preprocessing.StandardScaler. A zero-variance column gets scale 1."}


class FitOneHotConfig(StrictConfig):
    columns: list[str] = Field(default_factory=list, title="columns (empty = all string/bool)")
    handle_unknown: Literal["ignore", "error"] = "ignore"
    drop: Literal["first", "if_binary"] | None = Field(None, title="drop one category")


@register
class FitOneHot(TabularOperation):
    type = "tabular.fit_onehot"
    backend = "scikit-learn"
    inputs = ("train",)
    outputs = ("fit",)
    in_kinds = {"train": TABLE}
    out_kinds = {"fit": FIT_STATE}
    Config = FitOneHotConfig
    summary_kind = "fit_state"

    def infer(self, cfg, ins, node_id):
        t = ins["train"]
        leakage_check(t, "train", f"Fit one-hot encoding '{node_id}'")
        cols = _pick_columns(t, cfg.columns, ("string", "bool"), "train", f"Fit one-hot encoding '{node_id}'")
        if cols is not None and not cols:
            raise OpError("E_NO_COLUMNS", "There are no categorical (string/bool) columns to encode.", "train")
        return {"fit": VType(FIT_STATE, {"transform": "onehot", "columns": cols, "fittedOn": _fit_info(t)})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["train"]
        _require_train(t, f"Fit one-hot encoding '{ctx.node_id}'")
        cols = cfg.columns or [str(c) for c in t.df.columns if dtype_name(t.df[c]) in ("string", "bool")]
        enc = OneHotEncoder(sparse_output=False, handle_unknown=cfg.handle_unknown, drop=cfg.drop, dtype=np.float64).fit(t.df[cols])
        owner = _owner(t, ctx.node_id)
        details = {"transform": "onehot", "columns": cols, "fittedOn": owner, "params": {k: v for k, v in enc.get_params().items() if k != "dtype"},
                   "categories": {c: [clean(x) for x in cats] for c, cats in zip(cols, enc.categories_)},
                   "featureNamesOut": enc.get_feature_names_out().tolist()}
        fs = FitState("onehot", enc, cols, owner, clean(details))
        return {"fit": fs}, fs.details

    def explain(self, cfg, inputs, outputs):
        return {"equation": "x_c -> indicator(x == category) for every category seen in the training partition",
                "rule": "Categories are learned from the training partition only. Categories first seen in validation are all-zero rows (handle_unknown='ignore') or an error.",
                "note": "sklearn.preprocessing.OneHotEncoder. Output columns are named <column>_<category>; missing values form their own category."}


class FitImputeConfig(StrictConfig):
    columns: list[str] = Field(default_factory=list, title="columns (empty = all numeric)")
    strategy: Literal["mean", "median", "constant"] = "median"
    fill_value: float = Field(0.0, title="fill value (used by 'constant')")


@register
class FitImpute(TabularOperation):
    type = "tabular.fit_impute"
    backend = "scikit-learn"
    inputs = ("train",)
    outputs = ("fit",)
    in_kinds = {"train": TABLE}
    out_kinds = {"fit": FIT_STATE}
    Config = FitImputeConfig
    summary_kind = "fit_state"

    def infer(self, cfg, ins, node_id):
        t = ins["train"]
        leakage_check(t, "train", f"Fit imputation '{node_id}'")
        cols = _pick_columns(t, cfg.columns, NUMERIC, "train", f"Fit imputation '{node_id}'")
        if cols is not None:
            need_numeric(t, cols, "train", f"Fit imputation '{node_id}'")
            if not cols:
                raise OpError("E_NO_COLUMNS", "There are no numeric columns to impute.", "train")
        return {"fit": VType(FIT_STATE, {"transform": "impute", "columns": cols, "fittedOn": _fit_info(t)})}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["train"]
        _require_train(t, f"Fit imputation '{ctx.node_id}'")
        cols = cfg.columns or [str(c) for c in t.df.columns if dtype_name(t.df[c]) in NUMERIC]
        kw = {"fill_value": cfg.fill_value} if cfg.strategy == "constant" else {}
        imp = SimpleImputer(strategy=cfg.strategy, keep_empty_features=True, **kw).fit(t.df[cols].to_numpy(dtype="float64"))
        owner = _owner(t, ctx.node_id)
        details = {"transform": "impute", "columns": cols, "fittedOn": owner, "strategy": cfg.strategy,
                   "statistics": dict(zip(cols, imp.statistics_.tolist())),
                   "missingInTraining": {c: int(t.df[c].isna().sum()) for c in cols}}
        fs = FitState("impute", imp, cols, owner, clean(details))
        return {"fit": fs}, fs.details

    def explain(self, cfg, inputs, outputs):
        return {"equation": f"x_missing -> {cfg.strategy}_train(x)" if cfg.strategy != "constant" else f"x_missing -> {cfg.fill_value}",
                "rule": "The mean/median is computed from the training partition only and then used for validation rows too.",
                "note": "sklearn.impute.SimpleImputer."}


# ================================================================================================ apply
class ApplyConfig(StrictConfig):
    pass


@register
class ApplyTransform(TabularOperation):
    type = "tabular.apply_transform"
    backend = "scikit-learn"
    inputs = ("table", "fit")
    outputs = ("table",)
    in_kinds = {"table": TABLE, "fit": FIT_STATE}
    out_kinds = {"table": TABLE}
    Config = ApplyConfig
    summary_kind = "step"

    def infer(self, cfg, ins, node_id):
        t, f = ins["table"], ins["fit"]
        cols = f.info.get("columns")
        kind = f.info["transform"]
        if cols is not None:
            need_columns(t, cols, "table", f"Apply fitted {kind} '{node_id}'")
        if kind == "onehot":
            if cols is None:
                return {"table": _same_table(t, columnsComplete=False)}
            keep = [c for c in t.columns if c["name"] not in cols]
            return {"table": _same_table(t, columns=keep, columnsComplete=False, pending=t.info.get("pending", []) + [f"{c}_<category> (one-hot of {c}; categories known after fit)" for c in cols],
                                                                   pendingPrefixes=t.info.get("pendingPrefixes", []) + [f"{c}_" for c in cols])}
        newcols = [{"name": c["name"], "dtype": "float" if (cols is None or c["name"] in cols) and c["dtype"] in NUMERIC else c["dtype"]} for c in t.columns]
        return {"table": _same_table(t, columns=newcols)}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        fs: FitState = ins["fit"]
        df, cols = t.df, fs.columns
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ExecutionError("E_COLUMN_NOT_FOUND", f"The fitted {fs.transform} needs columns {miss} that this table lacks.")
        summary: dict[str, Any] = {"transform": fs.transform, "fittedOn": fs.fitted_on, "appliedToPartition": t.partition, "rows": len(df),
                                   "note": "Fitted statistics were learned on the training partition; nothing was refit here."}
        if fs.transform == "onehot":
            enc = fs.transformer
            try:
                arr = enc.transform(df[cols])
            except ValueError as e:
                raise ExecutionError("E_UNSEEN_CATEGORY", str(e))
            enc_df = pd.DataFrame(arr, index=df.index, columns=enc.get_feature_names_out())
            out = pd.concat([df.drop(columns=cols), enc_df], axis=1)
            cats = fs.details["categories"]
            summary["unseenRows"] = {c: int((~df[c].isin(cats[c]) & df[c].notna()).sum()) for c in cols}
            summary["addedColumns"] = list(enc_df.columns)
            summary["removedColumns"] = cols
        else:
            arr = fs.transformer.transform(df[cols].to_numpy(dtype="float64"))
            out = df.copy()
            out[cols] = arr
            if fs.transform == "impute":
                summary["cellsImputed"] = {c: int(df[c].isna().sum()) for c in cols}
        summary["before"], summary["after"] = sample_rows(df), sample_rows(out)
        return {"table": t.derive(out)}, clean(summary)

    def explain(self, cfg, inputs, outputs):
        return {"equation": "table_out = transform_fitted_on_train(table_in)", "rule": "Apply fitted transform: reads fitted state, never refits. Works on any partition.",
                "note": "One-hot output columns replace the encoded columns and are appended after the untouched ones."}


# ================================================================================================ predictions export
class PredictionsConfig(StrictConfig):
    include_features: bool = False
    include_probabilities: bool = True


def predict_frame(model, table: Table, include_features: bool, include_proba: bool) -> pd.DataFrame:
    m = model
    df = table.df
    miss = [c for c in m.features if c not in df.columns]
    if miss:
        raise ExecutionError("E_MODEL_FEATURES_MISSING", f"The model was fitted on features {miss} that the evaluation table lacks.")
    X = df[m.features]
    if X.isna().any().any():
        bad = {c: int(X[c].isna().sum()) for c in m.features if X[c].isna().any()}
        raise ExecutionError("E_MISSING_VALUES", f"Evaluation features contain missing values {bad}; impute them with the train-fitted imputer before predicting.")
    X = X.to_numpy(dtype="float64")
    pred = m.estimator.predict(X)
    out = pd.DataFrame({"observed": df[m.target].to_numpy() if m.target in df.columns else np.nan, "predicted": pred}, index=df.index)
    if m.task == "regression":
        out["residual"] = out["observed"] - out["predicted"]
    elif include_proba:
        proba = m.estimator.predict_proba(X)
        for i, cl in enumerate(m.estimator.classes_):
            out[f"proba_{cl}"] = proba[:, i]
    if include_features:
        out = pd.concat([out, df[m.features]], axis=1)
    return out


@register
class PredictionsExport(TabularOperation):
    type = "tabular.predictions_export"
    backend = "scikit-learn"
    inputs = ("model", "table")
    outputs = ("predictions",)
    in_kinds = {"model": "model", "table": TABLE}
    out_kinds = {"predictions": TABLE}
    Config = PredictionsConfig
    summary_kind = "step"

    def infer(self, cfg, ins, node_id):
        m, t = ins["model"], ins["table"]
        feats = m.info.get("features")
        if feats is not None:
            need_columns(t, feats, "table", f"Predictions '{node_id}'")
        cols = [{"name": "observed", "dtype": "float" if m.info["task"] == "regression" else "string"}, {"name": "predicted", "dtype": "float" if m.info["task"] == "regression" else "string"}]
        complete = True
        if m.info["task"] == "regression":
            cols.append({"name": "residual", "dtype": "float"})
        elif cfg.include_probabilities:
            complete = False
        if cfg.include_features and feats is not None:
            cols += [{"name": f, "dtype": "float"} for f in feats]
        return {"predictions": table_type(cols, t.info.get("rows"), t.partition, complete and (feats is not None or not cfg.include_features),
                                          ["proba_<class> columns (classes known after fit)"] if not complete else [], rowsExact=t.info.get("rowsExact", True))}

    def execute(self, cfg, ins, ctx):
        t: Table = ins["table"]
        out = predict_frame(ins["model"], t, cfg.include_features, cfg.include_probabilities)
        return ({"predictions": Table(out, t.partition, {**t.lineage, "export": True})},
                clean({"rows": len(out), "partition": t.partition, "columns": list(out.columns), "model": ins["model"].details.get("estimator"),
                       "sample": sample_rows(out, 8)}))

    def explain(self, cfg, inputs, outputs):
        return {"equation": "predicted = estimator.predict(table[features]); residual = observed - predicted",
                "note": "Indexed by the source row id. The run stores the full table as a CSV artifact; the preview shows a bounded slice."}
