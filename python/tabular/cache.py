"""Dependency-scoped node result cache for `tabular` graphs (acceptance A09).

A node's cache key covers everything its result depends on: operation type and version, node id (lineage records it), resolved
configuration (after a run-level seed override), the implementation source of the tabular operation family, the native environment,
and the identity of every input. An input's identity is its producer's cache key when the producer is cacheable, otherwise a hash of
the producer's actual output content (a CSV source's table, a connector's snapshot table). Editing one node therefore changes its own
key and the keys of everything downstream of it, and nothing else: siblings and upstream nodes keep their keys and are reused.

Only an explicit allowlist of deterministic operations is cacheable (seeded splits/fits, pure table steps, metrics, SciPy results).
Sources always run: files and databases can change under the same configuration, so their content is re-read and hashed every run.
Connector sources, code blocks, plugin operations and `domain` nodes (which record checkpoints) are never cached.

Entries are pickles of native objects written by the worker into this workbench's own content-addressed store and indexed in its
SQLite `node_cache` table. They are loaded only after their sha256 is verified; there is no import or upload path into the index."""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Plain, Table, dumps, schema_of

FORMAT = "void-node-cache/1"
MAX_ENTRY_BYTES = 64 * 1024 * 1024

CACHEABLE = frozenset({
    "tabular.profile", "tabular.duplicates", "tabular.select_columns", "tabular.drop_missing", "tabular.train_validation_split",
    "tabular.fit_standardize", "tabular.fit_onehot", "tabular.fit_impute", "tabular.apply_transform", "tabular.predictions_export",
    "sklearn.linear_regression", "sklearn.logistic_regression", "sklearn.metrics",
    "scipy.gamma_function", "scipy.gamma_distribution", "scipy.tail_probability", "scipy.hypothesis_test", "scipy.two_group_comparison",
    "sklearn.kmeans", "sklearn.gaussian_mixture", "sklearn.dbscan", "sklearn.pca", "sklearn.projection", "sklearn.cluster_diagnostics",
})
SOURCE_TYPES = frozenset({"tabular.csv_source"})

_PY = Path(__file__).resolve().parents[1]
# Every file whose code can change a cacheable node's result. Editing any of them changes every cacheable key (conservative).
IMPLEMENTATION_FILES = ("tabular/core.py", "tabular/engine.py", "tabular/cache.py", "operations/_common.py", "operations/tabular_ops.py",
                        "operations/sklearn_ops.py", "operations/stats_ops.py", "operations/unsup_ops.py", "unsup/methods.py")


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def implementation_identity() -> dict[str, str]:
    return {f: _sha((_PY / f).read_bytes()) for f in IMPLEMENTATION_FILES}


def environment_identity(libraries: dict[str, str]) -> dict[str, str]:
    return {"python": platform.python_version(), "implementation": platform.python_implementation(), **{k: libraries[k] for k in sorted(libraries)}}


def content_identity(outs: dict[str, Any]) -> str | None:
    """Identity of a non-cacheable node's actual output. Tables and JSON results are hashed by content; any other value has none."""
    parts = {}
    for port, v in sorted(outs.items()):
        if isinstance(v, Table):
            parts[port] = {"kind": "table", "csv": _sha(v.df.to_csv(index=True, index_label="row_id")), "index": str(v.df.index.dtype),
                           "columns": schema_of(v.df), "dtypes": [str(t) for t in v.df.dtypes], "partition": v.partition, "lineage": json.loads(dumps(v.lineage))}
        elif isinstance(v, Plain) and v.obj is None:
            parts[port] = {"kind": v.kind, "data": json.loads(dumps(v.data))}
        else:
            return None
    return _sha(json.dumps({"content": parts}, sort_keys=True, separators=(",", ":")))


@dataclass
class Decision:
    status: str  # hit | miss | bypass
    key: str | None
    reason: str
    changed: list[str]
    entry: dict[str, Any] | None = None
    parts: dict[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        out = {"status": self.status, "key": self.key, "reason": self.reason, "changed": self.changed}
        if self.entry is not None:
            out.update(fromRun=self.entry["run_id"], fromNode=self.entry["node_id"], entrySha256=self.entry["sha256"])
        return out


class NodeCache:
    """One run's view of the workbench cache. `decide` before a node executes, then `store_result` or `load`, then `identity`."""

    def __init__(self, store: Any, project_id: str | None, libraries: dict[str, str], graph_kind: str):
        self.store, self.project_id, self.graph_kind = store, project_id, graph_kind
        self.impl = _sha(json.dumps(implementation_identity(), sort_keys=True))
        self.env = _sha(json.dumps(environment_identity(libraries), sort_keys=True))
        self.ids: dict[str, str | None] = {}

    def describe(self) -> dict[str, Any]:
        return {"mode": "reuse", "implementationSha256": self.impl, "environmentSha256": self.env, "format": FORMAT}

    def decide(self, nid: str, op: Any, cfg: Any, inputs: dict[str, tuple[str, str]]) -> Decision:
        typ = op.type
        if self.graph_kind != "tabular":
            return Decision("bypass", None, "Domain nodes train models and record checkpoints; their results are never cached.", [])
        if typ in SOURCE_TYPES:
            return Decision("bypass", None, "Sources re-read their data every run; the content hash of what was read identifies the output downstream.", [])
        if typ not in CACHEABLE:
            return Decision("bypass", None, "This operation is not declared cacheable (external source, side effects, or determinism not declared); it always runs.", [])
        missing = [f"{p} (from {src})" for p, (src, _) in inputs.items() if self.ids.get(src) is None]
        if missing:
            return Decision("bypass", None, f"Input {', '.join(missing)} has no content identity, so a cached result could not be matched safely.", [])
        config = json.loads(dumps(cfg.model_dump(mode="json")))
        node_part = _sha(json.dumps({"op": typ, "version": op.version, "node": nid, "config": config}, sort_keys=True, separators=(",", ":")))
        ins = {p: [self.ids[src], port] for p, (src, port) in sorted(inputs.items())}
        parts = {"format": FORMAT, "node": node_part, "implementation": self.impl, "environment": self.env, "inputs": ins}
        key = _sha(json.dumps(parts, sort_keys=True, separators=(",", ":")))
        entry = self.store.get_node_cache(key)
        if entry is not None:
            return Decision("hit", key, "Same operation, settings, inputs, implementation and environment as a recorded result; reused it.", [], entry, parts)
        prev = self.store.latest_node_cache(nid, typ, self.project_id)
        if prev is None:
            return Decision("miss", key, "No recorded result for this node with these settings and inputs; executed it.", [], None, parts)
        changed = []
        if prev["node_part"] != node_part:
            changed.append("settings")
        if prev["implementation"] != self.impl:
            changed.append("implementation")
        if prev["environment"] != self.env:
            changed.append("environment")
        old_ins = json.loads(prev["inputs"])
        changed += [f"input {p} (from {inputs[p][0]})" for p in ins if old_ins.get(p) != ins[p]]
        what = ", ".join(changed) or "dependencies"
        return Decision("miss", key, f"Invalidated: {what} changed since the last recorded result (run {prev['run_id']}); executed it.", changed, None, parts)

    def load(self, d: Decision) -> tuple[dict[str, Any], dict[str, Any]] | None:
        sha = d.entry["sha256"]
        if not self.store.verify(sha):
            return None
        rec = pickle.loads(self.store.read_artifact(sha))  # noqa: S301  (internal, hash-verified entry written by this workbench's worker)
        if rec.get("format") != FORMAT or rec.get("key") != d.key:
            return None
        return rec["outs"], rec["summary"]

    def store_result(self, d: Decision, run_id: str, nid: str, typ: str, outs: dict[str, Any], summary: dict[str, Any]) -> str | None:
        """Record a freshly computed result. Returns why it was not recorded, or None."""
        try:
            data = pickle.dumps({"format": FORMAT, "key": d.key, "outs": outs, "summary": summary}, protocol=5)
        except Exception as e:  # noqa: BLE001
            return f"not recorded: {type(e).__name__}: {e}"
        if len(data) > MAX_ENTRY_BYTES:
            return f"not recorded: {len(data)} bytes exceeds the {MAX_ENTRY_BYTES}-byte entry limit"
        sha = self.store.put_bytes(data)
        self.store.put_node_cache(d.key, sha, len(data), run_id, nid, typ, self.project_id, d.parts["node"], self.impl, self.env, json.dumps(d.parts["inputs"], sort_keys=True))
        return None

    def set_identity(self, nid: str, d: Decision, outs: dict[str, Any]) -> None:
        self.ids[nid] = d.key if d.key is not None else content_identity(outs)
