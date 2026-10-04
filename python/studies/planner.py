"""Turn a study request into explicit, validated trials (no trial is a hand-drawn copy of the experiment).

Trial = one resolved configuration (assignments to node-config / run-config fields) x one repeat identity (seed, fold).
A baseline group (no assignments) is added unless disabled. Sizes are checked against `limits.max_trials` BEFORE anything is stored:
a plan that does not fit is rejected, never silently truncated."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from graph_core import registry
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import validate
from worker.rl_run import RLRunConfig
from worker.tabular_run import TabularRunConfig
from worker.train import RunConfig

from .models import RESERVED_RUN_FIELDS, StudyCreate, Target, Variable


class PlanError(Exception):
    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.code, self.message, self.detail = code, message, detail


def run_config_model(graph: Graph):
    return {"tabular": TabularRunConfig, "rl": RLRunConfig}.get(graph.graphKind, RunConfig)


def run_fields(graph: Graph) -> set[str]:
    return set(run_config_model(graph).model_fields) - RESERVED_RUN_FIELDS


def check_target(graph: Graph, t: Target) -> None:
    if t.scope == "run":
        if t.field.split(".")[0] not in run_fields(graph):
            raise PlanError("E_SWEEP_TARGET", f"'{t.field}' is not a sweepable run-config field of a {graph.graphKind} graph (have {sorted(run_fields(graph))}).")
        return
    if not t.node:
        raise PlanError("E_SWEEP_TARGET", "A node target needs 'node'.")
    try:
        n = graph.node(t.node)
    except KeyError:
        raise PlanError("E_SWEEP_TARGET", f"Node '{t.node}' does not exist.")
    op = registry.get_op(n.type)
    if op is None:
        raise PlanError("E_SWEEP_TARGET", f"Node '{t.node}' has an unknown operation.")
    head = t.field.split(".")[0]
    if head not in op.Config.model_fields:
        raise PlanError("E_SWEEP_TARGET", f"'{t.field}' is not a config field of {n.type} (have {sorted(op.Config.model_fields)}).")


def _set_path(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = obj
    for i, p in enumerate(parts[:-1]):
        if isinstance(cur, list):
            cur = cur[int(p)]
        else:
            if p not in cur or cur[p] is None:
                cur[p] = {}
            cur = cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def get_path(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for p in path.split("."):
        try:
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return cur


def apply(graph: Graph, base_cfg: dict[str, Any], assignments: list[dict[str, Any]]) -> tuple[Graph, dict[str, Any]]:
    g = graph.model_copy(deep=True)
    cfg = copy.deepcopy(base_cfg)
    for a in assignments:
        t = Target.model_validate(a["target"])
        if t.scope == "node":
            _set_path(g.node(t.node).config, t.field, copy.deepcopy(a["value"]))
        else:
            _set_path(cfg, t.field, a["value"])
    return g, cfg


def _grid(vars_: list[Variable]) -> list[list[tuple[Variable, Any, str | None]]]:
    axes = []
    for v in vars_:
        if not v.values:
            raise PlanError("E_SWEEP_VALUES", f"Grid variable {v.target.key} needs a non-empty 'values' list (intervals are for random search).")
        if v.labels is not None and len(v.labels) != len(v.values):
            raise PlanError("E_SWEEP_VALUES", f"'labels' of {v.target.key} must match 'values' in length.")
        axes.append([(v, val, (v.labels[i] if v.labels else None)) for i, val in enumerate(v.values)])
    return [list(combo) for combo in itertools.product(*axes)]


def _random(vars_: list[Variable], n: int, seed: int) -> list[list[tuple[Variable, Any, str | None]]]:
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        combo = []
        for v in vars_:
            if v.values:
                i = int(rng.integers(len(v.values)))
                combo.append((v, v.values[i], v.labels[i] if v.labels else None))
            elif v.range:
                r = v.range
                if r.scale == "log":
                    if r.low <= 0 or r.high <= 0:
                        raise PlanError("E_SWEEP_VALUES", f"Log-scale range of {v.target.key} must be positive.")
                    x = math.exp(rng.uniform(math.log(r.low), math.log(r.high)))
                else:
                    x = rng.uniform(r.low, r.high)
                combo.append((v, int(round(x)) if r.type == "int" else float(x), None))
            else:
                raise PlanError("E_SWEEP_VALUES", f"Variable {v.target.key} needs 'values' or 'range'.")
        key = json.dumps([c[1] for c in combo], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(combo)
    if len(out) < n:
        raise PlanError("E_SWEEP_VALUES", f"Only {len(out)} distinct random configurations exist for this space; asked for {n}.")
    return out


def group_key(assignments: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(sorted(([a["target"]["scope"], a["target"].get("node"), a["target"]["field"], a["value"]] for a in assignments), key=str),
                                     sort_keys=True, default=str).encode()).hexdigest()[:12]


def label_of(assignments: list[dict[str, Any]], labels: list[str | None]) -> str:
    if not assignments:
        return "baseline"
    return ", ".join((lab or f"{a['target']['field'] if a['target']['scope'] == 'run' else a['target']['node'] + '.' + a['target']['field']}={json.dumps(a['value'])}")
                     for a, lab in zip(assignments, labels))


def plan(graph: Graph, req: StudyCreate) -> dict[str, Any]:
    """Return {trials: [...], groups: n, counts, limits}. Raises PlanError for structural problems (bad target, too many trials)."""
    kind = graph.graphKind
    base_cfg = dict(req.run_config)
    s = req.search
    for v in s.variables:
        check_target(graph, v.target)
    keys = [v.target.key for v in s.variables]
    if len(set(keys)) != len(keys):
        raise PlanError("E_SWEEP_TARGET_CONFLICT", "The same field is used by two variables.")
    if s.method == "single" and s.variables:
        raise PlanError("E_SWEEP_VALUES", "method 'single' takes no variables; use 'grid' or 'random'.")
    if s.method in ("grid", "random") and not s.variables:
        raise PlanError("E_SWEEP_VALUES", f"method '{s.method}' needs at least one variable.")

    # configurations (groups)
    if s.method == "grid":
        combos = _grid(s.variables)
    elif s.method == "random":
        if s.random_trials < 1:
            raise PlanError("E_SWEEP_VALUES", "Random search needs 'random_trials' >= 1.")
        combos = _random(s.variables, s.random_trials, s.sampler_seed)
    else:
        combos = []
    groups: list[tuple[list[dict[str, Any]], list[str | None]]] = []
    if req.include_baseline or not combos:
        groups.append(([], []))
    for combo in combos:
        groups.append(([{"target": v.target.model_dump(), "value": val} for v, val, _ in combo], [lab for _, _, lab in combo]))

    # repeats
    rp = req.repeats
    seed_target = None
    if rp.seeds:
        seed_target = rp.seed_target or Target(scope="run", field="seed")
        check_target(graph, seed_target)
        if seed_target.key in keys:
            raise PlanError("E_SWEEP_TARGET_CONFLICT", f"'{seed_target.key}' is both a swept variable and the seed target.")
    if rp.folds is not None:
        if kind != "tabular":
            raise PlanError("E_SWEEP_FOLD_UNSUPPORTED", "Folds are a repeat dimension of tabular graphs (train_validation_split n_folds/fold); model graphs have no fold support.")
        if not rp.fold_node:
            raise PlanError("E_SWEEP_FOLD_UNSUPPORTED", "Give 'fold_node': the train_validation_split node that receives n_folds/fold.")
        try:
            if graph.node(rp.fold_node).type != "tabular.train_validation_split":
                raise PlanError("E_SWEEP_FOLD_UNSUPPORTED", f"'{rp.fold_node}' is not a train_validation_split node.")
        except KeyError:
            raise PlanError("E_SWEEP_FOLD_UNSUPPORTED", f"Node '{rp.fold_node}' does not exist.")
        for f in ("n_folds", "fold"):
            if f"{rp.fold_node}.{f}" in keys:
                raise PlanError("E_SWEEP_TARGET_CONFLICT", f"{rp.fold_node}.{f} is both swept and the fold target.")
    seeds = rp.seeds or [None]
    folds = list(range(rp.folds)) if rp.folds else [None]
    total = len(groups) * len(seeds) * len(folds)
    if total > req.limits.max_trials:
        raise PlanError("E_SWEEP_LIMIT", f"The plan has {total} trials ({len(groups)} configurations x {len(seeds)} seeds x {len(folds)} folds) but max_trials is {req.limits.max_trials}. "
                        "Reduce the search or raise the limit explicitly; plans are never truncated silently.", {"trials": total, "maxTrials": req.limits.max_trials})

    trials = []
    cfg_ok: dict[int, list[dict[str, Any]]] = {}
    for gi, (assign, labs) in enumerate(groups):
        gkey = group_key(assign)
        # validate this configuration once (the interface/shape check) with the first repeat identity
        for seed, fold in itertools.product(seeds, folds):
            full = list(assign)
            if seed is not None:
                full.append({"target": seed_target.model_dump(), "value": seed})
            if fold is not None:
                full += [{"target": {"scope": "node", "node": rp.fold_node, "field": "n_folds"}, "value": rp.folds},
                         {"target": {"scope": "node", "node": rp.fold_node, "field": "fold"}, "value": fold}]
            if gi not in cfg_ok:
                cfg_ok[gi] = check_variant(graph, base_cfg, full)
            tid = f"t{len(trials) + 1:03d}"
            trials.append({"id": tid, "idx": len(trials), "groupIdx": gi, "groupKey": gkey, "isBaseline": not assign, "label": label_of(assign, labs),
                           "assignments": assign, "seed": seed, "fold": fold, "status": "invalid" if cfg_ok[gi] else "planned", "diagnostics": cfg_ok[gi]})
    return {"trials": trials, "groups": len(groups), "seeds": len(seeds), "folds": len(folds), "total": total,
            "invalid": sum(1 for t in trials if t["status"] == "invalid"), "graphHash": semantic_hash(graph)}


def check_variant(graph: Graph, base_cfg: dict[str, Any], assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diagnostics (empty = valid) for graph + run config after applying the assignments."""
    try:
        g, cfg = apply(graph, base_cfg, assignments)
    except Exception as e:  # noqa: BLE001
        return [{"code": "E_SWEEP_APPLY", "severity": "error", "message": f"Cannot apply the assignment: {type(e).__name__}: {e}"}]
    out = []
    try:
        run_config_model(g).model_validate(cfg)
    except ValidationError as e:
        out.append({"code": "E_RUN_CONFIG", "severity": "error", "message": "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())})
    rep = validate(g)
    out += [d.to_json() for d in rep.errors]
    if not rep.errors and not rep.ok:
        out.append({"code": "E_UNRESOLVED", "severity": "error", "message": "Some nodes could not be typed after applying the assignment."})
    return out
