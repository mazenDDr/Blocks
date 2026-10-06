"""Validation of `agent` graphs: state schema and reducers, node configs, state reads/writes, control edges, routes and predicates,
loops, parallel-branch write conflicts, indexes and memory policies. Same diagnostic contract as the other graph kinds."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from graph_core import registry
from graph_core.schema import SCHEMA_VERSION, Graph
from graph_core.types import Diagnostic, Fix
from graph_core.validate import ID_RE, Report, _config_message

from . import policy as pol
from .blocks import AgentOp
from .spec import (CMP_OPS, END, FIELD_TYPES, LISTY, NUM_OPS, NUMERIC, REDUCERS_FOR, RESERVED_NODE_IDS, START, UNARY, AgentSpec, PATH_RE, pred_fields, resolve_type,
                   template_vars)

BACKEND = "langgraph"
RECORD_FIELDS = {"id", "kind", "namespace", "scope", "text", "importance", "age_days", "generated", "store", "created_at"}


@dataclass
class AgentReport(Report):
    spec: AgentSpec | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    node_info: dict[str, dict[str, Any]] = field(default_factory=dict)


def validate_agent(graph: Graph) -> AgentReport:
    r = AgentReport()
    r.graph = graph
    diag = r.diagnostics

    def add(code, msg, node=None, port=None, path="", fixes=None, severity="error"):
        diag.append(Diagnostic(code, msg, severity, node, port, path, fixes or []))

    if graph.backend != BACKEND:
        add("E_UNSUPPORTED_BACKEND", f"Agent graphs use backend '{BACKEND}' (native LangGraph), got '{graph.backend}'.", path="/backend", fixes=[Fix("Set backend to 'langgraph'", None, "backend", BACKEND)])
    if graph.schemaVersion.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        add("E_UNSUPPORTED_SCHEMA", f"Schema version {graph.schemaVersion} is not supported.", path="/schemaVersion")
    try:
        spec = AgentSpec.model_validate(graph.agent or {})
    except ValidationError as e:
        add("E_AGENT_SPEC", f"Invalid agent section: {_config_message(e)}", path="/agent")
        return r
    r.spec = spec

    # ---------------------------------------------------------------- state schema
    names: set[str] = set()
    for i, f in enumerate(spec.state):
        p = f"/agent/state/{f.name}"
        if not PATH_RE.match(f.name) or "." in f.name or f.name in RESERVED_NODE_IDS:
            add("E_STATE_FIELD_NAME", f"State field name '{f.name}' must be an identifier (letters, digits, underscore).", path=p)
            continue
        if f.name in names:
            add("E_DUPLICATE_ID", f"State field '{f.name}' is declared more than once.", path=p)
            continue
        names.add(f.name)
        if f.reducer.kind not in REDUCERS_FOR[f.type]:
            add("E_REDUCER", f"Reducer '{f.reducer.kind}' does not apply to a '{f.type}' field (valid: {', '.join(REDUCERS_FOR[f.type])}).", path=p + "/reducer",
                fixes=[Fix("Use the 'replace' reducer", None, None, None)])
        if f.reducer.kind == "keep_last_n" and not f.reducer.n:
            add("E_REDUCER", f"Reducer 'keep_last_n' on '{f.name}' needs n >= 1.", path=p + "/reducer")
        if f.properties and f.type != "object":
            add("E_STATE_FIELD_NAME", f"Only object fields declare properties; '{f.name}' is '{f.type}'.", path=p, severity="warning")
        if f.default is not None:
            ok = {"text": isinstance(f.default, str), "integer": isinstance(f.default, int) and not isinstance(f.default, bool), "number": isinstance(f.default, (int, float)),
                  "boolean": isinstance(f.default, bool), "object": isinstance(f.default, dict), "json": True}.get(f.type, isinstance(f.default, list))
            if not ok:
                add("E_STATE_DEFAULT", f"Default {f.default!r} of '{f.name}' is not a '{f.type}'.", path=p + "/default")

    # ---------------------------------------------------------------- nodes
    ops: dict[str, AgentOp] = {}
    cfgs: dict[str, Any] = {}
    seen: set[str] = set()
    for n in graph.nodes:
        npath = f"/nodes/{n.id}"
        if n.id in seen:
            add("E_DUPLICATE_ID", f"Node id '{n.id}' is used more than once.", n.id, path=npath)
            continue
        seen.add(n.id)
        if not ID_RE.match(n.id) or n.id in RESERVED_NODE_IDS:
            add("E_BAD_ID", f"Node id '{n.id}' must be letters, digits and underscores, start with a letter and not be START or END.", n.id, path=npath)
            continue
        op = registry.get_op(n.type)
        if op is None:
            add("E_UNKNOWN_OP", f"Operation '{n.type}' is not available. The node is preserved but cannot execute until it is resolved.", n.id, path=npath,
                fixes=[Fix("Replace the node with a supported operation")])
            continue
        if op.graph_kind != "agent" or not isinstance(op, AgentOp):
            add("E_OP_GRAPH_KIND", f"'{n.type}' belongs to a '{op.graph_kind}' graph and cannot appear in an 'agent' graph.", n.id, path=npath)
            continue
        if n.version != op.version:
            add("E_UNSUPPORTED_VERSION", f"'{n.type}' version {n.version} is not available (have {op.version}).", n.id, path=npath)
            continue
        try:
            cfgs[n.id] = op.Config.model_validate(n.config)
        except ValidationError as e:
            add("E_CONFIG", f"Invalid config for {n.type}: {_config_message(e)}", n.id, path=f"{npath}/config")
            continue
        ops[n.id] = op  # type: ignore[assignment]
    node_ids = set(ops) | {n.id for n in graph.nodes}

    # state access, templates, model providers, extra checks
    writers: dict[str, list[str]] = {}
    for nid, op in ops.items():
        cfg = cfgs[nid]
        reads, writes = op.reads(cfg), op.writes(cfg)
        r.node_info[nid] = {"reads": reads, "writes": writes, "effects": op.effects(cfg)}
        for fld in reads:
            if resolve_type(spec, fld) is None and spec.field(fld.split(".")[0]) is None:
                add("E_STATE_FIELD_UNKNOWN", f"'{nid}' reads state field '{fld}', which the state schema does not declare.", nid, path=f"/nodes/{nid}/config",
                    fixes=[Fix(f"Add a '{fld.split('.')[0]}' field to the state schema")])
        for fld in writes:
            if spec.field(fld) is None:
                add("E_STATE_FIELD_UNKNOWN", f"'{nid}' writes state field '{fld}', which the state schema does not declare.", nid, path=f"/nodes/{nid}/config",
                    fixes=[Fix(f"Add a '{fld}' field to the state schema")])
            writers.setdefault(fld, []).append(nid)
        for loc, tpl in op.templates(cfg):
            for v in template_vars(tpl):
                if spec.field(v.split(".")[0]) is None:
                    add("E_TEMPLATE_VAR", f"Template variable {{{v}}} at {loc} is not a field of the state schema.", nid, path=f"/nodes/{nid}/config/{loc}")
        for code, msg in op.check(cfg, spec):
            add(code, f"{nid}: {msg}", nid, path=f"/nodes/{nid}/config")
        for ms in op.model_specs(cfg):
            if ms.provider == "anthropic" and not ms.api_key:
                add("E_MODEL_NO_KEY", "The Anthropic provider needs an API key given as a secret reference.", nid, path=f"/nodes/{nid}/config/model/api_key", severity="warning")
            if ms.provider == "openai_compatible":
                from .openai_compat import check_base_url
                try:
                    check_base_url(ms.base_url, bool(ms.api_key))
                except ValueError as e:
                    add("E_MODEL_ENDPOINT", str(e), nid, path=f"/nodes/{nid}/config/model/base_url")
            if ms.provider == "fixture":
                add("W_FIXTURE_MODEL", "This block uses the FIXTURE model: scripted replies for control-flow tests, not a language model.", nid, path=f"/nodes/{nid}/config/model", severity="warning")

    # ---------------------------------------------------------------- edges
    out_edges: dict[str, list[str]] = {n: [] for n in node_ids | {START}}
    in_edges: dict[str, list[str]] = {n: [] for n in node_ids | {END}}
    edge_ids: set[str] = set()
    for e in graph.edges:
        epath = f"/edges/{e.id}"
        if e.id in edge_ids:
            add("E_DUPLICATE_ID", f"Edge id '{e.id}' is used more than once.", path=epath)
            continue
        edge_ids.add(e.id)
        if e.kind != "control":
            add("E_EDGE_KIND", f"Edge '{e.id}' has kind '{e.kind}'; agent graphs use 'control' transitions.", path=epath, fixes=[Fix("Set the edge kind to 'control'")])
            continue
        bad = False
        for end, side in ((e.from_, "from"), (e.to, "to")):
            if end.node not in node_ids and end.node not in RESERVED_NODE_IDS:
                add("E_DANGLING_EDGE", f"Edge '{e.id}' {side} refers to missing node '{end.node}'.", path=f"{epath}/{side}")
                bad = True
        if bad:
            continue
        if e.from_.node == END or e.to.node == START:
            add("E_EDGE_ENDPOINT", f"Edge '{e.id}' goes {'out of END' if e.from_.node == END else 'into START'}.", path=epath)
            continue
        if e.from_.port != "out" or e.to.port != "in":
            add("E_UNKNOWN_PORT", f"Agent nodes have one input port 'in' and one output port 'out' (edge '{e.id}').", e.to.node, e.to.port, epath)
            continue
        out_edges[e.from_.node].append(e.to.node)
        in_edges[e.to.node].append(e.from_.node)

    if not out_edges[START]:
        add("E_NO_START", "No transition leaves START: the graph has no entry point.", path="/edges", fixes=[Fix("Connect START to the first node")])

    # ---------------------------------------------------------------- routes and predicates
    route_targets: dict[str, list[str]] = {}
    routed: set[str] = set()
    route_ids: set[str] = set()
    for rt in spec.routes:
        rp = f"/agent/routes/{rt.id}"
        if rt.id in route_ids:
            add("E_DUPLICATE_ID", f"Route id '{rt.id}' is used more than once.", rt.from_, path=rp)
            continue
        route_ids.add(rt.id)
        if rt.from_ not in node_ids:
            add("E_ROUTE_SOURCE", f"Route '{rt.id}' starts at unknown node '{rt.from_}'.", path=rp)
            continue
        if rt.from_ in routed:
            add("E_ROUTE_DUPLICATE", f"Node '{rt.from_}' has more than one route.", rt.from_, path=rp)
            continue
        routed.add(rt.from_)
        if out_edges[rt.from_]:
            add("E_ROUTE_AND_EDGE", f"Node '{rt.from_}' has both a conditional route and fixed transitions; a node leaves by one or the other.", rt.from_, path=rp)
        if not rt.cases:
            add("E_ROUTE_EMPTY", f"Route '{rt.id}' has no cases.", rt.from_, path=rp, severity="warning")
        targets = []
        case_ids: set[str] = set()
        for c in rt.cases + []:
            cp = f"{rp}/cases/{c.id}"
            if c.id in case_ids:
                add("E_DUPLICATE_ID", f"Case id '{c.id}' repeats in route '{rt.id}'.", rt.from_, path=cp)
            case_ids.add(c.id)
            if c.to not in node_ids and c.to != END:
                add("E_ROUTE_TARGET", f"Case '{c.label or c.id}' of route '{rt.id}' goes to unknown node '{c.to}'.", rt.from_, path=cp)
            targets.append(c.to)
            _check_predicate(c.when, spec, rt.from_, cp + "/when", add)
        if rt.default not in node_ids and rt.default != END:
            add("E_ROUTE_TARGET", f"The default branch of route '{rt.id}' goes to unknown node '{rt.default}'.", rt.from_, path=rp + "/default")
        targets.append(rt.default)
        route_targets[rt.from_] = targets
        for t in targets:
            if t in in_edges:
                in_edges[t].append(rt.from_)

    # ---------------------------------------------------------------- dead ends, reachability, loops
    succ = {n: list(out_edges.get(n, [])) + route_targets.get(n, []) for n in node_ids}
    succ[START] = out_edges[START]
    for nid in sorted(node_ids):
        if not succ.get(nid):
            add("E_DEAD_END", f"Node '{nid}' has no outgoing transition or route; connect it to another node or to END.", nid, path=f"/nodes/{nid}", fixes=[Fix(f"Connect {nid} to END")])
    reach, stack = set(), [START]
    while stack:
        x = stack.pop()
        for y in succ.get(x, []):
            if y not in reach and y != END:
                reach.add(y)
                stack.append(y)
    r.order = _bfs_order(succ, START, node_ids)
    for nid in sorted(node_ids - reach):
        add("E_UNREACHABLE_NODE", f"Node '{nid}' cannot be reached from START.", nid, path=f"/nodes/{nid}")
    if END not in {t for ts in succ.values() for t in ts} and node_ids:
        add("E_NO_END", "No transition reaches END: the graph can only stop by hitting its step limit.", path="/edges", severity="warning")
    sccs = _cycles(succ, node_ids)
    loops = []
    for comp in sccs:
        exits = [(n, t) for n in comp for t in succ.get(n, []) if t not in comp]
        has_route = any(n in route_targets for n in comp)
        loops.append({"nodes": sorted(comp), "hasConditionalExit": has_route and bool(exits), "boundedBy": "route predicate and step limit" if has_route and exits else "step limit only"})
        if not (has_route and exits):
            add("W_LOOP_STEP_LIMIT_ONLY", f"The cycle {sorted(comp)} has no conditional exit; it ends only when the step limit ({spec.limits.maxSteps}) is reached.",
                sorted(comp)[0], path="/agent/limits", severity="warning")
    r.analysis["loops"] = loops

    # ---------------------------------------------------------------- parallel branches and joins
    groups = [(START, out_edges[START])] + [(n, out_edges[n]) for n in node_ids if len(out_edges[n]) > 1]
    for src, tgts in groups:
        if len(set(tgts)) > 1:
            for fld, ws in writers.items():
                conc = [t for t in set(tgts) if t in ws]
                f = spec.field(fld)
                if len(conc) > 1 and f is not None and f.reducer.kind == "replace":
                    add("E_CONCURRENT_WRITE", f"Branches {sorted(conc)} start together after '{src}' and both write '{fld}', which uses the 'replace' reducer: LangGraph rejects concurrent "
                        "updates to such a field. Give it a reducer such as append or add, or write different fields.", conc[0], path=f"/agent/state/{fld}/reducer",
                        fixes=[Fix(f"Use an 'append' or 'add' reducer for '{fld}'")])
    for j in spec.joins:
        if j.node not in node_ids:
            add("E_JOIN", f"Join waits at unknown node '{j.node}'.", path="/agent/joins")
            continue
        for s in j.waitFor:
            if s not in node_ids or j.node not in out_edges.get(s, []):
                add("E_JOIN", f"Join at '{j.node}' waits for '{s}', which has no transition into it.", j.node, path="/agent/joins")
    r.analysis["parallel"] = [{"from": s, "branches": sorted(set(t))} for s, t in groups if len(set(t)) > 1]
    r.analysis["joins"] = [j.model_dump() for j in spec.joins]

    # ---------------------------------------------------------------- indexes and policies
    iids: set[str] = set()
    for ix in spec.indexes:
        if ix.id in iids or not ID_RE.match(ix.id):
            add("E_INDEX", f"Index id '{ix.id}' must be unique and an identifier.", path=f"/agent/indexes/{ix.id}")
        iids.add(ix.id)
        if ix.splitter.chunkOverlap >= ix.splitter.chunkSize:
            add("E_INDEX", f"Index '{ix.id}': chunk overlap must be smaller than the chunk size.", path=f"/agent/indexes/{ix.id}/splitter")
    pids: set[str] = set()
    for p in spec.policies:
        pp = f"/agent/policies/{p.id}"
        if p.id in pids or not ID_RE.match(p.id):
            add("E_POLICY", f"Policy id '{p.id}' must be unique and an identifier.", path=pp)
        pids.add(p.id)
        for code, msg, path in check_policy(p, spec):
            add(code, f"policy '{p.id}': {msg}", path=pp + path)
    r.analysis["inputFields"] = [f.name for f in spec.state if f.name not in writers]  # never written by a node: supplied when a run starts
    r.analysis["policies"] = [p.id for p in spec.policies]
    r.analysis["limits"] = spec.limits.model_dump()
    r.analysis["usesFixtureModel"] = any(ms.provider == "fixture" for nid, op in ops.items() for ms in op.model_specs(cfgs[nid]))
    r.resolved = cfgs
    return r


def check_policy(p, spec: AgentSpec) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if not p.stages:
        return [("E_POLICY", "a policy needs at least one stage", "/stages")]
    if p.stages[0].op != "retrieve":
        out.append(("E_POLICY_FIRST_STAGE", "the first stage must be 'retrieve' (it defines which stored records are eligible)", "/stages/0"))
    if sum(1 for s in p.stages if s.op == "retrieve") > 1:
        out.append(("E_POLICY", "only one retrieve stage is allowed", "/stages"))
    ids: set[str] = set()
    for i, s in enumerate(p.stages):
        sp = f"/stages/{s.id}"
        if s.id in ids or not ID_RE.match(s.id):
            out.append(("E_POLICY", f"stage id '{s.id}' must be unique and an identifier", sp))
        ids.add(s.id)
        for m in pol.validate_stage(s.op, s.config):
            out.append(("E_CONFIG", f"stage '{s.id}': {m}", sp + "/config"))
        if s.op == "filter":
            for f in pred_fields(s.config.get("where") or {}):
                if f.split(".")[0] not in RECORD_FIELDS and f.split(".")[0] != "metadata":
                    out.append(("E_PREDICATE_FIELD", f"stage '{s.id}': record field '{f}' is unknown (use {sorted(RECORD_FIELDS)} or metadata.<key>)", sp + "/config/where"))
        if s.op == "rank":
            w = s.config.get("weights") or {}
            if set(w) - {"recency", "relevance", "importance"}:
                out.append(("E_CONFIG", f"stage '{s.id}': weights may only name recency, relevance, importance", sp + "/config/weights"))
            if all(float(v) <= 0 for v in w.values()) and w:
                out.append(("E_CONFIG", f"stage '{s.id}': at least one weight must be positive", sp + "/config/weights"))
        if s.op in ("rank", "retrieve") and (s.config.get("method") == "similarity" or (s.op == "rank" and float((s.config.get("weights") or {}).get("relevance", 0)) > 0)) and not p.query.strip():
            out.append(("E_POLICY_QUERY", f"stage '{s.id}' uses relevance but the policy has no query template", "/query"))
    for v in template_vars(p.query):
        if spec.field(v.split(".")[0]) is None:
            out.append(("E_TEMPLATE_VAR", f"query variable {{{v}}} is not a state field", "/query"))
    return out


def _check_predicate(p: dict[str, Any], spec: AgentSpec, node: str, path: str, add) -> None:
    if not p or p.get("always") is True:
        return
    n_forms = sum(1 for k in ("all", "any", "not", "field") if k in p)
    if n_forms != 1:
        add("E_PREDICATE", "A predicate is exactly one of: all / any / not / a field comparison.", node, path=path)
        return
    for k in ("all", "any"):
        if k in p:
            if not isinstance(p[k], list) or not p[k]:
                add("E_PREDICATE", f"'{k}' needs a non-empty list of conditions.", node, path=path)
                return
            for i, q in enumerate(p[k]):
                _check_predicate(q, spec, node, f"{path}/{k}/{i}", add)
            return
    if "not" in p:
        _check_predicate(p["not"], spec, node, path + "/not", add)
        return
    fld, op = p["field"], p.get("op", "==")
    if not isinstance(fld, str) or not PATH_RE.match(fld):
        add("E_PREDICATE_FIELD", f"'{fld}' is not a state path.", node, path=path + "/field")
        return
    t = resolve_type(spec, fld)
    if t is None:
        known = spec.field(fld.split(".")[0])
        if known is None:
            add("E_PREDICATE_FIELD", f"Predicate reads '{fld}', which the state schema does not declare.", node, path=path + "/field")
            return
        add("W_PREDICATE_UNTYPED", f"'{fld}' is not a declared property of object field '{known.name}'; its type cannot be checked.", node, path=path + "/field", severity="warning")
        return
    if op not in CMP_OPS:
        add("E_PREDICATE_OP", f"Unknown operator '{op}'.", node, path=path + "/op")
        return
    eff = "integer" if p.get("fn") == "len" else t
    if p.get("fn") == "len":
        if t not in LISTY + ("text", "object"):
            add("E_PREDICATE_TYPE", f"len() applies to lists, text and objects, not '{t}'.", node, path=path + "/fn")
            return
    if op in NUM_OPS and eff not in NUMERIC:
        add("E_PREDICATE_TYPE", f"Operator '{op}' compares numbers, but '{fld}' is '{t}'.", node, path=path + "/op",
            fixes=[Fix("Use == / != for non-numeric values")])
    if op in ("is_true", "is_false") and eff != "boolean":
        add("E_PREDICATE_TYPE", f"'{op}' applies to boolean fields, but '{fld}' is '{t}'.", node, path=path + "/op")
    if op in ("contains", "not_contains") and eff not in LISTY + ("text",):
        add("E_PREDICATE_TYPE", f"'{op}' applies to text and lists, but '{fld}' is '{t}'.", node, path=path + "/op")
    if op == "startswith" and eff != "text":
        add("E_PREDICATE_TYPE", f"'startswith' applies to text, but '{fld}' is '{t}'.", node, path=path + "/op")
    if op in UNARY:
        return
    if isinstance(p.get("other"), str):
        ot = resolve_type(spec, p["other"])
        if ot is None:
            add("E_PREDICATE_FIELD", f"Predicate compares with '{p['other']}', which the state schema does not declare.", node, path=path + "/other")
        elif (ot in NUMERIC) != (eff in NUMERIC) and op in ("==", "!=", "<", "<=", ">", ">="):
            add("E_PREDICATE_TYPE", f"Cannot compare '{fld}' ({eff}) with '{p['other']}' ({ot}).", node, path=path + "/other")
        return
    if "value" not in p:
        add("E_PREDICATE", "A comparison needs a value (or another field).", node, path=path + "/value")
        return
    v = p["value"]
    if op in NUM_OPS or (op in ("==", "!=") and eff in NUMERIC):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            add("E_PREDICATE_TYPE", f"'{fld}' is numeric; the value {v!r} is not.", node, path=path + "/value")
    elif op in ("==", "!=") and eff == "boolean" and not isinstance(v, bool):
        add("E_PREDICATE_TYPE", f"'{fld}' is boolean; the value {v!r} is not.", node, path=path + "/value")
    elif op in ("==", "!=", "startswith") and eff == "text" and not isinstance(v, str):
        add("E_PREDICATE_TYPE", f"'{fld}' is text; the value {v!r} is not.", node, path=path + "/value")


def _bfs_order(succ: dict[str, list[str]], start: str, node_ids: set[str]) -> list[str]:
    order, seen, q = [], {start}, [start]
    while q:
        x = q.pop(0)
        for y in succ.get(x, []):
            if y in node_ids and y not in seen:
                seen.add(y)
                order.append(y)
                q.append(y)
    return order + sorted(node_ids - seen)


def _cycles(succ: dict[str, list[str]], node_ids: set[str]) -> list[set[str]]:
    """Strongly connected components with a cycle (Tarjan, iterative enough for small graphs)."""
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]

    def go(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in succ.get(v, []):
            if w not in node_ids:
                continue
            if w not in index:
                go(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            if len(comp) > 1 or v in succ.get(v, []):
                out.append(comp)

    for v in sorted(node_ids):
        if v not in index:
            go(v)
    return out
