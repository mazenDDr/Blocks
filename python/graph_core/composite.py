"""Composite modules, bounded repeat and typed select: expansion of structured regions to a flat graph.

A model graph may contain nodes of three structural types. They are *not* executable operations; `expand` replaces them
by the nodes they contain, so that validation, shape inference, lowering, code generation, run records and the debugger
all see the same flat graph, with every inner node addressed by its full identity path (`res1/conv_a`):

    core.composite   one instance of a ModuleDef (a reusable subgraph with a typed signature)
    core.repeat      a ModuleDef applied `count` times, with loop-carried state; unrolled, parameters tied by default
    core.select      two ModuleDef branches with identical signatures and a scalar bool predicate; BOTH branches are
                     evaluated and `tensor.where` picks the outputs (data-flow select, not lazy evaluation)

Cycles in the user's graph are still errors; a repeat is a structured, bounded region (the cycle is the loop-carried
wire between iterations, which are separate nodes).

Parameter sharing: instances clone parameters by default. `share: "<other instance>"` (same module and version) makes every
inner node use the other instance's tensors (flat `sharedWith`). A repeat ties its iterations unless `share: "clone"`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .schema import Edge, Endpoint, Graph, ModuleDef, Node, PortSpec
from .types import Diagnostic, Fix

COMPOSITE, REPEAT, SELECT = "core.composite", "core.repeat", "core.select"
CODE_BLOCK = "code.block"
STRUCTURAL = (COMPOSITE, REPEAT, SELECT)
MAX_DEPTH = 8
MAX_REPEAT = 64
MAX_FLAT_NODES = 5000


class ModuleRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module: str = ""
    version: str = "1.0.0"
    args: dict[str, Any] = Field(default_factory=dict)


class CompositeConfig(ModuleRef):
    share: str = "clone"  # "clone" (own parameters) or the id of another instance of the same module+version in this scope


class Carry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str
    output: str


class Termination(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = "fixed_count"


class RepeatConfig(ModuleRef):
    count: int = Field(2, ge=1, le=MAX_REPEAT)
    carry: list[Carry] = Field(default_factory=list)
    termination: Termination = Field(default_factory=Termination)
    share: str = "tied"  # "tied" (one set of parameters reused by every iteration) or "clone"


class SelectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    then: ModuleRef = Field(default_factory=ModuleRef)
    otherwise: ModuleRef = Field(default_factory=ModuleRef)  # JSON key "otherwise" (else is reserved in some host languages)


@dataclass
class Instance:
    """What the editor and the explain panel need to know about an expanded structural node."""

    path: str
    kind: str
    module: str | None
    version: str | None
    inputs: list[str]
    outputs: list[str]
    args: dict[str, Any] = field(default_factory=dict)
    members: list[str] = field(default_factory=list)  # flat node ids below this instance (all depths)
    children: list[str] = field(default_factory=list)  # nested instance paths (direct)
    in_src: dict[str, tuple[str, str] | None] = field(default_factory=dict)  # port -> flat source endpoint
    out_map: dict[str, tuple[str, str]] = field(default_factory=dict)  # port -> flat producer endpoint
    shared_with: str | None = None
    iterations: int | None = None
    termination: str | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "module": self.module, "version": self.version, "inputs": self.inputs,
                "outputs": self.outputs, "args": self.args, "members": self.members, "children": self.children,
                "outMap": {k: list(v) for k, v in self.out_map.items()}, "sharedWith": self.shared_with,
                "iterations": self.iterations, "termination": self.termination, "note": self.note}


@dataclass
class Expansion:
    graph: Graph
    diagnostics: list[Diagnostic] = field(default_factory=list)
    instances: dict[str, Instance] = field(default_factory=dict)
    sig_in: list[tuple[str, str, tuple[str, str] | None, PortSpec]] = field(default_factory=list)  # instance, port, source endpoint, spec
    sig_out: list[tuple[str, str, tuple[str, str], Any]] = field(default_factory=list)  # instance, port, producer endpoint, ModuleOutput
    branch_checks: list[tuple[str, str, tuple[str, str], tuple[str, str]]] = field(default_factory=list)
    pred_checks: list[tuple[str, tuple[str, str] | None]] = field(default_factory=list)
    carry_checks: list[tuple[str, int, str, tuple[str, str], str, tuple[str, str] | None]] = field(default_factory=list)
    expanded: bool = False

    def top_of(self, flat_id: str) -> str:
        """The top-level node a flat node belongs to (first path component)."""
        return flat_id.split("/", 1)[0]


def has_structure(graph: Graph) -> bool:
    return any(n.type in STRUCTURAL or n.type == CODE_BLOCK for n in graph.nodes) or any(n.sharedWith for n in graph.nodes)


def code_interface(defn) -> dict[str, Any]:
    """The part of a code block definition that decides its ports and behaviour, embedded in each node's resolved config.
    The source hash and pinned dependencies are included, so they take part in the node's identity."""
    import hashlib
    import json

    src_hash = hashlib.sha256(defn.source.encode("utf-8")).hexdigest()
    return {"id": defn.id, "version": defn.version, "inputs": [i.model_dump(mode="json") for i in defn.inputs],
            "outputs": [o.model_dump(mode="json") for o in defn.outputs], "config": [c.model_dump(mode="json") for c in defn.config],
            "state": [c.model_dump(mode="json") for c in defn.state], "effects": sorted(defn.effects), "randomness": defn.randomness,
            "differentiable": defn.differentiable, "dependencies": sorted(defn.dependencies), "limits": dict(defn.limits),
            "sourceSha256": src_hash, "source": defn.source,
            "identity": hashlib.sha256(json.dumps([src_hash, sorted(defn.dependencies), defn.randomness, defn.differentiable,
                                                   sorted(defn.effects)], sort_keys=True).encode()).hexdigest()}


def subst(value: Any, params: dict[str, Any]) -> Any:
    """Replace {"$param": name} anywhere inside a config value."""
    if isinstance(value, dict):
        if set(value) == {"$param"}:
            return params.get(value["$param"])
        return {k: subst(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [subst(v, params) for v in value]
    return value


def param_refs(value: Any) -> set[str]:
    if isinstance(value, dict):
        if set(value) == {"$param"}:
            return {value["$param"]}
        return set().union(*(param_refs(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(param_refs(v) for v in value)) if value else set()
    return set()


_TYPES = {"int": int, "float": (int, float), "str": str, "bool": bool, "list": list}


class _Scope:
    def __init__(self, prefix: str, nodes: list[Node], edges: list[Edge], bindings: dict[str, tuple[str, str] | None],
                 params: dict[str, Any], twin: str | None, owner: str, mod_stack: tuple[str, ...], depth: int):
        self.prefix, self.edges, self.bindings, self.params, self.twin = prefix, edges, bindings, params, twin
        self.owner, self.mod_stack, self.depth = owner, mod_stack, depth
        self.nodes = {n.id: n for n in nodes}
        self.state: dict[str, str] = {}
        self.inst: dict[str, Instance] = {}
        self.incoming: dict[tuple[str, str], list[Edge]] = {}
        for e in edges:
            self.incoming.setdefault((e.to.node, e.to.port), []).append(e)


class _Expander:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.out = Expansion(graph=graph)
        self.flat_nodes: list[Node] = []
        self.flat_edges: list[Edge] = []

    # ------------------------------------------------------------------ helpers
    def diag(self, code: str, msg: str, node: str | None = None, port: str | None = None, fixes: list[Fix] | None = None, sev="error"):
        self.out.diagnostics.append(Diagnostic(code, msg, sev, node, port, f"/nodes/{node}" if node else "", fixes or []))  # type: ignore[arg-type]

    def find_module(self, ref: ModuleRef, at: str) -> ModuleDef | None:
        if not ref.module:
            self.diag("E_MODULE_MISSING", f"'{at}' does not name a module.", at)
            return None
        same = [m for m in self.graph.modules if m.id == ref.module]
        for m in same:
            if m.version == ref.version:
                return m
        if same:
            have = ", ".join(sorted(m.version for m in same))
            self.diag("E_MODULE_VERSION", f"'{at}' uses {ref.module} version {ref.version}, but this project has version(s) {have}. "
                      "No silent upgrade is performed.", at, fixes=[Fix(f"Use version {same[-1].version}", at, "version", same[-1].version)])
        else:
            self.diag("E_MODULE_MISSING", f"Module '{ref.module}' is not defined in this project (import it from the module library).", at)
        return None

    def bind_params(self, mod: ModuleDef, args: dict[str, Any], at: str, outer: dict[str, Any]) -> dict[str, Any] | None:
        out, ok = {}, True
        declared = {p.name for p in mod.params}
        for a in args:
            if a not in declared:
                self.diag("E_CONFIG", f"'{at}' passes argument '{a}' that module {mod.id} does not declare (has {sorted(declared)}).", at)
                ok = False
        for p in mod.params:
            v = subst(args[p.name], outer) if p.name in args else p.default
            if p.type in _TYPES and v is not None:
                good = isinstance(v, _TYPES[p.type]) and not (p.type != "bool" and isinstance(v, bool))
                if not good:
                    self.diag("E_CONFIG", f"'{at}' argument '{p.name}'={v!r} is not of the declared type {p.type}.", at)
                    ok = False
            out[p.name] = v
        return out if ok else None

    # ------------------------------------------------------------------ scope
    def run_scope(self, sc: _Scope) -> None:
        for nid in sc.nodes:
            if sc.nodes[nid].type in STRUCTURAL:
                self.ensure(sc, nid)
        for n in sc.nodes.values():
            if n.type in STRUCTURAL:
                continue
            twin = None
            if sc.twin is not None:
                twin = sc.twin + n.id
            elif n.sharedWith:
                twin = sc.prefix + n.sharedWith
            cfg = subst(n.config, sc.params)
            if n.type == CODE_BLOCK:
                d = next((c for c in self.graph.codeBlocks if c.id == cfg.get("block") and c.version == cfg.get("version", "1.0.0")), None)
                if d is None:
                    have = sorted(f"{c.id}@{c.version}" for c in self.graph.codeBlocks)
                    self.diag("E_CODE_BLOCK_MISSING", f"Code block '{cfg.get('block')}' version {cfg.get('version', '1.0.0')} is not defined in this project (has {have}).", sc.prefix + n.id)
                    self.flat_nodes.append(Node(id=sc.prefix + n.id, type=f"unresolved.{n.type}", version=n.version, config=cfg))
                    continue
                cfg = {**cfg, "interface": code_interface(d)}
            self.flat_nodes.append(Node(id=sc.prefix + n.id, type=n.type, version=n.version, config=cfg,
                                        stateRef=n.stateRef, sharedWith=twin))
        for e in sc.edges:
            dst = sc.nodes.get(e.to.node)
            if dst is None or dst.type in STRUCTURAL:
                continue
            src = self.resolve(sc, e.from_.node, e.from_.port, e.id)
            if src is None:
                continue
            self.flat_edges.append(Edge(id=sc.prefix + e.id, kind=e.kind, **{"from": Endpoint(node=src[0], port=src[1])},
                                        to=Endpoint(node=sc.prefix + e.to.node, port=e.to.port)))

    def resolve(self, sc: _Scope, node: str, port: str, edge_id: str = "") -> tuple[str, str] | None:
        if node == "$in":
            return sc.bindings.get(port)
        n = sc.nodes.get(node)
        if n is None:  # dangling: keep the name so the validator reports E_DANGLING_EDGE
            return (sc.prefix + node, port)
        if n.type not in STRUCTURAL:
            return (sc.prefix + node, port)
        self.ensure(sc, node)
        inst = sc.inst.get(node)
        if inst is None:
            return None
        if port not in inst.out_map:
            self.diag("E_UNKNOWN_PORT", f"Node '{inst.path}' has no output port '{port}' (has {list(inst.outputs)}).", inst.path, port)
            return None
        return inst.out_map[port]

    def ensure(self, sc: _Scope, nid: str) -> None:
        st = sc.state.get(nid)
        if st == "done":
            return
        path = sc.prefix + nid
        if st == "active":
            self.diag("E_CYCLE", f"The graph contains a cycle through the structural node '{path}'.", path)
            return
        sc.state[nid] = "active"
        node = sc.nodes[nid]
        try:
            if len(self.flat_nodes) > MAX_FLAT_NODES:
                self.diag("E_TOO_LARGE", f"Expansion exceeded {MAX_FLAT_NODES} nodes.", path)
            elif sc.depth >= MAX_DEPTH:
                self.diag("E_MODULE_RECURSION", f"Modules are nested deeper than {MAX_DEPTH} levels at '{path}'.", path)
            elif node.type == COMPOSITE:
                self.composite(sc, node, path)
            elif node.type == REPEAT:
                self.repeat(sc, node, path)
            else:
                self.select(sc, node, path)
        finally:
            sc.state[nid] = "done"

    def inputs_for(self, sc: _Scope, nid: str, ports: list[str], path: str, edge_ids=None) -> dict[str, tuple[str, str] | None]:
        b: dict[str, tuple[str, str] | None] = {}
        for p in ports:
            es = sc.incoming.get((nid, p), [])
            if len(es) > 1:
                self.diag("E_MULTIPLE_INPUTS", f"Input port '{p}' of '{path}' has more than one incoming edge.", path, p)
            src = self.resolve(sc, es[0].from_.node, es[0].from_.port, es[0].id) if es else None
            if not es:
                self.diag("E_MISSING_INPUT", f"Input '{p}' of '{path}' is not connected.", path, p, fixes=[Fix(f"Connect an output to {path}.{p}")])
            b[p] = src
        return b

    def instantiate(self, sc: _Scope, mod: ModuleDef, path: str, bindings: dict, args: dict, twin: str | None, info: Instance) -> dict[str, tuple[str, str]] | None:
        """Expand one body of `mod` below `path/` and return its output endpoints."""
        if mod.id in sc.mod_stack:
            self.diag("E_MODULE_RECURSION", f"Module '{mod.id}' contains itself ({' -> '.join(sc.mod_stack + (mod.id,))}).", path)
            return None
        params = self.bind_params(mod, args, path, sc.params)
        if params is None:
            return None
        for ps in mod.nodes:
            if ps.id == "$in" or not ps.id or "/" in ps.id:
                self.diag("E_BAD_ID", f"Module {mod.id} has a node id '{ps.id}' that cannot be used.", path)
                return None
        before = len(self.flat_nodes)
        sub = _Scope(path + "/", mod.nodes, mod.edges, bindings, params, twin, path, sc.mod_stack + (mod.id,), sc.depth + 1)
        self.run_scope(sub)
        info.members += [n.id for n in self.flat_nodes[before:]]
        info.children += [c.path for c in sub.inst.values()]
        for c in sub.inst.values():
            self.out.instances[c.path] = c
        outs: dict[str, tuple[str, str]] = {}
        for o in mod.outputs:
            ep = self.resolve(sub, o.from_.node, o.from_.port)
            if ep is None:
                self.diag("E_MODULE_OUTPUT", f"Output '{o.name}' of module {mod.id} (instance '{path}') cannot be resolved from {o.from_.node}.{o.from_.port}.", path)
                continue
            outs[o.name] = ep
            self.out.sig_out.append((path, o.name, ep, o))
        return outs

    def twin_of(self, sc: _Scope, share: str, node: Node, ref: ModuleRef, path: str) -> str | None:
        if share in ("clone", "tied", ""):
            return None
        t = sc.nodes.get(share)
        if t is None or t.type != COMPOSITE:
            self.diag("E_SHARE_TARGET", f"'{path}' shares parameters with '{share}', which is not a composite instance in the same scope.", path)
            return None
        tcfg = t.config
        if tcfg.get("module") != ref.module or tcfg.get("version", "1.0.0") != ref.version:
            self.diag("E_SHARE_TARGET", f"'{path}' shares with '{share}', but they are different modules or versions "
                      f"({tcfg.get('module')}@{tcfg.get('version')} vs {ref.module}@{ref.version}).", path)
            return None
        if tcfg.get("share", "clone") not in ("clone", ""):
            self.diag("E_SHARE_CHAIN", f"'{path}' shares with '{share}', which itself shares. Share with the original instance instead.", path)
            return None
        if share == node.id:
            self.diag("E_SHARE_TARGET", f"'{path}' cannot share parameters with itself.", path)
            return None
        self.ensure(sc, share)
        return sc.prefix + share + "/"

    # ------------------------------------------------------------------ composite
    def composite(self, sc: _Scope, node: Node, path: str) -> None:
        try:
            cfg = CompositeConfig.model_validate(node.config)
        except ValidationError as e:
            self.diag("E_CONFIG", f"Invalid config for {COMPOSITE}: {e.errors()[0]['msg']}", path)
            return self.unresolved(sc, node, path)
        mod = self.find_module(cfg, path)
        if mod is None:
            return self.unresolved(sc, node, path)
        ins = [p.name for p in mod.inputs]
        info = Instance(path, COMPOSITE, mod.id, mod.version, ins, [o.name for o in mod.outputs], dict(cfg.args))
        sc.inst[node.id] = info
        self.out.instances[path] = info
        bindings = self.inputs_for(sc, node.id, ins, path)
        info.in_src = dict(bindings)
        for p in mod.inputs:
            self.out.sig_in.append((path, p.name, bindings[p.name], p))
        twin = self.twin_of(sc, cfg.share, node, cfg, path)
        if cfg.share not in ("clone", "") and twin is None:
            return self.unresolved(sc, node, path)
        if twin is None and sc.twin is not None:
            twin = sc.twin + node.id + "/"  # the enclosing instance shares with another one: so does everything nested inside it
        info.shared_with = twin[:-1] if twin else None
        info.note = (f"shares parameters with '{info.shared_with}'" if twin else "own parameters (default instantiation clones: this instance is initialised independently)")
        outs = self.instantiate(sc, mod, path, bindings, cfg.args, twin, info)
        if outs is None:
            return self.unresolved(sc, node, path)
        info.out_map = outs

    def unresolved(self, sc: _Scope, node: Node, path: str) -> None:
        """Keep the node (as an unavailable operation) so loading never loses information and execution is blocked."""
        sc.inst.pop(node.id, None)
        self.out.instances.pop(path, None)
        self.flat_nodes.append(Node(id=path, type=f"unresolved.{node.type}", version=node.version, config=node.config))

    # ------------------------------------------------------------------ repeat
    def repeat(self, sc: _Scope, node: Node, path: str) -> None:
        try:
            cfg = RepeatConfig.model_validate(node.config)
        except ValidationError as e:
            self.diag("E_CONFIG", f"Invalid config for {REPEAT}: {e.errors()[0]['msg']}", path)
            return self.unresolved(sc, node, path)
        if cfg.termination.kind != "fixed_count":
            self.diag("E_UNSUPPORTED_TERMINATION", f"'{path}' declares termination '{cfg.termination.kind}'. This backend slice supports only the static "
                      "'fixed_count' (a bounded repeat); data-dependent loop termination is not implemented.", path,
                      fixes=[Fix("Use fixed_count", path, "termination", {"kind": "fixed_count"})])
            return self.unresolved(sc, node, path)
        mod = self.find_module(cfg, path)
        if mod is None:
            return self.unresolved(sc, node, path)
        ins, outs_names = [p.name for p in mod.inputs], [o.name for o in mod.outputs]
        for c in cfg.carry:
            if c.input not in ins or c.output not in outs_names:
                self.diag("E_CONFIG", f"carry {c.input}->{c.output} of '{path}' does not match module {mod.id} ports (inputs {ins}, outputs {outs_names}).", path)
                return self.unresolved(sc, node, path)
        if len({c.input for c in cfg.carry}) != len(cfg.carry):
            self.diag("E_CONFIG", f"'{path}' carries the same input twice.", path)
            return self.unresolved(sc, node, path)
        info = Instance(path, REPEAT, mod.id, mod.version, ins, outs_names, dict(cfg.args), iterations=cfg.count, termination="fixed_count")
        sc.inst[node.id] = info
        self.out.instances[path] = info
        ext = self.inputs_for(sc, node.id, ins, path)
        info.in_src = dict(ext)
        carried = {c.input: c.output for c in cfg.carry}
        prev: dict[str, tuple[str, str]] | None = None
        for k in range(cfg.count):
            b = {p: (prev[carried[p]] if (p in carried and prev is not None and carried[p] in prev) else ext[p]) for p in ins}
            if prev is not None:
                for p, o in carried.items():
                    if o in prev:
                        self.out.carry_checks.append((path, k, o, prev[o], p, ext[p]))
            twin = None if (k == 0 or cfg.share == "clone") else f"{path}/it0/"
            if sc.twin is not None and cfg.share == "clone":
                twin = f"{sc.twin}{node.id}/it{k}/"
            elif sc.twin is not None:
                twin = f"{sc.twin}{node.id}/it{k}/"
            sub_path = f"{path}/it{k}"
            iter_info = Instance(sub_path, COMPOSITE, mod.id, mod.version, ins, outs_names, dict(cfg.args))
            self.out.instances[sub_path] = iter_info
            info.children.append(sub_path)
            for p in mod.inputs:
                self.out.sig_in.append((sub_path, p.name, b[p.name], p))
            o = self.instantiate(sc, mod, sub_path, b, cfg.args, twin, iter_info)
            if o is None:
                return self.unresolved(sc, node, path)
            iter_info.out_map, iter_info.in_src = o, dict(b)
            iter_info.shared_with = f"{path}/it0" if twin else None
            info.members += iter_info.members
            prev = o
        info.out_map = dict(prev or {})
        info.note = (f"{cfg.count} iterations, termination fixed_count; parameters "
                     + ("tied across iterations (one tensor set reused)" if cfg.share != "clone" else "cloned per iteration"))

    # ------------------------------------------------------------------ select
    def select(self, sc: _Scope, node: Node, path: str) -> None:
        try:
            cfg = SelectConfig.model_validate(node.config)
        except ValidationError as e:
            self.diag("E_CONFIG", f"Invalid config for {SELECT}: {e.errors()[0]['msg']}", path)
            return self.unresolved(sc, node, path)
        mt, me = self.find_module(cfg.then, path + " (then)"), self.find_module(cfg.otherwise, path + " (otherwise)")
        if mt is None or me is None:
            return self.unresolved(sc, node, path)
        in_t, in_e = [p.name for p in mt.inputs], [p.name for p in me.inputs]
        out_t, out_e = [o.name for o in mt.outputs], [o.name for o in me.outputs]
        if sorted(in_t) != sorted(in_e) or sorted(out_t) != sorted(out_e):
            self.diag("E_BRANCH_MISMATCH", f"Branches of '{path}' must have the same input and output port names: then {in_t}->{out_t}, "
                      f"otherwise {in_e}->{out_e}.", path)
            return self.unresolved(sc, node, path)
        ins = ["pred"] + in_t
        info = Instance(path, SELECT, f"{mt.id}|{me.id}", f"{mt.version}|{me.version}", ins, out_t,
                        {"then": dict(cfg.then.args), "otherwise": dict(cfg.otherwise.args)})
        sc.inst[node.id] = info
        self.out.instances[path] = info
        b = self.inputs_for(sc, node.id, ins, path)
        info.in_src = dict(b)
        self.out.pred_checks.append((path, b["pred"]))
        for br, mod, ref in (("then", mt, cfg.then), ("otherwise", me, cfg.otherwise)):
            for p in mod.inputs:
                self.out.sig_in.append((f"{path}/{br}", p.name, b[p.name], p))
        sub_t, sub_e = Instance(f"{path}/then", COMPOSITE, mt.id, mt.version, in_t, out_t), Instance(f"{path}/otherwise", COMPOSITE, me.id, me.version, in_e, out_e)
        for s_ in (sub_t, sub_e):
            self.out.instances[s_.path] = s_
            s_.in_src = {p: b[p] for p in s_.inputs}
        tw_t = f"{sc.twin}{node.id}/then/" if sc.twin is not None else None
        tw_e = f"{sc.twin}{node.id}/otherwise/" if sc.twin is not None else None
        ot = self.instantiate(sc, mt, f"{path}/then", {p: b[p] for p in in_t}, cfg.then.args, tw_t, sub_t)
        oe = self.instantiate(sc, me, f"{path}/otherwise", {p: b[p] for p in in_e}, cfg.otherwise.args, tw_e, sub_e)
        if ot is None or oe is None:
            return self.unresolved(sc, node, path)
        sub_t.out_map, sub_e.out_map = ot, oe
        info.children += [sub_t.path, sub_e.path]
        info.members += sub_t.members + sub_e.members
        if b["pred"] is None:
            return self.unresolved(sc, node, path)
        for name in out_t:
            wid = f"{path}/{name}"
            self.flat_nodes.append(Node(id=wid, type="tensor.where", config={}))
            info.members.append(wid)
            for port, ep in (("cond", b["pred"]), ("a", ot[name]), ("b", oe[name])):
                self.flat_edges.append(Edge(id=f"{wid}.{port}", kind="tensor", **{"from": Endpoint(node=ep[0], port=ep[1])}, to=Endpoint(node=wid, port=port)))
            self.out.branch_checks.append((path, name, ot[name], oe[name]))
            info.out_map[name] = (wid, "output")
        info.note = "both branches are evaluated; the predicate selects which outputs are used (gradients flow only through the selected branch)"


def expand(graph: Graph) -> Expansion:
    """Flatten structural nodes. Graphs without structure are returned unchanged (same object)."""
    if graph.graphKind != "model" or not has_structure(graph):
        return Expansion(graph=graph)
    ex = _Expander(graph)
    top = _Scope("", graph.nodes, graph.edges, {}, {}, None, "", (), 0)
    ex.run_scope(top)
    for c in top.inst.values():
        ex.out.instances[c.path] = c
    # edges that end at a structural node were consumed as bindings; edges whose destination is missing pass through
    for e in graph.edges:
        if e.to.node not in top.nodes:
            ex.flat_edges.append(e)
    flat = graph.model_copy(update={"nodes": ex.flat_nodes, "edges": ex.flat_edges})
    ex.out.graph, ex.out.expanded = flat, True
    # make members of every instance complete (nested instances are included by `instantiate`)
    return ex.out


def module_harness(graph: Graph, module_id: str, version: str | None = None, shapes: dict[str, list[int | str]] | None = None) -> tuple[Graph, str]:
    """A graph that instantiates one module on tensor inputs, for validating and testing a module on its own.
    Returns (graph, instance id). Input shapes default to the declared PortSpec shapes with unknown dims set to 4."""
    mod = graph.module(module_id, version)
    if mod is None:
        raise KeyError(module_id)
    shapes = shapes or {}
    nodes = [Node(id="m", type=COMPOSITE, config={"module": mod.id, "version": mod.version, "args": {}})]
    edges = []
    for p in mod.inputs:
        sh = shapes.get(p.name) or [("N" if i == 0 else 4) if d is None else d for i, d in enumerate(p.shape or ["N", 4])]
        nodes.append(Node(id=f"in_{p.name}", type="core.tensor_input", config={"shape": sh, "dtype": p.dtype}))
        edges.append(Edge(id=f"w_{p.name}", **{"from": Endpoint(node=f"in_{p.name}", port="value")}, to=Endpoint(node="m", port=p.name)))
    return graph.model_copy(update={"nodes": nodes, "edges": edges}), "m"
