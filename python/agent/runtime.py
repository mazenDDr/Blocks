"""Compile an `agent` graph to a native LangGraph StateGraph and run it with a SQLite checkpointer.

Native semantics are kept: the state is a TypedDict whose reduced fields carry their reducer as `Annotated` metadata, nodes are plain
LangGraph node functions, transitions are `add_edge`, routes are `add_conditional_edges`, joins are multi-source `add_edge`,
interrupts are `langgraph.types.interrupt` and resumption is `Command(resume=...)`. What is added around that is observation: every
node records the state it read and the diff it produced, every route records the predicate values, every model call records the
exact message list it sent (context inspector)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from artifact_store import ArtifactStore, IllegalTransition
from graph_core import registry
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph

from .blocks import AgentOp, NodeCtx, NodeFailure
from .index import IndexStore
from .memory import MemoryStore, args_hash, new_id
from .models import ModelSpec, ModelUnavailable, estimate_tokens, invoke_chat, ollama_context_length, resolve_settings
from .spec import END, START, AgentSpec, apply_reducer, eval_predicate, get_path, reducer_fn, render_predicate, pred_fields

PREVIEW_LIMIT = 4000


class BudgetExhausted(Exception):
    def __init__(self, kind: str, limit: Any, used: Any):
        super().__init__(f"budget '{kind}' exhausted: used {used} of {limit}")
        self.kind, self.limit, self.used = kind, limit, used


def preview(v: Any) -> Any:
    """A JSON-safe value for events; very large values are replaced by a labelled preview with the hash of the full value."""
    try:
        s = json.dumps(v, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(v)
    if len(s) <= PREVIEW_LIMIT:
        return json.loads(s) if s else None
    return {"truncated": True, "bytes": len(s), "sha256": hashlib.sha256(s.encode()).hexdigest(), "preview": s[:300]}


class Emitter:
    """Ordered, thread-safe event appender that continues the run's sequence (a resumed run keeps counting)."""

    def __init__(self, store: ArtifactStore, run_id: str, graph_hash: str):
        self.store, self.run_id, self.graph_hash = store, run_id, graph_hash
        self.seq = store.max_seq(run_id) + 1
        self.lock = threading.Lock()

    def emit(self, type_: str, node_id: str | None = None, **data: Any) -> int:
        with self.lock:
            s = self.seq
            self.seq += 1
            self.store.append_event(self.run_id, s, time.time(), type_, self.graph_hash, node_id, data)
            return s


class Runtime:
    def __init__(self, graph: Graph, spec: AgentSpec, store: ArtifactStore, workbench: Path, run_id: str, thread_id: str, graph_hash: str, replays: set[tuple[str, int]] | None = None):
        self.graph, self.spec, self.store, self.workbench = graph, spec, store, Path(workbench)
        self.run_id, self.thread_id, self.graph_hash = run_id, thread_id, graph_hash
        self.em = Emitter(store, run_id, graph_hash)
        self.memory = MemoryStore(workbench)
        self.indexes = IndexStore(workbench)
        self.outbox = self.workbench / "agent" / "outbox"
        self.replays = replays or set()
        self.t0 = time.monotonic()
        self.model_calls = 0
        self.tool_calls = 0
        self.tokens = 0
        self.fixture_counters: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()
        self._built: dict[str, dict[str, Any]] = {}
        self.prior_seconds = 0.0

    # ------------------------------------------------------------------ plumbing
    def emit(self, type_: str, node: str | None = None, **data: Any) -> int:
        return self.em.emit(type_, node, **data)

    def resolve_dir(self, d: str) -> Path:
        p = Path(d).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    def effect_key(self, ctx: NodeCtx, tool: str, args: dict[str, Any]) -> str:
        """Identity of one protected effect: the same thread, node, tool and arguments. A replay (resume after an interrupt, or a time-travel re-run
        from an earlier checkpoint) re-executes the node, finds this key in the ledger and skips the effect. The step number is deliberately NOT part
        of the key (a replay can run the node at a different step). To perform the same call repeatedly, make its arguments differ (e.g. include a counter)."""
        return f"{self.thread_id}:{ctx.node_id}:{tool}:{args_hash(args)}"

    def count_tool(self) -> None:
        with self._lock:
            self.tool_calls += 1

    def ensure_index(self, ispec) -> dict[str, Any]:
        with self._lock:
            if ispec.id not in self._built:
                man = self.indexes.ensure(ispec)
                self._built[ispec.id] = man
                self.emit("index_ready", None, index=ispec.id, action=man["action"], chunks=man["chunks"], documents=len(man["documents"]), embedding=man["embedding"]["identity"],
                          embeddingsReused=man.get("embeddingsReused"), embeddingsComputed=man.get("embeddingsComputed"), identity=man["identity"][:16],
                          scoreInterpretation=man["scoreInterpretation"])
            return self._built[ispec.id]

    # ------------------------------------------------------------------ budgets
    def check_budget(self, kind: str | None = None) -> None:
        lim = self.spec.limits
        if lim.maxSeconds is not None and time.monotonic() - self.t0 > lim.maxSeconds:
            raise BudgetExhausted("maxSeconds", lim.maxSeconds, round(time.monotonic() - self.t0, 2))
        if kind == "model" and lim.maxModelCalls is not None and self.model_calls >= lim.maxModelCalls:
            raise BudgetExhausted("maxModelCalls", lim.maxModelCalls, self.model_calls)
        if kind == "model" and lim.maxTokens is not None and self.tokens >= lim.maxTokens:
            raise BudgetExhausted("maxTokens", lim.maxTokens, self.tokens)
        if kind == "tool" and lim.maxToolCalls is not None and self.tool_calls >= lim.maxToolCalls:
            raise BudgetExhausted("maxToolCalls", lim.maxToolCalls, self.tool_calls)

    # ------------------------------------------------------------------ model calls + context record
    def model_call(self, ctx: NodeCtx, spec: ModelSpec, messages: list[dict[str, Any]], purpose: str, attempt: int = 1, json_schema: dict | None = None,
                   validate: Callable[[str], list[str]] | None = None, messages_field: str | None = None, tools: list[dict] | None = None) -> dict[str, Any]:
        with self._lock:
            self.check_budget("model")
        resolved, ignored = resolve_settings(spec)
        sent = [{"role": m["role"], "content": m["content"], **{k: m[k] for k in ("tool_calls", "tool_call_id") if k in m}} for m in messages]
        call_id = new_id("call")
        counter = self.fixture_counters.setdefault(ctx.node_id, {})
        try:
            res = invoke_chat(spec, sent, counter, json_schema, tools)
        except ModelUnavailable as e:
            self.emit("model_call_failed", ctx.node_id, callId=call_id, provider=spec.provider, model=spec.model, code=e.code, message=e.message, purpose=purpose, attempt=attempt)
            raise NodeFailure(e.code, e.message)
        with self._lock:
            self.model_calls += 1
            u = res["usage"]
            self.tokens += (u["inputTokens"] or 0) + (u["outputTokens"] or 0) if u["source"] == "provider" else estimate_tokens("".join(m["content"] for m in sent)) + estimate_tokens(res["text"])
        ctxrec = self.build_context(call_id, ctx, spec, messages, res, purpose, attempt, resolved, ignored, messages_field)
        sha = self.store.add_artifact(self.run_id, "model_context", json.dumps(ctxrec, default=str).encode(), "complete", None,
                                      {"callId": call_id, "node": ctx.node_id, "graph_hash": self.graph_hash})["sha256"]
        vres = validate(res["text"]) if validate else None
        cost = ({"amount": 0.0, "currency": None, "basis": "local inference: no per-token price"} if spec.provider == "ollama" else
                {"amount": None, "currency": None, "basis": "unknown: no pricing table is configured, so no cost is claimed"})
        self.emit("model_call", ctx.node_id, callId=call_id, purpose=purpose, attempt=attempt, provider=spec.provider, model=spec.model, fixture=spec.provider == "fixture",
                  resolved=resolved, ignored=ignored, latencyMs=res["latencyMs"], usage=res["usage"], cost=cost, contextSha256=sha, messages=len(sent),
                  tokensEstimate=ctxrec["tokens"]["estimateTotal"], responseChars=len(res["text"]), response=res["text"][:2000], validationErrors=vres,
                  responseMetadata=res["responseMetadata"], step=ctx.step, **({"toolCalls": res["toolCalls"], "toolsOffered": [t["function"]["name"] for t in tools]} if tools else {}))
        return {"text": res["text"], "callId": call_id, "usage": res["usage"], "toolCalls": res.get("toolCalls", [])}

    def linked_sources(self, messages_field: str | None) -> tuple[list[str], list[str]]:
        """Selections and retrievals that FED this call even when nothing from them reached it (an empty selection leaves no segment to point at):
        follow the graph statically from the messages field to the prompt node(s) that write it, the state fields they read, and the memory-select /
        retrieve nodes that write those fields; take each such node's latest recorded result in this run."""
        if not messages_field:
            return [], []
        prompts = [n for n in self.graph.nodes if n.type == "agent.prompt" and n.config.get("output_field", "prompt_messages") == messages_field]
        fields = {it["field"] for n in prompts for it in n.config.get("items", []) if it.get("kind") != "template" and it.get("field")}
        apps, rets = [], []
        for n in self.graph.nodes:
            if n.type == "agent.memory_select" and n.config.get("output_field", "memory") in fields:
                a = [x for x in self.memory.applications(run_id=self.run_id) if x.get("node") == n.id]
                if a:
                    apps.append(a[-1]["id"])
            elif n.type == "agent.retrieve" and n.config.get("output_field", "docs") in fields:
                with_ret = [e for e in self.store.events(self.run_id, -1, ("retrieval",)) if e["node_id"] == n.id]
                if with_ret:
                    rets.append(with_ret[-1]["data"]["retrievalId"])
        return apps, rets

    def build_context(self, call_id, ctx, spec, messages, res, purpose, attempt, resolved, ignored, messages_field=None) -> dict[str, Any]:
        """Exactly what was sent, with every segment linked to its source, plus the stored-but-unused records that did NOT reach it."""
        apps, rets = {}, {}
        la, lr = self.linked_sources(messages_field)
        apps.update({a: None for a in la})
        rets.update({r: None for r in lr})
        for m in messages:
            for s in m.get("segments", []):
                src = s.get("source", {})
                if src.get("applicationId"):
                    apps[src["applicationId"]] = None
                if src.get("retrievalId"):
                    rets[src["retrievalId"]] = None
        excluded, included_ids = [], {s["source"].get("recordId") for m in messages for s in m.get("segments", []) if s.get("source", {}).get("recordId")}
        for aid in apps:
            app = self.memory.get_application(aid)
            if not app:
                continue
            apps[aid] = {"policy": app["policyId"], "node": app.get("node"), "universe": len(app["records"])}
            for rid, r in app["records"].items():
                if r["status"] in ("excluded", "summarized") and rid not in included_ids:
                    excluded.append({"kind": "memory_record", "id": rid, "applicationId": aid, "store": r["store"], "text": r["text"][:300], "status": r["status"],
                                     "stage": (r["excludedAt"] or {}).get("stage"), "op": (r["excludedAt"] or {}).get("op"), "reason": (r["excludedAt"] or {}).get("reason"),
                                     "scores": r["scores"], "tokensEstimate": r["tokens"]})
        shown = {s["source"].get("chunkId") for m in messages for s in m.get("segments", []) if s.get("source", {}).get("chunkId")}
        for rid in rets:
            ret = self.memory.get_retrieval(rid)
            if not ret:
                continue
            rets[rid] = {"index": ret["index"], "query": ret["query"], "node": ret.get("node")}
            for d in ret["excluded"]:
                if d["chunk_id"] not in shown:
                    excluded.append({"kind": "retrieved_chunk", "id": d["chunk_id"], "retrievalId": rid, "text": d["text"][:300], "status": "excluded", "stage": "retrieve",
                                     "op": "retrieve", "reason": d["reason"], "scores": {"similarity": d["score"]}, "tokensEstimate": estimate_tokens(d["text"])})
        est = [estimate_tokens(m["content"]) for m in messages]
        limit = ollama_context_length(spec.model, spec.base_url) if spec.provider == "ollama" else None
        return {"callId": call_id, "runId": self.run_id, "threadId": self.thread_id, "graphHash": self.graph_hash, "node": ctx.node_id, "step": ctx.step, "purpose": purpose, "attempt": attempt,
                "provider": spec.provider, "model": spec.model, "fixture": spec.provider == "fixture", "resolved": resolved, "ignored": ignored,
                "messages": [{"index": i, "role": m["role"], "content": m["content"], "segments": m.get("segments", []), "tokensEstimate": est[i]} for i, m in enumerate(messages)],
                "providerRequest": [{"role": m["role"], "content": m["content"]} for m in messages],
                "response": res["text"], "usage": res["usage"], "latencyMs": res["latencyMs"],
                "tokens": {"estimateTotal": sum(est), "estimateBasis": "ceil(characters / 4); an estimate, not a tokenizer count",
                           "providerInput": res["usage"]["inputTokens"], "providerOutput": res["usage"]["outputTokens"], "providerReported": res["usage"]["source"] == "provider",
                           "modelLimit": limit, "modelLimitSource": "ollama /api/show" if limit else "unknown for this provider/model", "reservedOutput": resolved.get("max_tokens")},
                "excluded": excluded, "applications": apps, "retrievals": rets,
                "boundary": "This is the request the application sent and the metadata the provider returned. Provider-side formatting and hidden tokens are not observable."}


# ===================================================================================================== compile
def build_state_type(spec: AgentSpec):
    anns: dict[str, Any] = {}
    for f in spec.state:
        fn = reducer_fn(f.reducer)
        anns[f.name] = Annotated[Any, fn] if fn else Any
    return TypedDict("AgentState", anns)  # type: ignore[operator]


def fill_defaults(spec: AgentSpec, state: dict[str, Any]) -> dict[str, Any]:
    out = dict(state)
    for f in spec.state:
        if f.name not in out or out[f.name] is None and f.type != "json":
            out[f.name] = f.initial()
    return out


def compile_graph(graph: Graph, rt: Runtime, checkpointer):
    from langgraph.errors import GraphBubbleUp
    from langgraph.graph import END as LG_END
    from langgraph.graph import START as LG_START
    from langgraph.graph import StateGraph

    def lg(n: str) -> str:
        return LG_START if n == START else LG_END if n == END else n

    spec = rt.spec
    sg = StateGraph(build_state_type(spec))

    def make_node(nid: str, op: AgentOp, cfg: Any):
        def node(state: dict[str, Any], config) -> dict[str, Any]:
            step = (config.get("metadata") or {}).get("langgraph_step")
            full = fill_defaults(spec, state)
            ctx = NodeCtx(nid, step, config)
            replay = (nid, step) in rt.replays
            rt.check_budget("tool" if op.type == "agent.tool_call" else None)
            rt.emit("node_started", nid, type=op.type, step=step, replay=replay)
            t0 = time.perf_counter()
            try:
                update = op.execute(rt, ctx, cfg, full)
            except GraphBubbleUp:
                raise  # interrupt: not a failure
            except (BudgetExhausted, NodeFailure):
                raise
            except Exception as e:  # noqa: BLE001
                rt.emit("node_failed", nid, code="E_NODE_EXCEPTION", message=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-1500:], step=step)
                raise
            dur = round((time.perf_counter() - t0) * 1000, 1)
            changes = []
            for k, v in update.items():
                f = spec.field(k)
                before = full.get(k)
                after = apply_reducer(f.reducer, before, v) if f else v
                changes.append({"field": k, "reducer": f.reducer.kind if f else "replace", "written": preview(v), "before": preview(before), "after": preview(after), "changed": after != before})
            reads = {r.split(".")[0]: preview(full.get(r.split(".")[0])) for r in op.reads(cfg)}
            rt.emit("state_update", nid, step=step, reads=reads, changes=changes, durationMs=dur)
            rt.emit("node_finished", nid, type=op.type, step=step, durationMs=dur, wrote=list(update))
            return update
        node.__name__ = nid
        return node

    nodes_by_id = {n.id: n for n in graph.nodes}
    for n in graph.nodes:
        op = registry.get_op(n.type)
        sg.add_node(n.id, make_node(n.id, op, op.Config.model_validate(n.config)))  # type: ignore[arg-type]

    joined: dict[str, list[str]] = {j.node: j.waitFor for j in spec.joins}
    consumed: set[tuple[str, str]] = {(s, j) for j, srcs in joined.items() for s in srcs}
    for e in graph.edges:
        a, b = e.from_.node, e.to.node
        if (a, b) in consumed:
            continue
        sg.add_edge(lg(a), lg(b))
    for j, srcs in joined.items():
        sg.add_edge(srcs, j)

    for route in spec.routes:
        sg.add_conditional_edges(route.from_, make_router(rt, route), {**{c.id: lg(c.to) for c in route.cases}, "__default__": lg(route.default)})
    return sg.compile(checkpointer=checkpointer)


def make_router(rt: Runtime, route):
    spec = rt.spec

    def router(state: dict[str, Any], config) -> str:
        full = fill_defaults(spec, state)
        evaluated, taken = [], None
        for c in route.cases:
            trace: dict[str, Any] = {}
            res = eval_predicate(c.when, full, trace)
            evaluated.append({"case": c.id, "label": c.label or c.id, "to": c.to, "predicate": render_predicate(c.when), "values": preview(trace), "result": res})
            if res and taken is None:
                taken = c
                break  # first matching case wins; later cases are not evaluated
        key = taken.id if taken else "__default__"
        rt.emit("route_taken", route.from_, route=route.id, step=(config.get("metadata") or {}).get("langgraph_step"), evaluated=evaluated, taken=key,
                label=(taken.label or taken.id) if taken else route.defaultLabel, to=taken.to if taken else route.default, via="case" if taken else "default")
        return key
    return router


# ===================================================================================================== checkpointer
def open_checkpointer(workbench: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    p = Path(workbench) / "agent" / "checkpoints.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False, timeout=30)
    from storage.schema import enable_wal

    enable_wal(conn)
    return SqliteSaver(conn)


def build_input(spec: AgentSpec, user_input: dict[str, Any], thread_exists: bool) -> dict[str, Any]:
    """Initial values for a new thread; on an existing thread turn-scoped fields are reset (`Overwrite` for reduced fields) and thread-scoped
    fields (conversation history) keep what the checkpoint holds."""
    from langgraph.types import Overwrite

    out: dict[str, Any] = {}
    for f in spec.state:
        reduced = f.reducer.kind != "replace"
        if not thread_exists:
            out[f.name] = user_input[f.name] if f.name in user_input else f.initial()
        elif f.scope == "turn":
            v = user_input[f.name] if f.name in user_input else f.initial()
            out[f.name] = Overwrite(v) if reduced else v
        elif f.name in user_input:
            out[f.name] = user_input[f.name]
    return out
