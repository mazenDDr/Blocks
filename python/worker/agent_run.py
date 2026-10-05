"""Run (or resume) an `agent` graph in a worker process.

Events: run_queued, run_preparing, run_started (thread, graph hash, library versions, limits), index_ready, node_started, state_update,
node_finished, route_taken, prompt_rendered, model_call, structured_attempt, retrieval, citations_checked, tool_call, memory_selection,
memory_write, interrupt_raised, interrupt_resumed, run_paused, budget_exhausted, node_failed, cancel_acknowledged, error, run_finished.
Each carries run id, graph hash, sequence and time (the existing event store). The final state is the `final_state` artifact."""
from __future__ import annotations

import importlib.metadata as md
import json
import traceback
from typing import Any, Callable

from pydantic import BaseModel

from agent.blocks import NodeFailure
from agent.runtime import BudgetExhausted, Runtime, build_input, compile_graph, open_checkpointer, preview
from agent.spec import agent_spec
from artifact_store import ArtifactStore, IllegalTransition
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, require_executable


class AgentRunConfig(BaseModel):
    model_config = {"extra": "forbid"}
    kind: str = "agent"
    project_id: str | None = None
    thread_id: str = ""
    input: dict[str, Any] = {}
    rerun_of: str | None = None


def _advance(store: ArtifactStore, run_id: str, new: str) -> None:
    try:
        store.set_status(run_id, new)
    except IllegalTransition:
        if store.get_run(run_id)["status"] != "cancelling":
            raise


def _libs() -> dict[str, str]:
    out = {}
    for p in ("langgraph", "langgraph-checkpoint-sqlite", "langchain-core", "langchain-ollama", "langchain-anthropic", "faiss-cpu"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def run_agent(graph: Graph, cfg: AgentRunConfig, store: ArtifactStore, run_id: str, should_cancel: Callable[[], bool] = lambda: False, resume: dict[str, Any] | None = None) -> str:
    from langgraph.types import Command

    graph_hash = semantic_hash(graph)
    spec = agent_spec(graph)
    workbench = store.root
    replays: set[tuple[str, int]] = set()
    if resume is not None:
        last = store.last_event(run_id, "interrupt_raised")
        if last and last["data"].get("step") is not None:
            replays = {(last["node_id"], last["data"]["step"])}
    rt = Runtime(graph, spec, store, workbench, run_id, cfg.thread_id, graph_hash, replays)
    em = rt.em
    if resume is None:
        em.emit("run_queued", config=cfg.model_dump())

    def finish(status: str, error: str | None = None, **data) -> str:
        em.emit("run_finished", status=status, error=error, **data)
        store.set_status(run_id, status, error)
        return status

    try:
        if resume is None:
            _advance(store, run_id, "preparing")
            em.emit("run_preparing")
            try:
                report = require_executable(graph)
            except ExecutionBlocked as e:
                for d in e.diagnostics:
                    em.emit("validation_error", d.nodeId, **d.to_json())
                return finish("failed", str(e))
            _advance(store, run_id, "running")
        saver = open_checkpointer(workbench)
        compiled = compile_graph(graph, rt, saver)
        config = {"configurable": {"thread_id": cfg.thread_id}, "recursion_limit": spec.limits.maxSteps}
        snap = compiled.get_state(config)
        exists = bool(snap.values) or bool(snap.next)
        if resume is None and snap.next:
            return finish("failed", f"thread '{cfg.thread_id}' has a pending interrupt at {list(snap.next)}; resume or cancel it before starting a new turn")
        em.emit("run_started" if resume is None else "run_resumed", kind="agent", threadId=cfg.thread_id, mode="start" if resume is None else "resume", threadExisted=exists,
                libraries=_libs(), limits=spec.limits.model_dump(), checkpointer=str(workbench / "agent" / "checkpoints.sqlite"), recursionLimit=spec.limits.maxSteps,
                inputFields=sorted(cfg.input), resumeValue=preview(resume) if resume is not None else None, order=(report.order if resume is None else None),
                fixtureModelInGraph=any(n.config.get("model", {}).get("provider") == "fixture" for n in graph.nodes))
        inp: Any = Command(resume=resume) if resume is not None else build_input(spec, cfg.input, exists)
        stopped_by = "end"
        paused = False
        try:
            for chunk in compiled.stream(inp, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    it = chunk["__interrupt__"][0]
                    snap = compiled.get_state(config)
                    payload = it.value if isinstance(it.value, dict) else {"value": it.value}
                    em.emit("interrupt_raised", payload.get("node"), interruptId=it.id, payload=preview(payload), step=(snap.metadata or {}).get("step", -1) + 1, next=list(snap.next),
                            checkpointId=(snap.config.get("configurable") or {}).get("checkpoint_id"))
                    paused = True
                    break
                if should_cancel():
                    raise _Cancelled()
        except BudgetExhausted as b:
            stopped_by = f"budget:{b.kind}"
            em.emit("budget_exhausted", None, kind=b.kind, limit=b.limit, used=b.used, message=str(b))
        except _Cancelled:
            if store.get_run(run_id)["status"] != "cancelling":
                store.set_status(run_id, "cancelling")
            em.emit("cancel_acknowledged")
            em.emit("run_finished", status="cancelled", error=None)
            store.set_status(run_id, "cancelled")
            return "cancelled"
        except NodeFailure as e:
            em.emit("node_failed", None, code=e.code, message=e.message)
            return finish("failed", f"{e.code}: {e.message}")
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "GraphRecursionError":
                stopped_by = "recursion_limit"
                em.emit("budget_exhausted", None, kind="maxSteps", limit=spec.limits.maxSteps, used=spec.limits.maxSteps, message=f"step limit {spec.limits.maxSteps} reached (LangGraph recursion limit)")
            else:
                raise
        snap = compiled.get_state(config)
        if paused:
            store.set_status(run_id, "paused")
            em.emit("run_paused", None, threadId=cfg.thread_id, next=list(snap.next), checkpointId=(snap.config.get("configurable") or {}).get("checkpoint_id"))
            return "paused"
        final = store.add_artifact(run_id, "final_state", json.dumps(snap.values, default=str).encode(), "complete", None,
                                   {"threadId": cfg.thread_id, "graph_hash": graph_hash, "checkpointId": (snap.config.get("configurable") or {}).get("checkpoint_id")})
        return finish("completed", None, stoppedBy=stopped_by, threadId=cfg.thread_id, finalStateSha256=final["sha256"], modelCalls=rt.model_calls, toolCalls=rt.tool_calls,
                      tokens=rt.tokens, checkpointId=(snap.config.get("configurable") or {}).get("checkpoint_id"))
    except Exception as e:  # noqa: BLE001
        em.emit("error", message=str(e), traceback=traceback.format_exc())
        try:
            return finish("failed", f"{type(e).__name__}: {e}")
        except IllegalTransition:
            return "failed"


class _Cancelled(Exception):
    pass
