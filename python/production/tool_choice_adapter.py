"""Served model-chosen calculator turns (ADR 0082).

A stateless release over a graph of prompt / set_state nodes and one `agent.tool_agent` offering only the effect-free
calculator, with a pinned local Ollama model. Each request runs the native graph in an isolated temporary store; every tool call
the model requested (offered, refused or executed) is returned in the request evidence and therefore recorded in the trace.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from agent.blocks import NodeFailure, ToolAgentConfig
from agent.runtime import BudgetExhausted, build_input, compile_graph
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.schema import Graph
from graph_core.validate import require_executable
from tabular.core import dumps
from . import conversation_adapter as conv
from .pipeline import ProductionError, read_verified

NODE = "__agent_tool_choice_graph__"
ALLOWED = {"agent.prompt", "agent.set_state", "agent.tool_agent"}
FILES = (*conv.FILES, "agent/tools.py", "agent/openai_compat.py", "production/tool_choice_adapter.py")


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def contract(graph, inputs):
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes) or spec.policies or spec.indexes:
        raise ProductionError("E_AGENT_SCOPE", "Tool-choice serving accepts prompt, set_state and one tool_agent; no memory, retrieval, interrupts or other tools.")
    if any(f.scope == "thread" for f in spec.state):
        raise ProductionError("E_AGENT_SCOPE", "Tool-choice releases are stateless per turn (no thread-scoped fields).")
    agents = [n for n in graph.nodes if n.type == "agent.tool_agent"]
    if len(agents) != 1:
        raise ProductionError("E_AGENT_SCOPE", "Declare exactly one tool_agent node.")
    cfg = ToolAgentConfig.model_validate(agents[0].config)
    if cfg.tools != ["calculator"] or cfg.allowed_dir:
        raise ProductionError("E_AGENT_SCOPE", "Only the effect-free calculator is served; no file tools.")
    if cfg.max_tool_calls > 4:
        raise ProductionError("E_AGENT_LIMITS", "Declare max_tool_calls <= 4.")
    m = cfg.model
    if m.provider != "ollama" or m.base_url not in (None, conv.OLLAMA_URL, "http://127.0.0.1:11434") or m.api_key or m.think \
            or m.max_tokens is None or m.max_tokens > 256 or m.timeout_s > 60:
        raise ProductionError("E_AGENT_PROVIDER", "Use installed local Ollama, think=false, max_tokens<=256 and timeout_s<=60.")
    lim = spec.limits
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 120 and lim.maxModelCalls is not None
            and cfg.max_tool_calls + 1 <= lim.maxModelCalls <= 5 and lim.maxToolCalls is not None and lim.maxToolCalls <= 4 and lim.maxTokens is not None and lim.maxTokens <= 8192):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps<=25, maxSeconds<=120, max_tool_calls+1 <= maxModelCalls <= 5, maxToolCalls<=4, maxTokens<=8192.")
    if len(graph.nodes) > 12 or len(dumps(graph.to_json()).encode()) > 32768:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds 12 nodes / 32KiB.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" or spec.field(k).scope != "turn" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "Declare 1–4 turn-scoped text input fields.")
    output = cfg.output_field
    if not spec.field(output) or spec.field(output).type != "text" or output in inputs:
        raise ProductionError("E_AGENT_OUTPUT", "Served text is the tool_agent's separate text output field.")
    for node in graph.nodes:
        op = get_op(node.type)
        if any(len(t) > 2000 or t.count("{") > 16 for _, t in op.templates(op.Config.model_validate(node.config))):
            raise ProductionError("E_AGENT_LIMITS", "Templates are bounded to 2000 characters / 16 references.")
    return spec, output, cfg


def is_candidate(store, row):
    if row["status"] != "completed" or row["config"].get("kind") != "agent":
        return None
    try:
        graph, _ = conv._graph(store, row["id"])
        contract(graph, row["config"].get("input", {}))
        finish = store.last_event(row["id"], "run_finished")
        if not finish or finish["data"].get("stoppedBy") != "end" or not store.artifacts(row["id"], "final_state"):
            return None
        return NODE
    except (ProductionError, ValueError):
        return None


def build_manifest(store, run_id, node):
    row = store.get_run(run_id)
    if not row or node != NODE or is_candidate(store, row) != NODE:
        raise ProductionError("E_AGENT_SOURCE", f"Register a completed END tool-choice run using {NODE}.")
    graph, graph_sha = conv._graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, output, cfg = contract(graph, inputs)
    m = {"adapter": "native-langgraph-tool-choice-local", "family": "agent_turn", "conversation": False, "runId": run_id, "node": node,
         "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
         "sourceFinalStateSha256": store.artifacts(run_id, "final_state")[-1]["sha256"], "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
         "inputFields": sorted(inputs), "outputField": output, "limits": spec.limits.model_dump(),
         "inputContract": "One record with exactly the pinned turn text fields, 1–2000 characters each, <=8KiB JSON.",
         "outputSchema": {"task": "agent_turn", "classes": None, "target": output},
         "tools": {"offered": cfg.tools, "maxToolCalls": cfg.max_tool_calls, "chosenBy": "model", "effects": "none (calculator only)"},
         "provider": conv.provider_identity(cfg.model.model), "environment": conv.environment(), "implementation": implementation(),
         "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()), "semantics": "fresh declared defaults per turn"},
         "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source input; not ground truth or a benchmark"}
    ToolChoicePipeline(store, m).validate_records([inputs])
    return m


def verify(store, m):
    if m["environment"] != conv.environment() or m["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned tool-choice adapter/dependency environment changed; register a new version.", 409)
    for sha in (m["graphSha256"], m["sourceFinalStateSha256"], m["referenceSha256"], m["source"]["configSha256"]):
        read_verified(store, sha)
    graph, sha = conv._graph(store, m["runId"])
    row = store.get_run(m["runId"])
    if sha != m["graphSha256"] or semantic_hash(graph) != m["graphHash"] or not row or row["status"] != "completed" \
            or dumps(row["config"]).encode() != read_verified(store, m["source"]["configSha256"]):
        raise ProductionError("E_AGENT_SOURCE", "Pinned tool-choice source graph/configuration differs.", 409)
    _, _, cfg = contract(graph, m["inputFields"])
    if conv.provider_identity(cfg.model.model) != m["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Pinned local chat model differs; register a new version.", 409)


class ToolChoicePipeline:
    reference_records = conv.AgentPipeline.reference_records
    validate_records = conv.AgentPipeline.validate_records

    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, self.output, self.cfg = contract(self.graph, manifest["inputFields"])

    def predict(self, records, *, capture=False, deadline=None, cancelled=lambda: False):
        from langgraph.checkpoint.memory import InMemorySaver

        self.validate_records(records)
        verify(self.store, self.manifest)
        start = time.perf_counter()
        deadline = min(deadline or start + self.spec.limits.maxSeconds, start + self.spec.limits.maxSeconds)
        execution, thread = uuid.uuid4().hex, uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="void-tool-choice-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            rt = conv.ServingRuntime(self.graph, self.spec, local, local.root, execution, thread, self.manifest["graphHash"])
            rt.deadline, rt.cancelled = deadline, cancelled
            config = {"configurable": {"thread_id": thread}, "recursion_limit": self.spec.limits.maxSteps}
            try:
                compiled = compile_graph(self.graph, rt, InMemorySaver())
                rt.emit("run_started", threadId=thread, mode="isolated tool-choice turn", sourceRunId=self.manifest["runId"], inputFields=self.manifest["inputFields"])
                final = None
                for state in compiled.stream(build_input(self.spec, records[0], False), config, stream_mode="values"):
                    rt.check_budget()
                    if len(dumps(state).encode()) > 65_536:
                        raise ProductionError("E_AGENT_STATE_BOUNDS", "Native state exceeded 64 KiB; no result committed.", 413)
                    final = state
                rt.check_budget()
                result = (final or {}).get(self.output)
                if not isinstance(result, str):
                    raise ProductionError("E_AGENT_OUTPUT", "Native END output is not text.", 413)
            except BudgetExhausted as e:
                raise ProductionError("E_AGENT_BUDGET", str(e), 422) from e
            except NodeFailure as e:
                rt.check_budget()
                raise ProductionError(e.code, e.message, 503) from e
            except Exception as e:
                if type(e).__name__ == "GraphRecursionError":
                    raise ProductionError("E_AGENT_BUDGET", "Native LangGraph recursion limit exhausted; no successful output committed.") from e
                raise
            rt.emit("run_finished", status="completed", stoppedBy="end", modelCalls=rt.model_calls, tokens=rt.tokens)
            events = local.events(execution)
            tool_calls = [{k: e["data"].get(k) for k in ("tool", "args", "status", "result", "callId", "modelCallId", "chosenBy")} for e in events if e["type"] == "tool_call"]
            contexts = []
            for a in local.artifacts(execution, "model_context"):
                raw_ctx = local.read_artifact(a["sha256"])
                value = json.loads(raw_ctx)
                contexts.append({"sha256": a["sha256"], "callId": value["callId"], "nodeId": value["node"], "usage": value["usage"], "latencyMs": value["latencyMs"],
                                 "value": value if capture else None})
                if capture:
                    self.store.put_bytes(raw_ctx)
            evidence = {"executionId": execution, "threadId": thread, "sourceRunId": self.manifest["runId"], "graphHash": self.manifest["graphHash"],
                        "provider": self.manifest["provider"], "modelCalls": rt.model_calls, "contexts": contexts, "toolCalls": tool_calls,
                        "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "capturePolicy": "Tool calls (tool, arguments, status, result) always; full state/events/model context only when release captureInputs=true.",
                        "isolation": "fresh declared defaults per turn; temporary store; only the effect-free calculator can run"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Tool-choice turn exceeded its deadline before result commit.", 504)
        return {"predictions": [result], "family": self.manifest["family"], "agent": evidence}, {
            "inferenceMs": (time.perf_counter() - start) * 1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph tool-choice turn; provider latency recorded per call"}


AgentPipeline = ToolChoicePipeline
