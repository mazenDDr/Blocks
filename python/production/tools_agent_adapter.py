"""Bounded pure calculator turns, optionally over pinned retrieval indexes (ADR0062).

Native compiler/blocks stay unchanged. The serving contract additionally bounds
calculator expressions and reserves parallel tool calls atomically.
"""
from __future__ import annotations

import ast
import math
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from agent.blocks import ChatModelConfig, NodeFailure, RetrieveConfig, ToolCallConfig
from agent.runtime import BudgetExhausted, build_input, compile_graph
from agent.blocks import template_vars, render_template
from agent.tools import TOOLS, ToolError, safe_eval
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.schema import Graph
from graph_core.validate import require_executable
from tabular.core import dumps

from . import conversation_adapter as conv
from . import retrieval_agent_adapter as retrieval
from .pipeline import ProductionError, read_verified

NODE = "__agent_tools_graph__"
ALLOWED = {"agent.prompt", "agent.set_state", "agent.chat_model", "agent.retrieve", "agent.tool_call"}
FILES = (*retrieval.FILES, "agent/tools.py", "production/tools_agent_adapter.py")


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def contract(graph, inputs):
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes) or spec.policies:
        raise ProductionError("E_AGENT_SCOPE", "Pure-tool serving accepts calculator, prompt, set_state, retrieve and one chat_model; no file tools, memory policies or interrupts.")
    tools = [n for n in graph.nodes if n.type == "agent.tool_call"]
    if not tools:
        raise ProductionError("E_AGENT_SCOPE", "Declare at least one pure calculator node; otherwise use the plain/retrieval adapter.")
    if spec.limits.maxToolCalls is None or spec.limits.maxToolCalls > 8:
        raise ProductionError("E_AGENT_LIMITS", "Declare maxToolCalls between 1 and 8.")
    written = {f.split(".")[0] for n in graph.nodes for f in get_op(n.type).writes(get_op(n.type).Config.model_validate(n.config))}
    for n in tools:
        cfg = ToolCallConfig.model_validate(n.config)
        if cfg.tool != "calculator" or TOOLS[cfg.tool].effects or cfg.allowed_dir:
            raise ProductionError("E_AGENT_SCOPE", "Only the effect-free calculator with no directory is served.")
        refs = {f.split(".")[0] for f in template_vars(cfg.args["expression"])}
        if refs & written:
            raise ProductionError("E_AGENT_TOOL_INPUT", "Calculator arguments must read only immutable input/default fields, never node outputs.")
    retrieves = [n for n in graph.nodes if n.type == "agent.retrieve"]
    used = {RetrieveConfig.model_validate(n.config).index for n in retrieves}
    if len(spec.indexes) > 2 or used != {i.id for i in spec.indexes}:
        raise ProductionError("E_AGENT_SCOPE", "Optional indexes: declare at most two, each used by a retrieve node.")
    if any(RetrieveConfig.model_validate(n.config).k > 8 for n in retrieves):
        raise ProductionError("E_AGENT_LIMITS", "Retrieve at most 8 chunks per node.")
    if any(f.scope != "turn" for f in spec.state):
        raise ProductionError("E_AGENT_SCOPE", "Pure-tool turns are stateless: every state field must be turn-scoped.")
    lim = spec.limits
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 30
            and lim.maxModelCalls is not None and lim.maxModelCalls <= 2 and lim.maxTokens is not None and lim.maxTokens <= 4096):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps <= 25, maxSeconds <= 30, maxModelCalls <= 2 and maxTokens <= 4096.")
    if len(graph.nodes) > 16 or len(spec.state) > 32 or len(dumps(graph.to_json()).encode()) > 32_768 or len(dumps([f.initial() for f in spec.state]).encode()) > 8192:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds 16 nodes / 32 KiB or state defaults exceed 8 KiB.")
    for n in graph.nodes:
        op = get_op(n.type)
        if any(len(t) > 2000 or t.count("{") > 16 for _, t in op.templates(op.Config.model_validate(n.config))):
            raise ProductionError("E_AGENT_LIMITS", "Each template is bounded to 2000 characters and 16 field references.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "The recorded input must declare 1–4 text state fields.")
    chats = [n for n in graph.nodes if n.type == "agent.chat_model"]
    if len(chats) > 1:
        raise ProductionError("E_AGENT_SCOPE", "At most one chat-model node is supported.")
    model = None
    if chats:
        cfg = ChatModelConfig.model_validate(chats[0].config)
        model = cfg.model
        if (model.provider != "ollama" or model.base_url not in (None, conv.OLLAMA_URL, "http://127.0.0.1:11434")
                or model.api_key or model.fixture or model.think or model.max_tokens is None or model.max_tokens > 128 or model.timeout_s > 30):
            raise ProductionError("E_AGENT_PROVIDER", "Use local Ollama on port 11434, think=false, max_tokens <= 128, timeout_s <= 30.")
        output = cfg.output_field
    else:
        outputs = sorted({a["field"] for n in graph.nodes if n.type == "agent.set_state" for a in n.config.get("assignments", [])
                          if spec.field(a["field"]) and spec.field(a["field"]).type == "text" and a["field"] not in inputs})
        if len(outputs) != 1:
            raise ProductionError("E_AGENT_OUTPUT", "A model-free pure-tool workflow needs exactly one written text output field.")
        output = outputs[0]
    if not spec.field(output) or spec.field(output).type != "text" or output in inputs:
        raise ProductionError("E_AGENT_OUTPUT", "The served output must be a text state field separate from input fields.")
    return spec, output, model


def is_candidate(store, row):
    if row["status"] != "completed" or row["config"].get("kind") != "agent":
        return None
    try:
        graph, _ = conv._graph(store, row["id"])
        contract(graph, row["config"].get("input", {}))
        finished = store.last_event(row["id"], "run_finished")
        return NODE if finished and finished["data"].get("stoppedBy") == "end" else None
    except (ProductionError, ValueError):
        return None


def build_manifest(store, run_id, node):
    row = store.get_run(run_id)
    if node != NODE or not row or is_candidate(store, row) != NODE:
        raise ProductionError("E_AGENT_SOURCE", "Register a completed END pure-tool run using __agent_tools_graph__.")
    graph, graph_sha = conv._graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, output, model = contract(graph, inputs)
    finals = store.artifacts(run_id, "final_state")
    if not finals:
        raise ProductionError("E_AGENT_SOURCE", "The source run recorded no final state.")
    m = {"adapter": "native-langgraph-tools-local", "family": "agent_turn", "runId": run_id, "node": NODE,
         "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
         "sourceFinalStateSha256": finals[-1]["sha256"], "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
         "inputFields": sorted(inputs), "outputField": output, "limits": spec.limits.model_dump(),
         "inputContract": "One record with exactly the pinned text input fields; 1–2000 characters each, <= 8 KiB JSON.",
         "outputSchema": {"task": "agent_turn", "classes": None, "target": output},
         "indexes": retrieval._snapshot(store, run_id, spec),
         "provider": conv.provider_identity(model.model) if model else None, "environment": conv.environment(), "implementation": implementation(),
         "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()),
                    "semantics": "fresh stateless native turn; effect-free bounded calculator; optional retrieval searches pinned snapshots"},
         "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source run input (not ground truth or a benchmark)"}
    ToolsPipeline(store, m).validate_records([inputs])
    return m


def verify(store, m):
    if m["environment"] != conv.environment() or m["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned pure-tool implementation or native dependencies changed; register a new version.", 409)
    for sha in (m["graphSha256"], m["sourceFinalStateSha256"], m["referenceSha256"], m["source"]["configSha256"],
                *(sha for ix in m["indexes"].values() for sha in ix["files"].values())):
        read_verified(store, sha)
    graph, sha = conv._graph(store, m["runId"])
    if sha != m["graphSha256"] or semantic_hash(graph) != m["graphHash"]:
        raise ProductionError("E_AGENT_SOURCE", "Pinned pure-tool source graph differs.", 409)
    row = store.get_run(m["runId"])
    if (not row or row["status"] != "completed"
            or dumps(row["config"]).encode() != read_verified(store, m["source"]["configSha256"])
            or not any(a["sha256"] == m["sourceFinalStateSha256"] and a["status"] == "complete"
                       for a in store.artifacts(m["runId"], "final_state"))):
        raise ProductionError("E_AGENT_SOURCE", "Pinned tool source membership/configuration differs.", 409)
    spec, _, model = contract(graph, m["inputFields"])
    for ispec in spec.indexes:
        if retrieval.embedding_identity(ispec) != m["indexes"][ispec.id]["embeddingProvider"]:
            raise ProductionError("E_AGENT_MODEL_CHANGED", f"Embedding model for index '{ispec.id}' changed; register a new version.", 409)
    if model and conv.provider_identity(model.model) != m["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Ollama chat model digest/runtime changed; register and warm a new version.", 409)


class ToolRuntime(retrieval.PinnedIndexRuntime):
    def check_budget(self, kind=None):
        if kind != "tool":
            return super().check_budget(kind)
        with self._lock:
            super().check_budget("tool")
            self.tool_calls += 1  # reserve before native execution, including parallel nodes

    def count_tool(self):
        pass  # the call was already reserved before execution


def check_expression(expression):
    """Bound CPU/magnitude before calling the unchanged native calculator."""
    if len(expression) > 512:
        raise ProductionError("E_AGENT_TOOL_BOUNDS", "Rendered calculator expression exceeds 512 characters.", 413)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return  # native ToolError remains recorded data
    nodes = list(ast.walk(tree))
    if len(nodes) > 128 or any(isinstance(n, ast.Pow) for n in nodes):
        raise ProductionError("E_AGENT_TOOL_BOUNDS", "Calculator allows at most 128 AST nodes and no exponentiation.", 413)
    def depth(node):
        return 1 + max((depth(c) for c in ast.iter_child_nodes(node)), default=0)
    if depth(tree) > 32:
        raise ProductionError("E_AGENT_TOOL_BOUNDS", "Calculator AST depth exceeds 32.", 413)
    for n in nodes:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            if abs(n.value) > 1e100 or not math.isfinite(n.value):
                raise ProductionError("E_AGENT_TOOL_BOUNDS", "Calculator literals must be finite with magnitude <= 1e100.", 413)
    try:
        value = safe_eval(expression)
    except ToolError:
        return  # native calculator reports its original error result
    if abs(value) > 1e100 or not math.isfinite(value):
        raise ProductionError("E_AGENT_TOOL_BOUNDS", "Calculator result must be finite with magnitude <= 1e100.", 413)


class ToolsPipeline:
    reference_records = conv.AgentPipeline.reference_records
    def validate_records(self, records, max_batch=1):
        conv.AgentPipeline.validate_records(self, records, max_batch)
        state = {f.name: f.initial() for f in self.spec.state}
        state.update(records[0])
        for node in self.graph.nodes:
            if node.type == "agent.tool_call":
                cfg = ToolCallConfig.model_validate(node.config)
                check_expression(render_template(cfg.args["expression"], state))

    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, _, _ = contract(self.graph, manifest["inputFields"])

    def predict(self, records, *, capture=False, deadline=None, cancelled=lambda: False):
        from langgraph.checkpoint.memory import InMemorySaver

        self.validate_records(records)
        verify(self.store, self.manifest)
        start = time.perf_counter()
        deadline = min(deadline or start + self.spec.limits.maxSeconds, start + self.spec.limits.maxSeconds)
        execution = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="void-retrieval-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            rt = ToolRuntime(self.graph, self.spec, local, local.root, execution, execution, self.manifest["graphHash"])
            rt.pinned = self.manifest["indexes"]
            for iid, snap in self.manifest["indexes"].items():  # restore the hash-verified snapshot into this turn's private workbench
                d = rt.indexes.dir(iid)
                d.mkdir(parents=True, exist_ok=True)
                for name, sha in snap["files"].items():
                    (d / name).write_bytes(read_verified(self.store, sha))
            rt.deadline, rt.cancelled = deadline, cancelled
            config = {"configurable": {"thread_id": execution}, "recursion_limit": self.spec.limits.maxSteps}
            try:
                compiled = compile_graph(self.graph, rt, InMemorySaver())
                rt.emit("run_started", threadId=execution, mode="isolated retrieval serving turn", threadExisted=False,
                        sourceRunId=self.manifest["runId"], inputFields=self.manifest["inputFields"], limits=self.manifest["limits"])
                final = None
                for state in compiled.stream(build_input(self.spec, records[0], False), config, stream_mode="values"):
                    rt.check_budget()
                    if len(dumps(state).encode()) > 65_536:
                        raise ProductionError("E_AGENT_STATE_BOUNDS", "Native state exceeded 64 KiB; no result committed.", 413)
                    final = state
                rt.check_budget()
            except BudgetExhausted as e:
                raise ProductionError("E_AGENT_BUDGET", str(e), 422) from e
            except NodeFailure as e:
                rt.check_budget()
                raise ProductionError(e.code, e.message, 503) from e
            except Exception as e:
                if type(e).__name__ == "GraphRecursionError":
                    raise ProductionError("E_AGENT_BUDGET", "Native LangGraph recursion limit exhausted; no successful output committed.") from e
                raise
            if len(dumps(final).encode()) > 65_536 or not isinstance(final.get(self.manifest["outputField"]), str):
                raise ProductionError("E_AGENT_STATE_BOUNDS", "Final state exceeds 64 KiB or output is not text.", 413)
            rt.emit("run_finished", status="completed", stoppedBy="end", modelCalls=rt.model_calls, toolCalls=rt.tool_calls, tokens=rt.tokens)
            contexts = []
            for a in local.artifacts(execution, "model_context"):
                raw = local.read_artifact(a["sha256"])
                value = json.loads(raw)
                contexts.append({"sha256": a["sha256"], "callId": value["callId"], "nodeId": value["node"],
                                 "usage": value["usage"], "latencyMs": value["latencyMs"], "value": value if capture else None})
                if capture:
                    self.store.put_bytes(raw)
            events = local.events(execution)
            retrievals = [{"node": e["node_id"], "index": e["data"]["index"], "included": e["data"]["included"], "embedding": e["data"]["embedding"]}
                          for e in events if e["type"] == "retrieval"]
            evidence = {"executionId": execution, "threadId": execution, "sourceRunId": self.manifest["runId"],
                        "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"], "modelCalls": rt.model_calls, "toolCalls": rt.tool_calls,
                        "indexes": {iid: {"identity": s["identity"], "chunks": s["chunks"]} for iid, s in self.manifest["indexes"].items()},
                        "retrievals": retrievals, "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "tools": [{"node": e["node_id"], **e["data"]} for e in events if e["type"] == "tool_call"] if capture else None,
                        "contexts": contexts, "capturePolicy": "Full state/events/context/tool arguments/results only when captureInputs=true; tool count and retrieved chunk ids/scores always.",
                        "isolation": "fresh in-memory checkpoint; index snapshot restored privately; no research index, memory or thread mutation"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Pure-tool turn exceeded its deadline before result commit.", 504)
        return {"predictions": [final[self.manifest["outputField"]]], "family": "agent_turn", "agent": evidence}, {
            "inferenceMs": (time.perf_counter() - start) * 1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph calculator/retrieval/prompt/model execution together; provider latency recorded per call"}
