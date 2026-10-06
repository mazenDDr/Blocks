"""Per-release, per-user long-term memory for served agent turns (ADR0075).

A release owns its memory: records live in `production.sqlite` (`release_memory`), keyed by release and user. Each turn runs the
native graph against a fresh temporary memory store seeded with ONLY the requesting user's records of that release, so policies cannot
see other users' memory or the research `agent/memory.db`. The turn's new long-term records are returned to the runtime, which commits
them in the same SQLite transaction as the successful request trace; failed, cancelled, timed-out and replayed requests commit nothing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from agent.blocks import ChatModelConfig, MemorySelectConfig, MemoryWriteConfig, NodeFailure
from agent.policy import STAGE_CONFIG
from agent.runtime import BudgetExhausted, build_input, compile_graph
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.validate import require_executable
from graph_core.schema import Graph
from tabular.core import dumps
from . import conversation_adapter as conv
from .pipeline import ProductionError, read_verified

NODE = "__agent_memory_graph__"
ALLOWED = {"agent.prompt", "agent.set_state", "agent.chat_model", "agent.memory_select", "agent.memory_write"}
FILES = (*conv.FILES, "production/memory_agent_adapter.py", "production/runtime.py", "production/store.py", "production/pipeline.py")
from .runtime import MEMORY_MAX_RECORDS as MAX_RECORDS  # a turn that would exceed it commits nothing (E_AGENT_MEMORY_FULL)
MAX_CHARS = 1000
USER_SCOPE = "user"  # the only declared scope; at serving time it means "the authenticated/declared request user of this release"


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def contract(graph, inputs):
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes):
        raise ProductionError("E_AGENT_SCOPE", "Memory serving accepts prompt, set_state, one chat model, memory_select and one long-term memory_write; no tools, retrieval indexes or interrupts.")
    if any(f.scope == "thread" for f in spec.state) or spec.indexes:
        raise ProductionError("E_AGENT_SCOPE", "Memory releases are stateless per turn (no thread-scoped fields) and declare no document indexes.")
    writes = [MemoryWriteConfig.model_validate(n.config) for n in graph.nodes if n.type == "agent.memory_write"]
    if len(writes) != 1:
        raise ProductionError("E_AGENT_MEMORY_SCOPE", "Declare exactly one memory_write node.")
    w = writes[0]
    if w.target != "long_term" or w.mode != "direct" or w.scope != USER_SCOPE or w.max_chars > MAX_CHARS or len(w.metadata) > 4:
        raise ProductionError("E_AGENT_MEMORY_SCOPE", f"The memory_write is long_term, direct, scope '{USER_SCOPE}', max_chars <= {MAX_CHARS}, at most 4 metadata keys.")
    selects = [MemorySelectConfig.model_validate(n.config) for n in graph.nodes if n.type == "agent.memory_select"]
    if not 1 <= len(selects) <= 2 or any(s.short_term_field for s in selects) or {s.policy for s in selects} != {p.id for p in spec.policies}:
        raise ProductionError("E_AGENT_MEMORY_SCOPE", "Declare 1–2 memory_select nodes over long-term memory only, each with its own declared policy.")
    for policy in spec.policies:
        if not 1 <= len(policy.stages) <= 6 or policy.stages[0].op != "retrieve" or any(stage.op == "retrieve" for stage in policy.stages[1:]):
            raise ProductionError("E_AGENT_MEMORY_SCOPE", "Policies have 1–6 native stages starting with one retrieve stage.")
        if policy.embeddings.provider != "local_hash" or policy.embeddings.dimension > 512:
            raise ProductionError("E_AGENT_MEMORY_SCOPE", "Memory policies use bounded local_hash <=512 dimensions; lexical hashing is not semantic embedding.")
        for stage in policy.stages:
            cfg = STAGE_CONFIG[stage.op].model_validate(stage.config)
            if stage.op == "retrieve" and (cfg.sources != ["long_term"] or cfg.scopes != [USER_SCOPE] or cfg.k is None or cfg.k > 8):
                raise ProductionError("E_AGENT_MEMORY_SCOPE", f"Policy retrieve reads long_term only, scopes ['{USER_SCOPE}'], k <= 8.")
            if stage.op == "summarize" and (cfg.method != "extractive" or cfg.max_chars > 2000):
                raise ProductionError("E_AGENT_MEMORY_SCOPE", "Memory summaries are extractive <=2000 characters; no hidden model calls.")
            if stage.op == "budget" and cfg.max_tokens > 4096:
                raise ProductionError("E_AGENT_LIMITS", "Policy token-estimate budget must be <=4096.")
        if len(policy.query) > 2000 or policy.query.count("{") > 16:
            raise ProductionError("E_AGENT_LIMITS", "Policy templates are bounded to 2000 characters / 16 references.")
    lim = spec.limits
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 30 and lim.maxModelCalls is not None and lim.maxModelCalls <= 2 and lim.maxTokens is not None and lim.maxTokens <= 4096):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps<=25, maxSeconds<=30, maxModelCalls<=2 and maxTokens<=4096.")
    if len(graph.nodes) > 16 or len(spec.state) > 32 or len(dumps(graph.to_json()).encode()) > 32768 or len(dumps([f.initial() for f in spec.state]).encode()) > 8192:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds 16 nodes / 32KiB or state defaults exceed 8KiB.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" or spec.field(k).scope != "turn" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "Declare 1–4 turn-scoped text input fields.")
    chats = [n for n in graph.nodes if n.type == "agent.chat_model"]
    if len(chats) > 1:
        raise ProductionError("E_AGENT_SCOPE", "At most one chat model node is served.")
    model = ChatModelConfig.model_validate(chats[0].config).model if chats else None
    if chats:
        output = ChatModelConfig.model_validate(chats[0].config).output_field
    else:
        outputs = sorted({a["field"] for n in graph.nodes if n.type == "agent.set_state" for a in n.config.get("assignments", [])
                          if spec.field(a["field"]) and spec.field(a["field"]).type == "text" and a["field"] not in inputs})
        if len(outputs) != 1:
            raise ProductionError("E_AGENT_OUTPUT", "Model-free memory turns need exactly one written text output.")
        output = outputs[0]
    if not spec.field(output) or spec.field(output).type != "text" or output in inputs:
        raise ProductionError("E_AGENT_OUTPUT", "Served text is a separate text field.")
    if model and (model.provider != "ollama" or model.base_url not in (None, conv.OLLAMA_URL, "http://127.0.0.1:11434") or model.api_key or model.fixture or model.think
                  or model.max_tokens is None or model.max_tokens > 128 or model.timeout_s > 30):
        raise ProductionError("E_AGENT_PROVIDER", "Use installed local Ollama, think=false, max_tokens<=128 and timeout_s<=30.")
    for node in graph.nodes:
        op = get_op(node.type)
        if any(len(t) > 2000 or t.count("{") > 16 for _, t in op.templates(op.Config.model_validate(node.config))):
            raise ProductionError("E_AGENT_LIMITS", "Templates are bounded to 2000 characters / 16 references.")
    return spec, output, model, w


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
        raise ProductionError("E_AGENT_SOURCE", f"Register a completed END long-term memory run using {NODE}.")
    graph, graph_sha = conv._graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, output, model, write = contract(graph, inputs)
    m = {"adapter": "native-langgraph-release-memory-local", "family": "agent_turn", "conversation": False, "runId": run_id, "node": node,
         "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
         "sourceFinalStateSha256": store.artifacts(run_id, "final_state")[-1]["sha256"], "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
         "inputFields": sorted(inputs), "outputField": output, "limits": spec.limits.model_dump(),
         "inputContract": "One record with exactly the pinned turn text fields, 1–2000 characters each, <=8KiB JSON.",
         "outputSchema": {"task": "agent_turn", "classes": None, "target": output},
         "memory": {"store": "production.sqlite release_memory", "ownership": "per release and request user", "scope": USER_SCOPE, "namespace": write.namespace,
                    "maxRecordsPerUser": MAX_RECORDS, "maxChars": MAX_CHARS, "writesPerTurn": 1,
                    "commit": "only with a successful request trace, in the same transaction; failed/cancelled/timed-out/replayed requests write nothing",
                    "research": "the source run's research memory is neither copied nor read"},
         "provider": conv.provider_identity(model.model) if model else None, "environment": conv.environment(), "implementation": implementation(),
         "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()),
                    "semantics": "fresh declared defaults per turn; long-term records of the requesting user in this release only"},
         "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source input; not ground truth or a benchmark"}
    MemoryPipeline(store, m).validate_records([inputs])
    return m


def verify(store, m):
    if m["environment"] != conv.environment() or m["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned memory adapter/dependency environment changed; register a new version.", 409)
    for sha in (m["graphSha256"], m["sourceFinalStateSha256"], m["referenceSha256"], m["source"]["configSha256"]):
        read_verified(store, sha)
    graph, sha = conv._graph(store, m["runId"])
    row = store.get_run(m["runId"])
    if sha != m["graphSha256"] or semantic_hash(graph) != m["graphHash"] or not row or row["status"] != "completed" \
            or dumps(row["config"]).encode() != read_verified(store, m["source"]["configSha256"]):
        raise ProductionError("E_AGENT_SOURCE", "Pinned memory source graph/configuration differs.", 409)
    _, _, model, _ = contract(graph, m["inputFields"])
    if model and conv.provider_identity(model.model) != m["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Pinned local chat model differs; register a new version.", 409)


def _seed(rt, records, namespace):
    for r in records:
        rt.memory.put_record({"id": r["id"], "namespace": r["namespace"], "scope": USER_SCOPE, "kind": r["kind"], "text": r["text"], "metadata": r["metadata"],
                              "importance": r["importance"], "generated": r["generated"], "created_at": r["created"], "source": {"releaseMemory": True, "requestId": r["request"]}})


class MemoryPipeline:
    reference_records = conv.AgentPipeline.reference_records
    validate_records = conv.AgentPipeline.validate_records

    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, self.output, _, self.write = contract(self.graph, manifest["inputFields"])

    def predict(self, records, *, capture=False, deadline=None, cancelled=lambda: False, memory=()):
        """`memory`: the requesting user's live records in this release (from the runtime). Returns new records as `_memory` for the
        runtime's transactional commit; nothing is written here."""
        from langgraph.checkpoint.memory import InMemorySaver

        self.validate_records(records)
        verify(self.store, self.manifest)
        start = time.perf_counter()
        deadline = min(deadline or start + self.spec.limits.maxSeconds, start + self.spec.limits.maxSeconds)
        execution, thread = uuid.uuid4().hex, uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="void-memory-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            rt = conv.ServingRuntime(self.graph, self.spec, local, local.root, execution, thread, self.manifest["graphHash"])
            rt.deadline, rt.cancelled = deadline, cancelled
            _seed(rt, memory, self.write.namespace)
            seeded = {r["id"] for r in memory}
            config = {"configurable": {"thread_id": thread}, "recursion_limit": self.spec.limits.maxSteps}
            try:
                compiled = compile_graph(self.graph, rt, InMemorySaver())
                rt.emit("run_started", threadId=thread, mode="isolated release-memory turn", sourceRunId=self.manifest["runId"],
                        inputFields=self.manifest["inputFields"], limits=self.manifest["limits"], memoryRecords=len(memory))
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
            new = [r for r in rt.memory.list_records() if r["id"] not in seeded]
            if len(new) > 1:
                raise ProductionError("E_AGENT_MEMORY_SCOPE", "A memory turn wrote more than one long-term record; nothing committed.", 409)
            writes = [{"id": r["id"], "namespace": r["namespace"], "kind": r["kind"], "text": r["text"], "metadata": r["metadata"], "importance": r["importance"],
                       "generated": r["generated"], "evidence": next((w["evidence"] for w in rt.memory.writes(record_id=r["id"])), None)} for r in new]
            events = local.events(execution)
            selections = []
            for event in events:
                if event["type"] == "memory_selection":
                    application = rt.memory.get_application(event["data"]["applicationId"])
                    raw = dumps(application).encode()
                    if len(raw) > 131_072:
                        raise ProductionError("E_AGENT_MEMORY_BOUNDS", "Memory application evidence exceeds 128 KiB.", 413)
                    if capture:
                        self.store.put_bytes(raw)
                    selections.append({"node": event["node_id"], "applicationSha256": hashlib.sha256(raw).hexdigest(), "selected": application["final"],
                                       "universe": len(application["records"]), "value": application if capture else None})
            contexts = []
            for a in local.artifacts(execution, "model_context"):
                raw_ctx = local.read_artifact(a["sha256"])
                value = json.loads(raw_ctx)
                contexts.append({"sha256": a["sha256"], "callId": value["callId"], "nodeId": value["node"],
                                 "usage": value["usage"], "latencyMs": value["latencyMs"], "value": value if capture else None})
                if capture:
                    self.store.put_bytes(raw_ctx)
            writes_ev = [{"node": e["node_id"], "status": e["data"]["status"], "recordId": e["data"].get("recordId")} for e in events if e["type"] == "memory_write"]
            evidence = {"executionId": execution, "threadId": thread, "sourceRunId": self.manifest["runId"], "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"],
                        "modelCalls": rt.model_calls, "contexts": contexts, "memoryRecordsVisible": len(memory), "memorySelections": selections, "memoryWrites": writes_ev,
                        "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "capturePolicy": "Full state/events/policy applications only when release captureInputs=true; selected record ids always (they are this user's own records).",
                        "isolation": "fresh declared defaults; temporary memory store seeded with only this user's records in this release; research memory untouched"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Memory turn exceeded its deadline before result commit.", 504)
        return {"predictions": [result], "family": self.manifest["family"], "agent": evidence}, {
            "_memory": writes, "inferenceMs": (time.perf_counter() - start) * 1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph memory turn; provider latency recorded per call"}


AgentPipeline = MemoryPipeline
