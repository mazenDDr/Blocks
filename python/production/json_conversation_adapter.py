"""Native conversations whose turns return validated JSON objects (ADR 0059).

Combines the conversation adapter's per-release/user/session native checkpoints (ADR 0024) with the JSON adapter's
structured-output contract (ADR 0045). Both existing adapters pin their own source files in registered identities, so this
path lives in its own module and only reuses their functions. Thread-scoped state carries the conversation; each turn's
served object is a turn-scoped field written only by the single structured-output node and validated against its native
schema before the turn's checkpoint can be committed.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from agent.blocks import NodeFailure, StructuredOutputConfig, json_schema_of
from agent.runtime import BudgetExhausted, build_input, compile_graph
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.schema import Graph
from graph_core.validate import require_executable
from tabular.core import dumps

from . import conversation_adapter as conv
from . import json_agent_adapter as jsa
from .pipeline import ProductionError, read_verified

NODE = "__agent_json_conversation__"
ALLOWED = jsa.ALLOWED
FILES = (*conv.FILES, "production/json_agent_adapter.py", "production/json_conversation_adapter.py")


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def contract(graph, inputs):
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes) or spec.indexes or spec.policies:
        raise ProductionError("E_AGENT_JSON_SCOPE", "JSON conversations accept only prompt/set_state/one structured_output; no tools/retrieval/memory/interrupts.")
    if not any(f.scope == "thread" for f in spec.state):
        raise ProductionError("E_AGENT_SCOPE", "JSON conversations must declare thread-scoped state; use the stateless JSON adapter otherwise.")
    lim = spec.limits
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 30
            and lim.maxModelCalls is not None and 1 <= lim.maxModelCalls <= 2 and lim.maxTokens is not None and lim.maxTokens <= 4096):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps <= 25, maxSeconds <= 30, maxModelCalls 1–2 and maxTokens <= 4096.")
    if len(graph.nodes) > 16 or len(spec.state) > 32 or len(dumps(graph.to_json()).encode()) > 32_768 or len(dumps([f.initial() for f in spec.state]).encode()) > 8192:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds 16 nodes / 32 KiB or state defaults exceed 8 KiB.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" or spec.field(k).scope != "turn" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "Declare 1–4 turn-scoped text input fields; thread state is owned by native checkpoints.")
    structured = [n for n in graph.nodes if n.type == "agent.structured_output"]
    if len(structured) != 1:
        raise ProductionError("E_AGENT_JSON_SCOPE", "Exactly one structured-output node is required.")
    cfg = StructuredOutputConfig.model_validate(structured[0].config)
    output = spec.field(cfg.output_field)
    if not output or output.type != "object" or cfg.output_field in inputs or output.reducer.kind != "replace" or output.scope != "turn":
        raise ProductionError("E_AGENT_OUTPUT", "The served JSON must be a separate turn-scoped object field with a replace reducer.")
    if cfg.on_failure != "fail" or cfg.retry.maxRetries + 1 > lim.maxModelCalls:
        raise ProductionError("E_AGENT_JSON_FAILURE", "Use on_failure=fail and retries within the declared model-call budget.")
    if not 1 <= len(cfg.schema_fields) <= 12 or len(dumps(json_schema_of(cfg.schema_fields)).encode()) > 4096:
        raise ProductionError("E_AGENT_JSON_SCHEMA", "Declare 1–12 flat native schema fields within 4 KiB.")
    model = cfg.model
    if (model.provider != "ollama" or model.base_url not in (None, conv.OLLAMA_URL, "http://127.0.0.1:11434") or model.api_key
            or model.fixture or model.think or model.max_tokens is None or model.max_tokens > 128 or model.timeout_s > 30):
        raise ProductionError("E_AGENT_PROVIDER", "Use installed local Ollama, think=false, max_tokens <= 128 and timeout_s <= 30.")
    for node in graph.nodes:
        op = get_op(node.type)
        parsed = op.Config.model_validate(node.config)
        if any(len(t) > 2000 or t.count("{") > 16 for _, t in op.templates(parsed)):
            raise ProductionError("E_AGENT_LIMITS", "Templates are bounded to 2000 characters and 16 references.")
        if node.id != structured[0].id and cfg.output_field in op.writes(parsed):
            raise ProductionError("E_AGENT_OUTPUT", "Only the structured-output node may write the served object.")
    return spec, cfg


def is_candidate(store, row):
    if row["status"] != "completed" or row["config"].get("kind") != "agent":
        return None
    try:
        graph, _ = conv._graph(store, row["id"])
        _, cfg = contract(graph, row["config"].get("input", {}))
        finish, finals = store.last_event(row["id"], "run_finished"), store.artifacts(row["id"], "final_state")
        if not finish or finish["data"].get("stoppedBy") != "end" or not finals:
            return None
        jsa.source_evidence(store, row["id"], graph)
        jsa.output_value(json.loads(read_verified(store, finals[-1]["sha256"])).get(cfg.output_field), cfg)
        return NODE
    except (ProductionError, ValueError):
        return None


def build_manifest(store, run_id, node):
    row = store.get_run(run_id)
    if node != NODE or not row or is_candidate(store, row) != NODE:
        raise ProductionError("E_AGENT_SOURCE", "Register a completed END conversation run with bounded valid JSON output using __agent_json_conversation__.")
    graph, graph_sha = conv._graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, cfg = contract(graph, inputs)
    attempts, contexts = jsa.source_evidence(store, run_id, graph)
    manifest = {"adapter": "native-langgraph-json-conversation-local", "family": "agent_json", "runId": run_id, "node": NODE,
                "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
                "sourceFinalStateSha256": store.artifacts(run_id, "final_state")[-1]["sha256"], "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
                "inputFields": sorted(inputs), "outputField": cfg.output_field, "limits": spec.limits.model_dump(),
                "inputContract": "One record with exactly the pinned turn text fields, 1–2000 characters each, <= 8 KiB JSON.",
                "outputSchema": {"task": "agent_json", "classes": None, "target": cfg.output_field, "jsonSchema": json_schema_of(cfg.schema_fields)},
                "provider": conv.provider_identity(cfg.model.model), "environment": conv.environment(), "implementation": implementation(),
                "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()),
                           "structuredEventsSha256": store.put_bytes(dumps(attempts).encode()), "contextSha256": contexts,
                           "semantics": "new conversations start at declared defaults; later turns restore the last successful native checkpoint; the source thread is never copied"},
                "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source run input; not ground truth or a benchmark"}
    JsonConversationPipeline(store, manifest).validate_records([inputs])
    return manifest


def verify(store, m):
    if m["environment"] != conv.environment() or m["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned JSON conversation implementation or native dependencies changed; register a new version.", 409)
    for sha in (m["graphSha256"], m["sourceFinalStateSha256"], m["referenceSha256"], m["source"]["configSha256"],
                m["source"]["structuredEventsSha256"], *m["source"]["contextSha256"]):
        read_verified(store, sha)
    graph, sha = conv._graph(store, m["runId"])
    row = store.get_run(m["runId"])
    finals = store.artifacts(m["runId"], "final_state")
    if (sha != m["graphSha256"] or semantic_hash(graph) != m["graphHash"] or not row or row["status"] != "completed"
            or dumps(row["config"]).encode() != read_verified(store, m["source"]["configSha256"])
            or not any(a["sha256"] == m["sourceFinalStateSha256"] and a["status"] == "complete" for a in finals)):
        raise ProductionError("E_AGENT_SOURCE", "Pinned JSON conversation source membership/configuration differs.", 409)
    _, cfg = contract(graph, m["inputFields"])
    if conv.provider_identity(cfg.model.model) != m["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Local Ollama digest/runtime changed; register and warm a new version.", 409)


class JsonConversationPipeline:
    reference_records = conv.AgentPipeline.reference_records
    validate_records = conv.AgentPipeline.validate_records
    checkpoint_state = conv.AgentPipeline.checkpoint_state

    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, self.cfg = contract(self.graph, manifest["inputFields"])

    def predict(self, records, *, capture=False, deadline=None, cancelled=lambda: False, checkpoint=None):
        from langgraph.checkpoint.memory import InMemorySaver

        self.validate_records(records)
        verify(self.store, self.manifest)
        start = time.perf_counter()
        deadline = min(deadline or start + self.spec.limits.maxSeconds, start + self.spec.limits.maxSeconds)
        execution, saver, thread = uuid.uuid4().hex, InMemorySaver(), uuid.uuid4().hex
        pin = hashlib.sha256(dumps(self.manifest).encode()).hexdigest()
        if checkpoint is not None:
            payload = json.loads(read_verified(self.store, checkpoint))
            if payload["manifestSha256"] != pin:
                raise ProductionError("E_AGENT_CHECKPOINT", "Checkpoint belongs to another pinned version.", 409)
            thread = payload["threadId"]
            native = saver.serde.loads_typed((payload["type"], base64.b64decode(payload["data"], validate=True)))
            saver.put({"configurable": {"thread_id": thread, "checkpoint_ns": ""}}, native["checkpoint"], native["metadata"], native["checkpoint"]["channel_versions"])
        with tempfile.TemporaryDirectory(prefix="void-json-conversation-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            rt = conv.ServingRuntime(self.graph, self.spec, local, local.root, execution, thread, self.manifest["graphHash"])
            rt.deadline, rt.cancelled = deadline, cancelled
            config = {"configurable": {"thread_id": thread}, "recursion_limit": self.spec.limits.maxSteps}
            try:
                compiled = compile_graph(self.graph, rt, saver)
                rt.emit("run_started", threadId=thread, mode="isolated JSON conversation candidate", threadExisted=checkpoint is not None,
                        sourceRunId=self.manifest["runId"], inputFields=self.manifest["inputFields"], limits=self.manifest["limits"])
                final = None
                for state in compiled.stream(build_input(self.spec, records[0], checkpoint is not None), config, stream_mode="values"):
                    rt.check_budget()
                    if len(dumps(state).encode()) > 65_536:
                        raise ProductionError("E_AGENT_STATE_BOUNDS", "Native state exceeded 64 KiB; no result committed.", 413)
                    final = state
                rt.check_budget()
                # The object is validated before any checkpoint can be committed for this turn.
                result = jsa.output_value((final or {}).get(self.cfg.output_field), self.cfg)
                jsa.source_evidence(local, execution, self.graph)
            except BudgetExhausted as e:
                raise ProductionError("E_AGENT_BUDGET", str(e), 422) from e
            except NodeFailure as e:
                rt.check_budget()
                raise ProductionError(e.code, e.message, 503) from e
            except Exception as e:
                if type(e).__name__ == "GraphRecursionError":
                    raise ProductionError("E_AGENT_BUDGET", "Native LangGraph recursion limit exhausted; no successful output committed.") from e
                raise
            if compiled.get_state(config).next:
                raise ProductionError("E_AGENT_CHECKPOINT", "Only complete END checkpoints may be committed.", 409)
            saved = saver.get_tuple(config)
            typ, raw = saver.serde.dumps_typed({"checkpoint": saved.checkpoint, "metadata": saved.metadata})
            candidate = {"manifestSha256": pin, "threadId": thread, "type": typ, "data": base64.b64encode(raw).decode()}
            if len(dumps(candidate).encode()) > 131_072:
                raise ProductionError("E_AGENT_CHECKPOINT_BOUNDS", "Serialized native checkpoint exceeds 128 KiB.", 413)
            rt.emit("run_finished", status="completed", stoppedBy="end", modelCalls=rt.model_calls, tokens=rt.tokens)
            contexts = []
            for a in local.artifacts(execution, "model_context"):
                raw_ctx = local.read_artifact(a["sha256"])
                value = json.loads(raw_ctx)
                contexts.append({"sha256": a["sha256"], "callId": value["callId"], "nodeId": value["node"],
                                 "usage": value["usage"], "latencyMs": value["latencyMs"], "value": value if capture else None})
                if capture:
                    self.store.put_bytes(raw_ctx)
            events = local.events(execution)
            evidence = {"executionId": execution, "threadId": thread, "sourceRunId": self.manifest["runId"],
                        "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"], "modelCalls": rt.model_calls,
                        "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "contexts": contexts, "capturePolicy": "Full state/events/context only when release captureInputs=true; otherwise hashes/usage only.",
                        "isolation": "candidate restored from the last successful per-release/user/session native checkpoint; no research-thread mutation"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "JSON conversation turn exceeded its deadline before result commit.", 504)
        return {"predictions": [result], "family": "agent_json", "agent": evidence}, {
            "_checkpoint": candidate, "inferenceMs": (time.perf_counter() - start) * 1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph structured-output conversation turn; provider latency recorded per call"}
