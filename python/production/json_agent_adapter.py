"""Additive isolated native structured-output serving; legacy identities stay unchanged.

Each served object requires a successful native structured model call and pinned source evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from agent.blocks import NodeFailure, StructuredOutputConfig, json_schema_of, validate_structured
from agent.runtime import BudgetExhausted, build_input, compile_graph
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.schema import Graph
from graph_core.validate import require_executable
from tabular.core import dumps

from . import agent_adapter as legacy
from .pipeline import ProductionError, read_verified

NODE = "__agent_json_graph__"
ALLOWED = {"agent.prompt", "agent.set_state", "agent.structured_output"}
FILES = (*legacy.FILES, "production/json_agent_adapter.py", "production/runtime.py", "production/monitor.py",
         "production/models.py", "production/store.py", "production/pipeline.py", "../services/control/production_api.py")


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in FILES}


def contract(graph, inputs):
    """Keep the source graph intact; reject unsupported effects and output ambiguity."""
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes) or spec.indexes or spec.policies or any(f.scope != "turn" for f in spec.state):
        raise ProductionError("E_AGENT_JSON_SCOPE", "JSON serving accepts only prompt/set_state/one structured_output, turn-scoped state and no tools/retrieval/memory/interrupts.")
    lim = spec.limits
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 30
            and lim.maxModelCalls is not None and 1 <= lim.maxModelCalls <= 2 and lim.maxTokens is not None and lim.maxTokens <= 4096):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps <=25, maxSeconds <=30, maxModelCalls 1–2 and maxTokens <=4096.")
    if len(graph.nodes) > 16 or len(spec.state) > 32 or len(dumps(graph.to_json()).encode()) > 32768 or len(dumps([f.initial() for f in spec.state]).encode()) > 8192:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds16nodes/32KiB or defaults exceed8KiB.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "Declare1–4text input fields.")
    structured = [n for n in graph.nodes if n.type == "agent.structured_output"]
    if len(structured) != 1:
        raise ProductionError("E_AGENT_JSON_SCOPE", "Exactly one structured-output node is required.")
    cfg = StructuredOutputConfig.model_validate(structured[0].config)
    output = spec.field(cfg.output_field)
    if not output or output.type != "object" or cfg.output_field in inputs or output.reducer.kind != "replace":
        raise ProductionError("E_AGENT_OUTPUT", "JSON output must be a separate object state field with replace reducer.")
    if cfg.on_failure != "fail" or cfg.retry.maxRetries + 1 > lim.maxModelCalls:
        raise ProductionError("E_AGENT_JSON_FAILURE", "Use on_failure=fail and retries within the declared model-call budget; empty fallback objects are not success.")
    if not 1 <= len(cfg.schema_fields) <= 12 or len(dumps(json_schema_of(cfg.schema_fields)).encode()) > 4096:
        raise ProductionError("E_AGENT_JSON_SCHEMA", "Declare1–12flat native schema fields within4KiB.")
    model = cfg.model
    if (model.provider != "ollama" or model.base_url not in (None, legacy.OLLAMA_URL, "http://127.0.0.1:11434") or model.api_key
            or model.fixture or model.think or model.max_tokens is None or model.max_tokens > 128 or model.timeout_s > 30):
        raise ProductionError("E_AGENT_PROVIDER", "Use installed local Ollama, think=false, max_tokens<=128 and timeout_s<=30; no fixture/API/remote provider.")
    for node in graph.nodes:
        op = get_op(node.type)
        parsed = op.Config.model_validate(node.config)
        if any(len(t) > 2000 or t.count("{") > 16 for _, t in op.templates(parsed)):
            raise ProductionError("E_AGENT_LIMITS", "Templates are bounded to2000characters/16references.")
        if node.id != structured[0].id and cfg.output_field in op.writes(parsed):
            raise ProductionError("E_AGENT_OUTPUT", "Only the structured-output node may write the served object.")
    return spec, cfg


def output_value(value, cfg):
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ProductionError("E_AGENT_JSON_OUTPUT", "Output must be finite JSON.") from exc
    if len(encoded.encode()) > 8192:
        raise ProductionError("E_AGENT_JSON_OUTPUT", "Output exceeds8KiB.", 413)
    parsed, errors = validate_structured(encoded, cfg.schema_fields)
    if parsed is None:
        raise ProductionError("E_AGENT_JSON_OUTPUT", "Native structured-output contract failed: " + "; ".join(errors))
    return parsed


def source_evidence(store, run_id, graph):
    node = next(n.id for n in graph.nodes if n.type == "agent.structured_output")
    attempts = [e for e in store.events(run_id, types=("structured_attempt",)) if e["node_id"] == node]
    contexts = []
    for artifact in store.artifacts(run_id, "model_context"):
        value = json.loads(read_verified(store, artifact["sha256"]))
        if value["node"] == node and value["purpose"] == "structured_output" and value["provider"] == "ollama" and not value["fixture"]:
            contexts.append((artifact["sha256"], value["callId"]))
    if not any(e["data"].get("valid") is True and e["data"].get("callId") in {c[1] for c in contexts} for e in attempts):
        raise ProductionError("E_AGENT_SOURCE", "Source must record a successful native non-fixture structured model call; declared defaults are not model evidence.")
    return attempts, [c[0] for c in contexts]


def is_candidate(store, row):
    if row["status"] != "completed" or row["config"].get("kind") != "agent":
        return None
    try:
        graph, _ = legacy._graph(store, row["id"])
        _, cfg = contract(graph, row["config"].get("input", {}))
        finish = store.last_event(row["id"], "run_finished")
        finals = store.artifacts(row["id"], "final_state")
        if not finish or finish["data"].get("stoppedBy") != "end" or not finals:
            return None
        source_evidence(store, row["id"], graph)
        output_value(json.loads(read_verified(store, finals[-1]["sha256"])).get(cfg.output_field), cfg)
        return NODE
    except (ProductionError, ValueError):
        return None


def build_manifest(store, run_id, node):
    row = store.get_run(run_id)
    if node != NODE or not row or is_candidate(store, row) != NODE:
        raise ProductionError("E_AGENT_SOURCE", "Register a completed END run with bounded valid JSON output using __agent_json_graph__.")
    graph, graph_sha = legacy._graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, cfg = contract(graph, inputs)
    final_sha = store.artifacts(run_id, "final_state")[-1]["sha256"]
    attempts, contexts = source_evidence(store, run_id, graph)
    manifest = {"adapter": "native-langgraph-json-local", "family": "agent_json", "runId": run_id, "node": NODE,
        "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
        "sourceFinalStateSha256": final_sha, "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
        "inputFields": sorted(inputs), "outputField": cfg.output_field, "limits": spec.limits.model_dump(),
        "inputContract": "One record with exactly the pinned text fields,1–2000characters each,<=8KiBJSON.",
        "outputSchema": {"task": "agent_json", "classes": None, "target": cfg.output_field, "jsonSchema": json_schema_of(cfg.schema_fields)},
        "provider": legacy.provider_identity(cfg.model.model), "environment": legacy.environment(), "implementation": implementation(),
        "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()),
            "structuredEventsSha256": store.put_bytes(dumps(attempts).encode()), "contextSha256": contexts,
            "semantics": "fresh native JSON turn from defaults; source checkpoint/thread never copied"},
        "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source run input; not ground truth or benchmark"}
    JsonAgentPipeline(store, manifest).validate_records([inputs])
    return manifest


def verify(store, manifest):
    if manifest["environment"] != legacy.environment() or manifest["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned JSON adapter/native dependencies changed; register a new JSON version.", 409)
    for sha in (manifest["graphSha256"], manifest["sourceFinalStateSha256"], manifest["referenceSha256"], manifest["source"]["configSha256"],
                manifest["source"]["structuredEventsSha256"], *manifest["source"]["contextSha256"]):
        read_verified(store, sha)
    graph, sha = legacy._graph(store, manifest["runId"])
    row = store.get_run(manifest["runId"])
    finals = store.artifacts(manifest["runId"], "final_state")
    finish = store.last_event(manifest["runId"], "run_finished")
    if (not finish or finish["data"].get("stoppedBy") != "end" or sha != manifest["graphSha256"] or semantic_hash(graph) != manifest["graphHash"] or not row or row["status"] != "completed"
            or dumps(row["config"]).encode() != read_verified(store, manifest["source"]["configSha256"])
            or not any(a["sha256"] == manifest["sourceFinalStateSha256"] and a["status"] == "complete" for a in finals)):
        raise ProductionError("E_AGENT_SOURCE", "Pinned JSON source membership/configuration differs.", 409)
    spec, cfg = contract(graph, manifest["inputFields"])
    attempts, contexts = source_evidence(store, manifest["runId"], graph)
    if dumps(attempts).encode() != read_verified(store, manifest["source"]["structuredEventsSha256"]) or contexts != manifest["source"]["contextSha256"]:
        raise ProductionError("E_AGENT_SOURCE", "Pinned native structured events/context membership differs.", 409)
    output_value(json.loads(read_verified(store, manifest["sourceFinalStateSha256"])).get(cfg.output_field), cfg)
    schema = {"task": "agent_json", "classes": None, "target": cfg.output_field, "jsonSchema": json_schema_of(cfg.schema_fields)}
    if (manifest["outputField"] != cfg.output_field or manifest["outputSchema"] != schema or manifest["limits"] != spec.limits.model_dump()
            or manifest["inputFields"] != sorted(row["config"].get("input", {})) or manifest["source"]["runId"] != row["id"]
            or manifest["family"] != "agent_json" or manifest["node"] != NODE or manifest["adapter"] != "native-langgraph-json-local"):
        raise ProductionError("E_AGENT_SOURCE", "Pinned JSON output contract differs from the source.", 409)
    if legacy.provider_identity(cfg.model.model) != manifest["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Local Ollama digest/runtime changed; register and warm a new JSON version.", 409)


class JsonAgentPipeline:
    # These input/reference contracts are identical to legacy isolated text turns.
    reference_records = legacy.AgentPipeline.reference_records
    validate_records = legacy.AgentPipeline.validate_records

    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, self.cfg = contract(self.graph, manifest["inputFields"])

    def predict(self, records, *, capture=False, deadline=None, cancelled=lambda: False):
        from langgraph.checkpoint.memory import InMemorySaver

        self.validate_records(records)
        verify(self.store, self.manifest)
        start = time.perf_counter()
        deadline = min(deadline or start + self.spec.limits.maxSeconds, start + self.spec.limits.maxSeconds)
        execution = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="void-json-agent-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            runtime = legacy.ServingRuntime(self.graph, self.spec, local, local.root, execution, execution, self.manifest["graphHash"])
            runtime.deadline, runtime.cancelled = deadline, cancelled
            config = {"configurable": {"thread_id": execution}, "recursion_limit": self.spec.limits.maxSteps}
            runtime.emit("run_started", threadId=execution, mode="isolated native JSON serving turn", threadExisted=False,
                         sourceRunId=self.manifest["runId"], inputFields=self.manifest["inputFields"], limits=self.manifest["limits"])
            try:
                compiled = compile_graph(self.graph, runtime, InMemorySaver())
                final = None
                for state in compiled.stream(build_input(self.spec, records[0], False), config, stream_mode="values"):
                    runtime.check_budget()
                    if len(dumps(state).encode()) > 65536:
                        raise ProductionError("E_AGENT_STATE_BOUNDS", "Native JSON state exceeds64KiB; no result committed.", 413)
                    final = state
                runtime.check_budget()
                result = output_value((final or {}).get(self.cfg.output_field), self.cfg)
                source_evidence(local, execution, self.graph)
            except BudgetExhausted as exc:
                raise ProductionError("E_AGENT_BUDGET", str(exc)) from exc
            except NodeFailure as exc:
                runtime.check_budget()
                raise ProductionError(exc.code, exc.message, 503) from exc
            except Exception as exc:
                if type(exc).__name__ == "GraphRecursionError":
                    raise ProductionError("E_AGENT_BUDGET", "Native recursion budget exhausted; no JSON output committed.") from exc
                raise
            runtime.emit("run_finished", status="completed", stoppedBy="end", modelCalls=runtime.model_calls, tokens=runtime.tokens)
            contexts = []
            for artifact in local.artifacts(execution, "model_context"):
                raw = local.read_artifact(artifact["sha256"])
                value = json.loads(raw)
                contexts.append({"sha256": artifact["sha256"], "callId": value["callId"], "nodeId": value["node"],
                    "usage": value["usage"], "latencyMs": value["latencyMs"], "value": value if capture else None})
                if capture:
                    self.store.put_bytes(raw)
            events = local.events(execution)
            evidence = {"executionId": execution, "threadId": execution, "sourceRunId": self.manifest["runId"],
                "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"], "modelCalls": runtime.model_calls,
                "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                "contexts": contexts, "capturePolicy": "State/events/context only with captureInputs=true; hashes/usage otherwise.",
                "isolation": "fresh in-memory native checkpoint; source/research/conversation state never copied"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "JSON turn exceeded deadline before result commit.", 504)
        return {"predictions": [result], "family": "agent_json", "agent": evidence}, {
            "inferenceMs": (time.perf_counter()-start)*1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "actual native LangGraph structured-output execution; provider latency recorded per call"}
