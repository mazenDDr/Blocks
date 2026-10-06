"""Native committed approval checkpoints in effect-free conversations (ADR0063)."""
from __future__ import annotations

import base64
import hashlib
import json
import platform
import tempfile
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import httpx

from agent.blocks import ChatModelConfig, HumanInterruptConfig, NodeFailure
from agent.models import OLLAMA_URL
from agent.runtime import BudgetExhausted, Runtime, build_input, compile_graph
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import require_executable
from tabular.core import dumps

from .pipeline import ProductionError, read_verified

NODE = "__agent_approval_conversation__"
ALLOWED = {"agent.prompt", "agent.chat_model", "agent.set_state", "agent.human_interrupt"}
FILES = ("production/approval_adapter.py", "production/approval_store.py", "production/approval_requests.py", "production/approval_monitor.py", "../services/control/approval_api.py", "production/models.py", "production/conversation_adapter.py", "production/agent_adapter.py", "agent/runtime.py", "agent/blocks.py", "agent/spec.py", "agent/models.py",
         "agent/validate.py", "agent/memory.py", "agent/index.py", "agent/policy.py", "graph_core/schema.py", "graph_core/hashing.py",
         "graph_core/registry.py", "graph_core/validate.py", "artifact_store/store.py", "storage/schema.py", "tabular/core.py")


def environment():
    return {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}", **{p: version(p) for p in
            ("langgraph", "langgraph-checkpoint", "langchain-core", "langchain-ollama", "ollama", "pydantic", "httpx")}}


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in FILES}


def provider_identity(model):
    """No pull, remote URL, credentials or model loading in identity checks."""
    try:
        with httpx.Client(base_url=OLLAMA_URL, timeout=3, trust_env=False) as c:
            tags = c.get("/api/tags")
            tags.raise_for_status()
            name = model if ":" in model else f"{model}:latest"
            found = next((r for r in tags.json()["models"] if r["name"] == name), None)
            if not found or not found.get("digest"):
                raise ProductionError("E_AGENT_MODEL_MISSING", f"Local Ollama model {name} is not installed; no model is downloaded.", 409)
            response = c.get("/api/version")
            response.raise_for_status()
            return {"provider": "ollama", "baseUrl": OLLAMA_URL, "model": name, "digest": found["digest"],
                    "runtimeVersion": response.json()["version"], "pinnedAt": "registration (source runs did not record model digests)"}
    except ProductionError:
        raise
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
        raise ProductionError("E_AGENT_PROVIDER_UNAVAILABLE", f"Cannot verify local Ollama identity: {type(e).__name__}.", 503) from e


def _graph(store, run_id):
    arts = store.artifacts(run_id, "graph")
    if not arts:
        raise ProductionError("E_AGENT_SOURCE", "Agent run has no recorded graph.")
    sha = arts[0]["sha256"]
    return Graph.model_validate(json.loads(read_verified(store, sha))), sha


def contract(graph, inputs):
    """Refuse unsupported semantics; never prune/rewrite a source graph to make it servable."""
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as e:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {e}.") from e
    lim = spec.limits
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes) or spec.indexes or spec.policies:
        raise ProductionError("E_AGENT_SCOPE", "Serving accepts only prompt, chat_model and set_state nodes; file tools, retrieval and memory effects are not supported.")
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 30
            and lim.maxModelCalls is not None and lim.maxModelCalls <= 2 and lim.maxTokens is not None and lim.maxTokens <= 4096):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps <= 25, maxSeconds <= 30, maxModelCalls <= 2 and maxTokens <= 4096.")
    if len(graph.nodes) > 16 or len(spec.state) > 32 or len(dumps(graph.to_json()).encode()) > 32_768 or len(dumps([f.initial() for f in spec.state]).encode()) > 8192:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds 16 nodes / 32 KiB or state defaults exceed 8 KiB.")
    from graph_core.registry import get_op
    for n in graph.nodes:
        op = get_op(n.type)
        cfg = op.Config.model_validate(n.config)
        if set(inputs) & {field.split(".")[0] for field in op.writes(cfg)}:
            raise ProductionError("E_AGENT_INPUT", "Approval input fields must stay immutable during the turn.")
        if any(len(t) > 2000 or t.count("{") > 16 for _, t in op.templates(cfg)):
            raise ProductionError("E_AGENT_LIMITS", "Each template is bounded to 2000 characters and 16 field references.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "The recorded input must declare 1–4 text state fields.")
    reviews = [n for n in graph.nodes if n.type == "agent.human_interrupt"]
    if len(reviews) != 1:
        raise ProductionError("E_AGENT_SCOPE", "Approval conversations require exactly one human_interrupt.")
    review = HumanInterruptConfig.model_validate(reviews[0].config)
    if not review.actions or len(review.actions) != len(set(review.actions)):
        raise ProductionError("E_AGENT_SCOPE", "Declare at least one approval action, without duplicates.")
    for field in (review.decision_field, review.edit_field):
        if field and (not spec.field(field) or spec.field(field).type != "text" or spec.field(field).reducer.kind != "replace"):
            raise ProductionError("E_AGENT_SCOPE", "Approval decision/edit fields must be replace-reduced text fields.")
    # A single interrupt in an acyclic graph bounds one logical turn to two segments.
    adjacent = {n.id: [] for n in graph.nodes}
    fixed = {}
    for edge in graph.edges:
        fixed[edge.from_.node] = fixed.get(edge.from_.node, 0) + 1
        if edge.from_.node in adjacent:
            adjacent[edge.from_.node].append(edge.to.node)
    if spec.joins or any(count > 1 for count in fixed.values()):
        raise ProductionError("E_AGENT_SCOPE", "Approval graphs use one active path: conditional routes are supported, fixed parallel forks/joins are not.")
    for route in spec.routes:
        adjacent[route.from_].extend([c.to for c in route.cases] + [route.default])
    seen, active = set(), set()
    def visit(node):
        if node in active:
            raise ProductionError("E_AGENT_SCOPE", "Approval graphs must be acyclic; repeated interrupts are not served.")
        if node in seen or node not in adjacent:
            return
        active.add(node)
        for target in adjacent[node]:
            visit(target)
        active.remove(node)
        seen.add(node)
    for node in adjacent:
        visit(node)
    if not any(f.scope == "thread" for f in spec.state):
        raise ProductionError("E_AGENT_SCOPE", "Conversation graphs must declare thread-scoped state.")
    if any(spec.field(k).scope != "turn" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "HTTP text inputs must be turn-scoped; thread state is owned by native checkpoints.")
    chats = [n for n in graph.nodes if n.type == "agent.chat_model"]
    if len(chats) > 1:
        raise ProductionError("E_AGENT_SCOPE", "At most one chat-model node is supported (it may run twice in bounded control flow).")
    model = None
    if chats:
        cfg = ChatModelConfig.model_validate(chats[0].config)
        model = cfg.model
        if (model.provider != "ollama" or model.base_url not in (None, OLLAMA_URL, "http://127.0.0.1:11434")
                or model.api_key or model.fixture or model.think or model.max_tokens is None or model.max_tokens > 128 or model.timeout_s > 30):
            raise ProductionError("E_AGENT_PROVIDER", "Use local Ollama on port 11434, think=false, max_tokens <= 128, timeout_s <= 30; fixture/API/remote providers are refused.")
        output = cfg.output_field
    else:
        # Pure StateGraph workflows are valid native agents; they make zero model calls.
        outputs = sorted({a["field"] for n in graph.nodes if n.type == "agent.set_state" for a in n.config.get("assignments", [])
                          if spec.field(a["field"]) and spec.field(a["field"]).type == "text" and a["field"] not in inputs})
        if len(outputs) != 1:
            raise ProductionError("E_AGENT_OUTPUT", "A model-free workflow needs exactly one written text output field.")
        output = outputs[0]
    if not spec.field(output) or spec.field(output).type != "text" or output in inputs:
        raise ProductionError("E_AGENT_OUTPUT", "The served output must be a text state field separate from input fields.")
    return spec, output, model


def is_candidate(store, row):
    if row["status"] != "completed" or row["config"].get("kind") != "agent":
        return None
    try:
        graph, _ = _graph(store, row["id"])
        contract(graph, row["config"].get("input", {}))
        finished = store.last_event(row["id"], "run_finished")
        return NODE if finished and finished["data"].get("stoppedBy") == "end" else None
    except (ProductionError, ValueError):
        return None


def build_manifest(store, run_id, node):
    row = store.get_run(run_id)
    if not row or row["status"] != "completed" or row["config"].get("kind") != "agent" or node != NODE:
        raise ProductionError("E_AGENT_SOURCE", "Register a completed agent run using __agent_approval_conversation__.")
    graph, graph_sha = _graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, output, model = contract(graph, inputs)
    finished = store.last_event(run_id, "run_finished")
    finals = store.artifacts(run_id, "final_state")
    if not finished or finished["data"].get("stoppedBy") != "end" or not finals:
        raise ProductionError("E_AGENT_SOURCE", "Budget/recursion-stopped runs cannot be registered as successful end-to-end workflows.")
    read_verified(store, finals[-1]["sha256"])
    m = {"adapter": "native-langgraph-approval-local", "family": "agent_turn", "runId": run_id, "node": NODE,
         "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
         "sourceFinalStateSha256": finals[-1]["sha256"], "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
         "inputFields": sorted(inputs), "outputField": output, "limits": spec.limits.model_dump(),
         "inputContract": "One record with exactly the pinned text input fields; 1–2000 characters each, <= 8 KiB JSON.",
         "outputSchema": {"task": "agent_turn", "classes": None, "target": output},
         "provider": provider_identity(model.model) if model else None, "environment": environment(), "implementation": implementation(),
         "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()),
                    "semantics": "new conversations start at declared defaults; subsequent turns restore the last successful native checkpoint; paused native checkpoints also commit; source research checkpoint is never copied"},
         "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source run input (not ground truth or a benchmark)"}
    AgentPipeline(store, m).validate_records([inputs])
    return m


def verify(store, m):
    if m["environment"] != environment() or m["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned agent implementation or native dependencies changed; register a new version.", 409)
    for sha in (m["graphSha256"], m["sourceFinalStateSha256"], m["referenceSha256"], m["source"]["configSha256"]):
        read_verified(store, sha)
    graph, sha = _graph(store, m["runId"])
    row = store.get_run(m["runId"])
    finals = store.artifacts(m["runId"], "final_state")
    if (sha != m["graphSha256"] or semantic_hash(graph) != m["graphHash"] or not row or row["status"] != "completed"
            or dumps(row["config"]).encode() != read_verified(store, m["source"]["configSha256"])
            or not any(a["sha256"] == m["sourceFinalStateSha256"] and a["status"] == "complete" for a in finals)):
        raise ProductionError("E_AGENT_SOURCE", "Pinned agent source membership/configuration differs.", 409)
    if m["provider"] and provider_identity(m["provider"]["model"]) != m["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Ollama model digest/runtime changed; register and warm a new version.", 409)


class ServingRuntime(Runtime):
    """Keep the native graph semantics; narrow HTTP timeouts to the remaining serving deadline."""
    def check_budget(self, kind=None):
        if self.cancelled():
            raise ProductionError("E_REQUEST_CANCELLED", "Cancelled before committing the isolated turn.", 409)
        if time.perf_counter() >= self.deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Isolated agent turn exceeded its serving deadline.", 504)
        return super().check_budget(kind)

    def model_call(self, ctx, spec, messages, *args, **kwargs):
        self.check_budget("model")
        if len(dumps(messages).encode()) > 16_384:
            raise ProductionError("E_AGENT_CONTEXT_BOUNDS", "Rendered model context exceeds 16 KiB.", 413)
        effective = spec.model_copy(update={"timeout_s": min(spec.timeout_s, max(.001, self.deadline-time.perf_counter()))})
        result = super().model_call(ctx, effective, messages, *args, **kwargs)
        self.check_budget()
        if self.spec.limits.maxTokens is not None and self.tokens > self.spec.limits.maxTokens:
            raise BudgetExhausted("maxTokens", self.spec.limits.maxTokens, self.tokens)
        return result


class AgentPipeline:
    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, _, _ = contract(self.graph, manifest["inputFields"])

    def reference_records(self, n=1):
        return json.loads(read_verified(self.store, self.manifest["referenceSha256"]))[:n]

    def validate_records(self, records, max_batch=1):
        if len(records) != 1 or max_batch < 1 or len(dumps(records).encode()) > 8192:
            raise ProductionError("E_REQUEST_BOUNDS", "Agent serving accepts exactly one turn within 8 KiB.", 413)
        r = records[0]
        if set(r) != set(self.manifest["inputFields"]) or any(not isinstance(v, str) or not 1 <= len(v) <= 2000 for v in r.values()):
            raise ProductionError("E_REQUEST_SCHEMA", "Supply exactly the pinned input fields, each a 1–2000-character text value.")

    def decode(self, sha):
        from langgraph.checkpoint.memory import InMemorySaver
        payload = json.loads(read_verified(self.store, sha))
        if payload["manifestSha256"] != hashlib.sha256(dumps(self.manifest).encode()).hexdigest():
            raise ProductionError("E_AGENT_CHECKPOINT", "Checkpoint belongs to another pinned version.", 409)
        native = InMemorySaver().serde.loads_typed((payload["type"], base64.b64decode(payload["data"], validate=True)))
        return payload, native

    def inspect_checkpoint(self, sha):
        payload, native = self.decode(sha)
        state = {f.name: native["checkpoint"]["channel_values"].get(f.name) for f in self.spec.state}
        return {"state": state, "pending": payload.get("pending"), "budget": payload.get("budget")}

    def checkpoint_state(self, sha):
        inspected = self.inspect_checkpoint(sha)
        if inspected["pending"]:
            raise ProductionError("E_APPROVAL_PENDING", "Resume the reviewed approval before resetting, forking, restoring or replaying this paused checkpoint.", 409)
        return inspected["state"]

    def predict(self, records, *, capture=False, deadline=None, cancelled=lambda: False, checkpoint=None, approval=None):
        from langgraph.checkpoint.memory import InMemorySaver

        self.validate_records(records)
        verify(self.store, self.manifest)
        start = time.perf_counter()
        deadline = min(deadline or start + self.spec.limits.maxSeconds, start + self.spec.limits.maxSeconds)
        execution = uuid.uuid4().hex
        saver = InMemorySaver()
        thread = uuid.uuid4().hex
        prior = {"modelCalls": 0, "tokens": 0, "activeSeconds": 0.0}
        pending = None
        if checkpoint is not None:
            payload, native = self.decode(checkpoint)
            pending = payload.get("pending")
            thread = payload["threadId"]
            saved_config = saver.put({"configurable": {"thread_id": thread, "checkpoint_ns": ""}}, native["checkpoint"],
                                     native["metadata"], native["checkpoint"]["channel_versions"])
            for task_id, channel, value in native.get("pendingWrites", []):
                saver.put_writes(saved_config, [(channel, value)], task_id)
            if pending:
                prior = payload["budget"]
                if approval is None:
                    raise ProductionError("E_APPROVAL_PENDING", "This session is paused; send a reviewed resume request.", 409)
                if approval.interruptId != pending["id"] or approval.action not in pending["value"]["actions"]:
                    raise ProductionError("E_APPROVAL_DECISION", "Decision does not match the pending native interrupt.", 409)
                if records[0] != {key: native["checkpoint"]["channel_values"].get(key) for key in self.manifest["inputFields"]}:
                    raise ProductionError("E_APPROVAL_INPUT", "A resume preserves the paused turn inputs; it cannot start another turn.", 409)
        if approval is not None and not pending:
            raise ProductionError("E_APPROVAL_EMPTY", "No committed pending interrupt to resume.", 409)
        remaining = self.spec.limits.maxSeconds - prior["activeSeconds"]
        if remaining <= 0:
            raise ProductionError("E_AGENT_BUDGET", "Logical turn active-time budget exhausted.", 422)
        deadline = min(deadline, start + remaining)
        # Neither checkpoints, memory, events nor contexts enter the research workbench.
        with tempfile.TemporaryDirectory(prefix="void-agent-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            rt = ServingRuntime(self.graph, self.spec, local, local.root, execution, thread, self.manifest["graphHash"])
            rt.deadline, rt.cancelled = deadline, cancelled
            rt.model_calls, rt.tokens = prior["modelCalls"], prior["tokens"]
            config = {"configurable": {"thread_id": thread}, "recursion_limit": self.spec.limits.maxSteps}
            try:
                compiled = compile_graph(self.graph, rt, saver)
                rt.emit("run_started", threadId=thread, mode="isolated conversation candidate", threadExisted=checkpoint is not None,
                        sourceRunId=self.manifest["runId"], inputFields=self.manifest["inputFields"], limits=self.manifest["limits"])
                final = None
                from langgraph.types import Command
                turn_input = Command(resume={pending["id"]: {"action": approval.action, **({"value": approval.value} if approval.action == "edit" else {})}}) if approval else build_input(self.spec, records[0], checkpoint is not None)
                for state in compiled.stream(turn_input, config, stream_mode="values"):
                    rt.check_budget()
                    declared = {f.name: state.get(f.name) for f in self.spec.state}
                    if len(dumps(declared).encode()) > 65_536:
                        raise ProductionError("E_AGENT_STATE_BOUNDS", "Native state exceeded 64 KiB; no result committed.", 413)
                    final = state
                rt.check_budget()
            except BudgetExhausted as e:
                raise ProductionError("E_AGENT_BUDGET", str(e), 422) from e
            except NodeFailure as e:
                rt.check_budget()  # deadline/cancellation takes precedence over provider timeout
                raise ProductionError(e.code, e.message, 503) from e
            except Exception as e:
                if type(e).__name__ == "GraphRecursionError":
                    raise ProductionError("E_AGENT_BUDGET", "Native LangGraph recursion limit exhausted; no successful output committed.") from e
                raise
            snapshot = compiled.get_state(config)
            interrupts = [i for task in snapshot.tasks for i in task.interrupts]
            if len(interrupts) > 1 or (snapshot.next and not interrupts):
                raise ProductionError("E_AGENT_CHECKPOINT", "Only one genuine pending native interrupt or END may commit.", 409)
            pending = {"id": interrupts[0].id, "value": interrupts[0].value} if interrupts else None
            if pending and len(dumps(pending).encode()) > 8192:
                raise ProductionError("E_APPROVAL_BOUNDS", "Pending review exceeds 8 KiB.", 413)
            final = {f.name: snapshot.values.get(f.name) for f in self.spec.state}
            if len(dumps(final).encode()) > 65_536 or (not pending and not isinstance(final.get(self.manifest["outputField"]), str)):
                raise ProductionError("E_AGENT_STATE_BOUNDS", "Final state exceeds 64 KiB or END output is not text.", 413)
            saved = saver.get_tuple(config)
            typ, raw = saver.serde.dumps_typed({"checkpoint": saved.checkpoint, "metadata": saved.metadata,
                                             "pendingWrites": saved.pending_writes if pending else []})
            candidate = {"manifestSha256": hashlib.sha256(dumps(self.manifest).encode()).hexdigest(),
                         "threadId": thread, "type": typ, "data": base64.b64encode(raw).decode(), "pending": pending,
                         "budget": {"modelCalls": rt.model_calls, "tokens": rt.tokens, "activeSeconds": prior["activeSeconds"] + time.perf_counter()-start}}
            if len(dumps(candidate).encode()) > 131_072:
                raise ProductionError("E_AGENT_CHECKPOINT_BOUNDS", "Serialized native checkpoint exceeds 128 KiB.", 413)
            rt.emit("run_finished", status="paused" if pending else "completed", stoppedBy="interrupt" if pending else "end", modelCalls=rt.model_calls-prior["modelCalls"], tokens=rt.tokens-prior["tokens"])
            contexts = []
            for a in local.artifacts(execution, "model_context"):
                raw = local.read_artifact(a["sha256"])
                value = json.loads(raw)
                contexts.append({"sha256": a["sha256"], "callId": value["callId"], "nodeId": value["node"],
                                 "usage": value["usage"], "latencyMs": value["latencyMs"], "value": value if capture else None})
                if capture:
                    self.store.put_bytes(raw)
            events = local.events(execution)
            evidence = {"executionId": execution, "threadId": thread, "sourceRunId": self.manifest["runId"],
                        "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"], "modelCalls": rt.model_calls-prior["modelCalls"], "logicalTurnModelCalls": rt.model_calls,
                        "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "contexts": contexts, "capturePolicy": "Full state/events/context only when release captureInputs=true; otherwise hashes/usage only.",
                        "isolation": "candidate restored from the last successful per-release/user/session native checkpoint; no research-thread mutation"}
        verify(self.store, self.manifest)  # do not accept a model swapped during this turn
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Agent turn exceeded its deadline before result commit.", 504)
        return {"predictions": [] if pending else [final[self.manifest["outputField"]]], "pending": pending, "family": "agent_turn", "agent": evidence}, {
            "_checkpoint": candidate, "_status": 202 if pending else 200, "inferenceMs": (time.perf_counter()-start)*1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph prompt/control/state/model execution together; provider latency recorded per call"}
