"""Stateless agent turns that retrieve from pinned index snapshots (ADR 0060).

The research workbench rebuilds an index whenever its documents change. A served version must not: registration copies the
exact FAISS index, chunks and index manifest the source run used into the CAS (checked against the identity the run recorded),
and every serving turn restores that snapshot into its private temporary workbench and searches it. Serving never reads the
document directory, never rebuilds and refuses a changed embedding model. The existing agent adapters pin their own source
files in registered identities, so this path lives in its own module and reuses their functions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid

from agent.blocks import ChatModelConfig, NodeFailure, RetrieveConfig
from agent.index import IndexStore
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

NODE = "__agent_retrieval_graph__"
ALLOWED = {"agent.prompt", "agent.set_state", "agent.chat_model", "agent.retrieve"}
FILES = (*conv.FILES, "production/retrieval_agent_adapter.py")
SNAPSHOT_FILES = ("faiss.index", "chunks.json", "manifest.json")
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024


def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in FILES}


def embedding_identity(spec):
    """Embedding provider identity: the local Ollama model digest, or the deterministic local_hash parameters."""
    e = spec.embeddings
    if e.provider == "ollama":
        if getattr(e, "baseUrl", None) not in (None, conv.OLLAMA_URL, "http://127.0.0.1:11434"):
            raise ProductionError("E_AGENT_PROVIDER", "Embeddings must use installed local Ollama on port 11434.")
        return conv.provider_identity(e.model)
    return {"provider": "local_hash", "dimension": e.dimension, "normalize": e.normalize}


def contract(graph, inputs):
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes) or spec.policies:
        raise ProductionError("E_AGENT_SCOPE", "Retrieval serving accepts prompt, set_state, retrieve and one chat_model; no tools, memory policies or interrupts.")
    retrieves = [n for n in graph.nodes if n.type == "agent.retrieve"]
    used = {RetrieveConfig.model_validate(n.config).index for n in retrieves}
    if not retrieves or not 1 <= len(spec.indexes) <= 2 or used != {i.id for i in spec.indexes}:
        raise ProductionError("E_AGENT_SCOPE", "Declare 1–2 indexes, each used by a retrieve node (otherwise use the plain agent adapter).")
    if any(RetrieveConfig.model_validate(n.config).k > 8 for n in retrieves):
        raise ProductionError("E_AGENT_LIMITS", "Retrieve at most 8 chunks per node.")
    if any(f.scope != "turn" for f in spec.state):
        raise ProductionError("E_AGENT_SCOPE", "Retrieval turns are stateless: every state field must be turn-scoped.")
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
            raise ProductionError("E_AGENT_OUTPUT", "A model-free retrieval workflow needs exactly one written text output field.")
        output = outputs[0]
    if not spec.field(output) or spec.field(output).type != "text" or output in inputs:
        raise ProductionError("E_AGENT_OUTPUT", "The served output must be a text state field separate from input fields.")
    return spec, output, model


def _snapshot(store, run_id, spec):
    """Copy each index's exact built files into the CAS, tied to the identity the source run recorded."""
    recorded = {e["data"]["index"]: e["data"]["identity"] for e in store.events(run_id, types=("index_ready",))}
    live = IndexStore(store.root)
    out = {}
    for ispec in spec.indexes:
        manifest = live.manifest(ispec.id)
        if ispec.id not in recorded or not manifest or not manifest["identity"].startswith(recorded[ispec.id]):
            raise ProductionError("E_AGENT_INDEX", f"Index '{ispec.id}' differs from the one the source run used (documents, splitter or embeddings changed); rerun the source.", 409)
        files = {}
        for name in SNAPSHOT_FILES:
            p = live.dir(ispec.id) / name
            if not p.is_file() or p.stat().st_size > MAX_SNAPSHOT_BYTES:
                raise ProductionError("E_AGENT_INDEX", f"Index '{ispec.id}' snapshot file {name} is missing or larger than 32 MiB.", 413)
            files[name] = store.put_bytes(p.read_bytes())
        out[ispec.id] = {"identity": manifest["identity"], "chunks": manifest["chunks"], "embedding": manifest["embedding"],
                         "documents": manifest["documents"], "files": files, "embeddingProvider": embedding_identity(ispec)}
    return out


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
        raise ProductionError("E_AGENT_SOURCE", "Register a completed END retrieval run using __agent_retrieval_graph__.")
    graph, graph_sha = conv._graph(store, run_id)
    inputs = row["config"].get("input", {})
    spec, output, model = contract(graph, inputs)
    finals = store.artifacts(run_id, "final_state")
    if not finals:
        raise ProductionError("E_AGENT_SOURCE", "The source run recorded no final state.")
    m = {"adapter": "native-langgraph-retrieval-local", "family": "agent_turn", "runId": run_id, "node": NODE,
         "graphHash": semantic_hash(graph), "graphSha256": graph_sha, "modelSha256": None,
         "sourceFinalStateSha256": finals[-1]["sha256"], "referenceSha256": store.put_bytes(dumps([inputs]).encode()),
         "inputFields": sorted(inputs), "outputField": output, "limits": spec.limits.model_dump(),
         "inputContract": "One record with exactly the pinned text input fields; 1–2000 characters each, <= 8 KiB JSON.",
         "outputSchema": {"task": "agent_turn", "classes": None, "target": output},
         "indexes": _snapshot(store, run_id, spec),
         "provider": conv.provider_identity(model.model) if model else None, "environment": conv.environment(), "implementation": implementation(),
         "source": {"runId": run_id, "configSha256": store.put_bytes(dumps(row["config"]).encode()),
                    "semantics": "fresh stateless turn; retrieval searches the index snapshot pinned at registration, never the live document folder"},
         "fitArtifacts": {}, "evaluationArtifacts": [], "referencePartition": "source run input (not ground truth or a benchmark)"}
    RetrievalPipeline(store, m).validate_records([inputs])
    return m


def verify(store, m):
    if m["environment"] != conv.environment() or m["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned retrieval implementation or native dependencies changed; register a new version.", 409)
    for sha in (m["graphSha256"], m["sourceFinalStateSha256"], m["referenceSha256"], m["source"]["configSha256"],
                *(sha for ix in m["indexes"].values() for sha in ix["files"].values())):
        read_verified(store, sha)
    graph, sha = conv._graph(store, m["runId"])
    if sha != m["graphSha256"] or semantic_hash(graph) != m["graphHash"]:
        raise ProductionError("E_AGENT_SOURCE", "Pinned retrieval source graph differs.", 409)
    spec, _, model = contract(graph, m["inputFields"])
    for ispec in spec.indexes:
        if embedding_identity(ispec) != m["indexes"][ispec.id]["embeddingProvider"]:
            raise ProductionError("E_AGENT_MODEL_CHANGED", f"Embedding model for index '{ispec.id}' changed; register a new version.", 409)
    if model and conv.provider_identity(model.model) != m["provider"]:
        raise ProductionError("E_AGENT_MODEL_CHANGED", "Ollama chat model digest/runtime changed; register and warm a new version.", 409)


class PinnedIndexRuntime(conv.ServingRuntime):
    """Serving runtime whose indexes are the registered snapshots: no document reads, no rebuilds."""
    pinned: dict = {}

    def ensure_index(self, ispec):
        with self._lock:
            if ispec.id not in self._built:
                man = self.indexes.manifest(ispec.id)
                if not man or man["identity"] != self.pinned[ispec.id]["identity"]:
                    raise ProductionError("E_AGENT_INDEX", f"Pinned snapshot for index '{ispec.id}' is unavailable.", 409)
                self._built[ispec.id] = {**man, "action": "pinned"}
                self.emit("index_ready", None, index=ispec.id, action="pinned", chunks=man["chunks"], documents=len(man["documents"]),
                          embedding=man["embedding"]["identity"], identity=man["identity"][:16], scoreInterpretation=man["scoreInterpretation"])
            return self._built[ispec.id]


class RetrievalPipeline:
    reference_records = conv.AgentPipeline.reference_records
    validate_records = conv.AgentPipeline.validate_records

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
            rt = PinnedIndexRuntime(self.graph, self.spec, local, local.root, execution, execution, self.manifest["graphHash"])
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
            rt.emit("run_finished", status="completed", stoppedBy="end", modelCalls=rt.model_calls, tokens=rt.tokens)
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
                        "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"], "modelCalls": rt.model_calls,
                        "indexes": {iid: {"identity": s["identity"], "chunks": s["chunks"]} for iid, s in self.manifest["indexes"].items()},
                        "retrievals": retrievals, "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "contexts": contexts, "capturePolicy": "Full state/events/context only when release captureInputs=true; retrieved chunk ids/scores always.",
                        "isolation": "fresh in-memory checkpoint; index snapshot restored privately; no research index, memory or thread mutation"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Retrieval turn exceeded its deadline before result commit.", 504)
        return {"predictions": [final[self.manifest["outputField"]]], "family": "agent_turn", "agent": evidence}, {
            "inferenceMs": (time.perf_counter() - start) * 1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph retrieval/prompt/model execution together; provider latency recorded per call"}
