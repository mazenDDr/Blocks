"""Native conversation/JSON turns with pinned retrieval and short-term policies (ADR0064)."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import time
import uuid
from agent.blocks import ChatModelConfig, MemorySelectConfig, MemoryWriteConfig, NodeFailure, RetrieveConfig, StructuredOutputConfig, json_schema_of
from agent.policy import STAGE_CONFIG, RetrieveCfg, SummarizeCfg, BudgetCfg
from agent.runtime import BudgetExhausted, build_input, compile_graph
from agent.spec import agent_spec
from artifact_store import ArtifactStore
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.schema import Graph
from graph_core.validate import require_executable
from tabular.core import dumps
from . import conversation_adapter as conv, json_agent_adapter as jsa, retrieval_agent_adapter as retrieval
from .pipeline import ProductionError, read_verified

NODES = {"agent_context": "__agent_context_graph__", "conversation_context": "__agent_context_conversation__",
         "agent_context_json": "__agent_context_json__", "conversation_context_json": "__agent_context_json_conversation__"}
ALLOWED = {"agent.prompt", "agent.set_state", "agent.chat_model", "agent.structured_output", "agent.retrieve", "agent.memory_select", "agent.memory_write"}
FILES = (*retrieval.FILES, "production/json_agent_adapter.py", "production/context_agent_adapter.py")

def implementation():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in FILES}

def name_of(spec, cfg):
    return ("conversation_context" if any(f.scope == "thread" for f in spec.state) else "agent_context") + ("_json" if cfg else "")

def contract(graph, inputs):
    try:
        require_executable(graph)
        spec = agent_spec(graph)
    except Exception as exc:
        raise ProductionError("E_AGENT_GRAPH", f"Agent graph is not executable: {exc}.") from exc
    if graph.graphKind != "agent" or any(n.type not in ALLOWED for n in graph.nodes):
        raise ProductionError("E_AGENT_SCOPE", "Context serving accepts native text/JSON turns, pinned retrieval and short-term memory policies; no tools, interrupts or long-term writes.")
    retrieves = [RetrieveConfig.model_validate(n.config) for n in graph.nodes if n.type == "agent.retrieve"]
    selects = [MemorySelectConfig.model_validate(n.config) for n in graph.nodes if n.type == "agent.memory_select"]
    if not retrieves and not selects:
        raise ProductionError("E_AGENT_SCOPE", "Declare at least one retrieve or memory_select node; otherwise use an existing plain adapter.")
    if len(spec.indexes) > 2 or {r.index for r in retrieves} != {i.id for i in spec.indexes} or any(r.k > 8 for r in retrieves):
        raise ProductionError("E_AGENT_LIMITS", "Declare 0–2 used indexes and at most 8 retrieved chunks per node.")
    if len(spec.policies) > 2 or {s.policy for s in selects} != {p.id for p in spec.policies}:
        raise ProductionError("E_AGENT_MEMORY_SCOPE", "Declare at most two policies, each used by a memory_select node.")
    memory_fields = set()
    for select in selects:
        field = spec.field(select.short_term_field)
        if not field or field.type != "messages" or field.reducer.kind != "keep_last_n" or not field.reducer.n or field.reducer.n > 8:
            raise ProductionError("E_AGENT_MEMORY_SCOPE", "Policy input must be a messages field with keep_last_n <= 8.")
        memory_fields.add(field.name)
    for node in graph.nodes:
        if node.type == "agent.memory_write":
            cfg = MemoryWriteConfig.model_validate(node.config)
            if cfg.target != "short_term" or cfg.mode != "direct" or cfg.field not in memory_fields:
                raise ProductionError("E_AGENT_MEMORY_SCOPE", "Memory writes only append directly to a declared bounded short-term policy input; no long-term writes or approvals.")
    for policy in spec.policies:
        if not 1 <= len(policy.stages) <= 6 or policy.stages[0].op != "retrieve" or any(stage.op == "retrieve" for stage in policy.stages[1:]):
            raise ProductionError("E_AGENT_MEMORY_SCOPE", "Policies have 1–6 native stages starting with one retrieve stage.")
        if policy.embeddings.provider != "local_hash" or policy.embeddings.dimension > 512:
            raise ProductionError("E_AGENT_MEMORY_SCOPE", "Memory policies use bounded local_hash <=512 dimensions; lexical hashing is not semantic embedding.")
        templates = [policy.query]
        for stage in policy.stages:
            cfg = STAGE_CONFIG[stage.op].model_validate(stage.config)
            if stage.op == "retrieve":
                if cfg.sources != ["short_term"] or cfg.k is None or cfg.k > 8:
                    raise ProductionError("E_AGENT_MEMORY_SCOPE", "Policy retrieve selects short_term only, with k <= 8; no research/long-term corpus is copied.")
                templates.extend(cfg.scopes)
            if stage.op == "summarize" and (cfg.method != "extractive" or cfg.max_chars > 2000):
                raise ProductionError("E_AGENT_MEMORY_SCOPE", "Memory summaries are extractive <=2000 characters; no hidden model calls.")
            if stage.op == "budget" and cfg.max_tokens > 4096:
                raise ProductionError("E_AGENT_LIMITS", "Policy token-estimate budget must be <=4096.")
        if any(len(t) > 2000 or t.count("{") > 16 for t in templates):
            raise ProductionError("E_AGENT_LIMITS", "Policy templates are bounded to 2000 characters / 16 references.")
    lim = spec.limits
    if not (1 <= lim.maxSteps <= 25 and lim.maxSeconds is not None and lim.maxSeconds <= 30 and lim.maxModelCalls is not None and lim.maxModelCalls <= 2 and lim.maxTokens is not None and lim.maxTokens <= 4096):
        raise ProductionError("E_AGENT_LIMITS", "Declare maxSteps<=25, maxSeconds<=30, maxModelCalls<=2 and maxTokens<=4096.")
    if len(graph.nodes)>16 or len(spec.state)>32 or len(dumps(graph.to_json()).encode())>32768 or len(dumps([f.initial() for f in spec.state]).encode())>8192:
        raise ProductionError("E_AGENT_LIMITS", "Graph exceeds 16 nodes / 32KiB or state defaults exceed 8KiB.")
    if not 1 <= len(inputs) <= 4 or any(not spec.field(k) or spec.field(k).type != "text" or spec.field(k).scope != "turn" for k in inputs):
        raise ProductionError("E_AGENT_INPUT", "Declare 1–4 turn-scoped text input fields.")
    chats = [n for n in graph.nodes if n.type == "agent.chat_model"]
    structured = [n for n in graph.nodes if n.type == "agent.structured_output"]
    if len(chats)+len(structured)>1:
        raise ProductionError("E_AGENT_SCOPE", "At most one chat OR structured-output model node is served.")
    cfg = StructuredOutputConfig.model_validate(structured[0].config) if structured else None
    model = cfg.model if cfg else ChatModelConfig.model_validate(chats[0].config).model if chats else None
    if cfg:
        output=cfg.output_field;field=spec.field(output)
        if not field or field.type!="object" or field.scope!="turn" or field.reducer.kind!="replace" or output in inputs:
            raise ProductionError("E_AGENT_OUTPUT", "JSON output is a separate turn-scoped replace object field.")
        if cfg.on_failure!="fail" or cfg.retry.maxRetries+1>lim.maxModelCalls or not 1<=len(cfg.schema_fields)<=12 or len(dumps(json_schema_of(cfg.schema_fields)).encode())>4096:
            raise ProductionError("E_AGENT_JSON_SCHEMA", "Use on_failure=fail, retries within model budget and 1–12 flat schema fields within 4KiB.")
    elif chats:
        output=ChatModelConfig.model_validate(chats[0].config).output_field
    else:
        outputs=sorted({a['field'] for n in graph.nodes if n.type=='agent.set_state' for a in n.config.get('assignments',[]) if spec.field(a['field']) and spec.field(a['field']).type=='text' and a['field'] not in inputs})
        if len(outputs)!=1:
            raise ProductionError("E_AGENT_OUTPUT", "Model-free context turns need exactly one written text output.")
        output=outputs[0]
    if not cfg and (not spec.field(output) or spec.field(output).type!="text" or output in inputs):
        raise ProductionError("E_AGENT_OUTPUT", "Served text is a separate text field.")
    if model and (model.provider!="ollama" or model.base_url not in (None,conv.OLLAMA_URL,"http://127.0.0.1:11434") or model.api_key or model.fixture or model.think or model.max_tokens is None or model.max_tokens>128 or model.timeout_s>30):
        raise ProductionError("E_AGENT_PROVIDER", "Use installed local Ollama, think=false, max_tokens<=128 and timeout_s<=30.")
    for node in graph.nodes:
        op=get_op(node.type);parsed=op.Config.model_validate(node.config)
        if any(len(t)>2000 or t.count('{')>16 for _,t in op.templates(parsed)):
            raise ProductionError("E_AGENT_LIMITS", "Templates are bounded to 2000 characters / 16 references.")
        if cfg and node.id!=structured[0].id and output in op.writes(parsed):
            raise ProductionError("E_AGENT_OUTPUT", "Only structured_output may write the served JSON object.")
    return spec,output,model,cfg

def is_candidate(store,row):
    if row['status']!='completed' or row['config'].get('kind')!='agent': return None
    try:
        graph,_=conv._graph(store,row['id']);spec,output,_,cfg=contract(graph,row['config'].get('input',{}))
        finish=store.last_event(row['id'],'run_finished');finals=store.artifacts(row['id'],'final_state')
        if not finish or finish['data'].get('stoppedBy')!='end' or not finals: return None
        if cfg:
            jsa.source_evidence(store,row['id'],graph)
            jsa.output_value(json.loads(read_verified(store,finals[-1]['sha256'])).get(output),cfg)
        return NODES[name_of(spec,cfg)]
    except (ProductionError,ValueError): return None

def build_manifest(store,run_id,node):
    row=store.get_run(run_id)
    if not row or is_candidate(store,row)!=node:
        raise ProductionError("E_AGENT_SOURCE","Register a completed END context run with its declared stateless/conversation/text/JSON candidate.")
    graph,graph_sha=conv._graph(store,run_id);inputs=row['config'].get('input',{})
    spec,output,model,cfg=contract(graph,inputs)
    events,contexts=jsa.source_evidence(store,run_id,graph) if cfg else ([],[])
    m={"adapter":"native-langgraph-context-local","family":"agent_json" if cfg else "agent_turn","conversation":name_of(spec,cfg).startswith('conversation'),
       "runId":run_id,"node":node,"graphHash":semantic_hash(graph),"graphSha256":graph_sha,"modelSha256":None,
       "sourceFinalStateSha256":store.artifacts(run_id,'final_state')[-1]['sha256'],"referenceSha256":store.put_bytes(dumps([inputs]).encode()),
       "inputFields":sorted(inputs),"outputField":output,"limits":spec.limits.model_dump(),
       "inputContract":"One record with exactly the pinned turn text fields, 1–2000 characters each, <=8KiB JSON.",
       "outputSchema":{"task":"agent_json" if cfg else "agent_turn","classes":None,"target":output,**({"jsonSchema":json_schema_of(cfg.schema_fields)} if cfg else {})},
       "indexes":retrieval._snapshot(store,run_id,spec),"provider":conv.provider_identity(model.model) if model else None,
       "environment":conv.environment(),"implementation":implementation(),
       "source":{"runId":run_id,"configSha256":store.put_bytes(dumps(row['config']).encode()),"structuredEventsSha256":store.put_bytes(dumps(events).encode()),"contextSha256":contexts,
                 "semantics":"fresh defaults or prior native END conversation; pinned retrieval snapshots; policies select bounded short-term state only; no research memory/history copied"},
       "fitArtifacts":{},"evaluationArtifacts":[],"referencePartition":"source input; not ground truth or a benchmark"}
    ContextPipeline(store,m).validate_records([inputs])
    return m

def verify(store,m):
    if m['environment']!=conv.environment() or m['implementation']!=implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT","Pinned context adapter/dependency environment changed; register a new version.",409)
    for sha in (m['graphSha256'],m['sourceFinalStateSha256'],m['referenceSha256'],m['source']['configSha256'],m['source']['structuredEventsSha256'],*m['source']['contextSha256'],*(sha for snapshot in m['indexes'].values() for sha in snapshot['files'].values())):
        read_verified(store,sha)
    graph,sha=conv._graph(store,m['runId']);row=store.get_run(m['runId'])
    if sha!=m['graphSha256'] or semantic_hash(graph)!=m['graphHash'] or not row or row['status']!='completed' or dumps(row['config']).encode()!=read_verified(store,m['source']['configSha256']) or not any(a['sha256']==m['sourceFinalStateSha256'] and a['status']=='complete' for a in store.artifacts(m['runId'],'final_state')):
        raise ProductionError("E_AGENT_SOURCE","Pinned context source membership/configuration differs.",409)
    spec,_,model,_=contract(graph,m['inputFields'])
    for ispec in spec.indexes:
        if retrieval.embedding_identity(ispec)!=m['indexes'][ispec.id]['embeddingProvider']:
            raise ProductionError("E_AGENT_MODEL_CHANGED","Pinned embedding model differs; register a new version.",409)
    if model and conv.provider_identity(model.model)!=m['provider']:
        raise ProductionError("E_AGENT_MODEL_CHANGED","Pinned local chat model differs; register a new version.",409)

class ContextPipeline:
    reference_records = conv.AgentPipeline.reference_records
    validate_records = conv.AgentPipeline.validate_records
    checkpoint_state = conv.AgentPipeline.checkpoint_state

    def __init__(self, store, manifest):
        self.store, self.manifest = store, manifest
        verify(store, manifest)
        self.graph = Graph.model_validate(json.loads(read_verified(store, manifest["graphSha256"])))
        self.spec, self.output, _, self.cfg = contract(self.graph, manifest["inputFields"])

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
        with tempfile.TemporaryDirectory(prefix="void-context-turn-") as directory:
            local = ArtifactStore(Path(directory))
            local.create_run(execution, self.manifest["graphHash"], {"kind": "agent", "input": records[0]})
            rt = retrieval.PinnedIndexRuntime(self.graph, self.spec, local, local.root, execution, thread, self.manifest["graphHash"])
            rt.pinned = self.manifest["indexes"]
            for iid, snapshot in self.manifest["indexes"].items():
                directory = rt.indexes.dir(iid)
                directory.mkdir(parents=True, exist_ok=True)
                for name, sha in snapshot["files"].items():
                    (directory / name).write_bytes(read_verified(self.store, sha))
            rt.deadline, rt.cancelled = deadline, cancelled
            config = {"configurable": {"thread_id": thread}, "recursion_limit": self.spec.limits.maxSteps}
            try:
                compiled = compile_graph(self.graph, rt, saver)
                rt.emit("run_started", threadId=thread, mode="isolated context turn candidate", threadExisted=checkpoint is not None,
                        sourceRunId=self.manifest["runId"], inputFields=self.manifest["inputFields"], limits=self.manifest["limits"])
                final = None
                for state in compiled.stream(build_input(self.spec, records[0], checkpoint is not None), config, stream_mode="values"):
                    rt.check_budget()
                    if len(dumps(state).encode()) > 65_536:
                        raise ProductionError("E_AGENT_STATE_BOUNDS", "Native state exceeded 64 KiB; no result committed.", 413)
                    final = state
                rt.check_budget()
                # The object is validated before any checkpoint can be committed for this turn.
                result = jsa.output_value((final or {}).get(self.output), self.cfg) if self.cfg else (final or {}).get(self.output)
                if self.cfg:
                    jsa.source_evidence(local, execution, self.graph)
                elif not isinstance(result, str):
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
            retrievals = [{"node": e["node_id"], "index": e["data"]["index"], "included": e["data"]["included"], "embedding": e["data"]["embedding"]}
                          for e in events if e["type"] == "retrieval"]
            selections = []
            for event in events:
                if event["type"] == "memory_selection":
                    application = rt.memory.get_application(event["data"]["applicationId"])
                    raw = dumps(application).encode()
                    if len(raw) > 131_072:
                        raise ProductionError("E_AGENT_MEMORY_BOUNDS", "Memory application evidence exceeds 128 KiB.", 413)
                    sha = hashlib.sha256(raw).hexdigest()
                    if capture:
                        self.store.put_bytes(raw)
                    selections.append({"node": event["node_id"], "applicationSha256": sha, "summary": event["data"], "value": application if capture else None})
            evidence = {"executionId": execution, "threadId": thread, "sourceRunId": self.manifest["runId"],
                        "graphHash": self.manifest["graphHash"], "provider": self.manifest["provider"], "modelCalls": rt.model_calls, "retrievals": retrievals, "memorySelections": selections,
                        "stateSha256": hashlib.sha256(dumps(final).encode()).hexdigest(), "finalState": final if capture else None,
                        "eventsSha256": hashlib.sha256(dumps(events).encode()).hexdigest(), "events": events if capture else None,
                        "contexts": contexts, "capturePolicy": "Full state/events/model context/policy applications only when release captureInputs=true; otherwise hashes/usage and retrieval/policy decision summaries only.",
                        "isolation": "fresh declared defaults or the last successful per-release/user/session native END checkpoint; pinned indexes and bounded private short-term policies; no research-thread mutation"}
        verify(self.store, self.manifest)
        if time.perf_counter() >= deadline:
            raise ProductionError("E_REQUEST_TIMEOUT", "Context turn exceeded its deadline before result commit.", 504)
        return {"predictions": [result], "family": self.manifest["family"], "agent": evidence}, {
            "_checkpoint": candidate if self.manifest["conversation"] else None, "inferenceMs": (time.perf_counter() - start) * 1000, "preprocessingMs": None, "postprocessingMs": None,
            "timingNote": "native LangGraph context turn; provider latency recorded per call"}

AgentPipeline = ContextPipeline
