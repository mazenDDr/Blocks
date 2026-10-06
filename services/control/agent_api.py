"""Milestone 4 routes: agent catalog, runs (resume / rerun), trace, model-call context records, threads and checkpoints, memory stores and
policy preview, document indexes, effect ledger. Inspection routes only read recorded data; the only routes that start model calls are
POST /api/runs (and /api/runs/{id}/resume, /rerun), which are explicit run actions."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from agent import samples
from agent import tools as toolmod
from agent.index import IndexStore
from agent.memory import KINDS, MemoryStore
from agent.models import CAPABILITIES, DEFAULT_ANTHROPIC_MODEL, ModelSpec, ollama_status
from agent.policy import STAGE_CONFIG, stage_defaults, stage_schemas
from agent.preview import PreviewError, preview_policy_edit
from agent.runtime import open_checkpointer
from agent.spec import CMP_OPS, FIELD_TYPES, REDUCER_DOC, REDUCERS_FOR, SYMBOLS, agent_spec, template_vars
from agent.trace import build_trace
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import validate
from worker.agent_run import AgentRunConfig
from worker.process import submit_resume


def err(status: int, message: str, diagnostics: list | None = None, code: str = "request_invalid") -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "diagnostics": diagnostics or []})


def agent_node_views(graph: Graph, report) -> dict[str, Any]:
    from graph_core import registry

    diags: dict[str, list] = {}
    for d in report.diagnostics:
        if d.nodeId:
            diags.setdefault(d.nodeId, []).append(d.to_json())
    out = {}
    for n in graph.nodes:
        op = registry.get_op(n.type)
        e: dict[str, Any] = {"known": op is not None, "diagnostics": diags.get(n.id, []), "type": n.type, "typed": n.id in report.resolved, "inputPorts": ["in"], "outputPorts": ["out"]}
        if n.id in report.resolved and op is not None:
            cfg = report.resolved[n.id]
            e["resolvedConfig"] = cfg.model_dump(mode="json")
            e.update(report.node_info.get(n.id, {}))
            e["explain"] = op.explain(cfg)
            e["fixtureModel"] = any(m.provider == "fixture" for m in op.model_specs(cfg))
        out[n.id] = e
    return out


def agent_validation(graph: Graph, report) -> dict[str, Any]:
    return {"nodes": agent_node_views(graph, report), "agent": {**report.analysis, "state": [f.model_dump(mode="json") for f in report.spec.state] if report.spec else []}}


class SeedRequest(BaseModel):
    example: str = "memory_debugging"


class RecordBody(BaseModel):
    model_config = {"extra": "forbid"}
    id: str | None = None
    namespace: str = "default"
    scope: str = "global"
    kind: str = "semantic"
    text: str = Field(min_length=1, max_length=5000)
    importance: float = Field(0.5, ge=0, le=1)
    metadata: dict[str, Any] = {}
    expires_at: float | None = None
    generated: bool = False
    created_at: float | None = None


class ResumeBody(BaseModel):
    value: dict[str, Any] = Field(default_factory=lambda: {"action": "approve"})


class RerunBody(BaseModel):
    threadId: str | None = None


class PreviewBody(BaseModel):
    graph: Graph
    runId: str
    callId: str
    policy: dict[str, Any] | None = None  # an edited policy that is not saved in the graph yet


class IndexBody(BaseModel):
    graph: Graph
    indexId: str
    query: str = ""
    k: int = Field(3, ge=1, le=50)
    threshold: float | None = None


class ForkBody(BaseModel):
    checkpointId: str | None = None
    newThreadId: str | None = None
    edits: dict[str, Any] = {}


def register(app: FastAPI, sv) -> None:
    wb: Path = sv.workbench
    memory = MemoryStore(wb)
    indexes = IndexStore(wb)
    store = sv.store

    def agent_run(rid: str) -> dict[str, Any]:
        row = store.get_run(rid)
        if row is None:
            raise HTTPException(404, {"code": "not_found", "message": f"unknown run '{rid}'"})
        if row["config"].get("kind") != "agent":
            raise err(422, f"run '{rid}' is not an agent run", code="not_agent_run")
        return row

    def graph_of(rid: str) -> Graph:
        a = store.artifacts(rid, "graph")
        return Graph.model_validate(json.loads(store.read_artifact(a[0]["sha256"])))

    # ------------------------------------------------------------------ catalog
    @app.get("/api/agent/catalog")
    def catalog():
        oll = ollama_status()
        return {
            "providers": {k: {**v, "ollama": oll if k == "ollama" else None} for k, v in CAPABILITIES.items()},
            "defaults": {"anthropicModel": DEFAULT_ANTHROPIC_MODEL, "ollamaModel": "qwen3.5:2b"},
            "reducers": [{"kind": k, "doc": d, "appliesTo": [t for t, rs in REDUCERS_FOR.items() if k in rs]} for k, d in REDUCER_DOC.items()],
            "fieldTypes": list(FIELD_TYPES),
            "predicate": {"operators": [{"op": o, "label": SYMBOLS[o], "unary": o in ("is_empty", "not_empty", "is_true", "is_false")} for o in CMP_OPS], "functions": ["len"]},
            "tools": [t.describe() for t in toolmod.TOOLS.values()],
            "effects": {"external": list(toolmod.EXTERNAL_EFFECTS), "policy": "External effects require an explicit approval interrupt and run once (effect ledger)."},
            "stages": {"defaults": stage_defaults(), "schemas": stage_schemas(), "order": ["retrieve", "filter", "rank", "dedupe", "budget", "summarize"]},
            "memoryKinds": list(KINDS),
            "embeddings": [{"provider": "local_hash", "label": "local_hash: deterministic hashed bag-of-words (lexical overlap, NOT a semantic model)"},
                           {"provider": "ollama", "label": "Ollama embeddings (e.g. nomic-embed-text)", "available": "nomic-embed-text:latest" in oll.get("models", [])}],
            "fixtureNote": "Provider 'fixture' is a scripted test model. Its replies are labelled FIXTURE everywhere and are never real model output.",
        }

    # ------------------------------------------------------------------ resume / rerun / threads
    @app.post("/api/runs/{rid}/resume")
    def resume(rid: str, body: ResumeBody):
        row = agent_run(rid)
        if row["status"] != "paused":
            raise HTTPException(409, {"code": "not_paused", "message": f"run is {row['status']}; only a paused run can be resumed"})
        try:
            h = submit_resume(rid, body.value, wb)
        except Exception as e:  # noqa: BLE001 - a second concurrent resume loses the paused -> running transition
            from artifact_store import IllegalTransition
            if isinstance(e, IllegalTransition):
                raise HTTPException(409, {"code": "not_paused", "message": "the run was already resumed"})
            raise
        sv.handles[rid] = h
        return {"runId": rid, "status": "running", "resumeValue": body.value}

    @app.post("/api/runs/{rid}/rerun", status_code=201)
    def rerun(rid: str, body: RerunBody):
        """A NEW run with the same graph version and input on a new thread: new model calls (not a replay of recorded output)."""
        from worker.process import submit_run

        row = agent_run(rid)
        graph = graph_of(rid)
        cfg = AgentRunConfig.model_validate({**row["config"], "thread_id": body.threadId or f"th-{uuid.uuid4().hex[:8]}", "rerun_of": rid})
        with sv.lock:
            h = submit_run(graph, cfg, wb)
            sv.handles[h.run_id] = h
        return {"runId": h.run_id, "threadId": cfg.thread_id, "rerunOf": rid, "note": "new model calls; a hosted or stochastic model is not guaranteed to repeat its earlier response"}

    def thread_runs(tid: str) -> list[dict[str, Any]]:
        return [r for r in store.list_runs() if r["config"].get("kind") == "agent" and r["config"].get("thread_id") == tid]

    @app.get("/api/agent/threads")
    def threads(project: str | None = None):
        saver = open_checkpointer(wb)
        by: dict[str, dict[str, Any]] = {}
        for r in store.list_runs():
            c = r["config"]
            if c.get("kind") != "agent" or (project and c.get("project_id") != project):
                continue
            t = by.setdefault(c["thread_id"], {"threadId": c["thread_id"], "projectId": c.get("project_id"), "runs": [], "status": None})
            t["runs"].append({"runId": r["id"], "status": r["status"], "createdAt": r["created_at"]})
            t["status"] = r["status"]
        for t in by.values():
            tup = saver.get_tuple({"configurable": {"thread_id": t["threadId"]}})
            t["hasCheckpoint"] = tup is not None
            t["checkpointId"] = tup.config["configurable"]["checkpoint_id"] if tup else None
            t["step"] = (tup.metadata or {}).get("step") if tup else None
            t["pendingInterrupt"] = bool(tup and tup.pending_writes and any(w[1] == "__interrupt__" for w in tup.pending_writes))
        return {"threads": sorted(by.values(), key=lambda t: t["runs"][-1]["createdAt"])}

    @app.get("/api/agent/threads/{tid}/state")
    def thread_state(tid: str, checkpointId: str | None = None):
        cfg = {"configurable": {"thread_id": tid, **({"checkpoint_id": checkpointId} if checkpointId else {})}}
        tup = open_checkpointer(wb).get_tuple(cfg)
        if tup is None:
            raise HTTPException(404, {"code": "not_found", "message": f"thread '{tid}' has no checkpoint"})
        return {"threadId": tid, "checkpointId": tup.config["configurable"]["checkpoint_id"], "parentCheckpointId": (tup.parent_config or {}).get("configurable", {}).get("checkpoint_id"),
                "step": (tup.metadata or {}).get("step"), "source": (tup.metadata or {}).get("source"), "values": tup.checkpoint["channel_values"],
                "ts": tup.checkpoint.get("ts"), "pendingInterrupt": any(w[1] == "__interrupt__" for w in (tup.pending_writes or [])),
                "provenance": {"threadId": tid, "source": "LangGraph SQLite checkpoint (the thread's persisted state; short-term memory lives here)", "runs": [r["id"] for r in thread_runs(tid)]}}

    @app.get("/api/agent/threads/{tid}/history")
    def thread_history(tid: str, limit: int = 100):
        out = []
        for tup in open_checkpointer(wb).list({"configurable": {"thread_id": tid}}, limit=limit):
            md = tup.metadata or {}
            out.append({"checkpointId": tup.config["configurable"]["checkpoint_id"], "parentCheckpointId": (tup.parent_config or {}).get("configurable", {}).get("checkpoint_id"),
                        "step": md.get("step"), "source": md.get("source"), "writes": list((md.get("writes") or {}).keys()) if isinstance(md.get("writes"), dict) else None,
                        "ts": tup.checkpoint.get("ts"), "fields": sorted(tup.checkpoint["channel_values"])})
        return {"threadId": tid, "checkpoints": out}

    @app.post("/api/agent/threads/{tid}/fork", status_code=201)
    def fork(tid: str, body: ForkBody):
        """Copy a checkpoint to a NEW thread, applying edits to chosen state fields: an edited branch with its own execution identity (the original thread is untouched)."""
        import copy

        from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

        saver = open_checkpointer(wb)
        cfg = {"configurable": {"thread_id": tid, **({"checkpoint_id": body.checkpointId} if body.checkpointId else {})}}
        tup = saver.get_tuple(cfg)
        if tup is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no checkpoint for thread '{tid}'"})
        new_tid = body.newThreadId or f"th-{uuid.uuid4().hex[:8]}"
        if saver.get_tuple({"configurable": {"thread_id": new_tid}}) is not None:
            raise HTTPException(409, {"code": "thread_exists", "message": f"thread '{new_tid}' already exists"})
        ck = copy.deepcopy(tup.checkpoint)
        versions = dict(ck["channel_versions"])
        for k, v in body.edits.items():
            ck["channel_values"][k] = v
            versions[k] = saver.get_next_version(versions.get(k), None) if k in versions else saver.get_next_version(None, None)
        ck["channel_versions"] = versions
        ck["versions_seen"] = {}
        ck["id"] = str(uuid.uuid4()) if False else ck["id"]
        md = {**(tup.metadata or {}), "forkedFrom": {"threadId": tid, "checkpointId": tup.config["configurable"]["checkpoint_id"]}, "edits": sorted(body.edits)}
        saver.put({"configurable": {"thread_id": new_tid, "checkpoint_ns": ""}}, ck, md, versions)
        return {"threadId": new_tid, "forkedFrom": {"threadId": tid, "checkpointId": tup.config["configurable"]["checkpoint_id"]}, "edited": sorted(body.edits),
                "note": "an edited copy of recorded state on a new thread; the original thread and its runs are unchanged"}

    # ------------------------------------------------------------------ trace and context
    @app.get("/api/agent/runs/{rid}/trace")
    def trace(rid: str):
        agent_run(rid)
        return build_trace(store, rid)

    @app.get("/api/agent/runs/{rid}/model-calls")
    def model_calls(rid: str):
        agent_run(rid)
        out = []
        for e in store.events(rid, -1, ("model_call", "model_call_failed")):
            d = e["data"]
            out.append({"seq": e["seq"], "node": e["node_id"], "failed": e["type"] == "model_call_failed", **{k: d.get(k) for k in (
                "callId", "purpose", "attempt", "provider", "model", "fixture", "latencyMs", "usage", "tokensEstimate", "validationErrors", "code", "message")}})
        return {"runId": rid, "calls": out}

    @app.get("/api/agent/runs/{rid}/model-calls/{call_id}")
    def model_call(rid: str, call_id: str):
        agent_run(rid)
        for a in store.artifacts(rid, "model_context"):
            if a["meta"].get("callId") == call_id:
                ctx = json.loads(store.read_artifact(a["sha256"]))
                ctx["provenance"] = {"runId": rid, "graphHash": ctx.get("graphHash"), "node": ctx.get("node"), "artifactSha256": a["sha256"], "source": "the request recorded by the worker at call time"}
                return ctx
        raise HTTPException(404, {"code": "not_found", "message": f"run {rid} recorded no model call '{call_id}'"})

    @app.get("/api/agent/runs/{rid}/final-state")
    def final_state(rid: str):
        agent_run(rid)
        a = store.artifacts(rid, "final_state")
        if not a:
            return {"available": False, "reason": "the run has not completed"}
        return {"available": True, "values": json.loads(store.read_artifact(a[-1]["sha256"])), "provenance": {"runId": rid, "artifactSha256": a[-1]["sha256"]}}

    # ------------------------------------------------------------------ evaluations (ADR 0077)
    @app.get("/api/agent/evals/{rid}")
    def evaluation(rid: str):
        row = store.get_run(rid)
        if row is None or row["config"].get("kind") != "agent_eval":
            raise HTTPException(404, {"code": "not_found", "message": f"no agent evaluation '{rid}'"})
        a = store.artifacts(rid, "agent_eval_report")
        cases = [e["data"] for e in store.events(rid, -1, ("eval_case",))]
        return {"runId": rid, "status": row["status"], "error": row["error"], "name": row["config"].get("name"), "caseCount": len(row["config"].get("cases", [])),
                "cases": cases, "report": json.loads(store.read_artifact(a[-1]["sha256"])) if a else None,
                "provenance": {"runId": rid, "graphHash": row["graph_hash"], "reportSha256": a[-1]["sha256"] if a else None}}

    # ------------------------------------------------------------------ memory
    @app.get("/api/agent/memory/records")
    def list_records(namespace: str | None = None, scope: str | None = None, kind: str | None = None, deleted: bool = False):
        recs = memory.list_records(namespace, scope, kind, include_deleted=deleted)
        return {"store": "long_term", "path": "agent/memory.db", "records": recs, "namespaces": sorted({r["namespace"] for r in recs}), "scopes": sorted({r["scope"] for r in recs}),
                "note": "Stored records. Stored does not mean used: see each model call's context for what was actually sent."}

    @app.post("/api/agent/memory/records", status_code=201)
    def add_record(body: RecordBody):
        if body.kind not in KINDS:
            raise err(422, f"kind must be one of {list(KINDS)}")
        rec = memory.put_record(body.model_dump(exclude_none=True) | {"source": {"kind": "manual", "via": "api"}}, evidence="manual edit in the memory browser")
        return rec

    @app.put("/api/agent/memory/records/{rid}")
    def put_record(rid: str, body: RecordBody):
        if memory.get_record(rid) is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no record '{rid}'"})
        if body.kind not in KINDS:
            raise err(422, f"kind must be one of {list(KINDS)}")
        return memory.put_record({**body.model_dump(exclude_none=True), "id": rid, "source": {"kind": "manual", "via": "api"}}, evidence="manual edit in the memory browser")

    @app.delete("/api/agent/memory/records/{rid}")
    def delete_record(rid: str):
        """Deleting a stored record is NOT the same as removing it from an assembled context: report what still holds a retained snapshot."""
        rec = memory.get_record(rid)
        if rec is None or rec["deleted_at"]:
            raise HTTPException(404, {"code": "not_found", "message": f"no record '{rid}'"})
        retained = retained_snapshots(rid)
        memory.delete_record(rid, evidence="manual delete in the memory browser")
        return {"deleted": rid, "retainedSnapshots": retained, "futureCalls": "No future memory selection can retrieve this record.",
                "note": "Historical runs keep the snapshot of this record that they recorded; those recorded model contexts are not rewritten."}

    def retained_snapshots(rid: str) -> list[dict[str, Any]]:
        out = []
        for app_ in memory.applications():
            r = app_["records"].get(rid)
            if r:
                out.append({"applicationId": app_["id"], "runId": app_.get("runId"), "threadId": app_.get("threadId"), "status": r["status"], "usedInModelContext": r["status"] == "included"})
        return out

    @app.get("/api/agent/memory/records/{rid}/history")
    def record_history(rid: str):
        return {"recordId": rid, "writes": memory.writes(record_id=rid), "retainedSnapshots": retained_snapshots(rid)}

    @app.get("/api/agent/memory/writes")
    def writes(runId: str | None = None):
        return {"writes": memory.writes(run_id=runId)}

    @app.get("/api/agent/memory/applications")
    def applications(runId: str | None = None, threadId: str | None = None):
        apps = memory.applications(runId, threadId)
        return {"applications": [{"id": a["id"], "policyId": a["policyId"], "runId": a.get("runId"), "threadId": a.get("threadId"), "node": a.get("node"), "asOf": a["asOf"],
                                  "universe": len(a["records"]), "selected": len(a["final"]), "tokensEstimate": a["tokensEstimate"]} for a in apps]}

    @app.get("/api/agent/memory/applications/{aid}")
    def application(aid: str):
        a = memory.get_application(aid)
        if a is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no recorded policy application '{aid}'"})
        return a

    @app.get("/api/agent/memory/trace")
    def trace_record(recordId: str, runId: str | None = None, callId: str | None = None):
        """Trace a stored record through storage, scope, retrieval, ranking, summary and budget for the selection(s) behind a call (VISION 12.7 step 3-4)."""
        apps = memory.applications(run_id=runId)
        if callId and runId:
            ctxs = [a for a in store.artifacts(runId, "model_context") if a["meta"].get("callId") == callId]
            if ctxs:
                ids = set(json.loads(store.read_artifact(ctxs[0]["sha256"]))["applications"])
                apps = [a for a in apps if a["id"] in ids]
        stored = memory.get_record(recordId)
        out = []
        for a in apps:
            r = a["records"].get(recordId)
            if r is None:
                out.append({"applicationId": a["id"], "policyId": a["policyId"], "node": a.get("node"), "present": False, "verdict": "this record was not in the record universe of this selection (it did not exist yet or is outside every store the policy reads)"})
                continue
            first = next((t for t in r["trail"] if t["action"] in ("excluded", "replaced")), None)
            out.append({"applicationId": a["id"], "policyId": a["policyId"], "node": a.get("node"), "present": True, "status": r["status"], "usedInContext": r["status"] == "included",
                        "trail": r["trail"], "firstDisappearedAt": first, "scores": r["scores"], "tokensEstimate": r["tokens"],
                        "verdict": ("included in the selection: the trace rules out omission by the memory policy" if r["status"] == "included" else
                                    f"excluded at stage '{first['stage']}' ({first['op']}): {first['reason']}" if first else r["status"])})
        return {"recordId": recordId, "stored": stored is not None and not stored["deleted_at"], "storedRecord": stored, "applications": out,
                "caution": "Presence in context does not guarantee the model obeyed it; absence is the finding only for the calls listed."}

    @app.post("/api/agent/memory/seed")
    def seed(req: SeedRequest):
        """Write the SYNTHETIC memory records of an example into the long-term store (dated relative to now)."""
        if req.example != "memory_debugging":
            raise err(422, f"unknown example '{req.example}'")
        recs = samples.memory_seed()
        for r in recs:
            memory.put_record(r, evidence="example seed: SYNTHETIC records invented for this project")
        return {"seeded": len(recs), "ids": [r["id"] for r in recs], "synthetic": True, "note": samples.MEMORY_SEED_NOTE}

    @app.post("/api/agent/policy/preview")
    def policy_preview(body: PreviewBody):
        try:
            return preview_policy_edit(store, memory, body.graph, body.runId, body.callId, body.policy)
        except PreviewError as e:
            raise HTTPException(e.status, {"code": e.code, "message": e.message})

    # ------------------------------------------------------------------ indexes
    def index_spec(g: Graph, iid: str):
        s = agent_spec(g).index(iid)
        if s is None:
            raise err(422, f"index '{iid}' is not declared in this graph")
        return s

    @app.post("/api/agent/indexes/build")
    def build_index(body: IndexBody):
        spec = index_spec(body.graph, body.indexId)
        try:
            m = indexes.ensure(spec)
        except (FileNotFoundError, ValueError) as e:
            raise err(422, str(e), code="index_build_failed")
        except Exception as e:  # noqa: BLE001
            raise err(422, f"{type(e).__name__}: {e}", code="index_build_failed")
        return m

    @app.post("/api/agent/indexes/info")
    def index_info(body: IndexBody):
        spec = index_spec(body.graph, body.indexId)
        m = indexes.manifest(spec.id)
        return {"built": m is not None, "manifest": m, "declared": spec.model_dump(mode="json")}

    @app.post("/api/agent/indexes/chunks")
    def index_chunks(body: IndexBody):
        spec = index_spec(body.graph, body.indexId)
        return {"index": spec.id, "chunks": [{k: c[k] for k in ("chunk_id", "doc_id", "index", "text", "start")} for c in indexes.chunks(spec.id)]}

    @app.post("/api/agent/indexes/search")
    def index_search(body: IndexBody):
        spec = index_spec(body.graph, body.indexId)
        if not body.query.strip():
            raise err(422, "a query is required")
        try:
            return indexes.search(spec, body.query, body.k, body.threshold)
        except (FileNotFoundError, ValueError) as e:
            raise err(422, str(e), code="index_search_failed")

    @app.get("/api/agent/effects")
    def effects(threadId: str | None = None):
        return {"effects": memory.effects(threadId), "note": "Protected effects (external tool effects and long-term memory writes) run once per key; a replay finds the key and skips."}

    @app.get("/api/agent/retrievals/{rid}")
    def retrieval(rid: str):
        r = memory.get_retrieval(rid)
        if r is None:
            raise HTTPException(404, {"code": "not_found", "message": f"no retrieval '{rid}'"})
        return r
