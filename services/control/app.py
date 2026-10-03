"""FastAPI control service.

    uvicorn control.app:create_app --factory --app-dir services

Reading and inspecting never train: the only endpoint that starts a worker is POST /api/runs.
Runs live in their own OS process and write events straight to SQLite, so they continue when a client goes away."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from artifact_store import ArtifactStore, IllegalTransition
from graph_core import registry
from graph_core.codegen import generate_pytorch
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import Project, load_project, save_project, ui_path_for
from graph_core.schema import Graph, ProjectDocument
from graph_core.validate import ExecutionBlocked, validate
from worker.process import RunHandle, submit_run
from worker.train import RunConfig, UnsupportedGraph, _io_contract

from . import inspection as insp
from .registry_meta import META

REPO = Path(__file__).resolve().parents[2]
PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TERMINAL = ("completed", "failed", "cancelled")


# ---------------------------------------------------------------------------------------- request models
class ValidateRequest(BaseModel):
    graph: Graph


class InspectRequest(BaseModel):
    kind: Literal["weights", "activations", "sample_loss", "confusion"]
    runId: str | None = None
    node: str | None = None
    checkpointStep: int | None = None
    sample: int | None = None
    offset: int = Field(0, ge=0)
    limit: int | None = Field(None, ge=1)
    epoch: int | None = None


class InferRequest(BaseModel):
    runId: str
    checkpointStep: int | None = None
    sample: int | None = None
    imageBase64: str | None = None


class SubmitRequest(BaseModel):
    projectId: str | None = None
    graph: Graph | None = None
    config: RunConfig


# ---------------------------------------------------------------------------------------- helpers
def _err(status: int, message: str, diagnostics: list | None = None, code: str = "request_invalid") -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "diagnostics": diagnostics or []})


def node_view(graph: Graph, report) -> dict[str, Any]:
    """Per-node shapes, param counts, diagnostics, resolved config and explain data."""
    by_node: dict[str, list] = {}
    for d in report.diagnostics:
        if d.nodeId:
            by_node.setdefault(d.nodeId, []).append(d.to_json())
    out = {}
    for n in graph.nodes:
        op = registry.get_op(n.type)
        entry: dict[str, Any] = {"known": op is not None, "diagnostics": by_node.get(n.id, []), "typed": n.id in report.output_types}
        if n.id in report.output_types:
            entry["inputShapes"] = {p: t.to_json() for p, t in report.input_types[n.id].items()}
            entry["outputShapes"] = {p: t.to_json() for p, t in report.output_types[n.id].items()}
            entry["params"] = report.params[n.id]
            entry["resolvedConfig"] = report.resolved[n.id].model_dump(mode="json")
            try:
                entry["explain"] = jsonable_encoder(op.explain(report.resolved[n.id], report.input_types[n.id], report.output_types[n.id]))
            except Exception as e:  # noqa: BLE001  (an explain bug must not break validation)
                entry["explain"] = {"error": str(e)}
        out[n.id] = entry
    return out


def validation_json(graph: Graph) -> dict[str, Any]:
    report = validate(graph)
    j = report.to_json()
    j["graphHash"] = semantic_hash(graph)
    j["nodes"] = node_view(graph, report)
    j["order"] = report.order
    return j


class Services:
    def __init__(self, workbench: Path):
        self.workbench = workbench
        self.store = ArtifactStore(workbench)
        self.projects = workbench / "projects"
        self.projects.mkdir(parents=True, exist_ok=True)
        self.handles: dict[str, RunHandle] = {}
        self.lock = threading.Lock()  # serializes run submission so one idempotency key creates one run
        self.examples = REPO / "examples"

    def project_path(self, pid: str) -> Path:
        if not PROJECT_ID.match(pid):
            raise _err(422, f"invalid project id '{pid}' (letters, digits, '_' and '-', max 64)")
        return self.projects / f"{pid}.project.json"

    def reap(self) -> None:
        for rid, h in list(self.handles.items()):
            if not h.is_alive():
                h.process.join(0)
                del self.handles[rid]

    def run_data(self, run_id: str) -> insp.RunData:
        try:
            return insp.RunData(self.store, run_id)
        except insp.InspectError as e:
            raise HTTPException(e.status, {"code": "not_found", "message": e.message})

    def run_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        rid = row["id"]
        started = self.store.last_event(rid, "run_started")
        last_step = self.store.last_event(rid, "train_step")
        epochs = [e["data"] for e in self.store.events(rid, -1, ("epoch_end",))]
        cfg = row["config"]
        spe = math.ceil(started["data"]["n_train"] / cfg["batch_size"]) if started else None
        sd = started["data"] if started else None
        return {
            "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": cfg,
            "createdAt": row["created_at"], "updatedAt": row["updated_at"],
            "classes": sd["classes"] if sd else None,
            "totalParams": sd["total_params"] if sd else None,
            "split": ({"seed": sd["split_seed"], "valFraction": sd["val_fraction"], "nTrain": sd["n_train"], "nVal": sd["n_val"],
                       "datasetSha256": sd["dataset_sha256"]} if sd else None),
            "progress": {"step": last_step["data"]["step"] if last_step else 0, "epochsDone": len(epochs), "epochs": cfg["epochs"],
                         "stepsPerEpoch": spe},
            "final": epochs[-1] if epochs else None,
            "maxSeq": self.store.max_seq(rid),
        }


# ---------------------------------------------------------------------------------------- app
def create_app(workbench: str | Path | None = None) -> FastAPI:
    wb = Path(workbench or os.environ.get("VOID_WORKBENCH", ".workbench")).resolve()
    sv = Services(wb)
    app = FastAPI(title="Project Void control service", version="1.0.0")
    app.state.services = sv

    @app.exception_handler(insp.InspectError)
    async def _inspect_error(_: Request, e: insp.InspectError):
        return JSONResponse({"detail": {"code": "inspect_invalid", "message": e.message}}, status_code=e.status)

    # ------------------------------------------------------------------ registry / validate
    @app.get("/api/registry")
    def get_registry():
        ops = []
        for op in registry.all_ops():
            name, cat, purpose = META.get(op.type, (op.type, "Other", ""))
            ops.append({"type": op.type, "version": op.version, "backend": op.backend, "displayName": name, "category": cat, "purpose": purpose,
                        "inputs": list(op.inputs), "outputs": list(op.outputs), "configSchema": op.Config.model_json_schema(),
                        "defaults": op.Config().model_dump(mode="json")})
        return {"ops": ops}

    @app.post("/api/validate")
    def post_validate(req: ValidateRequest):
        return validation_json(req.graph)

    # ------------------------------------------------------------------ projects
    @app.get("/api/projects")
    def list_projects():
        return {"projects": sorted(p.name[: -len(".project.json")] for p in sv.projects.glob("*.project.json"))}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        path = sv.project_path(pid)
        if not path.exists():
            raise HTTPException(404, {"code": "not_found", "message": f"no project '{pid}'"})
        p = load_project(path)
        return {"id": pid, "graph": p.graph.to_json(), "ui": p.ui.to_json() if p.ui else None, "graphHash": semantic_hash(p.graph)}

    @app.put("/api/projects/{pid}")
    def put_project(pid: str, doc: ProjectDocument):
        path = sv.project_path(pid)
        save_project(Project(doc.graph, doc.ui), path)
        if doc.ui is None and ui_path_for(path).exists():
            ui_path_for(path).unlink()
        return {"id": pid, "graphHash": semantic_hash(doc.graph)}

    @app.get("/api/examples")
    def list_examples():
        return {"examples": sorted(p.name[: -len(".project.json")] for p in sv.examples.glob("*.project.json"))}

    @app.get("/api/examples/{name}")
    def get_example(name: str):
        path = sv.examples / f"{name}.project.json"
        if not PROJECT_ID.match(name) or not path.exists():
            raise HTTPException(404, {"code": "not_found", "message": f"no example '{name}'"})
        p = load_project(path)
        return {"id": name, "graph": p.graph.to_json(), "ui": p.ui.to_json() if p.ui else None, "graphHash": semantic_hash(p.graph)}

    @app.get("/api/projects/{pid}/export/pytorch")
    def export_pytorch(pid: str):
        path = sv.project_path(pid)
        if not path.exists():
            raise HTTPException(404, {"code": "not_found", "message": f"no project '{pid}'"})
        graph = load_project(path).graph
        try:
            code = generate_pytorch(graph)
        except ExecutionBlocked as e:
            raise _err(422, "the graph cannot be exported until its errors are fixed", [d.to_json() for d in e.diagnostics], "execution_blocked")
        return {"projectId": pid, "graphHash": semantic_hash(graph), "language": "python", "code": code}

    # ------------------------------------------------------------------ runs
    @app.post("/api/runs")
    def submit(req: SubmitRequest, idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
        if not idempotency_key or len(idempotency_key) > 200:
            raise _err(400, "an Idempotency-Key header (1-200 chars) is required so a double submit cannot start two runs", code="idempotency_key_required")
        if (req.graph is None) == (req.projectId is None):
            raise _err(422, "give exactly one of 'projectId' or 'graph'")
        if req.projectId is not None:
            path = sv.project_path(req.projectId)
            if not path.exists():
                raise HTTPException(404, {"code": "not_found", "message": f"no project '{req.projectId}'"})
            graph = load_project(path).graph
        else:
            graph = req.graph
        cfg = req.config.model_copy(update={"data": str(Path(req.config.data).expanduser().resolve()),
                                            "project_id": req.projectId or req.config.project_id})
        req_hash = hashlib.sha256(json.dumps({"graph": semantic_hash(graph), "config": cfg.model_dump()}, sort_keys=True).encode()).hexdigest()
        with sv.lock:
            prior = sv.store.get_idempotent(idempotency_key)
            if prior:
                if prior["request_hash"] != req_hash:
                    raise _err(409, "this Idempotency-Key was already used with a different request", code="idempotency_key_reused")
                return JSONResponse({"runId": prior["run_id"], "idempotentReplay": True, "status": sv.store.get_run(prior["run_id"])["status"]})
            if not Path(cfg.data).is_dir():
                raise _err(422, f"dataset directory '{cfg.data}' does not exist", code="dataset_missing")
            n_dirs = sum(1 for d in Path(cfg.data).iterdir() if d.is_dir())
            try:
                report = validate(graph)
                if not report.ok:
                    raise ExecutionBlocked(report.errors)
                _, n_classes = _io_contract(graph, report, lower_graph(graph, report))
                if n_dirs != n_classes:
                    raise UnsupportedGraph(f"dataset has {n_dirs} class folders but the graph outputs {n_classes} logits")
            except ExecutionBlocked as e:
                raise _err(422, "the graph has errors and cannot run", [d.to_json() for d in e.diagnostics], "execution_blocked")
            except UnsupportedGraph as e:
                raise _err(422, str(e), code="unsupported_graph")
            handle = submit_run(graph, cfg, sv.workbench)
            sv.handles[handle.run_id] = handle
            sv.store.put_idempotent(idempotency_key, req_hash, handle.run_id)
        return JSONResponse({"runId": handle.run_id, "idempotentReplay": False, "status": "queued", "graphHash": semantic_hash(graph)}, status_code=201)

    @app.get("/api/runs")
    def list_runs(project: str | None = None):
        sv.reap()
        rows = sv.store.list_runs()
        if project:
            rows = [r for r in rows if r["config"].get("project_id") == project]
        return {"runs": [sv.run_summary(r) for r in rows]}

    @app.get("/api/runs/{rid}")
    def get_run(rid: str):
        sv.reap()
        row = sv.store.get_run(rid)
        if row is None:
            raise HTTPException(404, {"code": "not_found", "message": f"unknown run '{rid}'"})
        return sv.run_summary(row)

    @app.post("/api/runs/{rid}/cancel", status_code=202)
    def cancel(rid: str):
        row = sv.store.get_run(rid)
        if row is None:
            raise HTTPException(404, {"code": "not_found", "message": f"unknown run '{rid}'"})
        try:
            sv.store.set_status(rid, "cancelling")
        except IllegalTransition:
            raise HTTPException(409, {"code": "illegal_transition", "message": f"cannot cancel a run that is {row['status']}"})
        h = sv.handles.get(rid)
        if h:
            h._cancel.set()
        return {"runId": rid, "status": "cancelling"}

    @app.get("/api/runs/{rid}/metrics")
    def run_metrics(rid: str):
        rd = sv.run_data(rid)
        evs = sv.store.events(rid, -1, ("train_step", "epoch_end"))
        return {"summary": sv.run_summary(rd.row), "graph": rd.graph.to_json() if rd.graph else None,
                "trainLoss": [[e["data"]["step"], e["data"]["loss"]] for e in evs if e["type"] == "train_step"],
                "epochs": [e["data"] for e in evs if e["type"] == "epoch_end"],
                "provenance": {"runId": rid, "graphHash": rd.graph_hash, "source": "events recorded by the worker"}}

    @app.get("/api/runs/{rid}/checkpoints")
    def checkpoints(rid: str):
        rd = sv.run_data(rid)
        return {"runId": rid, "graphHash": rd.graph_hash, "checkpoints": insp.checkpoint_list(rd)}

    @app.get("/api/runs/{rid}/samples")
    def samples(rid: str):
        rd = sv.run_data(rid)
        return {"runId": rid, "available": rd.split is not None, "classes": rd.split["classes"] if rd.split else None,
                "samples": insp.sample_list(rd),
                "split": ({k: rd.split[k] for k in ("split_seed", "val_fraction", "dataset_sha256")} if rd.split else None)}

    @app.get("/api/runs/{rid}/samples/{index}/image")
    def sample_image(rid: str, index: int):
        rd = sv.run_data(rid)
        return FileResponse(insp.val_sample_image_path(rd, index))

    @app.get("/api/runs/{rid}/events")
    async def events(rid: str, request: Request, after: int | None = Query(None), last_event_id: str | None = Header(None, alias="Last-Event-ID")):
        if sv.store.get_run(rid) is None:
            raise HTTPException(404, {"code": "not_found", "message": f"unknown run '{rid}'"})
        cursor = after if after is not None else -1
        if after is None and last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError:
                raise _err(400, "Last-Event-ID must be an event sequence number")

        async def gen():
            last, idle = cursor, 0
            while True:
                status = (await asyncio.to_thread(sv.store.get_run, rid))["status"]  # read first: events written before it are all seen
                batch = await asyncio.to_thread(sv.store.events, rid, last)
                for e in batch:
                    last = e["seq"]
                    payload = {k: e[k] for k in ("run_id", "seq", "ts", "type", "graph_hash", "node_id", "data")}
                    yield f"id: {e['seq']}\nevent: {e['type']}\ndata: {json.dumps(payload)}\n\n"
                if not batch and status in TERMINAL:
                    yield "event: end\ndata: {}\n\n"
                    return
                if await request.is_disconnected():
                    return  # the run is unaffected
                idle = 0 if batch else idle + 1
                if idle and idle % 50 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.05 if batch else 0.2)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ inspect / infer (read-only)
    def do_inspect(rid: str | None, req: InspectRequest) -> dict[str, Any]:
        if rid is None:
            return {"available": False, "kind": req.kind, "reason": "no_run", "message": "No run exists yet; nothing is recorded.",
                    "provenance": {"runId": None, "nodeId": req.node}}
        rd = sv.run_data(rid)
        if req.kind in ("weights", "activations") and not req.node:
            raise _err(422, "'node' is required for weights and activations")
        if req.kind == "weights":
            return insp.inspect_weights(rd, req.node, req.checkpointStep, req.offset, req.limit)
        if req.kind == "activations":
            return insp.inspect_activations(rd, req.node, req.checkpointStep, req.sample, req.offset, req.limit)
        if req.kind == "sample_loss":
            return insp.inspect_sample_loss(rd, req.epoch, req.limit or 12)
        return insp.inspect_confusion(rd, req.epoch)

    @app.post("/api/runs/{rid}/inspect")
    def inspect_run(rid: str, req: InspectRequest):
        return do_inspect(rid, req)

    @app.post("/api/inspect")
    def inspect_any(req: InspectRequest):
        """Like the run-scoped route; with no runId it uses the newest run, or reports that no run exists."""
        rid = req.runId
        if rid is None:
            runs = sv.store.list_runs()
            rid = runs[-1]["id"] if runs else None
        return do_inspect(rid, req)

    @app.post("/api/infer")
    def infer(req: InferRequest):
        return insp.infer(sv.run_data(req.runId), req.checkpointStep, req.sample, req.imageBase64)

    return app
