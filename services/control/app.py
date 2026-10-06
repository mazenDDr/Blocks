"""FastAPI control service.

    uvicorn control.app:create_app --factory --app-dir services

Reading and inspecting never train: the only endpoint that starts a worker is POST /api/runs.
Runs live in their own OS process and write events straight to SQLite, so they continue when a client goes away."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from artifact_store import ArtifactStore, IllegalTransition
from graph_core import registry
from graph_core.codegen import generate_pytorch
from graph_core.hashing import semantic_hash
from graph_core.lower import lower_graph
from graph_core.project_io import Project, load_project, save_project, ui_path_for
from graph_core.schema import Graph, ProjectDocument
from graph_core.validate import ExecutionBlocked, validate
from worker.process import RunHandle, submit_run
from worker.agent_run import AgentRunConfig
from worker.rl_run import RLRunConfig
from worker.procedure_run import ProcedureRunConfig
from worker.tabular_run import TabularRunConfig
from worker.train import RunConfig, UnsupportedGraph, _io_contract
from connectors import context as conn_context
from connectors.errors import SourceError
from connectors.registry import ConnectionRegistry
from studies.runner import StudyRunner
from studies.store import StudyStore
from storage.schema import inspect as inspect_schemas

from . import inspection as insp
from . import tabular_inspection as tinsp
from .registry_meta import META
from .m3_support import module_summaries, procedure_check
from .views import node_view, views

REPO = Path(__file__).resolve().parents[2]
PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
TERMINAL = ("completed", "failed", "cancelled")


# ---------------------------------------------------------------------------------------- request models
class ValidateRequest(BaseModel):
    graph: Graph


class CompatRequest(BaseModel):
    graph: Graph
    backend: str


class ExportRequest(BaseModel):
    graph: Graph
    backend: str = "pytorch"


class InspectRequest(BaseModel):
    kind: Literal["weights", "activations", "sample_loss", "confusion",
                  "table", "profile", "fit_state", "coefficients", "metrics", "test_result", "distribution", "tail", "number", "summary"]
    port: str | None = None
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
    config: dict[str, Any] = Field(default_factory=dict)  # RunConfig for model graphs, TabularRunConfig for tabular graphs


# ---------------------------------------------------------------------------------------- helpers
def _err(status: int, message: str, diagnostics: list | None = None, code: str = "request_invalid") -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, "diagnostics": diagnostics or []})


def model_preflight(graph: Graph, cfg: RunConfig) -> None:
    """Dataset and graph/dataset contract checks for a model run; raises HTTPException(422)."""
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


def validation_json(graph: Graph) -> dict[str, Any]:
    report = validate(graph)
    j = report.to_json()
    j["graphHash"] = semantic_hash(graph)
    j["graphKind"] = graph.graphKind
    if graph.graphKind == "agent":
        from .agent_api import agent_validation

        j.update(agent_validation(graph, report))
    elif graph.graphKind == "rl":
        from .rl_api import rl_validation

        j["nodes"] = node_view(graph, report)
        j["rl"] = rl_validation(graph, report)
    elif graph.graphKind == "model":
        v = views(graph, report)
        j["nodes"], j["flat"], j["instances"] = v["nodes"], v["flat"], v["instances"]
        j["modules"] = module_summaries(graph, report)
        if graph.training is not None:
            j["procedure"] = procedure_check(graph.training)
    else:
        j["nodes"] = node_view(graph, report)
    j["order"] = report.order
    return j


def preflight_procedure(graph: Graph, cfg: ProcedureRunConfig) -> None:
    """Everything that can be checked without training: procedure order rules, graph validity, data/graph contract, loss wiring. Raises HTTPException(422)."""
    from training.spec import ProcedureSpec, check_procedure
    from training.trainer import Trainer

    try:
        spec = ProcedureSpec.model_validate(cfg.procedure)
    except ValidationError as e:
        raise _err(422, "invalid training procedure: " + "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()))
    errs = [d for d in check_procedure(spec) if d.severity == "error"]
    if errs:
        raise _err(422, "the training procedure is not valid", [d.to_json() for d in errs], "procedure_invalid")
    try:
        Trainer(graph, spec)
    except ExecutionBlocked as e:
        raise _err(422, "the graph or its training setup has errors and cannot run", [d.to_json() for d in e.diagnostics], "execution_blocked")


def agent_preflight(graph: Graph, cfg: AgentRunConfig) -> None:
    """Checks that need no model call: the graph validates, the local runtime is reachable when a block uses it, the thread has no pending interrupt."""
    report = validate(graph)
    if not report.ok:
        raise _err(422, "the graph has errors and cannot run", [d.to_json() for d in report.errors], "execution_blocked")
    if any(n.config.get("model", {}).get("provider") == "ollama" for n in graph.nodes):
        from agent.models import ollama_status

        if not ollama_status()["reachable"]:
            raise _err(422, "a block uses the local Ollama runtime, which is not reachable at http://localhost:11434", code="model_unavailable")


class Services:
    def __init__(self, workbench: Path):
        inspect_schemas(workbench)  # refuse a future/foreign owned schema before startup mutations
        self.workbench = workbench
        self.store = ArtifactStore(workbench)
        self.projects = workbench / "projects"
        self.projects.mkdir(parents=True, exist_ok=True)
        self.handles: dict[str, RunHandle] = {}
        self.lock = threading.Lock()  # serializes run submission so one idempotency key creates one run
        self.examples = REPO / "examples"
        conn_context.set_workbench(workbench)
        self.connections = ConnectionRegistry(workbench)
        self.studies = StudyStore(workbench)
        self.study_runner = StudyRunner(self.studies, self.store, self.launch, self.cancel_run)

    def cancel_run(self, rid: str) -> None:
        try:
            self.store.set_status(rid, "cancelling")
        except IllegalTransition:
            return
        h = self.handles.get(rid)
        if h:
            h._cancel.set()

    def launch(self, graph: Graph, cfg_dict: dict[str, Any]) -> str:
        """Start one run for a study trial (same checks as POST /api/runs). Raises HTTPException with the reason when it cannot start."""
        with self.lock:
            if graph.graphKind in ("tabular", "rl", "domain"):
                cfg = (RLRunConfig if graph.graphKind == "rl" else TabularRunConfig).model_validate({**cfg_dict, "kind": graph.graphKind})
                report = validate(graph)
                if not report.ok:
                    raise _err(422, "the graph has errors and cannot run", [d.to_json() for d in report.errors], "execution_blocked")
            else:
                base = RunConfig.model_validate(cfg_dict)
                cfg = base.model_copy(update={"data": str(Path(base.data).expanduser().resolve())})
                model_preflight(graph, cfg)
            handle = submit_run(graph, cfg, self.workbench)
            self.handles[handle.run_id] = handle
            return handle.run_id

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

    def tabular_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        rid = row["id"]
        evs = self.store.events(rid, -1, ("run_started", "node_started", "node_finished", "node_failed", "source_recorded", "source_snapshot_recorded", "split_recorded", "validation_error"))
        order, status, failure, sources, splits, libs, snaps, seeded, cache = [], {}, None, [], [], None, [], None, None
        for e in evs:
            d, t = e["data"], e["type"]
            if t == "run_started":
                order, libs, seeded = d["order"], d.get("libraries"), {"seed": d.get("seed"), "applied": d.get("seedApplied"), "pins": d.get("sourcePins")}
                cache = d.get("cache")
            elif t == "node_started":
                status[e["node_id"]] = {"node": e["node_id"], "type": d["type"], "status": "running"}
            elif t == "node_finished":
                status[e["node_id"]] = {"node": e["node_id"], "type": d["type"], "status": "finished", "rows": d.get("rows", {}), **({"cache": d["cache"]} if "cache" in d else {})}
            elif t == "node_failed":
                status[e["node_id"]] = {**status.get(e["node_id"], {"node": e["node_id"]}), "status": "failed"}
                failure = {"node": e["node_id"], "code": d["code"], "message": d["message"]}
            elif t == "source_recorded":
                sources.append({"node": e["node_id"], **d})
            elif t == "source_snapshot_recorded":
                snaps.append({"node": e["node_id"], **d})
            elif t == "split_recorded":
                splits.append({"node": e["node_id"], **d})
            elif t == "validation_error":
                failure = failure or {"node": e["node_id"], "code": d["code"], "message": d["message"]}
        nodes = [status.get(n, {"node": n, "status": "pending"}) for n in order]
        return {"kind": row["config"].get("kind", "tabular"), "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": row["config"],
                "createdAt": row["created_at"], "updatedAt": row["updated_at"], "maxSeq": self.store.max_seq(rid), "nodes": nodes,
                "progress": {"nodesDone": sum(1 for n in nodes if n["status"] == "finished"), "nodes": len(order)},
                "sources": sources, "snapshots": snaps, "runSeed": seeded, "splits": splits, "failure": failure, "libraries": libs,
                "cache": cache}

    def rl_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        rid, cfg = row["id"], row["config"]
        started = self.store.last_event(rid, "run_started")
        up = self.store.last_event(rid, "train_update")
        ep = self.store.last_event(rid, "episode_end")
        ev = self.store.last_event(rid, "eval")
        return {"kind": "rl", "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": cfg, "createdAt": row["created_at"],
                "updatedAt": row["updated_at"], "maxSeq": self.store.max_seq(rid), "seed": cfg.get("seed"), "algorithm": started["data"]["algorithm"] if started else None,
                "totalSteps": started["data"]["totalSteps"] if started else None, "envSteps": (ep["data"]["tick"] if ep else 0), "updates": (up["data"]["update"] if up else 0),
                "lastEval": ({k: ev["data"][k] for k in ("tick", "final", "taskReturn", "return", "successRate")} if ev else None), "trial": cfg.get("trial")}

    def agent_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        rid, cfg = row["id"], row["config"]
        fin = self.store.last_event(rid, "run_finished")
        intr = self.store.last_event(rid, "interrupt_raised")
        calls = self.store.events(rid, -1, ("model_call",))
        return {"kind": "agent", "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": cfg, "createdAt": row["created_at"],
                "updatedAt": row["updated_at"], "maxSeq": self.store.max_seq(rid), "threadId": cfg.get("thread_id"), "input": cfg.get("input"), "rerunOf": cfg.get("rerun_of"),
                "stoppedBy": fin["data"].get("stoppedBy") if fin else None, "pendingInterrupt": ({"node": intr["node_id"], **intr["data"]} if intr and row["status"] == "paused" else None),
                "modelCalls": len(calls), "fixtureCalls": sum(1 for c in calls if c["data"].get("fixture")),
                "totalLatencyMs": round(sum(c["data"].get("latencyMs") or 0 for c in calls), 1)}

    def procedure_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        rid = row["id"]
        started = self.store.last_event(rid, "run_started")
        last = self.store.last_event(rid, "train_step")
        val = self.store.last_event(rid, "validation")
        fin = self.store.last_event(rid, "run_finished")
        spec = row["config"].get("procedure", {})
        epochs = len(self.store.events(rid, -1, ("epoch_end",)))
        sd = started["data"] if started else {}
        return {"kind": "procedure", "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": row["config"],
                "createdAt": row["created_at"], "updatedAt": row["updated_at"], "maxSeq": self.store.max_seq(rid),
                "totalParams": sd.get("total_params"), "synthetic": sd.get("synthetic"), "dataNote": sd.get("data"), "resumedFrom": sd.get("resumed_from"), "rerunOf": row["config"].get("rerun_of"),
                "progress": {"step": last["data"]["step"] if last else 0, "epochsDone": epochs, "epochs": spec.get("epochs"), "stepsPerEpoch": sd.get("steps_per_epoch")},
                "last": ({k: last["data"].get(k) for k in ("step", "loss", "lr", "grad_norm")} if last else None), "validation": val["data"] if val else None,
                "stoppedBy": fin["data"].get("stopped_by") if fin else None, "checkpoints": len([a for a in self.store.artifacts(rid, "checkpoint") if a["status"] != "pruned"]),
                "captures": [a["step"] for a in self.store.artifacts(rid, "capture")]}

    def run_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        rid = row["id"]
        if row["config"].get("kind") in ("tabular", "domain"):
            return self.tabular_summary(row)
        if row["config"].get("kind") == "rl":
            return self.rl_summary(row)
        if row["config"].get("kind") == "procedure":
            return self.procedure_summary(row)
        if row["config"].get("kind") == "agent":
            return self.agent_summary(row)
        if row["config"].get("kind") == "sandbox":
            return {"kind": "sandbox", "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": row["config"],
                    "createdAt": row["created_at"], "updatedAt": row["updated_at"], "maxSeq": self.store.max_seq(rid), "parent": row["config"].get("parent"), "step": row["config"].get("step")}
        started = self.store.last_event(rid, "run_started")
        last_step = self.store.last_event(rid, "train_step")
        epochs = [e["data"] for e in self.store.events(rid, -1, ("epoch_end",))]
        cfg = row["config"]
        spe = math.ceil(started["data"]["n_train"] / cfg["batch_size"]) if started else None
        sd = started["data"] if started else None
        return {
            "kind": "model", "id": rid, "status": row["status"], "error": row["error"], "graphHash": row["graph_hash"], "config": cfg,
            "createdAt": row["created_at"], "updatedAt": row["updated_at"],
            "classes": sd["classes"] if sd else None,
            "totalParams": sd["total_params"] if sd else None,
            "split": ({"seed": sd.get("split_seed"), "valFraction": sd.get("val_fraction"), "nTrain": sd["n_train"], "nVal": sd["n_val"],
                       "datasetSha256": sd.get("dataset_sha256")} if sd else None),
            "progress": {"step": last_step["data"]["step"] if last_step else 0, "epochsDone": len(epochs), "epochs": cfg["epochs"],
                         "stepsPerEpoch": spe},
            "final": epochs[-1] if epochs else None,
            "maxSeq": self.store.max_seq(rid),
        }


# ---------------------------------------------------------------------------------------- app
RECONCILE_SECONDS = 30.0


def create_app(workbench: str | Path | None = None, api_token: str | None = None, users_file: str | Path | None = None) -> FastAPI:
    wb = Path(workbench or os.environ.get("VOID_WORKBENCH", ".workbench")).resolve()
    sv = Services(wb)
    from maintenance.cache_retention import CacheRetention
    sv.cache_retention = CacheRetention(sv.store, idle_lock=sv.lock)

    stop_reconcile = threading.Event()

    def reconcile_workers():
        # A crash of this service (or of a worker) can leave runs active forever; fail those with no sign of life.
        while True:
            try:
                sv.recovered_runs = sv.store.reconcile_lost_workers()
            except Exception:  # noqa: BLE001 - retried on the next tick
                pass
            if stop_reconcile.wait(RECONCILE_SECONDS):
                return

    @asynccontextmanager
    async def lifespan(app):
        sv.cache_retention.start()
        reconciler = threading.Thread(target=reconcile_workers, daemon=True, name="worker-reconcile")
        reconciler.start()
        try:
            yield
        finally:
            stop_reconcile.set()
            reconciler.join(timeout=5)
            sv.cache_retention.stop()

    app = FastAPI(title="Project Void control service", version="1.0.0", lifespan=lifespan)
    token = api_token if api_token is not None else (os.environ.get("VOID_API_TOKEN") or None)
    users = users_file if users_file is not None else (os.environ.get("VOID_USERS_FILE") or None)
    sv.api_token = None if users else token
    if users:
        from .accounts import AccountAuth, load as load_accounts
        if token:
            raise ValueError("Set either VOID_USERS_FILE (named accounts) or VOID_API_TOKEN (one shared token), not both.")
        app.add_middleware(AccountAuth, accounts=load_accounts(users), audit_dir=wb / "audit")  # parsed now: start fails on a bad file
    elif token:
        from .auth import TokenAuth, check_token
        app.add_middleware(TokenAuth, token=check_token(token))  # checked now: middleware is only built on the first request
    app.state.services = sv

    from .accounts import AccountError, current as current_account

    @app.exception_handler(AccountError)
    async def _account_error(_: Request, e: AccountError):
        return JSONResponse({"detail": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.get("/api/whoami")
    def whoami():
        acc = current_account()
        return {"account": {"name": acc.name, "role": acc.role} if acc else None,
                "mode": "accounts" if acc else ("shared-token" if sv.api_token else "open")}

    @app.exception_handler(SourceError)
    async def _source_error(_: Request, e: SourceError):
        return JSONResponse({"detail": e.to_json()}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, e: RequestValidationError):
        # Python's JSON parser accepts NaN/Infinity, and FastAPI echoes the invalid input in the 422 detail; a strict JSON response
        # cannot encode those values, so report them as strings instead of failing while reporting the validation error.
        def safe(v):
            if isinstance(v, float) and not math.isfinite(v):
                return str(v)
            if isinstance(v, dict):
                return {k: safe(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [safe(x) for x in v]
            return v
        return JSONResponse({"detail": safe(jsonable_encoder(e.errors()))}, status_code=422)

    @app.exception_handler(insp.InspectError)
    async def _inspect_error(_: Request, e: insp.InspectError):
        return JSONResponse({"detail": {"code": "inspect_invalid", "message": e.message}}, status_code=e.status)

    # ------------------------------------------------------------------ registry / validate
    @app.get("/api/registry")
    def get_registry():
        ops = []
        for op in registry.all_ops():
            name, cat, purpose = META.get(op.type, (op.type, "Other", ""))
            from extensions.sdk import LOADED
            package = LOADED.get(op.type)
            if package:
                name = package["inspector"].get("displayName", name)
                cat = package["inspector"].get("category", "Community")
                purpose = package["documentation"]
            ops.append({"type": op.type, "version": op.version, "backend": op.backend, "graphKind": op.graph_kind, "summaryKind": getattr(op, "summary_kind", None), "displayName": name, "category": cat, "purpose": purpose,
                        "inputKinds": getattr(op, "in_kinds", None) or {p: "tensor" for p in op.inputs},
                        "outputKinds": getattr(op, "out_kinds", None) or {p: "tensor" for p in op.outputs},
                        "inputs": list(op.inputs), "outputs": list(op.outputs), "configSchema": op.Config.model_json_schema(),
                        "defaults": op.Config().model_dump(mode="json")})
        return {"ops": ops}

    @app.post("/api/validate")
    def post_validate(req: ValidateRequest):
        return validation_json(req.graph)

    # ------------------------------------------------------------------ projects
    def describe(path: Path, pid: str) -> dict[str, Any]:
        try:
            g = json.loads(path.read_text())
            ui_p = ui_path_for(path)
            ui = json.loads(ui_p.read_text()) if ui_p.exists() else {}
            return {"id": pid, "graphKind": g.get("graphKind", "model"), "description": ui.get("description"), "synthetic": bool(ui.get("synthetic"))}
        except (OSError, ValueError):
            return {"id": pid, "graphKind": "unknown", "description": None, "synthetic": False}

    @app.get("/api/projects")
    def list_projects():
        names = sorted(p.name[: -len(".project.json")] for p in sv.projects.glob("*.project.json"))
        return {"projects": names, "details": [describe(sv.projects / f"{n}.project.json", n) for n in names]}

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
        names = sorted(p.name[: -len(".project.json")] for p in sv.examples.glob("*.project.json"))
        return {"examples": names, "details": [describe(sv.examples / f"{n}.project.json", n) for n in names]}

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
        if graph.graphKind != "model":
            raise _err(422, f"'{graph.graphKind}' graphs have no PyTorch export", code="export_unsupported")
        try:
            code = generate_pytorch(graph)
        except ExecutionBlocked as e:
            raise _err(422, "the graph cannot be exported until its errors are fixed", [d.to_json() for d in e.diagnostics], "execution_blocked")
        return {"projectId": pid, "graphHash": semantic_hash(graph), "language": "python", "code": code}

    # ------------------------------------------------------------------ backends (Milestone 6a)
    def _export_for(graph: Graph, backend: str, pid: str | None = None) -> dict[str, Any]:
        import backends as B

        if graph.graphKind != "model":
            raise _err(422, f"'{graph.graphKind}' graphs have no native {backend} export", code="export_unsupported")
        if backend not in B.BACKEND_IDS:
            raise _err(404, f"unknown backend '{backend}' (known: {list(B.BACKEND_IDS)})", code="unknown_backend")
        try:
            code = B.export_code(graph, backend)
        except B.BackendError as e:
            diags = [{"code": v["code"], "nodeId": n, "message": v["reason"]} for n, v in (e.report.nodes.items() if e.report else []) if v["status"] == "unsupported"]
            diags += [{"code": d["code"], "nodeId": d.get("nodeId"), "message": d["message"]} for d in (e.report.structural if e.report else [])]
            raise _err(422, e.message, diags, e.code.lower())
        except ExecutionBlocked as e:
            raise _err(422, "the graph cannot be exported until its errors are fixed", [d.to_json() for d in e.diagnostics], "execution_blocked")
        return {"projectId": pid, "graphHash": semantic_hash(graph), "backend": backend, "language": "python", "code": code}

    @app.get("/api/backends")
    def get_backends():
        import backends as B

        return {"backends": [B.describe(b) for b in B.BACKEND_IDS]}

    @app.post("/api/backends/compat")
    def post_compat(req: CompatRequest):
        """Backend compatibility report for a model graph, computed before anything executes."""
        import backends as B

        if req.graph.graphKind != "model":
            raise _err(422, f"backend compatibility applies to model graphs, not '{req.graph.graphKind}'", code="compat_unsupported")
        try:
            rep = B.compat_report(req.graph, req.backend)
        except B.BackendError as e:
            raise _err(404, e.message, code="unknown_backend")
        return rep.to_json()

    @app.post("/api/export")
    def post_export(req: ExportRequest):
        return _export_for(req.graph, req.backend)

    def _register_backend_export(backend: str) -> None:
        # one literal route per backend: a `{backend}` path parameter would shadow sibling routes such as .../export/bundle
        @app.get(f"/api/projects/{{pid}}/export/{backend}", name=f"export_{backend}")
        def export_backend(pid: str):
            path = sv.project_path(pid)
            if not path.exists():
                raise HTTPException(404, {"code": "not_found", "message": f"no project '{pid}'"})
            return _export_for(load_project(path).graph, backend, pid)

    for _b in ("keras", "jax"):
        _register_backend_export(_b)

    @app.get("/api/coverage")
    def get_coverage():
        """The public coverage ledger (generated from the registry and adapters; docs/COVERAGE.md is the same data rendered)."""
        from backends import coverage as C

        L = C.build_ledger()
        L["markdown"] = C.render_markdown(L)
        return L

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
        if graph.graphKind == "model" and graph.backend != "pytorch":
            raise _err(422, f"This graph targets backend '{graph.backend}'. Training runs execute on PyTorch only: the {graph.backend} backend runs forward, loss, "
                       "gradients and one SGD step through the backends API, not worker training runs. Set the graph's backend to 'pytorch' to train it.",
                       code="backend_training_unsupported")
        tabular = graph.graphKind in ("tabular", "rl", "domain")   # typed-wire graph kinds: validate, then submit (their config models differ below)
        rl = graph.graphKind == "rl"
        agent = graph.graphKind == "agent"
        procedure = graph.graphKind == "model" and req.config.get("kind") == "procedure"
        try:
            if agent:
                cfg = AgentRunConfig.model_validate({**req.config, "kind": "agent", "project_id": req.projectId or req.config.get("project_id"),
                                                     "thread_id": req.config.get("thread_id") or f"th-{hashlib.sha256(idempotency_key.encode()).hexdigest()[:8]}"})
            elif procedure:
                body = {**req.config}
                body["procedure"] = body.get("procedure") or graph.training or {}
                cfg = ProcedureRunConfig.model_validate({**body, "project_id": req.projectId or body.get("project_id")})
            elif rl:
                cfg = RLRunConfig.model_validate({**req.config, "project_id": req.projectId or req.config.get("project_id")})
            elif tabular:
                cfg = TabularRunConfig.model_validate({**req.config, "kind": graph.graphKind, "project_id": req.projectId or req.config.get("project_id")})
            else:
                base = RunConfig.model_validate(req.config)
                cfg = base.model_copy(update={"data": str(Path(base.data).expanduser().resolve()), "project_id": req.projectId or base.project_id})
        except ValidationError as e:
            raise _err(422, "invalid run config: " + "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()))
        req_hash = hashlib.sha256(json.dumps({"graph": semantic_hash(graph), "config": cfg.model_dump()}, sort_keys=True).encode()).hexdigest()
        with sv.lock:
            prior = sv.store.get_idempotent(idempotency_key)
            if prior:
                if prior["request_hash"] != req_hash:
                    raise _err(409, "this Idempotency-Key was already used with a different request", code="idempotency_key_reused")
                return JSONResponse({"runId": prior["run_id"], "idempotentReplay": True, "status": sv.store.get_run(prior["run_id"])["status"]})
            if agent:
                agent_preflight(graph, cfg)
                busy = [r["id"] for r in sv.store.list_runs() if r["config"].get("kind") == "agent" and r["config"].get("thread_id") == cfg.thread_id
                        and r["status"] in ("queued", "preparing", "running", "paused", "cancelling")]
                if busy:
                    raise _err(409, f"thread '{cfg.thread_id}' is busy with run {busy[0]} (a paused run must be resumed or cancelled first)", code="thread_busy")
                handle = submit_run(graph, cfg, sv.workbench)
                sv.handles[handle.run_id] = handle
                sv.store.put_idempotent(idempotency_key, req_hash, handle.run_id)
                return JSONResponse({"runId": handle.run_id, "idempotentReplay": False, "status": "queued", "graphHash": semantic_hash(graph), "threadId": cfg.thread_id}, status_code=201)
            if procedure:
                preflight_procedure(graph, cfg)
                handle = submit_run(graph, cfg, sv.workbench)
                sv.handles[handle.run_id] = handle
                sv.store.put_idempotent(idempotency_key, req_hash, handle.run_id)
                return JSONResponse({"runId": handle.run_id, "idempotentReplay": False, "status": "queued", "graphHash": semantic_hash(graph)}, status_code=201)
            if tabular:
                report = validate(graph)
                if not report.ok:
                    raise _err(422, "the graph has errors and cannot run", [d.to_json() for d in report.errors], "execution_blocked")
                handle = submit_run(graph, cfg, sv.workbench)
                sv.handles[handle.run_id] = handle
                sv.store.put_idempotent(idempotency_key, req_hash, handle.run_id)
                return JSONResponse({"runId": handle.run_id, "idempotentReplay": False, "status": "queued", "graphHash": semantic_hash(graph)}, status_code=201)
            model_preflight(graph, cfg)
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

    @app.get("/api/runs/{rid}/tables/{node}/{port}.csv")
    def table_csv(rid: str, node: str, port: str):
        """Download a recorded table (e.g. exported predictions) exactly as stored."""
        path = tinsp.table_artifact_path(sv.store, rid, node, port)
        return FileResponse(path, media_type="text/csv", filename=f"{rid}_{node}_{port}.csv")

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
                if not batch and (status in TERMINAL or status == "paused"):
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
        row = sv.store.get_run(rid)
        if row is not None and row["config"].get("kind") in ("tabular", "domain"):
            return tinsp.inspect_tabular(sv.store, rid, req)
        if req.kind not in ("weights", "activations", "sample_loss", "confusion"):
            raise _err(422, f"'{req.kind}' inspection applies to tabular runs only")
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

    from . import agent_api, cache_api, connections_api, domain_api, domain_datasets_api, m3_api, production_api, repos_api, research_api, rl_api, scale_api, studies_api, unsup_api

    agent_api.register(app, sv)
    connections_api.register(app, sv)
    studies_api.register(app, sv)
    m3_api.register(app, sv)
    rl_api.register(app, sv)
    unsup_api.register(app, sv)
    production_api.register(app, sv)
    from . import memory_api
    memory_api.register(app, sv)
    scale_api.register(app, sv)
    domain_api.register(app, sv)
    domain_datasets_api.register(app, sv)
    repos_api.register(app, sv)
    cache_api.register(app, sv)
    from . import cache_retention_api
    cache_retention_api.register(app, sv)
    research_api.register(app, sv)
    return app
