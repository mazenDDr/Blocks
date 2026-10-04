"""Studies, sweeps, trials, retries, baseline pinning and run diffs."""
from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from graph_core.schema import Graph
from graph_core.validate import validate
from studies import metrics as smetrics
from studies import planner
from studies.diff import diff_runs
from studies.models import MetricSpec, StudyCreate
from studies.report import study_view

from . import app as app_mod


class BaselineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runId: str | None = None


def _err(status: int, code: str, message: str, detail: Any = None) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, **({"detail": detail} if detail is not None else {})})


def register(app: FastAPI, sv: "app_mod.Services") -> None:
    ss, rs, runner = sv.studies, sv.store, sv.study_runner

    def resolve(req: StudyCreate) -> tuple[Graph, dict[str, Any], Any]:
        if (req.graph is None) == (req.projectId is None):
            raise _err(422, "request_invalid", "give exactly one of 'projectId' or 'graph'")
        if req.projectId is not None:
            path = sv.project_path(req.projectId)
            if not path.exists():
                raise _err(404, "not_found", f"no project '{req.projectId}'")
            graph = load_project(path).graph
        else:
            try:
                graph = Graph.model_validate(req.graph)
            except ValidationError as e:
                raise _err(422, "request_invalid", f"invalid graph: {e.errors()[0]['msg']}")
        report = validate(graph)
        if not report.ok:
            raise _err(422, "execution_blocked", "the base graph has errors; fix them before planning a study", [d.to_json() for d in report.errors])
        cfg_model = planner.run_config_model(graph)
        try:
            cfg_model.model_validate(req.run_config)
        except ValidationError as e:
            raise _err(422, "request_invalid", "invalid run config: " + "; ".join(f"{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors()))
        if graph.graphKind == "model":
            try:
                base = cfg_model.model_validate(req.run_config)
                app_mod.model_preflight(graph, base)
            except HTTPException:
                raise
        bad = smetrics.check_spec("tabular" if graph.graphKind == "tabular" else "model", req.objective.metric, report, graph)
        if bad:
            raise _err(422, "objective_invalid", bad)
        return graph, report, None

    def make_plan(req: StudyCreate):
        graph, _, _ = resolve(req)
        try:
            p = planner.plan(graph, req)
        except planner.PlanError as e:
            raise _err(422, e.code, e.message, e.detail)
        return graph, p

    @app.post("/api/studies/plan")
    def plan_only(req: StudyCreate):
        """Validate and expand a study without storing or running anything."""
        graph, p = make_plan(req)
        return {"graphKind": graph.graphKind, "graphHash": p["graphHash"], "counts": {k: p[k] for k in ("total", "groups", "seeds", "folds", "invalid")},
                "limits": req.limits.model_dump(), "trials": p["trials"]}

    @app.post("/api/studies", status_code=201)
    def create_study(req: StudyCreate):
        graph, p = make_plan(req)
        sid = uuid.uuid4().hex[:10]
        labels = {t["groupKey"]: t["label"] for t in p["trials"]}
        spec = {**req.model_dump(mode="json", exclude={"graph", "start"}), "graph": graph.to_json(), "graphKind": graph.graphKind, "graphHash": p["graphHash"],
                "planSummary": {k: p[k] for k in ("total", "groups", "seeds", "folds", "invalid")}, "labels": labels}
        ss.create(sid, spec, p["trials"])
        if req.start and any(t["status"] == "planned" for t in p["trials"]):
            runner.start(sid)
        return study_view(ss, rs, sid, runner.is_running(sid))

    @app.get("/api/studies")
    def list_studies():
        out = []
        for st in ss.list()[-100:]:
            sp = st["spec"]
            ts = ss.trials(st["id"])
            counts: dict[str, int] = {}
            for t in ts:
                counts[t["status"]] = counts.get(t["status"], 0) + 1
            out.append({"id": st["id"], "name": sp["name"], "state": "running" if runner.is_running(st["id"]) else st["state"], "graphKind": sp["graphKind"],
                        "objective": sp["objective"], "createdAt": st["created_at"], "trials": len(ts), "counts": counts, "projectId": sp.get("projectId")})
        return {"studies": out, "bounds": {"max": 100}}

    def need(sid: str):
        if ss.get(sid) is None:
            raise _err(404, "not_found", f"unknown study '{sid}'")

    @app.get("/api/studies/{sid}")
    def get_study(sid: str):
        need(sid)
        return study_view(ss, rs, sid, runner.is_running(sid))

    @app.post("/api/studies/{sid}/start", status_code=202)
    def start_study(sid: str):
        need(sid)
        started = runner.start(sid)
        return {"studyId": sid, "started": started, "note": None if started else "already running"}

    @app.post("/api/studies/{sid}/cancel", status_code=202)
    def cancel_study(sid: str):
        need(sid)
        if not runner.is_running(sid):
            raise _err(409, "illegal_transition", "the study is not running")
        runner.request_cancel(sid)
        return {"studyId": sid, "cancelRequested": True}

    @app.post("/api/studies/{sid}/trials/{tid}/retry", status_code=202)
    def retry_trial(sid: str, tid: str):
        need(sid)
        t = ss.trial(sid, tid)
        if t is None:
            raise _err(404, "not_found", f"unknown trial '{tid}'")
        if t["status"] not in ("failed", "cancelled"):
            raise _err(409, "illegal_transition", f"only failed or cancelled trials can be retried (this one is {t['status']})")
        max_attempts = ss.get(sid)["spec"]["limits"]["max_attempts"]
        n = len(ss.attempts(sid, tid))
        if n >= max_attempts:
            raise _err(409, "attempt_limit", f"trial {tid} already has {n} attempts (max_attempts = {max_attempts})")
        ss.set_trial_status(sid, tid, "planned")
        runner.start(sid)
        return {"studyId": sid, "trialId": tid, "nextAttempt": n + 1}

    @app.put("/api/studies/{sid}/baseline")
    def pin_baseline(sid: str, req: BaselineRequest):
        need(sid)
        if req.runId is not None:
            mine = {a["run_id"] for a in ss.attempts(sid) if a["status"] == "completed"}
            if req.runId not in mine:
                raise _err(422, "request_invalid", "the baseline must be a completed run of this study")
        ss.set_baseline(sid, req.runId)
        return study_view(ss, rs, sid, runner.is_running(sid))

    @app.get("/api/runs/{a}/diff/{b}")
    def run_diff(a: str, b: str):
        for r in (a, b):
            if rs.get_run(r) is None:
                raise _err(404, "not_found", f"unknown run '{r}'")
        return diff_runs(rs, a, b)
