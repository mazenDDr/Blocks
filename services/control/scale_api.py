from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from graph_core.schema import Graph
from extensions.packages import build_package,inspect_package,import_package
from scale.remote import RemoteClient,RemoteRequest
from scale.common import IntegrationError
from scale.common import digest, encoded
from tracking.bridge import Bridge,ExportSelection

class PackageRequest(BaseModel):
    model_config={"extra":"forbid"}
    graph:Graph
    ui:dict
    includeCsv:bool=False
class PackageImport(BaseModel):
    model_config={"extra":"forbid"}
    package:dict
    projectId:str=Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
class ConformanceRequest(BaseModel):
    example:str=Field(pattern=r"^[a-z0-9_-]{1,64}$")
class EvidenceRequest(BaseModel):
    model_config={"extra":"forbid"}
    runIds:list[str]=Field(min_length=1,max_length=4)
    conclusion:str=Field("",max_length=4000)

def register(app,sv):
    repo=Path(__file__).resolve().parents[2]
    remote=RemoteClient(sv.workbench);bridge=Bridge(sv.workbench)
    sv.remote=remote;sv.tracker_bridge=bridge
    @app.exception_handler(IntegrationError)
    async def error(request,exc):
        return JSONResponse({"detail":{"code":exc.code,"message":exc.message}},status_code=422)
    @app.get("/api/integrations")
    def overview():
        runs=[]
        for r in sv.store.list_runs():
            if r["status"] in ("completed","failed","cancelled"):
                arts=[{"sha256":a["sha256"],"node":a["meta"].get("node")} for a in sv.store.artifacts(r["id"],"node_summary") if a["meta"].get("type")=="sklearn.metrics"]
                runs.append({"id":r["id"],"status":r["status"],"graphHash":r["graph_hash"],"shareableMetricsArtifacts":arts})
        return {"remote":remote.state.all("remote"),"exports":bridge.state.all("tracker"),"runs":runs,
                "trackerRuntimeAvailable":bridge.python.exists(),"sdkExamples":[p.parent.name for p in (repo/"examples/plugins").glob("*/manifest.json")],
                "capabilities":{"worker":"authenticated CPU tabular CSV/JSONL subset; separate process on loopback HTTP or another host over HTTPS with a pinned certificate (ADR 0071)", "trackers":"native local MLflow and W&B offline only", "package":"inert graph/UI/dependencies plus optional explicit CSV/JSONL snapshots", "multiHost":"one worker per job over pinned-certificate HTTPS; no scheduling across hosts","distributedGpu":False}}
    @app.post("/api/integrations/remote")
    def submit(req:RemoteRequest):return remote.submit(req)
    @app.post("/api/integrations/remote/{id}/refresh")
    def refresh(id:str):return remote.refresh(id)
    @app.post("/api/integrations/remote/{id}/cancel")
    def cancel(id:str):return remote.cancel(id)
    @app.post("/api/integrations/exports")
    def queue(req:ExportSelection):return bridge.queue(req)
    @app.post("/api/integrations/exports/{id}/sync")
    def sync(id:str):return bridge.sync(id)
    @app.post("/api/packages/export")
    def export(req:PackageRequest):return build_package(req.graph,req.ui,req.includeCsv)
    @app.post("/api/packages/inspect")
    def inspect(req:PackageImport):return inspect_package(req.package)
    @app.post("/api/packages/import")
    def import_(req:PackageImport):return import_package(req.package,sv.projects,req.projectId)
    @app.post("/api/extensions/conformance")
    def conformance(req:ConformanceRequest):
        manifest=repo/"examples/plugins"/req.example/"manifest.json"
        if not manifest.exists():raise IntegrationError("sdk_example_missing","No bundled SDK example with that name.")
        proc=subprocess.run([sys.executable,"-m","extensions.sdk",str(manifest)],capture_output=True,timeout=15)
        if proc.returncode:raise IntegrationError("sdk_conformance",proc.stderr.decode()[-2000:])
        return json.loads(proc.stdout)
    @app.get("/api/acceptance")
    def acceptance():return json.loads((repo/"docs/acceptance.json").read_text())
    def evidence(req):
        rows=[]
        for rid in req.runIds:
            row=sv.store.get_run(rid)
            if not row or row["status"] not in ("completed","failed","cancelled"):
                raise IntegrationError("evidence_run","Evidence requires recorded terminal runs.")
            refs=[]
            for a in sv.store.artifacts(rid,"node_summary"):
                if not sv.store.verify(a["sha256"]):raise IntegrationError("evidence_integrity","Recorded evidence changed.")
                refs.append({"sha256":a["sha256"],"node":a["meta"].get("node"),"type":a["meta"].get("type"),"summary":json.loads(sv.store.read_artifact(a["sha256"]))})
            rows.append({"runId":rid,"graphHash":row["graph_hash"],"status":row["status"],"summaries":refs})
        return {"runs":rows,"conclusion":req.conclusion,"source":"recorded node summaries; conclusion supplied by researcher"}
    @app.post("/api/evidence/compare")
    def compare(req:EvidenceRequest):return evidence(req)
    @app.post("/api/evidence")
    def record(req:EvidenceRequest):
        if not req.conclusion.strip():raise IntegrationError("conclusion_required","Enter a scientific conclusion.")
        body=evidence(req);id=digest(body);sha=sv.store.put_bytes(encoded(body))
        return remote.state.put("evidence",id,{"id":id,"sha256":sha,"runIds":req.runIds,"conclusion":req.conclusion})
