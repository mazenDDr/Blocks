"""Start with python -m scale.worker_server. Single owner, authenticated loopback only."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import threading
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from artifact_store import ArtifactStore, IllegalTransition
from worker.process import submit_run
from worker.tabular_run import TabularRunConfig
from .remote import materialize
from .common import IntegrationError, digest
from .state import State

class Job(BaseModel):
    model_config={"extra":"forbid"}
    id:str=Field(pattern=r"^[0-9a-f]{64}$")
    bundle:dict

def create_worker(root,token):
    if len(token)<16:raise ValueError("Worker token must have at least 16 characters")
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    store=ArtifactStore(root);state=State(root);lock=threading.RLock();handles={}
    app=FastAPI()
    @app.middleware("http")
    async def auth(request:Request,call_next):
        from fastapi.responses import JSONResponse
        if not secrets.compare_digest(request.headers.get("authorization",""),"Bearer "+token):
            return JSONResponse({"detail":{"code":"worker_auth","message":"Worker authorization required."}},status_code=401)
        chunks=[];size=0
        async for chunk in request.stream():
            size+=len(chunk)
            if size>12*1024*1024:return JSONResponse({"detail":{"code":"worker_body_limit"}},status_code=413)
            chunks.append(chunk)
        request._body=b"".join(chunks)
        return await call_next(request)
    @app.exception_handler(IntegrationError)
    async def error(request,exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail":{"code":exc.code,"message":exc.message}},status_code=422)
    @app.get("/v1/health")
    def health():
        return {"ready":True,"protocol":"void-worker/1","location":"separate local process over authenticated loopback HTTP","hardware":"CPU","maxActive":2}
    @app.post("/v1/jobs")
    def submit(req:Job):
        with lock:
            old=state.get("job",req.id)
            request_hash=digest(req.bundle)
            if old:
                if old["hash"]!=request_hash:raise HTTPException(409,"Job identity reused with different inputs")
                return {"runId":old["runId"],"replayed":True}
            if sum(r["status"] not in ("completed","failed","cancelled") for r in store.list_runs())>=2:raise HTTPException(429,"Worker has two active jobs")
            graph=materialize(req.bundle,root)
            rid="worker-"+req.id[:24]
            cfg=TabularRunConfig(seed=req.bundle.get("seed"))
            # Record intention before spawning; known run ID prevents duplicate execution on reconnect.
            state.put("job",req.id,{"runId":rid,"hash":request_hash})
            try:handles[rid]=submit_run(graph,cfg,root,run_id=rid)
            except Exception:
                if store.get_run(rid) is None:
                    store.create_run(rid,"unstarted",cfg.model_dump());store.set_status(rid,"failed","Worker submission failed; explicit new job required")
                raise
            return {"runId":rid,"replayed":False}
    def find(id):
        r=state.get("job",id)
        if not r:raise HTTPException(404,"No worker job")
        row=store.get_run(r["runId"])
        if not row:raise HTTPException(409,"Job intention exists without a recorded run; explicit new job required")
        h=handles.get(r["runId"])
        if h and not h.is_alive():
            h.process.join(0);handles.pop(r["runId"],None)
            if row["status"] not in ("completed","failed","cancelled"):
                store.set_status(r["runId"],"failed","Worker process exited without a terminal result")
                row=store.get_run(r["runId"])
        return row
    @app.get("/v1/jobs/{id}")
    def get(id):
        row=find(id)
        terminal=row["status"] in ("completed","failed","cancelled")
        return {"run":row,"events":store.events(row["id"]) if terminal else [],"artifacts":store.artifacts(row["id"]) if terminal else []}
    @app.post("/v1/jobs/{id}/cancel")
    def cancel(id):
        row=find(id)
        try:store.set_status(row["id"],"cancelling")
        except IllegalTransition:raise HTTPException(409,"Run already terminal")
        if row["id"] in handles:handles[row["id"]]._cancel.set()
        return {"status":"cancelling","policy":"between graph nodes; native calls finish before the next check"}
    @app.get("/v1/jobs/{id}/artifacts/{sha}")
    def artifact(id,sha):
        row=find(id)
        if not re.fullmatch(r"[0-9a-f]{64}",sha) or sha not in {a["sha256"] for a in store.artifacts(row["id"])}:raise HTTPException(404,"No artifact in this job")
        if not store.verify(sha):raise HTTPException(409,"Artifact integrity failed")
        return FileResponse(store.path_of(sha),media_type="application/octet-stream")
    app.state.worker_handles=handles
    return app

if __name__=="__main__":
    import uvicorn
    ap=argparse.ArgumentParser();ap.add_argument("--workbench",required=True);ap.add_argument("--port",type=int,default=8778);ap.add_argument("--token-env",default="VOID_WORKER_TOKEN")
    ap.add_argument("--host",default="127.0.0.1",help="bind address; anything but 127.0.0.1 requires --ssl-certfile/--ssl-keyfile (ADR 0071)")
    ap.add_argument("--ssl-certfile");ap.add_argument("--ssl-keyfile")
    a=ap.parse_args()
    if a.host!="127.0.0.1" and not (a.ssl_certfile and a.ssl_keyfile):
        ap.error("binding to another interface requires TLS: pass --ssl-certfile and --ssl-keyfile")
    token=os.environ.get(a.token_env,"")
    uvicorn.run(create_worker(a.workbench,token),host=a.host,port=a.port,ssl_certfile=a.ssl_certfile,ssl_keyfile=a.ssl_keyfile)
