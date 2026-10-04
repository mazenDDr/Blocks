"""Snapshot-based authenticated loopback worker protocol. No shared filesystem assumption."""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
import threading
from pathlib import Path
import httpx
from pydantic import BaseModel, Field
from graph_core.schema import Graph
from graph_core.hashing import semantic_hash
from graph_core.registry import get_op
from graph_core.validate import require_executable
from tabular.core import resolve_path
from artifact_store import ArtifactStore
from .common import IntegrationError, digest, encoded, loopback, secret_free
from .state import State

# Native pure tabular CPU subset. Arbitrary code, third-party operations and live credentials are excluded.
ALLOWED={"tabular.csv_source","tabular.select_columns","tabular.profile","tabular.drop_missing","tabular.duplicates",
         "tabular.train_validation_split","tabular.fit_standardize","tabular.fit_impute","tabular.fit_onehot","tabular.apply_transform",
         "sklearn.linear_regression","sklearn.logistic_regression","sklearn.metrics"}
MAX_BYTES=8*1024*1024
class RemoteRequest(BaseModel):
    model_config={"extra":"forbid"}
    graph:Graph
    endpoint:str
    tokenEnv:str=Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,80}$")
    seed:int|None=None
    projectId:str|None=None


def snapshot(graph):
    if graph.graphKind!="tabular" or len(graph.nodes)>64 or not graph.nodes or any(n.type not in ALLOWED for n in graph.nodes):
        raise IntegrationError("worker_subset","Worker supports at most 64 native tabular CSV/preprocessing/linear or logistic/metrics nodes.")
    secret_free(graph.to_json())
    require_executable(graph)
    files={}
    total=0
    for n in graph.nodes:
        if n.type=="tabular.csv_source":
            raw=resolve_path(n.config["path"]).read_bytes()
            total+=len(raw)
            if total>MAX_BYTES:raise IntegrationError("worker_input_limit","Snapshots are limited to 8 MiB.")
            files[n.id]={"sha256":hashlib.sha256(raw).hexdigest(),"base64":base64.b64encode(raw).decode()}
    return {"graph":graph.to_json(),"originalGraphHash":semantic_hash(graph),"files":files}


def materialize(bundle,root):
    graph=Graph.model_validate(bundle["graph"])
    if graph.graphKind!="tabular" or not graph.nodes or len(graph.nodes)>64 or any(n.type not in ALLOWED for n in graph.nodes):
        raise IntegrationError("worker_subset","Unsupported worker graph.")
    if semantic_hash(graph)!=bundle["originalGraphHash"]:raise IntegrationError("worker_graph_hash","Original graph hash differs.")
    secret_free(graph.to_json())
    files=bundle["files"]
    if set(files)!={n.id for n in graph.nodes if n.type=="tabular.csv_source"}:raise IntegrationError("worker_files","Snapshot source mapping differs.")
    size=0
    inputs=Path(root)/"inputs"
    inputs.mkdir(parents=True,exist_ok=True)
    for n in graph.nodes:
        if n.type=="tabular.csv_source":
            f=files[n.id]
            raw=base64.b64decode(f["base64"],validate=True)
            size+=len(raw)
            sha=hashlib.sha256(raw).hexdigest()
            if not re.fullmatch(r"[0-9a-f]{64}",f["sha256"]) or sha!=f["sha256"]:raise IntegrationError("worker_input_hash","Input content does not match its hash.")
            if size>MAX_BYTES:raise IntegrationError("worker_input_limit","Snapshots are limited to 8 MiB.")
            path=inputs/(sha+".csv")
            path.write_bytes(raw)
            # Preserve other CSV parsing settings; execution graph has its own recorded identity.
            n.config["path"]=str(path.resolve())
    require_executable(graph)
    return graph

class RemoteClient:
    def __init__(self,root):
        self.root=Path(root)
        self.store=ArtifactStore(root)
        self.state=State(root)
        self.lock=threading.RLock()
    def headers(self,r):
        token=os.environ.get(r["tokenEnv"])
        if not token or len(token)<16:raise IntegrationError("worker_token_missing","Set the referenced worker token environment variable (at least 16 characters).")
        return {"Authorization":"Bearer "+token}
    def submit(self,req):
        endpoint=loopback(req.endpoint)
        bundle=snapshot(req.graph)
        bundle["seed"]=req.seed
        id=digest({"bundle":bundle,"endpoint":endpoint,"projectId":req.projectId})
        with self.lock:
            prior=self.state.get("remote",id)
            if not prior:
                sha=self.store.put_bytes(encoded(bundle))
                prior=self.state.put("remote",id,{"id":id,"endpoint":endpoint,"tokenEnv":req.tokenEnv,"projectId":req.projectId,
                        "bundleSha256":sha,"originalGraphHash":bundle["originalGraphHash"],"workerRunId":None,"runId":None,"status":"pending","error":None,
                        "location":"separate local process over authenticated loopback HTTP; CPU"})
            return self.refresh(id,submit=True)
    def refresh(self,id,submit=False):
        with self.lock:
            r=self.state.get("remote",id)
            if not r:raise IntegrationError("worker_job_missing","Unknown worker job.")
            if r["status"] in ("completed","failed","cancelled") and r.get("runId"):return r
            try:
                with httpx.Client(base_url=loopback(r["endpoint"]),headers=self.headers(r),timeout=15,trust_env=False) as c:
                    if not r["workerRunId"]:
                        if not self.store.verify(r["bundleSha256"]):raise IntegrationError("worker_bundle_hash","Saved input bundle changed.")
                        body=json.loads(self.store.read_artifact(r["bundleSha256"]))
                        res=c.post("/v1/jobs",json={"id":id,"bundle":body})
                        res.raise_for_status()
                        r["workerRunId"]=res.json()["runId"]
                        self.state.put("remote",id,r) # checkpoint before result transfer
                    res=c.get("/v1/jobs/"+id)
                    res.raise_for_status()
                    body=res.json()
                    r.update(status=body["run"]["status"],error=None)
                    if r["status"] in ("completed","failed","cancelled"):
                        self.import_result(r,body,c)
                return self.state.put("remote",id,r)
            except Exception as e:
                r["error"]=f"{type(e).__name__}: {e}"
                return self.state.put("remote",id,r)
    def import_result(self,r,body,c):
        row=body["run"]
        if row["id"]!=r["workerRunId"]:raise IntegrationError("worker_identity","Worker run changed.")
        rid="remote-"+r["id"][:24]
        if sum(a["size"] for a in body["artifacts"])>32*1024*1024:raise IntegrationError("worker_output_limit","Output transfer exceeds 32 MiB.")
        # Download and verify all bytes before inserting any run/events metadata.
        blobs={}
        for a in body["artifacts"]:
            sha=a["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}",sha):raise IntegrationError("worker_artifact_hash","Invalid hash.")
            if self.store.verify(sha):continue
            raw=c.get("/v1/jobs/"+r["id"]+"/artifacts/"+sha)
            raw.raise_for_status()
            if len(raw.content)!=a["size"] or hashlib.sha256(raw.content).hexdigest()!=sha:raise IntegrationError("worker_artifact_hash","Worker artifact bytes differ.")
            blobs[sha]=raw.content
        for sha,raw in blobs.items():self.store.put_bytes(raw)
        with self.store._lock:
            cfg={**row["config"],"project_id":r["projectId"],"remote":{"jobId":r["id"],"workerRunId":row["id"],"endpoint":r["endpoint"],"originalGraphHash":r["originalGraphHash"],"bundleSha256":r["bundleSha256"],"location":r["location"]}}
            # Atomic metadata import makes reconnect/restart idempotent. Imported native pickles remain untrusted for serving.
            from contextlib import closing
            with closing(self.store._db()) as db,db:
                db.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?,?)",(rid,row["graph_hash"],row["status"],json.dumps(cfg),row["error"],row["created_at"],row["updated_at"]))
                for e in body["events"]:
                    db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)",(rid,e["seq"],e["ts"],e["type"],e["node_id"],e["graph_hash"],json.dumps(e["data"])))
                if not db.execute("SELECT 1 FROM artifacts WHERE run_id=?",(rid,)).fetchone():
                    for a in body["artifacts"]:
                        meta={**a["meta"],"remoteWorkerRunId":row["id"],"remoteJobId":r["id"]}
                        meta.pop("trustedWorkerArtifact",None)
                        db.execute("INSERT INTO artifacts(sha256,run_id,kind,status,step,size,meta,created_at) VALUES(?,?,?,?,?,?,?,?)",(a["sha256"],rid,a["kind"],a["status"],a["step"],a["size"],json.dumps(meta),a["created_at"]))
        r["runId"]=rid
    def cancel(self,id):
        r=self.state.get("remote",id)
        if not r or not r["workerRunId"]:raise IntegrationError("worker_job_missing","Worker has not accepted this job.")
        res=httpx.post(loopback(r["endpoint"])+"/v1/jobs/"+id+"/cancel",headers=self.headers(r),timeout=15,trust_env=False)
        res.raise_for_status()
        return self.refresh(id)
