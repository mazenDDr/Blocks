from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from pydantic import BaseModel, Field
from typing import Literal
from artifact_store import ArtifactStore
from scale.common import IntegrationError, digest, encoded
from scale.state import State

REPO=Path(__file__).resolve().parents[2]
class ExportSelection(BaseModel):
    model_config={"extra":"forbid"}
    runId:str
    adapter:Literal["mlflow","wandb"]
    project:str=Field("void-local",pattern=r"^[A-Za-z0-9_-]{1,64}$")
    shareMetrics:bool=False
    artifactSha256:list[str]=Field(default_factory=list,max_length=32)

class Bridge:
    def __init__(self,root):
        self.root=Path(root)
        self.store=ArtifactStore(root)
        self.state=State(root)
        self.lock=threading.RLock()
        self.python=REPO/".venv-trackers/bin/python"
    def queue(self,selection:ExportSelection):
        row=self.store.get_run(selection.runId)
        if not row or row["status"] not in ("completed","failed","cancelled"):
            raise IntegrationError("terminal_run_required","Select a recorded terminal run for an immutable export.")
        available={a["sha256"]:a for a in self.store.artifacts(selection.runId)}
        arts=[]
        for sha in sorted(set(selection.artifactSha256)):
            a=available.get(sha)
            # No raw data, graph/config, model pickles, or code sharing through this initial bridge.
            if not a or a["kind"]!="node_summary" or not self.store.verify(sha):
                raise IntegrationError("artifact_sharing_refused","Only verified numeric metrics node summaries may be shared.")
            body=json.loads(self.store.read_artifact(sha))
            if a["meta"].get("type") != "sklearn.metrics":
                raise IntegrationError("artifact_sharing_refused","Select a sklearn.metrics summary; raw data and code sharing are unsupported.")
            arts.append({"sha256":sha,"path":str(self.store.path_of(sha).resolve())})
        metrics=[]
        if selection.shareMetrics:
            for ev in self.store.events(selection.runId,-1,("train_step","epoch_end")):
                for k in ("loss","train_loss","val_loss","val_accuracy"):
                    v=ev["data"].get(k)
                    if isinstance(v,(int,float)) and math.isfinite(v):
                        metrics.append({"key":k,"value":v,"step":ev["data"].get("step",ev["seq"]),"timestamp":int(ev["ts"]*1000)})
            for a in available.values():
                if a["kind"]=="node_summary" and a["meta"].get("type")=="sklearn.metrics":
                    body=json.loads(self.store.read_artifact(a["sha256"]))
                    for k,v in body.get("values", {}).items():
                        if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v):
                            metrics.append({"key":a["meta"]["node"]+"."+k,"value":v,"step":0,"timestamp":int(row["updated_at"]*1000)})
        payload={"adapter":selection.adapter,"project":selection.project,"runId":selection.runId,"graphHash":row["graph_hash"],
                 "metadata":{"void_run":selection.runId,"void_graph":row["graph_hash"],"void_status":row["status"]},"metrics":metrics,"artifacts":arts}
        # Identity binds sharing scope, destination adapter/project and exact values.
        id=digest(payload)
        payload.update(id=id,destination=str((self.root/"trackers"/selection.adapter).resolve()))
        sha=self.store.put_bytes(encoded(payload))
        with self.lock:
            prior=self.state.get("tracker",id)
            if prior:return prior
            return self.state.put("tracker",id,{"id":id,"runId":selection.runId,"adapter":selection.adapter,"project":selection.project,"payloadSha256":sha,
                        "status":"pending","sharing":{"metrics":selection.shareMetrics,"artifactSha256":[a["sha256"] for a in arts]},"external":None,"error":None})
    def sync(self,id):
        with self.lock:
            r=self.state.get("tracker",id)
            if not r:raise IntegrationError("export_not_found","No queued export has this identity.")
            if r["status"]=="confirmed":return r
            if not self.python.exists():raise IntegrationError("tracker_runtime_missing","Install python/tracking/requirements.txt in .venv-trackers first.")
            if not self.store.verify(r["payloadSha256"]):raise IntegrationError("export_integrity","Export payload changed.")
            payload=json.loads(self.store.read_artifact(r["payloadSha256"]))
            for a in payload["artifacts"]:
                if not self.store.verify(a["sha256"]):raise IntegrationError("export_integrity","Selected artifact changed.")
            try:
                proc=subprocess.run([str(self.python),str(REPO/"python/tracking/native.py")],input=encoded(payload),capture_output=True,timeout=90,
                                    env={**os.environ,"WANDB_MODE":"offline","WANDB_SILENT":"true","MLFLOW_ENABLE_TELEMETRY":"false"})
                lines=[s for s in proc.stdout.decode().splitlines() if s.startswith("VOID_RESULT:")]
                if proc.returncode or not lines:raise RuntimeError(proc.stderr.decode()[-2000:])
                r.update(status="confirmed",external=json.loads(lines[-1][12:]),error=None)
            except Exception as e:
                r.update(status="pending",error=str(e))
            return self.state.put("tracker",id,r)
