"""JSON subprocess protocol. No graph/config/raw data is shared implicitly."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def mlflow_export(p):
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param
    root = Path(p["destination"])
    root.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{root/'tracking.sqlite'}")
    exp = client.get_experiment_by_name(p["project"])
    eid = exp.experiment_id if exp else client.create_experiment(p["project"], artifact_location=(root/"artifacts").as_uri())
    matches = client.search_runs([eid], f"tags.`void.export` = '{p['id']}'")
    if len(matches)>1:
        raise ValueError("duplicate external mappings require manual investigation")
    run = matches[0] if matches else client.create_run(eid, tags={"void.export": p["id"], "void.run": p["runId"], "void.graph": p["graphHash"]})
    rid = run.info.run_id
    for k,v in p["metadata"].items():
        prior = client.get_run(rid).data.params.get(k)
        if prior is not None and prior != str(v):
            raise ValueError("confirmed metadata differs")
        if prior is None:
            client.log_batch(rid, params=[Param(k,str(v))])
    for m in p["metrics"]:
        history = client.get_metric_history(rid,m["key"])
        if not any(x.step == m["step"] and x.timestamp == m["timestamp"] and x.value == m["value"] for x in history):
            client.log_batch(rid,metrics=[Metric(m["key"],m["value"],m["timestamp"],m["step"])])
    confirmed=[]
    for a in p["artifacts"]:
        target = f"cas/{a['sha256']}"
        existing = client.list_artifacts(rid,"cas")
        if any(x.path==target for x in existing):
            found=Path(client.download_artifacts(rid,target))
            if hashlib.sha256(found.read_bytes()).hexdigest()!=a["sha256"]:
                raise ValueError("external artifact hash differs")
        else:
            client.log_artifact(rid,a["path"],"cas")
        confirmed.append({"sha256":a["sha256"],"externalPath":target})
    client.set_terminated(rid,"FINISHED")
    return {"nativeRunId":rid,"experimentId":eid,"artifacts":confirmed,"mode":"native local MLflow SQLite"}


def wandb_export(p):
    import wandb
    root=Path(p["destination"])
    final=root/p["id"]
    marker=final/"confirmed.json"
    if marker.exists():
        return json.loads(marker.read_text())
    staging=root/(p["id"]+".pending")
    # Native offline resume is not supported. Rebuild only an unconfirmed immutable export.
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    run=wandb.init(mode="offline", project=p["project"], id=p["id"][:32], dir=str(staging), config=p["metadata"],
                   settings=wandb.Settings(disable_git=True, disable_code=True, x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True, console="off"))
    for m in p["metrics"]:
        run.log({m["key"]:m["value"],"void/source_step":m["step"],"void/source_timestamp":m["timestamp"]})
    confirmed=[]
    for a in p["artifacts"]:
        artifact=wandb.Artifact("void-"+a["sha256"],type="void-recorded-evidence",metadata={"sha256":a["sha256"],"runId":p["runId"]})
        artifact.add_file(a["path"],name=a["sha256"])
        run.log_artifact(artifact)
        confirmed.append({"sha256":a["sha256"],"externalPath":"offline-artifact:"+a["sha256"]})
    run.finish()
    out={"nativeRunId":p["id"][:32],"artifacts":confirmed,"mode":"native W&B offline; not uploaded", "directory":str(final)}
    (staging/"confirmed.json").write_text(json.dumps(out))
    os.replace(staging,final)
    return out

if __name__=="__main__":
    p=json.load(sys.stdin)
    out=mlflow_export(p) if p["adapter"]=="mlflow" else wandb_export(p)
    print("VOID_RESULT:"+json.dumps(out),flush=True)
