"""Real HTTP imports → fresh native CPU training → persisted inference. SYNTHETIC fixture evidence only."""
import argparse
import json
import os
import time
import uuid
from pathlib import Path
import httpx
from domain import samples


def journey(base, fixtures):
    root=Path(fixtures).resolve();out=[]
    headers={"Authorization":"Bearer "+os.environ["VOID_API_TOKEN"]} if os.environ.get("VOID_API_TOKEN") else {}
    with httpx.Client(base_url=base,headers=headers,timeout=90,trust_env=False) as c:
        def get(path):
            r=c.get(path);r.raise_for_status();return r.json()
        def post(path,body):
            r=c.post(path,json=body,headers={"Idempotency-Key":uuid.uuid4().hex});r.raise_for_status();return r.json()
        for kind,file,builder in [("coco","coco.json",samples.vision_graph),("conll","ner.conll",samples.nlp_graph),("wav","audio.json",samples.speech_graph)]:
            dataset=post("/api/domain/datasets",{"kind":kind,"path":str(root/file),"root":str(root),
                "license":"CC0-1.0; generated SYNTHETIC fixture", "synthetic":True,"flip_pairs":[[0,1]] if kind=="coco" else []})
            graph=builder(dataset["path"],epochs=2)
            graph["nodes"][-1]["config"].update({"width":4} if kind=="coco" else {"hidden":8,**({"embedding":8} if kind=="conll" else {})})
            project="import_"+dataset["family"];ui={"schemaVersion":"1.0.0","positions":{},"synthetic":True,"description":"SYNTHETIC imported "+kind+" teaching fixture; CC0-1.0"}
            r=c.put("/api/projects/"+project,json={"graph":graph,"ui":ui});r.raise_for_status()
            rid=post("/api/runs",{"projectId":project,"config":{}})["runId"];end=time.monotonic()+90
            while time.monotonic()<end:
                run=get("/api/runs/"+rid)
                if run["status"] in ("completed","failed","cancelled"):
                    assert run["status"]=="completed",run;break
                time.sleep(.1)
            else:raise TimeoutError(rid)
            model=get("/api/domain/models?runId="+rid)["models"][0]
            assert model["available"] and model["source"]["datasetId"]==dataset["id"] and model["source"]["synthetic"] is True
            recorded=get("/api/domain/models/"+model["modelId"]+"/example")
            prediction=post("/api/domain/models/"+model["modelId"]+"/predict",{"records":recorded["records"]})
            assert prediction["provenance"]["source"]["license"]==dataset["license"]
            out.append({"family":dataset["family"],"datasetId":dataset["id"],"runId":rid,"modelId":model["modelId"],
                "sourceSha256":model["source"]["sha256"],"synthetic":True,"license":dataset["license"],"predictions":len(prediction["predictions"])})
    return {"scope":"local CPU; labelled SYNTHETIC fixtures; no real-world accuracy claim","journeys":out}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",default="http://127.0.0.1:8776");p.add_argument("--fixtures",required=True);a=p.parse_args()
    print(json.dumps(journey(a.base,a.fixtures),indent=2))
