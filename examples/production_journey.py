"""Real HTTP journey on a running local workbench: train → register → serve → load → rollback.

Uses labelled SYNTHETIC sensors. Never claims a remote deployment or a production
accuracy benchmark. Changes only the chosen local namespace, and retains all traces.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from urllib.parse import urlparse

import os
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",default="http://127.0.0.1:8767")
    parser.add_argument("--namespace",default="m7-cli")
    args = parser.parse_args()
    if urlparse(args.base).hostname not in ("localhost","127.0.0.1"):
        parser.error("this example only targets a local workbench")
    with httpx.Client(headers=({'Authorization': 'Bearer ' + os.environ['VOID_API_TOKEN']} if os.environ.get('VOID_API_TOKEN') else {}), base_url=args.base,timeout=35,trust_env=False) as c:
        def call(method,path,**kw):
            r=c.request(method,path,**kw)
            r.raise_for_status()
            return r.json()
        graph=call("GET","/api/examples/production_sensors")["graph"]
        submitted=call("POST","/api/runs",json={"graph":graph,"config":{}},headers={"Idempotency-Key":uuid.uuid4().hex})
        rid=submitted.get("runId") or submitted["id"]
        deadline=time.monotonic()+60
        while True:
            run=call("GET",f"/api/runs/{rid}")
            if run["status"] in ("completed","failed","cancelled"):
                break
            if time.monotonic()>deadline:
                raise RuntimeError("worker exceeded this journey's 60-second bound")
            time.sleep(0.1)
        if run["status"]!="completed":
            raise RuntimeError(run)
        v=call("POST","/api/production/versions",json={"runId":rid,"node":"classifier","name":"SYNTHETIC sensors CLI","owner":"local scientist",
                                                       "intendedUse":"bounded fixture investigation","limitations":"SYNTHETIC training-reference inputs; not held-out production accuracy"})
        config={"namespace":args.namespace,"captureInputs":True,"sessionMode":"counter"}
        r1=call("POST","/api/production/releases",json={"versionId":v["id"],"config":config})
        overview=call("GET","/api/production")
        previous=next((r["release"] for r in overview["routes"] if r["target"]=="local" and r["namespace"]==args.namespace),None)
        call("POST",f"/api/production/releases/{r1['id']}/deploy",json={"expectedCurrent":previous})
        refs=call("GET",f"/api/production/versions/{v['id']}/reference-input")
        request_id=uuid.uuid4().hex
        trace=call("POST",f"/api/serve/local/{args.namespace}/predict",json={"requestId":request_id,"records":refs["records"],"session":"isolated", "expectedRelease":r1["id"]})
        call("POST",f"/api/production/requests/{request_id}/labels",json={"labels":refs["observedLabels"]})
        replay=call("POST",f"/api/production/requests/{request_id}/replay",json={})
        assert replay["result"]==trace["result"]
        job=call("POST","/api/production/traffic",json={"releaseId":r1["id"],"payloads":[refs["records"],refs["records"][:1]],
                                                      "pattern":"steady","rate":10,"durationSeconds":1,"concurrency":2,"maxRequests":20})
        deadline=time.monotonic()+40
        while job["state"]=="running":
            if time.monotonic()>deadline:
                raise RuntimeError("traffic exceeded its bounded deadline")
            time.sleep(0.1)
            job=call("GET",f"/api/production/traffic/{job['id']}")
        if job["state"]!="completed":
            raise RuntimeError(job)
        r2=call("POST","/api/production/releases",json={"versionId":v["id"],"config":{**config,"concurrency":3}})
        call("POST",f"/api/production/releases/{r2['id']}/deploy",json={"expectedCurrent":r1["id"]})
        call("POST",f"/api/production/releases/{r1['id']}/rollback",json={"expectedCurrent":r2["id"]})
        monitor=call("GET",f"/api/production/releases/{r1['id']}/monitor")
        print(json.dumps({"synthetic":True,"runId":rid,"versionId":v["id"],"releaseId":r1["id"],"traceSha256":trace["traceSha256"],
                          "lineage":trace["lineage"],"sessionState":trace["sessionState"],"trafficResultId":job["resultId"],
                          "traffic":job["result"]["observed"],"generator":job["result"]["generator"],
                          "quality":monitor["labelBasedQuality"],"qualityNote":refs["labelNote"],"rollback":"verified"},indent=2))


if __name__=="__main__":
    main()
