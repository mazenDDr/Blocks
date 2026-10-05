"""Real HTTP native conversation turns → exact history → isolated replay → traffic → rollout/rollback.

Teaching prompts are SYNTHETIC; generated answers and usage come from local Ollama.
No downloads or correctness benchmark. Only the chosen local namespace is changed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from urllib.parse import urlparse

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8779")
    parser.add_argument("--namespace", default="conversation-cli")
    args = parser.parse_args()
    if urlparse(args.base).hostname not in ("localhost", "127.0.0.1"):
        parser.error("this journey targets a local workbench only")
    headers = {"Authorization": "Bearer " + os.environ["VOID_API_TOKEN"]} if os.environ.get("VOID_API_TOKEN") else {}
    with httpx.Client(base_url=args.base, headers=headers, timeout=40, trust_env=False) as c:
        def call(method, path, **kw):
            response = c.request(method, path, **kw)
            response.raise_for_status()
            return response.json()
        graph = call("GET", "/api/examples/serving_conversation")["graph"]
        run = call("POST", "/api/runs", headers={"Idempotency-Key": uuid.uuid4().hex}, json={"graph": graph,
                   "config": {"input": {"question": "SYNTHETIC teaching prompt: reply with a short greeting."}}})
        rid = run["runId"]
        until = time.monotonic()+60
        while True:
            state = call("GET", f"/api/runs/{rid}")
            if state["status"] in ("completed", "failed", "cancelled", "paused"):
                break
            if time.monotonic() > until:
                raise RuntimeError("Source worker exceeded this journey's 60-second bound")
            time.sleep(.1)
        assert state["status"] == "completed", state
        version = call("POST", "/api/production/versions", json={"runId": rid, "node": "__agent_conversation__", "name": "SYNTHETIC agent teaching journey",
                       "owner": "local scientist", "intendedUse": "bounded native Ollama serving", "limitations": "bounded native conversation; teaching prompts, no semantic quality claim"})
        config = {"namespace": args.namespace, "captureInputs": True, "maxBatch": 1, "sessionMode": "conversation", "timeoutSeconds": 30}
        rel = call("POST", "/api/production/releases", json={"versionId": version["id"], "config": config})
        overview = call("GET", "/api/production")
        prior = next((r["release"] for r in overview["routes"] if r["target"] == "local" and r["namespace"] == args.namespace), None)
        call("POST", f"/api/production/releases/{rel['id']}/deploy", json={"expectedCurrent": prior})
        ref = call("GET", f"/api/production/versions/{version['id']}/reference-input")
        first_body = {"requestId": uuid.uuid4().hex, "records": [{"question": "SYNTHETIC teaching prompt: remember my favourite colour is blue. Reply briefly."}], "session": "cli-chat", "expectedRelease": rel["id"]}
        first = call("POST", f"/api/serve/local/{args.namespace}/predict", json=first_body)
        assert first["conversationState"]["revision"] == 1
        request_id = uuid.uuid4().hex
        body = {"requestId": request_id, "records": [{"question": "SYNTHETIC teaching prompt: which colour did I mention?"}], "session": "cli-chat", "expectedRelease": rel["id"]}
        trace = call("POST", f"/api/serve/local/{args.namespace}/predict", json=body)
        duplicate = call("POST", f"/api/serve/local/{args.namespace}/predict", json=body)
        assert duplicate["idempotentReplay"] and duplicate["result"] == trace["result"]
        evidence = trace["result"]["agent"]
        context = evidence["contexts"][0]["value"]
        assert context["usage"]["source"] == "provider" and not context["fixture"]
        assert context["providerRequest"][1:] == [{"role": "user", "content": first_body["records"][0]["question"]},
                 {"role": "assistant", "content": first["result"]["predictions"][0]}, {"role": "user", "content": body["records"][0]["question"]}]
        endpoint = f"/api/production/releases/{rel['id']}/conversation?session=cli-chat"
        head = call("GET", endpoint)
        assert head["head"]["revision"] == 2
        # Explicit independent reference. Exact string agreement may be zero; do not copy predictions into labels.
        call("POST", f"/api/production/requests/{request_id}/labels", json={"labels": ["red"]})
        replay = call("POST", f"/api/production/requests/{request_id}/replay", json={})
        assert replay["result"]["agent"]["executionId"] != evidence["executionId"]
        assert replay["result"]["agent"]["contexts"][0]["value"]["providerRequest"] == context["providerRequest"]
        assert call("GET", endpoint) == head
        job = call("POST", "/api/production/traffic", json={"releaseId": rel["id"], "payloads": [ref["records"]],
                   "pattern": "closed_loop", "durationSeconds": 1, "concurrency": 1, "maxRequests": 2, "warmupRequests": 0})
        until = time.monotonic()+40
        while job["state"] == "running":
            if time.monotonic() > until:
                raise RuntimeError("Bounded traffic did not drain within 40 seconds")
            time.sleep(.1)
            job = call("GET", f"/api/production/traffic/{job['id']}")
        assert job["state"] == "completed", job
        other = call("POST", "/api/production/releases", json={"versionId": version["id"], "config": {**config, "concurrency": 1}})
        call("POST", f"/api/production/releases/{other['id']}/deploy", json={"expectedCurrent": rel["id"]})
        isolated = call("POST", f"/api/serve/local/{args.namespace}/predict", json={**first_body, "requestId": uuid.uuid4().hex, "expectedRelease": other["id"]})
        assert isolated["conversationState"]["revision"] == 1
        call("POST", f"/api/production/releases/{rel['id']}/rollback", json={"expectedCurrent": other["id"]})
        assert call("GET", endpoint) == head
        monitor = call("GET", f"/api/production/releases/{rel['id']}/monitor")
        print(json.dumps({"teachingPrompts": "SYNTHETIC", "runId": rid, "versionId": version["id"], "releaseId": rel["id"],
                          "traceSha256": trace["traceSha256"], "provider": evidence["provider"], "executionId": evidence["executionId"],
                          "contextSha256": evidence["contexts"][0]["sha256"], "predictions": trace["result"]["predictions"], "usage": context["usage"],
                          "replay": replay["replayNote"], "quality": monitor["labelBasedQuality"], "traffic": job["result"]["observed"],
                          "rollback": "verified", "conversation": head["head"], "exactHistory": context["providerRequest"], "sourceThread": "untouched; native production checkpoint heads isolated by release/user/session"}, indent=2))


if __name__ == "__main__":
    main()
