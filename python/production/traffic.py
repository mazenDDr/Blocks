"""Bounded measured loopback HTTP traffic; generator limitations are recorded.

No arbitrary URL/host is accepted. The target is a recorded release on this server's
loopback port. No retries/redirects; each measured request pins the expected release.
"""
from __future__ import annotations

import concurrent.futures as futures
import hashlib
import os
import threading
import time
import uuid

import httpx
import numpy as np
import psutil

from tabular.core import dumps
from .pipeline import ProductionError


class TrafficRunner:
    def __init__(self, runtime, api_token: str | None = None):
        self.runtime = runtime
        self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}  # the server's own boundary, when configured
        self.lock = threading.Lock()
        self.jobs = {}

    def start(self, spec, port, authorization: str | None = None):
        """`authorization` replaces the shared-token header with the caller's own (named accounts, ADR 0055)."""
        release = self.runtime.ps.get("release", spec.releaseId)
        cfg = release["config"]
        if self.runtime.ps.route(cfg["target"], cfg["namespace"])["id"] != spec.releaseId:
            raise ProductionError("E_TRAFFIC_ROUTE", "Deploy this release before testing its endpoint.", 409)
        for payload in spec.payloads:
            self.runtime.pipeline(release["versionId"]).validate_records(payload, cfg["maxBatch"])
        with self.lock:
            if any(j["state"] == "running" for j in self.jobs.values()):
                raise ProductionError("E_TRAFFIC_BUSY", "One bounded load generator runs at a time.", 409)
            id_ = uuid.uuid4().hex[:12]
            cancel = threading.Event()
            self.jobs[id_] = {"id": id_, "state": "running", "spec": spec.model_dump(), "cancel": cancel}
        thread = threading.Thread(target=self._run, args=(id_, spec, release, port, cancel, authorization), daemon=True)
        thread.start()
        return self.view(id_)

    def view(self, id_):
        with self.lock:
            job = self.jobs.get(id_)
            if job:
                return {k: v for k, v in job.items() if k != "cancel"}
        for record in self.runtime.ps.list("traffic"):
            if record["id"] == id_ or record["jobId"] == id_:
                return record
        raise ProductionError("E_TRAFFIC_NOT_FOUND", "Unknown traffic result/job.", 404)

    def cancel(self, id_):
        with self.lock:
            if id_ not in self.jobs or self.jobs[id_]["state"] != "running":
                raise ProductionError("E_TRAFFIC_TERMINAL", "No running traffic job to cancel.", 409)
            self.jobs[id_]["cancel"].set()

    def _run(self, id_, spec, release, port, cancel, authorization=None):
        cfg = release["config"]
        url = f"http://127.0.0.1:{port}/api/serve/{cfg['target']}/{cfg['namespace']}/predict"
        process = psutil.Process(os.getpid())
        cpu0 = sum(process.cpu_times()[:2])
        rss = [process.memory_info().rss]
        rows, warmup, pending, dropped, scheduled = [], [], set(), 0, 0
        def send(i, warm=False):
            payload = spec.payloads[i % len(spec.payloads)]
            reqid = f"load-{id_}-{'warm' if warm else 'r'}-{i}"
            data = {"requestId": reqid, "records": payload, "user": f"traffic-{id_}", "session": f"user-{i % spec.concurrency}", "expectedRelease": spec.releaseId}
            t = time.perf_counter()
            try:
                response = client.post(url, json=data)
                body = response.json()
                return {"requestId": reqid, "status": response.status_code, "latencyMs": (time.perf_counter()-t)*1000,
                        "queueMs": body.get("queueMs"), "releaseId": body.get("releaseId"), "error": body.get("error"),
                        "payloadBytes": len(dumps(payload).encode()), "batchSize": len(payload), "traceSha256": body.get("traceSha256")}
            except (httpx.HTTPError, ValueError) as e:
                return {"requestId": reqid, "status": 0, "latencyMs": (time.perf_counter()-t)*1000, "error": str(e), "queueMs": None,
                        "payloadBytes": len(dumps(payload).encode()), "batchSize": len(payload)}
        result = None
        try:
            with httpx.Client(timeout=cfg["timeoutSeconds"]+2, trust_env=False, follow_redirects=False, headers={"Authorization": authorization} if authorization else self.headers,
                              limits=httpx.Limits(max_connections=spec.concurrency)) as client:
                for i in range(spec.warmupRequests):
                    if cancel.is_set():
                        break
                    warmup.append(send(i, True))
                start = time.perf_counter()
                next_arrival = start
                with futures.ThreadPoolExecutor(max_workers=spec.concurrency) as pool:
                    while time.perf_counter()-start < spec.durationSeconds and scheduled < spec.maxRequests and not cancel.is_set():
                        now = time.perf_counter()
                        ready = {p for p in pending if p.done()}
                        rows.extend(p.result() for p in ready)
                        pending -= ready
                        if spec.pattern == "closed_loop":
                            if len(pending) < spec.concurrency:
                                pending.add(pool.submit(send, scheduled))
                                scheduled += 1
                            else:
                                cancel.wait(0.001)
                        elif now >= next_arrival:
                            if len(pending) < spec.concurrency:
                                pending.add(pool.submit(send, scheduled))
                            else:
                                dropped += 1
                            scheduled += 1
                            elapsed = now-start
                            rate = spec.rate if spec.pattern == "steady" else spec.rate*(0.1+0.9*elapsed/spec.durationSeconds) if spec.pattern == "ramp" else spec.rate*(2 if elapsed<spec.durationSeconds/2 else 0.5)
                            next_arrival += 1/rate
                        else:
                            cancel.wait(min(0.005, next_arrival-now))
                        rss.append(process.memory_info().rss)
                    offer_elapsed = time.perf_counter()-start
                    rows.extend(p.result() for p in futures.as_completed(pending))
                total_elapsed = time.perf_counter()-start
                lat = [r["latencyMs"] for r in rows]
                statuses = {str(s): sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}
                success = sum(r["status"] == 200 for r in rows)
                queues = [r["queueMs"] for r in rows if r.get("queueMs") is not None]
                result = {"jobId": id_, "state": "cancelled" if cancel.is_set() else "completed", "spec": spec.model_dump(), "releaseId": spec.releaseId,
                          "versionId": release["versionId"], "endpoint": url, "mode": "measured HTTP against real local endpoint",
                          "payloadSha256": hashlib.sha256(dumps(spec.payloads).encode()).hexdigest(), "warmup": warmup,
                          "generator": {"mode": "response-driven" if spec.pattern == "closed_loop" else "rate-driven", "scheduledArrivals": scheduled,
                                        "sentRequests": len(rows), "droppedAtGenerator": dropped, "concurrency": spec.concurrency,
                                        "offerDurationSeconds": offer_elapsed, "includingDrainSeconds": total_elapsed, "configuredRate": spec.rate,
                                        "offeredRps": scheduled/offer_elapsed, "sentRps": len(rows)/offer_elapsed,
                                        "limits": "one local thread pool; drops arrivals when all generator slots are occupied; max duration/requests; measured in same process as server; no retries"},
                          "observed": {"successfulRequests": success, "achievedRps": success/total_elapsed, "errors": len(rows)-success, "statuses": statuses,
                                       "unexpectedStatuses": sum(r["status"] not in spec.expectedStatuses for r in rows),
                                       "p50Ms": float(np.percentile(lat, 50)) if lat else None, "p95Ms": float(np.percentile(lat, 95)) if lat else None,
                                       "p99Ms": float(np.percentile(lat, 99)) if lat else None, "meanQueueMs": float(np.mean(queues)) if queues else None,
                                       "timeouts": sum(r["status"] in (0,504) for r in rows)},
                          "resources": {"cpuSeconds": sum(process.cpu_times()[:2])-cpu0, "peakSampledRssBytes": max(rss), "rssSampleCount": len(rss),
                                        "scope": "combined local control server and generator process including warmup/drain; sampled, not instantaneous peak", "cost": "not measured"}, "requests": rows}
        except Exception as e:
            result = {"jobId": id_, "state": "failed", "error": f"{type(e).__name__}: {e}", "spec": spec.model_dump(), "requests": rows}
        saved = self.runtime.ps.save("traffic", result)
        # The immutable result hash is distinct from the mutable job's short id.
        with self.lock:
            self.jobs[id_].update(state=result["state"], result=saved, resultId=saved["id"])
