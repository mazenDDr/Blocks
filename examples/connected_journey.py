"""Connected-data journey (VISION 7.10) on LOCAL TEST SERVICES with SYNTHETIC data.

    python examples/connected_journey.py                 # seed, register connections, save the project, run regression + a small sweep
    python examples/connected_journey.py --serve         # ... then keep the services and the control API (127.0.0.1:8000) up for the editor

What it starts (see python/connectors/localtest.py): a real PostgreSQL 16 (pgserver) and an S3-compatible server (moto, a mock of the S3 API, not AWS).
The connectors are generic; only the endpoints below point at the local services. S3 credentials are passed to the control process
through environment variables referenced by the connections - they are never written to the workbench."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "python"), str(REPO / "services")]

from connectors import journey, synthetic  # noqa: E402
from connectors.localtest import LocalPostgres, LocalS3  # noqa: E402
from connectors.registry import ConnectionRegistry  # noqa: E402

BUCKET = "labdata"
KEY_ENV, SECRET_ENV = "VOID_LAB_S3_KEY", "VOID_LAB_S3_SECRET"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbench", default=os.environ.get("VOID_WORKBENCH", ".workbench"))
    ap.add_argument("--serve", action="store_true", help="keep services and the control API running for the editor")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--port", type=int, default=8000, help="port of the control API with --serve")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # run the cleanup below (stop the API and both local services)
    wb = Path(a.workbench).resolve()
    wb.mkdir(parents=True, exist_ok=True)

    print(f"[{synthetic.LABEL}]")
    pg, s3 = LocalPostgres(), LocalS3()
    api = None
    try:
        synthetic.seed_postgres(pg)
        synthetic.seed_s3(s3, BUCKET, versioned=True)
        os.environ[KEY_ENV], os.environ[SECRET_ENV] = s3.ACCESS, s3.SECRET
        print(f"PostgreSQL (local pgserver) socket {pg.host}; S3-compatible (moto) {s3.endpoint}, bucket {BUCKET} (versioned)")

        reg = ConnectionRegistry(wb)
        for cid in ("lab_db", "lab_files"):
            if reg.get_row(cid):
                reg.delete(cid)
        reg.create("lab_db", "Lab database (local PostgreSQL, synthetic)", "postgres", pg.settings())
        reg.create("lab_files", "Lab files (local S3-compatible, synthetic)", "s3", {"bucket": BUCKET, "endpoint_url": s3.endpoint},
                   {"access_key_id": {"kind": "env", "name": KEY_ENV}, "secret_access_key": {"kind": "env", "name": SECRET_ENV}})

        from fastapi.testclient import TestClient

        from control.app import create_app

        g = journey.graph()
        with TestClient(create_app(wb)) as c:
            c.put("/api/projects/connected_journey", json={"graph": g, "ui": journey.ui(g)})
            for cid in ("lab_db", "lab_files"):
                t = c.post(f"/api/connections/{cid}/test").json()
                print(f"test {cid}: ok={t['ok']} {t.get('serverVersion') or t.get('versioning') or t.get('error')}")
            v = c.post("/api/validate", json={"graph": g}).json()
            print(f"project saved; validation ok={v['ok']} ({len(v['diagnostics'])} diagnostics)")
            rid = c.post("/api/runs", json={"projectId": "connected_journey", "config": {}}, headers={"Idempotency-Key": f"journey-{time.time()}"}).json()["runId"]
            run = wait(c, f"/api/runs/{rid}")
            if run["status"] != "completed":
                print("run failed:", run["failure"])
                return 1
            m = c.post(f"/api/runs/{rid}/inspect", json={"kind": "metrics", "node": "metrics"}).json()["data"]["values"]
            print(f"regression run {rid}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
            j = c.post(f"/api/runs/{rid}/inspect", json={"kind": "summary", "node": "join_spectra"}).json()["data"]
            print(f"join DB assays x S3 spectra: {j['cardinality']}, unmatched left {j['left']['unmatchedRows']}, unmatched right {j['right']['unmatchedRows']}, rows {j['resultRows']}")
            pins = {s["node"]: s["snapshotId"] for s in run["snapshots"]}
            for s in run["snapshots"]:
                print(f"  snapshot {s['node']}: {s['connector']} {s['snapshotId'][:12]} ({s['reproducibility']['level']})")
            if not a.no_sweep:
                sweep = {"name": "feature ablation (pinned sources)", "projectId": "connected_journey", "run_config": {"source_pins": pins},
                         "hypothesis": "the image-size feature carries no signal beyond pH, temperature and absorbance",
                         "objective": {"metric": {"name": "rmse", "node": "metrics"}, "direction": "minimize"},
                         "search": {"method": "grid", "variables": [{"target": {"scope": "node", "node": "ols", "field": "features"},
                                                                     "values": [["ph", "temp_c", "absorbance_a", "absorbance_b"], ["ph", "temp_c", "absorbance_a", "size"]],
                                                                     "labels": ["drop image size", "drop absorbance_b"]}]},
                         "repeats": {"seeds": [7, 8, 9]}, "limits": {"max_trials": 12}}
                sid = c.post("/api/studies", json=sweep).json()["id"]
                while True:
                    s = c.get(f"/api/studies/{sid}").json()
                    if s["state"] != "running" and not any(t["status"] == "running" for t in s["trials"]):
                        break
                    time.sleep(1)
                print(f"study {sid}: {s['state']}, {s['counts']}; baseline {s['baseline']['value']:.4f} (mean of {s['baseline']['n']} seeds)")
                for gr in s["groups"]:
                    print(f"  {gr['label']:<22} n={gr['n']} mean rmse={gr['mean']:.4f} std={gr['std'] if gr['std'] is None else round(gr['std'], 4)} delta vs baseline={gr['delta']:+.4f}")
        if a.serve:
            env = {**os.environ, "VOID_WORKBENCH": str(wb)}
            api = subprocess.Popen([sys.executable, "-m", "uvicorn", "control.app:create_app", "--factory", "--app-dir", str(REPO / "services"), "--host", "127.0.0.1", "--port", str(a.port)], env=env)
            print(f"control API on http://127.0.0.1:{a.port} - start the editor with `pnpm -C apps/editor dev`, open project 'connected_journey', Ctrl-C to stop everything")
            api.wait()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if api and api.poll() is None:
            api.terminate()
        s3.stop()
        pg.stop()


def wait(c, url: str, timeout: float = 300) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(url).json()
        if r["status"] in ("completed", "failed", "cancelled"):
            return r
        time.sleep(0.3)
    raise TimeoutError(url)


if __name__ == "__main__":
    raise SystemExit(main())
