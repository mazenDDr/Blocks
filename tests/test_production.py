"""A50/A59–A63: native fitted serving, measured HTTP, isolation and lineage."""
from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.stats import ks_2samp

from artifact_store import ArtifactStore
from control.app import create_app
from graph_core.hashing import semantic_hash
from graph_core.project_io import load_project
from graph_core.validate import require_executable
from production.models import PredictRequest, RegisterVersion, ReleaseCreate, ServingConfig
from production.monitor import categorical_compare, distribution_compare, monitoring
from production.pipeline import Pipeline, ProductionError, capture_pipelines
from production.runtime import ProductionRuntime
from tabular.engine import run_graph
from worker.tabular_run import TabularRunConfig, run_tabular
from conftest import EXAMPLES


@pytest.fixture
def lab(tmp_path):
    graph = load_project(EXAMPLES / "production_sensors.project.json").graph
    store = ArtifactStore(tmp_path / "wb")
    store.create_run("trained", semantic_hash(graph), TabularRunConfig().model_dump())
    store.add_artifact("trained", "graph", graph.model_dump_json(by_alias=True).encode(), "complete", None, {})
    assert run_tabular(graph, TabularRunConfig(), store, "trained") == "completed"
    rt = ProductionRuntime(store)
    version = rt.register_version(RegisterVersion(runId="trained", node="classifier", name="synthetic sensors", owner="local scientist",
                                                 intendedUse="Synthetic fixture evidence", limitations="Not a real sensor benchmark"))
    return rt, version, graph


def release(lab, **config):
    rt, version, _ = lab
    r = rt.create_release(ReleaseCreate(versionId=version["id"], config=ServingConfig(**config)))
    rt.activate(r["id"], None)
    return r


def request(i="r1", **kw):
    return PredictRequest(requestId=i, records=[{"signal": 1.2, "background": -0.5}], **kw)


def test_pipeline_matches_native_training_state_and_pins_all_identities(lab):
    rt, v, graph = lab
    done = run_graph(graph, require_executable(graph))
    fs = done["scale"].outs["fit"].transformer
    est = done["classifier"].outs["model"].estimator
    rows = [{"signal": 1.2, "background": -0.5}, {"signal": -1.3, "background": 0.8}]
    p = rt.pipeline(v["id"])
    out, _ = p.predict(rows)
    native = fs.transform(pd.DataFrame(rows)[["signal", "background"]].to_numpy())
    np.testing.assert_allclose(out["probabilities"], est.predict_proba(native), atol=1e-12)
    np.testing.assert_equal(out["predictions"], est.predict(native))
    m = v["manifest"]
    assert m["featureOrder"] == ["signal", "background"] and m["outputSchema"]["classes"] == [0, 1]
    assert m["source"]["summary"]["sha256"] and m["evaluationArtifacts"] and m["fitArtifacts"]
    assert m["environment"]["scikit-learn"] and m["graphHash"] == semantic_hash(graph)
    assert rt.store.verify(v["pipelineSha256"])


def test_regression_pipeline_preserves_imputation_onehot_scaling_and_order(tmp_path):
    graph = load_project(EXAMPLES / "tabular_regression.project.json").graph
    store = ArtifactStore(tmp_path / "wb")
    store.create_run("ols", semantic_hash(graph), {})
    assert run_tabular(graph, TabularRunConfig(), store, "ols") == "completed"
    art = store.artifacts("ols", "inference_pipeline")[0]
    p = Pipeline(store, json.loads(store.read_artifact(art["sha256"])))
    df = pd.read_csv(EXAMPLES / "fixtures" / "synthetic_housing.csv").iloc[:5]
    rows = df[[c["name"] for c in p.manifest["inputSchema"]]].astype(object).where(lambda x:x.notna(), None).to_dict("records")
    p.validate_records(rows)
    got = p.predict(rows)[0]["predictions"]
    done = run_graph(graph, require_executable(graph))
    from tabular.core import Table, ExecCtx
    from graph_core import registry
    t = Table(pd.DataFrame(rows))
    for node, fit in (("imp_val","imp_fit"),("oh_val","oh_fit"),("sc_val","sc_fit")):
        op = registry.get_op("tabular.apply_transform")
        t = op.execute(op.Config(), {"table":t,"fit":done[fit].outs["fit"]}, ExecCtx(node))[0]["table"]
    m = done["ols"].outs["model"]
    np.testing.assert_allclose(got, m.estimator.predict(t.df[m.features].to_numpy()), atol=1e-10)


@pytest.mark.parametrize("record", [{"signal":1}, {"signal":1,"background":2,"label":1}, {"signal":"1","background":2},
                                   {"signal":None,"background":2}, {"signal":float("inf"),"background":2}, {"signal":True,"background":2}])
def test_schema_refusals_record_errors_without_state_changes(lab, record):
    r = release(lab, sessionMode="counter")
    rt = lab[0]
    out = rt.predict("local","lab",PredictRequest(requestId="invalid",records=[record],session="s"))
    assert out["status"] == 422 and out["error"]["code"] == "E_REQUEST_SCHEMA"
    assert not rt.ps.query("SELECT * FROM sessions")
    assert rt.ps.trace("local-user","invalid")["status"] == 422


def test_batch_bounds_and_missing_session(lab):
    release(lab, maxBatch=1, sessionMode="counter")
    rt = lab[0]
    assert rt.predict("local","lab",request())["error"]["code"] == "E_SESSION_REQUIRED"
    req = request("big",session="s").model_copy(update={"records": [{"signal":1,"background":2}]*2})
    assert rt.predict("local","lab",req)["status"] == 413


def test_exact_lineage_capture_policy_and_isolated_replay_api(lab):
    rt, v, _ = lab
    r = release(lab,captureInputs=True,sessionMode="counter")
    with TestClient(create_app(rt.store.root)) as c:
        out = c.post("/api/serve/local/lab/predict",json=request(session="s").model_dump()).json()
        assert out["lineage"]["runId"] == "trained" and out["releaseId"] == r["id"] and out["versionId"] == v["id"]
        assert out["lineage"]["source"]["summary"]["sha256"] and out["timings"]["stages"][0]["node"] == "train_scaled"
        replay = c.post("/api/production/requests/r1/replay",json={"user":"local-user"})
        assert replay.status_code == 200 and replay.json()["result"] == out["result"]
        assert rt.ps.query("SELECT count FROM sessions")[0]["count"] == 1
        assert c.get("/api/serve/local/lab/health").json()["ready"]
    # Restart control runtime and prove durable route/version/state.
    fresh = ProductionRuntime(rt.store)
    assert fresh.ps.route("local","lab")["id"] == r["id"]
    assert fresh.predict("local","lab",request("after-restart",session="s"))["sessionState"]["count"] == 2


def test_capture_disabled_has_hash_only_and_no_replay(lab):
    release(lab)
    rt = lab[0]
    out = rt.predict("local","lab",request())
    assert out["records"] is None and out["inputSha256"] and "transformedFeatures" not in out["timings"]
    with TestClient(create_app(rt.store.root)) as c:
        assert c.post("/api/production/requests/r1/replay",json={}).status_code == 409


def test_idempotent_request_conflict_and_counter_isolation_under_concurrency(lab):
    release(lab,sessionMode="counter",concurrency=4,queueLimit=64)
    rt = lab[0]
    def send(i):
        return rt.predict("local","lab",request(f"r{i}",user="alice" if i%2 else "bob",session="same-name"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(send,range(20)))
    assert all(r["status"] == 200 for r in rows)
    for user in ("alice","bob"):
        counts = sorted(r["sessionState"]["count"] for r in rows if r["user"] == user)
        assert counts == list(range(1,11))
    replay = send(0)
    assert replay["idempotentReplay"] and len(rt.ps.traces()) == 20
    assert rt.predict("local","lab",request("different-session",user="bob",session="other"))["sessionState"]["count"] == 1
    with pytest.raises(ProductionError, match="different input"):
        rt.predict("local","lab",request("r0",user="bob",session="changed"))


def test_cancel_active_forward_pass_prevents_session_commit_and_other_user_cannot_cancel(lab, monkeypatch):
    r = release(lab,sessionMode="counter")
    rt = lab[0]
    p = rt.pipeline(r["versionId"])
    entered, proceed = threading.Event(),threading.Event()
    native = p.predict
    def gated(records):
        entered.set()
        assert proceed.wait(3)
        return native(records)
    monkeypatch.setattr(p,"predict",gated)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(rt.predict,"local","lab",request(session="s",user="alice"))
        assert entered.wait(2)
        with pytest.raises(ProductionError) as e:
            rt.ps.cancel("bob","r1")
        assert e.value.status == 404
        rt.ps.cancel("alice","r1")
        proceed.set()
        out = future.result()
    assert out["status"] == 409 and out["error"]["code"] == "E_REQUEST_CANCELLED"
    assert not rt.ps.query("SELECT * FROM sessions")
    with pytest.raises(ProductionError):
        rt.ps.cancel("alice","r1")


def test_queue_admission_limit_and_timeout_leave_no_state(lab, monkeypatch):
    r = release(lab,sessionMode="counter",concurrency=1,queueLimit=0,timeoutSeconds=0.05)
    rt = lab[0]
    native = rt.pipeline(r["versionId"]).predict
    entered, proceed = threading.Event(),threading.Event()
    def gated(records):
        entered.set()
        assert proceed.wait(3)
        return native(records)
    monkeypatch.setattr(rt.pipeline(r["versionId"]),"predict",gated)
    with ThreadPoolExecutor() as pool:
        first = pool.submit(rt.predict,"local","lab",request(session="s"))
        assert entered.wait(2)
        other = rt.predict("local","lab",request("overflow",session="another"))
        assert other["status"] == 429 and other["error"]["code"] == "E_QUEUE_FULL"
        time.sleep(0.06)
        proceed.set()
        assert first.result()["status"] == 504
    assert not rt.ps.query("SELECT * FROM sessions")


def test_rollout_rollback_alias_resolution_and_stale_route_conflict(lab):
    rt,v,_ = lab
    prior = release(lab)
    rt.ps.alias("candidate",v["id"])
    next_ = rt.create_release(ReleaseCreate(versionId="candidate",config=ServingConfig(concurrency=3)))
    assert next_["versionId"] == v["id"] and next_["id"] != prior["id"]
    with pytest.raises(ProductionError) as e:
        rt.activate(next_["id"],None)
    assert e.value.code == "E_ROUTE_CONFLICT"
    rt.activate(next_["id"],prior["id"])
    assert rt.predict("local","lab",request())["releaseId"] == next_["id"]
    with pytest.raises(ProductionError):
        rt.predict("local","lab",request("stale",expectedRelease=prior["id"]))
    rt.activate(prior["id"],next_["id"],True)
    assert rt.predict("local","lab",request("rollback"))["releaseId"] == prior["id"]
    events = rt.ps.query("SELECT * FROM lifecycle WHERE type='rolled_back'")
    assert json.loads(events[0]["data"])["previousRelease"] == next_["id"]
    undeployed = rt.create_release(ReleaseCreate(versionId=v["id"],config=ServingConfig(concurrency=4)))
    with pytest.raises(ProductionError) as e:
        rt.activate(undeployed["id"],prior["id"],True)
    assert e.value.code == "E_ROLLBACK_UNKNOWN"


def test_local_and_staging_routes_are_independent(lab):
    local = release(lab,target="local")
    stage = release(lab,target="staging")
    rt = lab[0]
    assert rt.predict("local","lab",request("loc"))["releaseId"] == local["id"]
    assert rt.predict("staging","lab",request("stage"))["releaseId"] == stage["id"]


def test_integrity_and_environment_mismatches_refused_before_native_load(lab):
    rt,v,_ = lab
    manifest = copy.deepcopy(v["manifest"])
    manifest["environment"]["scikit-learn"] = "wrong"
    with pytest.raises(ProductionError) as e:
        Pipeline(rt.store,manifest)
    assert e.value.code == "E_SERVING_ENVIRONMENT"
    rt.store.path_of(manifest["modelSha256"]).write_bytes(b"tampered")
    with pytest.raises(ProductionError) as e:
        rt.pipeline(v["id"])
    assert e.value.code == "E_ARTIFACT_INTEGRITY"


def test_unrecorded_or_unfinished_model_registration_refused(lab):
    rt,v,_ = lab
    req = RegisterVersion(runId="missing",node="classifier",name="a",owner="b",intendedUse="c",limitations="d")
    with pytest.raises(ProductionError) as e:
        rt.register_version(req)
    assert e.value.code == "E_REGISTER_RUN"
    with pytest.raises(ProductionError) as e:
        rt.register_version(req.model_copy(update={"runId":"trained","node":"metrics"}))
    assert e.value.code == "E_PIPELINE_NOT_RECORDED"


def test_drift_references_and_label_quality_are_separate_and_evidence_linked(lab):
    r = release(lab,captureInputs=True)
    rt = lab[0]
    for i in range(4):
        req = request(f"shift-{i}").model_copy(update={"records":[{"signal":100+i,"background":80}]})
        assert rt.predict("local","lab",req)["status"] == 200
    m = monitoring(rt,r["id"])
    assert m["inputDrift"]["signal"]["ksStatistic"] == 1
    assert m["labelBasedQuality"]["available"] is False
    with TestClient(create_app(rt.store.root)) as c:
        assert c.post("/api/production/requests/shift-0/labels",json={"labels":[0]}).status_code == 200
        assert c.post("/api/production/requests/shift-0/labels",json={"labels":[1]}).status_code == 409
        assert c.post("/api/production/requests/shift-1/labels",json={"labels":["invalid"]}).status_code == 422
        assert c.post("/api/production/requests/shift-1/labels",json={"labels":[0,1]}).status_code == 422
    m = monitoring(rt,r["id"])
    assert m["labelBasedQuality"]["values"]["accuracy"] == 0
    assert m["labelBasedQuality"]["meanLabelDelaySeconds"] >= 0
    assert m["labelBasedQuality"]["evidence"][0]["traceSha256"]
    empty = monitoring(rt,r["id"],since=time.time()+1)
    assert empty["health"]["p95Ms"] is None and not empty["labelBasedQuality"]["available"]


def test_distribution_math_matches_native_and_hand_calculation():
    a,b = [0,1,2],[3,4,5]
    result = distribution_compare(a,b)
    native = ks_2samp(a,b)
    assert result["ksStatistic"] == native.statistic and result["ksPValue"] == native.pvalue
    assert result["meanChange"] == 3
    assert categorical_compare([0,0,1,1],[1,1,1,1])["totalVariation"] == 0.5


def test_rollout_between_different_native_weights_pins_inflight_release(lab, monkeypatch):
    rt,v,graph = lab
    prior = release(lab)
    changed = graph.model_copy(deep=True)
    changed.node("classifier").config["C"] = 0.01
    rt.store.create_run("changed",semantic_hash(changed),{})
    rt.store.add_artifact("changed","graph",changed.model_dump_json(by_alias=True).encode(),"complete",None,{})
    assert run_tabular(changed,TabularRunConfig(),rt.store,"changed") == "completed"
    v2 = rt.register_version(RegisterVersion(runId="changed",node="classifier",name="second",owner="scientist",intendedUse="synthetic comparison",limitations="fixture only"))
    assert v2["manifest"]["modelSha256"] != v["manifest"]["modelSha256"]
    second = rt.create_release(ReleaseCreate(versionId=v2["id"]))
    p = rt.pipeline(v["id"])
    native = p.predict
    entered, proceed = threading.Event(),threading.Event()
    def gated(records):
        entered.set()
        assert proceed.wait(3)
        return native(records)
    monkeypatch.setattr(p,"predict",gated)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(rt.predict,"local","lab",request("inflight"))
        assert entered.wait(2)
        rt.activate(second["id"],prior["id"])
        newer = rt.predict("local","lab",request("new"))
        proceed.set()
        older = future.result()
    assert older["releaseId"] == prior["id"] and older["versionId"] == v["id"]
    assert newer["releaseId"] == second["id"] and newer["versionId"] == v2["id"]
    assert older["result"]["probabilities"] != newer["result"]["probabilities"]
    rt.activate(prior["id"],second["id"],True)
    monkeypatch.setattr(p,"predict",native)
    restored = rt.predict("local","lab",request("restored"))
    assert restored["result"] == older["result"]
    rt.ps.alias("candidate",v2["id"])
    assert rt.ps.get("release",prior["id"])["versionId"] == v["id"]


def test_restart_marks_uncommitted_requests_failed_without_resuming_state(lab):
    rt,_,_ = lab
    r = release(lab,sessionMode="counter")
    rt.ps.begin_request("user","abandoned","fp",r["id"])
    restarted = ProductionRuntime(rt.store)
    trace = restarted.ps.trace("user","abandoned")
    assert trace["status"] == 503 and trace["error"]["code"] == "E_SERVING_RESTART"
    assert trace["sessionState"] is None and not restarted.ps.query("SELECT * FROM sessions")
    assert restarted.ps.begin_request("user","abandoned","fp",r["id"])["status"] == 503


def test_serving_does_not_reread_source_or_modify_fitted_artifacts(lab, tmp_path):
    rt,v,graph = lab
    source = tmp_path/"source.csv"
    source.write_bytes((EXAMPLES/"fixtures/synthetic_serving_sensors.csv").read_bytes())
    copied = graph.model_copy(deep=True)
    copied.node("sensors").config["path"] = str(source)
    rt.store.create_run("copied",semantic_hash(copied),{})
    assert run_tabular(copied,TabularRunConfig(),rt.store,"copied") == "completed"
    version = rt.register_version(RegisterVersion(runId="copied",node="classifier",name="snapshot",owner="scientist",intendedUse="source independence",limitations="fixture"))
    source.unlink()
    p = rt.pipeline(version["id"])
    pinned = [p.manifest["modelSha256"],*p.manifest["fitArtifacts"].values()]
    before = [rt.store.read_artifact(sha) for sha in pinned]
    result = p.predict(request().records)[0]
    assert result["probabilities"]
    assert before == [rt.store.read_artifact(sha) for sha in pinned]


def test_registration_and_release_api_validation_and_refusals(lab):
    rt,v,_ = lab
    with TestClient(create_app(rt.store.root)) as c:
        req = {"runId":"trained","node":"classifier","name":"model","owner":"scientist","intendedUse":"fixture","limitations":"synthetic"}
        registered = c.post("/api/production/versions",json=req)
        assert registered.status_code == 201
        vid = registered.json()["id"]
        assert c.put("/api/production/aliases/candidate",json={"versionId":vid}).status_code == 200
        assert c.post("/api/production/releases",json={"versionId":vid,"config":{"target":"remote"}}).status_code == 422
        assert c.post("/api/production/releases",json={"versionId":vid,"config":{"concurrency":17}}).status_code == 422
        r = c.post("/api/production/releases",json={"versionId":"candidate"}).json()
        assert c.get("/api/serve/local/lab/health").status_code == 404
        assert c.post(f"/api/production/releases/{r['id']}/deploy",json={}).status_code == 200
        assert c.get("/api/production").json()["capabilities"]["replicas"] == 1
        assert c.get(f"/api/production/versions/{vid}/reference-input").json()["provenance"]["referenceSha256"]


@pytest.mark.parametrize("pattern",["steady","ramp","burst","closed_loop"])
def test_arrival_patterns_and_generator_saturation_are_measured(lab,pattern,monkeypatch):
    import socket
    import uvicorn
    import httpx
    from production.models import TrafficSpec
    from production.traffic import TrafficRunner
    rt,_,_ = lab
    r = release(lab)
    app = create_app(rt.store.root)
    p = app.state.services.production.pipeline(r["versionId"])
    native = p.predict
    def instrumented(records):
        time.sleep(0.035)
        return native(records)
    monkeypatch.setattr(p,"predict",instrumented)
    sock = socket.socket()
    sock.bind(("127.0.0.1",0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app,log_level="error",lifespan="off"))
    thread = threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
    thread.start()
    try:
        deadline = time.monotonic()+5
        while not server.started and time.monotonic()<deadline:
            time.sleep(0.01)
        runner = app.state.services.traffic
        spec = TrafficSpec(releaseId=r["id"],payloads=[request().records],pattern=pattern,rate=100,durationSeconds=0.2,concurrency=1,warmupRequests=0,maxRequests=30)
        job = runner.start(spec,port)
        with pytest.raises(ProductionError) as e:
            runner.start(spec,port)
        assert e.value.code == "E_TRAFFIC_BUSY"
        deadline = time.monotonic()+5
        while job["state"] == "running" and time.monotonic()<deadline:
            time.sleep(0.02)
            job = runner.view(job["id"])
        out = job["result"]
        assert out["state"] == "completed" and out["observed"]["successfulRequests"]>0
        assert out["observed"]["errors"] == 0
        if pattern == "closed_loop":
            assert out["generator"]["droppedAtGenerator"] == 0 and out["generator"]["mode"] == "response-driven"
        else:
            assert out["generator"]["droppedAtGenerator"]>0 and out["generator"]["mode"] == "rate-driven"
        restarted = TrafficRunner(ProductionRuntime(rt.store))
        assert restarted.view(job["id"])["id"] == out["id"]
    finally:
        server.should_exit=True
        thread.join(5)
        sock.close()
    assert not thread.is_alive()


def test_traffic_cancel_stops_arrivals_drains_and_records_result(lab):
    import socket
    import uvicorn
    from production.models import TrafficSpec
    rt,_,_ = lab
    r = release(lab)
    app = create_app(rt.store.root)
    sock = socket.socket()
    sock.bind(("127.0.0.1",0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app,log_level="error",lifespan="off"))
    thread = threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
    thread.start()
    try:
        deadline = time.monotonic()+5
        while not server.started and time.monotonic()<deadline:
            time.sleep(0.01)
        runner = app.state.services.traffic
        job = runner.start(TrafficSpec(releaseId=r["id"],payloads=[request().records],durationSeconds=5,warmupRequests=0),port)
        runner.cancel(job["id"])
        deadline = time.monotonic()+5
        while job["state"] == "running" and time.monotonic()<deadline:
            time.sleep(0.02)
            job = runner.view(job["id"])
        assert job["state"] == "cancelled" and job["result"]["generator"]["scheduledArrivals"] < 25
    finally:
        server.should_exit=True
        thread.join(5)
        sock.close()
    assert not thread.is_alive()


def test_real_loopback_http_load_generator_measures_and_pins(lab):
    import socket
    import uvicorn
    rt,_,_ = lab
    r = release(lab,captureInputs=True)
    sock = socket.socket()
    sock.bind(("127.0.0.1",0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(rt.store.root),log_level="error",lifespan="off"))
    thread = threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
    thread.start()
    try:
        import httpx
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",trust_env=False,timeout=10) as c:
            deadline = time.monotonic()+5
            while not server.started and time.monotonic()<deadline:
                time.sleep(0.01)
            spec = {"releaseId":r["id"],"payloads":[request().records],"durationSeconds":0.3,"rate":20,"maxRequests":8,"concurrency":2}
            job = c.post("/api/production/traffic",json=spec)
            assert job.status_code == 202,job.text
            job = job.json()
            deadline = time.monotonic()+5
            while job["state"] == "running" and time.monotonic()<deadline:
                time.sleep(0.03)
                job = c.get(f"/api/production/traffic/{job['id']}").json()
            assert job["state"] == "completed",job
            out = job["result"]
            assert out["observed"]["successfulRequests"] > 0 and out["observed"]["errors"] == 0
            assert out["generator"]["scheduledArrivals"] <= 8
            assert out["generator"]["sentRequests"]+out["generator"]["droppedAtGenerator"] == out["generator"]["scheduledArrivals"]
            assert out["resources"]["peakSampledRssBytes"] > 0 and out["resources"]["cpuSeconds"] >= 0
            assert all(x["releaseId"] == r["id"] and x["traceSha256"] for x in out["requests"])
            assert out["observed"]["achievedRps"] == out["observed"]["successfulRequests"]/out["generator"]["includingDrainSeconds"]
            assert out["warmup"] and out["observed"]["p95Ms"] >= out["observed"]["p50Ms"]
            persisted = ProductionRuntime(rt.store).ps.get("traffic",out["id"])
            assert persisted["observed"] == out["observed"]
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()
    assert not thread.is_alive()
