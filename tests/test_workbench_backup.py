"""Real SQLite/WAL/CAS and native recovery; no substituted learned predictions."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from artifact_store import ArtifactStore
from workbench_backup import core as backup


def minimal(tmp_path):
    store = ArtifactStore(tmp_path / "source")
    store.create_run("finished", "semantic-graph-hash", {})
    store.set_status("finished", "preparing")
    store.set_status("finished", "running")
    store.set_status("finished", "completed")
    store.add_artifact("finished", "SYNTHETIC_fixture", b"SYNTHETIC persisted bytes", "complete", None, {})
    (store.root / "projects").mkdir()
    (store.root / "projects/demo.json").write_text(json.dumps({"path": str(tmp_path / "external.csv"), "synthetic": True}))
    (store.root / "empty").mkdir()
    return store


def expect(code, fn):
    with pytest.raises(backup.BackupError) as e:
        fn()
    assert e.value.code == code, str(e.value)


def test_wal_all_sqlite_and_file_tree_roundtrip_preserves_committed_values(tmp_path):
    store = minimal(tmp_path)
    dbs = set(backup.REQUIRED_DBS) - {"meta.db"}
    dbs.add("trackers/mlflow/mlflow.sqlite")  # Discover nested native tracker DBs too.
    with closing(sqlite3.connect(store.db_path)) as held:
        held.execute("PRAGMA wal_autocheckpoint=0")
        held.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", ("finished", 1, 1., "wal_evidence", None, "hash", '"SYNTHETIC WAL committed"'))
        held.commit()
        assert Path(str(store.db_path) + "-wal").stat().st_size > 0
        for rel in dbs:
            p = store.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(p)) as db, db:
                db.execute("CREATE TABLE fixture(value TEXT)")
                db.execute("INSERT INTO fixture VALUES (?)", (rel,))
        for rel in ("library/modules/pinned/1.0.0.json", "agent/indexes/fixture/index.faiss",
                    "repos/mirror.git/HEAD", "repos/imports/inert.py", "domain_datasets/fixture.json"):
            p = store.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("SYNTHETIC raw persisted fixture; not an executable model")
        made = backup.create(store.root, tmp_path / "backup", offline=True)
        m = backup.verify(tmp_path / "backup", manifest_sha256=made["manifestSha256"])
        assert set(m["databases"]) == dbs | {"meta.db"}
        assert not any(rel.endswith(("-wal", "-shm")) for rel in m["files"])
        assert (tmp_path / "backup").stat().st_mode & 0o777 == 0o700
    shutil.rmtree(store.root)  # Recovery must work with the source physically gone.
    backup.restore(tmp_path / "backup", tmp_path / "restored", trusted=True)
    out = tmp_path / "restored"
    for rel in m["files"]:
        assert backup.digest(out / rel) == m["files"][rel]["sha256"]
    with closing(sqlite3.connect(out / "meta.db")) as db:
        assert db.execute("SELECT data FROM events WHERE type='wal_evidence'").fetchall() == [('"SYNTHETIC WAL committed"',)]
    assert ArtifactStore(out).get_run("finished")["status"] == "completed"
    assert (out / "empty").is_dir()
    assert json.loads((out / "projects/demo.json").read_text())["path"] == str(tmp_path / "external.csv")


def test_offline_trust_disjoint_destination_and_no_overwrite(tmp_path):
    store = minimal(tmp_path)
    expect("E_BACKUP_OFFLINE", lambda: backup.create(store.root, tmp_path / "backup"))
    expect("E_BACKUP_DESTINATION", lambda: backup.create(store.root, store.root / "backup", offline=True))
    backup.create(store.root, tmp_path / "backup", offline=True)
    before = backup.digest(tmp_path / "backup/manifest.json")
    expect("E_BACKUP_DESTINATION", lambda: backup.create(store.root, tmp_path / "backup", offline=True))
    assert backup.digest(tmp_path / "backup/manifest.json") == before
    expect("E_BACKUP_TRUST", lambda: backup.restore(tmp_path / "backup", tmp_path / "restored"))
    expect("E_BACKUP_DESTINATION", lambda: backup.restore(tmp_path / "backup", store.root, trusted=True))
    expect("E_BACKUP_MANIFEST", lambda: backup.verify(tmp_path / "backup", manifest_sha256="0" * 64))


@pytest.mark.parametrize("kind", ["run", "request", "study", "writer"])
def test_active_records_and_sqlite_writer_refused_without_publishing(tmp_path, kind):
    store = minimal(tmp_path)
    held = None
    if kind == "run":
        store.create_run("unfinished", "hash", {})
    elif kind == "request":
        from production.store import ProductionStore
        ProductionStore(store).begin_request("user", "id", "fingerprint", "release")
    elif kind == "study":
        from studies.store import StudyStore
        s = StudyStore(store.root)
        s.create("s", {}, [])
        s.set_state("s", "running")
    else:
        held = sqlite3.connect(store.db_path)
        held.execute("BEGIN IMMEDIATE")
    try:
        expect("E_BACKUP_BUSY" if held else "E_BACKUP_ACTIVE",
               lambda: backup.create(store.root, tmp_path / "backup", offline=True))
    finally:
        if held:
            held.close()
    assert not (tmp_path / "backup").exists()
    assert not list(tmp_path.glob(".void-backup-*"))


def test_all_database_write_reservations_held_during_file_copy(tmp_path, monkeypatch):
    store = minimal(tmp_path)
    from production.store import ProductionStore
    ps = ProductionStore(store)
    original = shutil.copyfile
    seen = []
    def while_copying(src, dst):
        for p in (store.db_path, ps.path):
            with closing(sqlite3.connect(p, timeout=.01)) as contender:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    contender.execute("BEGIN IMMEDIATE")
                seen.append(str(p))
        return original(src, dst)
    monkeypatch.setattr(shutil, "copyfile", while_copying)
    backup.create(store.root, tmp_path / "backup", offline=True)
    assert len(set(seen)) == 2
    # Reservations are released after success, without modifying logical source rows.
    with closing(sqlite3.connect(ps.path)) as db:
        db.execute("BEGIN IMMEDIATE")
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0


@pytest.mark.parametrize("damage", ["cas", "missing_reference", "bad_database", "symlink", "orphan_wal"])
def test_damaged_source_refused(tmp_path, damage):
    store = minimal(tmp_path)
    sha = store.artifacts("finished")[0]["sha256"]
    if damage == "cas":
        store.path_of(sha).write_bytes(b"corrupt")
    elif damage == "missing_reference":
        store.path_of(sha).unlink()
    elif damage == "bad_database":
        (store.root / "production.sqlite").write_bytes(b"not SQLite")
    elif damage == "symlink":
        (store.root / "escape").symlink_to(tmp_path)
    else:
        (store.root / "orphan-wal").write_bytes(b"stray")
    code = {"cas": "E_BACKUP_CAS", "missing_reference": "E_BACKUP_REFERENCE", "bad_database": "E_BACKUP_SQLITE",
            "symlink": "E_BACKUP_FILE_TYPE", "orphan_wal": "E_BACKUP_SQLITE"}[damage]
    expect(code, lambda: backup.create(store.root, tmp_path / "backup", offline=True))
    assert not (tmp_path / "backup").exists()


@pytest.mark.parametrize("damage", ["file", "extra", "missing", "escape", "database_inventory", "format", "source_root", "symlink"])
def test_damaged_backup_never_publishes_restore(tmp_path, damage):
    store = minimal(tmp_path)
    path = tmp_path / "backup"
    backup.create(store.root, path, offline=True)
    m = json.loads((path / "manifest.json").read_text())
    f = path / "data/projects/demo.json"
    if damage == "file":
        f.write_bytes(b"corrupted")
    elif damage == "extra":
        (path / "data/unknown").write_bytes(b"extra")
    elif damage == "missing":
        f.unlink()
    elif damage == "escape":
        m["files"]["../../escape"] = {"sha256": "0" * 64, "size": 0}
    elif damage == "database_inventory":
        m["databases"].append("projects/demo.json")
    elif damage == "format":
        m["format"] = "future-version"
    elif damage == "source_root":
        del m["sourceRoot"]
    else:
        f.unlink()
        f.symlink_to(tmp_path / "outside")
    (path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(backup.BackupError):
        backup.restore(path, tmp_path / "restored", trusted=True)
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".void-restore-*"))


@pytest.mark.parametrize("failure", ["change", "copy_error"])
def test_changed_source_or_failed_io_cleans_staging(tmp_path, monkeypatch, failure):
    store = minimal(tmp_path)
    original = shutil.copyfile
    def changed(src, dst):
        if failure == "copy_error":
            raise OSError("forced disk write failure")
        result = original(src, dst)
        (store.root / "new-file").write_bytes(b"concurrent producer")
        return result
    monkeypatch.setattr(shutil, "copyfile", changed)
    with pytest.raises((backup.BackupError, OSError)):
        backup.create(store.root, tmp_path / "backup", offline=True)
    assert not (tmp_path / "backup").exists()
    assert not list(tmp_path.glob(".void-backup-*"))


def test_cli_success_and_stable_error_json(tmp_path):
    store = minimal(tmp_path)
    def cli(*args):
        return subprocess.run([sys.executable, "-m", "workbench_backup", *map(str, args)], capture_output=True, text=True)
    bad = cli("create", store.root, tmp_path / "backup")
    assert bad.returncode == 1 and json.loads(bad.stderr)["error"]["code"] == "E_BACKUP_OFFLINE"
    made = cli("create", store.root, tmp_path / "backup", "--offline")
    assert made.returncode == 0, made.stderr
    sha = json.loads(made.stdout)["manifestSha256"]
    assert cli("verify", tmp_path / "backup", "--manifest-sha256", sha).returncode == 0
    assert cli("restore", tmp_path / "backup", tmp_path / "restored", "--trusted-local", "--manifest-sha256", sha).returncode == 0


def test_native_conversation_fork_reset_replay_and_research_resume_after_source_loss(tmp_path):
    from test_production_conversation import setup, req, head, graph
    from test_conversation_actions import body
    from production.conversations import action
    from production.runtime import ProductionRuntime
    from worker.agent_run import AgentRunConfig, run_agent
    from graph_core.hashing import semantic_hash
    lab, rt, version, release = setup(tmp_path)
    first = rt.predict("local", "lab", req("one", "SYNTHETIC first"))
    rt.predict("local", "lab", req("two", "SYNTHETIC second"))
    original_head = head(rt, release)
    fork_req = body(rt, release, "fork", destination="branch")
    fork = action(rt, release["id"], "fork", fork_req)
    reset_req = body(rt, release, "reset")
    reset = action(rt, release["id"], "reset", reset_req)
    old_state = rt.pipeline(version["id"]).checkpoint_state(original_head["checkpointSha256"])
    backup.create(lab.wb, tmp_path / "backup", offline=True)
    shutil.rmtree(lab.wb)
    backup.restore(tmp_path / "backup", tmp_path / "recovered", trusted=True)
    store = ArtifactStore(tmp_path / "recovered")
    recovered = ProductionRuntime(store)
    assert recovered.ps.get("version", version["id"]) == version
    assert recovered.ps.route("local", "lab")["id"] == release["id"]
    assert recovered.pipeline(version["id"]).checkpoint_state(original_head["checkpointSha256"]) == old_state
    assert head(recovered, release) == reset["head"]
    assert action(recovered, release["id"], "fork", fork_req)["actionSha256"] == fork["actionSha256"]
    assert action(recovered, release["id"], "reset", reset_req)["actionSha256"] == reset["actionSha256"]
    assert recovered.predict("local", "lab", req("one", "SYNTHETIC first"))["traceSha256"] == first["traceSha256"]
    branch = recovered.predict("local", "lab", req("next-branch", "SYNTHETIC branch", session="branch"))
    assert branch["status"] == 200 and branch["result"]["predictions"] == ["SYNTHETIC branch: 3/1"]
    fresh = recovered.predict("local", "lab", req("fresh", "SYNTHETIC fresh"))
    assert fresh["status"] == 200 and fresh["result"]["predictions"] == ["SYNTHETIC fresh: 1/1"]
    # Research native SQLite saver is also relocated; its thread has its own count.
    cfg = AgentRunConfig(thread_id="t1", input={"question": "SYNTHETIC research next"})
    store.create_run("research", semantic_hash(graph()), cfg.model_dump())
    assert run_agent(graph(), cfg, store, "research", lambda: False) == "completed"
    final = json.loads(store.read_artifact(store.artifacts("research", "final_state")[-1]["sha256"]))
    assert final["n"] == 2


def test_fitted_native_pipeline_predictions_identical_after_source_loss(tmp_path, monkeypatch):
    from graph_core.hashing import semantic_hash
    from graph_core.project_io import load_project
    from production.models import PredictRequest, RegisterVersion, ReleaseCreate
    from production.runtime import ProductionRuntime
    from worker.tabular_run import TabularRunConfig, run_tabular
    graph = load_project(Path(__file__).resolve().parents[1] / "examples/production_sensors.project.json").graph
    store = ArtifactStore(tmp_path / "source")
    store.create_run("trained", semantic_hash(graph), TabularRunConfig().model_dump())
    store.add_artifact("trained", "graph", graph.model_dump_json(by_alias=True).encode(), "complete", None, {})
    assert run_tabular(graph, TabularRunConfig(), store, "trained") == "completed"
    rt = ProductionRuntime(store)
    version = rt.register_version(RegisterVersion(runId="trained", node="classifier", name="SYNTHETIC recovery",
        owner="tests", intendedUse="native fitted recovery evidence", limitations="SYNTHETIC fixture, not a benchmark"))
    release = rt.create_release(ReleaseCreate(versionId=version["id"], config={"captureInputs": True}))
    rt.activate(release["id"], None)
    records = [{"signal": 1.2, "background": -.5}, {"signal": -1.3, "background": .8}]
    before = rt.predict("local", "lab", PredictRequest(requestId="before", records=records))
    assert before["status"] == 200
    backup.create(store.root, tmp_path / "backup", offline=True)
    shutil.rmtree(store.root)
    backup.restore(tmp_path / "backup", tmp_path / "recovered", trusted=True)
    recovered = ProductionRuntime(ArtifactStore(tmp_path / "recovered"))
    assert recovered.ps.get("version", version["id"]) == version
    after = recovered.predict("local", "lab", PredictRequest(requestId="after", records=records))
    assert after["status"] == 200 and before["result"] == after["result"]
    assert after["lineage"] == before["lineage"]
    assert recovered.predict("local", "lab", PredictRequest(requestId="before", records=records))["traceSha256"] == before["traceSha256"]
    from production import runtime
    monkeypatch.setattr(runtime, "environment", lambda: {"incompatible-upgrade": True})
    from production.pipeline import ProductionError
    with pytest.raises(ProductionError) as e:
        recovered.pipeline(version["id"])
    assert e.value.code == "E_SERVING_ENVIRONMENT"  # recovery does not bypass upgrade safety


@pytest.mark.parametrize("family", ["vision", "nlp", "speech"])
def test_native_domain_predictions_and_lineage_after_source_loss(tmp_path, domain_fixtures, family):
    from domain import samples
    from graph_core.schema import Graph
    from graph_core.hashing import semantic_hash
    from production.models import PredictRequest, RegisterVersion, ReleaseCreate
    from production.runtime import ProductionRuntime
    from worker.tabular_run import TabularRunConfig, run_tabular
    builder = {"vision": samples.vision_graph, "nlp": samples.nlp_graph, "speech": samples.speech_graph}[family]
    graph = Graph.model_validate(builder(epochs=2))
    graph.nodes[0].config["n"] = 12
    if family == "vision":
        graph.nodes[-1].config["width"] = 4
    elif family == "nlp":
        graph.nodes[-1].config.update(hidden=8, embedding=8)
    else:
        graph.nodes[-1].config["hidden"] = 8
    store = ArtifactStore(tmp_path / "source")
    store.create_run("trained", semantic_hash(graph), TabularRunConfig().model_dump())
    store.add_artifact("trained", "graph", graph.model_dump_json(by_alias=True).encode(), "complete", None, {})
    assert run_tabular(graph, TabularRunConfig(), store, "trained") == "completed"
    model = store.artifacts("trained", "domain_model")[-1]
    rt = ProductionRuntime(store)
    version = rt.register_version(RegisterVersion(runId="trained", node=model["meta"]["node"], name="SYNTHETIC domain recovery",
        owner="tests", intendedUse="native model recovery", limitations="small SYNTHETIC fixture, not a benchmark"))
    release = rt.create_release(ReleaseCreate(versionId=version["id"], config={"maxBatch": 4, "captureInputs": True}))
    rt.activate(release["id"], None)
    records = rt.pipeline(version["id"]).reference_records()
    before = rt.predict("local", "lab", PredictRequest(requestId="before", records=records))
    assert before["status"] == 200
    backup.create(store.root, tmp_path / "backup", offline=True)
    shutil.rmtree(store.root)
    backup.restore(tmp_path / "backup", tmp_path / "recovered", trusted=True)
    recovered = ProductionRuntime(ArtifactStore(tmp_path / "recovered"))
    assert recovered.ps.get("version", version["id"]) == version
    after = recovered.predict("local", "lab", PredictRequest(requestId="after", records=records))
    assert after["status"] == 200 and after["result"] == before["result"]
    assert after["lineage"] == before["lineage"]


def test_restore_mutation_and_publication_race_never_replace_existing_data(tmp_path, monkeypatch):
    store = minimal(tmp_path)
    path = tmp_path / "backup"
    backup.create(store.root, path, offline=True)
    original = shutil.copyfile
    def changed(src, dst):
        result = original(src, dst)
        (path / "manifest.json").write_text((path / "manifest.json").read_text() + "\n")
        return result
    monkeypatch.setattr(shutil, "copyfile", changed)
    expect("E_BACKUP_CHANGED", lambda: backup.restore(path, tmp_path / "restored", trusted=True))
    assert not (tmp_path / "restored").exists()
    stage = tmp_path / "staging"
    stage.mkdir()
    (stage / "new").write_text("new")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "old").write_text("old")
    expect("E_BACKUP_DESTINATION", lambda: backup.publish(stage, existing))
    assert (existing / "old").read_text() == "old" and not (existing / "new").exists()


def test_sqlite_bytes_in_cas_remain_opaque_and_identical(tmp_path):
    store = minimal(tmp_path)
    path = tmp_path / "opaque.sqlite"
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE fixture(value)")
        db.commit()
    raw = path.read_bytes()
    art = store.add_artifact("finished", "SYNTHETIC_opaque_database", raw, "complete", None, {})
    backup.create(store.root, tmp_path / "backup", offline=True)
    m = backup.verify(tmp_path / "backup")
    assert "artifacts/" + art["sha256"] not in m["databases"]
    assert (tmp_path / "backup/data/artifacts" / art["sha256"]).read_bytes() == raw
