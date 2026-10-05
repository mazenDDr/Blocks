"""Opt-in no-follow link snapshots and real native tracker recovery."""
import json
import os
from pathlib import Path
import shutil

import pytest

from artifact_store import ArtifactStore
from test_workbench_backup import minimal, expect
from test_scale import recorded, native_python
from tracking.bridge import Bridge, ExportSelection
from workbench_backup import core as backup


def test_internal_file_directory_links_restore_after_source_deletion(tmp_path):
    store = minimal(tmp_path)
    (store.root / "projects/current").symlink_to("demo.json")
    (store.root / "latest").symlink_to("projects", target_is_directory=True)
    made = backup.create(store.root, tmp_path / "backup", offline=True, links="internal")
    m = backup.verify(tmp_path / "backup", manifest_sha256=made["manifestSha256"])
    assert m["format"] == backup.LINK_FORMAT and made["links"] == 2
    assert not (tmp_path / "backup/data/latest").exists()  # Backup link data is inert.
    assert not (tmp_path / "backup/data/projects/current").exists()
    assert m["links"]["projects/current"] == {"target": "demo.json", "resolved": "projects/demo.json", "kind": "file"}
    shutil.rmtree(store.root)
    out = tmp_path / "restored"
    result = backup.restore(tmp_path / "backup", out, trusted=True)
    assert result["links"] == 2 and result["omittedLinks"] == {}
    assert os.readlink(out / "latest") == "projects"
    assert os.readlink(out / "projects/current") == "demo.json"
    assert (out / "latest/current").read_bytes() == (out / "projects/demo.json").read_bytes()
    assert ArtifactStore(out).get_run("finished")["status"] == "completed"


@pytest.mark.parametrize("member", ["artifacts", "artifacts/" + "b" * 64, "unknown-wal"])
def test_cas_root_members_and_sidecars_cannot_be_links(tmp_path, member):
    store = minimal(tmp_path)
    if member == "artifacts":
        (store.root / "artifacts").rename(store.root / "ordinary")
        (store.root / member).symlink_to("ordinary", target_is_directory=True)
    else:
        (store.root / member).symlink_to("projects/demo.json")
    expect("E_BACKUP_LINK", lambda: backup.create(store.root, tmp_path / "backup", offline=True, links="internal"))
    assert not (tmp_path / "backup").exists()


@pytest.mark.parametrize("target", ["/etc/passwd", "../../outside", "missing", "chain", "loop", "../meta.db", "chain/../demo.json"])
def test_unsafe_chained_dangling_database_links_refused(tmp_path, target):
    store = minimal(tmp_path)
    (store.root / "projects/chain").symlink_to("demo.json")
    (store.root / "projects/loop").symlink_to("loop")
    # Remove the loop unless it is itself the selected target.
    if target != "loop":
        (store.root / "projects/loop").unlink()
    (store.root / "projects/link").symlink_to(target)
    expect("E_BACKUP_LINK", lambda: backup.create(store.root, tmp_path / "backup", offline=True, links="internal"))
    assert not (tmp_path / "backup").exists()


@pytest.mark.parametrize("field", ["target", "kind", "resolved", "overlap", "parent", "omitted", "version"])
def test_tampered_link_manifest_never_publishes(tmp_path, field):
    store = minimal(tmp_path)
    (store.root / "projects/current").symlink_to("demo.json")
    path = tmp_path / "backup"
    backup.create(store.root, path, offline=True, links="internal")
    m = backup.verify(path)
    rec = m["links"]["projects/current"]
    if field in ("target", "kind", "resolved"):
        rec[field] = {"target": "../../outside", "kind": "directory", "resolved": "meta.db"}[field]
    elif field == "overlap":
        m["links"]["projects/demo.json"] = rec
    elif field == "parent":
        m["links"]["projects/current/child"] = rec
    elif field == "omitted":
        m["omittedLinks"]["unrelated"] = {"target": "/external", "reason": "external-wandb-debug-log"}
    else:
        m["format"] = backup.FORMAT
        m["policy"] = backup.POLICY
    (path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(backup.BackupError):
        backup.restore(path, tmp_path / "restored", trusted=True)
    assert not (tmp_path / "restored").exists()


def test_external_wandb_log_requires_explicit_omission_and_is_never_followed(tmp_path):
    store = minimal(tmp_path)
    rel = "trackers/wandb/" + "a" * 64 + "/wandb/offline-run-SYNTHETIC/logs/debug-core.log"
    link = store.root / rel
    link.parent.mkdir(parents=True)
    # Nonexistent target demonstrates create/verify/restore never read/stat it.
    link.symlink_to("/nonexistent/SYNTHETIC-external-debug-log")
    expect("E_BACKUP_LINK", lambda: backup.create(store.root, tmp_path / "refused", offline=True, links="internal"))
    expect("E_BACKUP_LINK_POLICY", lambda: backup.create(store.root, tmp_path / "refused", offline=True, omit_wandb_external_logs=True))
    made = backup.create(store.root, tmp_path / "backup", offline=True, links="internal", omit_wandb_external_logs=True)
    m = backup.verify(tmp_path / "backup")
    assert m["omittedLinks"] == made["omittedLinks"] == {rel: {"target": os.readlink(link), "reason": "external-wandb-debug-log"}}
    restored = backup.restore(tmp_path / "backup", tmp_path / "restored", trusted=True)
    assert restored["omittedLinks"] == m["omittedLinks"]
    assert not (tmp_path / "restored" / rel).is_symlink()


def test_link_mutation_during_backup_and_failed_restore_clean_staging(tmp_path, monkeypatch):
    store = minimal(tmp_path)
    link = store.root / "projects/current"
    link.symlink_to("demo.json")
    original = shutil.copyfile
    def mutate(src, dest):
        result = original(src, dest)
        link.unlink()
        link.symlink_to("../empty")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", mutate)
        expect("E_BACKUP_CHANGED", lambda: backup.create(store.root, tmp_path / "changed", offline=True, links="internal"))
    backup.create(store.root, tmp_path / "backup", offline=True, links="internal")
    def fail_link(*args, **kwargs):
        raise OSError("forced link creation failure")
    monkeypatch.setattr(Path, "symlink_to", fail_link)
    with pytest.raises(OSError):
        backup.restore(tmp_path / "backup", tmp_path / "restored", trusted=True)
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".void-restore-*"))
    assert not list(tmp_path.glob(".void-backup-*"))


def export(store, adapter):
    bridge = Bridge(store.root)
    sha = next(a["sha256"] for a in store.artifacts("native", "node_summary") if a["meta"]["node"] == "metrics")
    queued = bridge.queue(ExportSelection(runId="native", adapter=adapter, shareMetrics=True, artifactSha256=[sha]))
    row = bridge.sync(queued["id"])
    assert row["status"] == "confirmed", row["error"]
    return row


def test_actual_wandb_export_binary_history_and_links_survive_new_root(recorded, tmp_path):
    store, _ = recorded
    row = export(store, "wandb")
    files, _, links = backup.inventory(store.root, allow_links=True)
    binaries = {rel: backup.digest(p) for rel, p in files.items() if rel.endswith(".wandb")}
    assert len(binaries) == 1
    assert any(Path(rel).name == "latest-run" for rel in links)
    expect("E_BACKUP_FILE_TYPE", lambda: backup.create(store.root, tmp_path / "strict", offline=True))
    made = backup.create(store.root, tmp_path / "backup", offline=True, links="internal", omit_wandb_external_logs=True)
    assert made["links"] >= 3
    # New SDK releases must be measured if they change this exact external link policy.
    assert made["omittedLinks"] and all(Path(rel).name == "debug-core.log" for rel in made["omittedLinks"])
    shutil.rmtree(store.root)
    root = tmp_path / "restored"
    backup.restore(tmp_path / "backup", root, trusted=True)
    for rel, sha in binaries.items():
        assert backup.digest(root / rel) == sha  # Exact SDK binary records, including checksummed history/artifact records.
    m = backup.verify(tmp_path / "backup")
    for rel, info in m["links"].items():
        link = root / rel
        assert link.exists() and os.readlink(link) == info["target"]
        assert link.resolve().is_relative_to(root)
    recovered = Bridge(root)
    assert recovered.sync(row["id"]) == row  # Confirmed export mapping remains idempotent.
    markers = list((root / "trackers/wandb").glob("*/confirmed.json"))
    assert len(markers) == 1 and json.loads(markers[0].read_text()) == row["external"]
    # Immutable absolute provenance is preserved, not falsely advertised as relocated.
    assert row["external"]["directory"].startswith(str(store.root))


MLFLOW_READ = '''import hashlib,json,sys
from mlflow import MlflowClient
c=MlflowClient(tracking_uri='sqlite:///'+sys.argv[1]+'/tracking.sqlite')
r=c.get_run(sys.argv[2])
out={'info':dict(r.info),'params':r.data.params,'tags':r.data.tags,'metrics':r.data.metrics,
     'history':{k:[{'value':v.value,'step':v.step,'timestamp':v.timestamp} for v in c.get_metric_history(r.info.run_id,k)] for k in r.data.metrics}}
try:
 out['artifactSha256']=hashlib.sha256(open(c.download_artifacts(r.info.run_id,'cas/'+sys.argv[3]),'rb').read()).hexdigest()
except Exception as e:out['artifactError']=type(e).__name__
print(json.dumps(out))'''


def test_actual_mlflow_original_root_recovery_and_relocated_path_limitation(recorded, tmp_path):
    store, _ = recorded
    row = export(store, "mlflow")
    rid = row["external"]["nativeRunId"]
    sha = row["sharing"]["artifactSha256"][0]
    before = native_python(MLFLOW_READ, [store.root / "trackers/mlflow", rid, sha])
    assert before["artifactSha256"] == sha and before["history"]
    backup.create(store.root, tmp_path / "backup", offline=True)
    shutil.rmtree(store.root)
    relocated = tmp_path / "relocated"
    backup.restore(tmp_path / "backup", relocated, trusted=True)
    moved = native_python(MLFLOW_READ, [relocated / "trackers/mlflow", rid, sha])
    assert moved.pop("artifactError") == "MlflowException"
    assert moved == {k: v for k, v in before.items() if k != "artifactSha256"}
    backup.restore(tmp_path / "backup", store.root, trusted=True)
    assert native_python(MLFLOW_READ, [store.root / "trackers/mlflow", rid, sha]) == before
    bridge = Bridge(store.root)
    # Retry a lost confirmation against the restored original native DB/artifacts.
    bridge.state.put("tracker", row["id"], {**row, "status": "pending", "external": None})
    assert bridge.sync(row["id"]) == row
