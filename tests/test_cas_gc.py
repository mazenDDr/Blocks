"""Offline conservative CAS garbage collection on SYNTHETIC workbenches."""
import json
import os
import sqlite3
import time
from contextlib import closing

import pytest

from artifact_store import ArtifactStore
from maintenance.cas_gc import GcError, collect, main


def old(store: ArtifactStore, sha: str, age: float = 7 * 86400):
    t = time.time() - age
    os.utime(store.path_of(sha), (t, t))


def put_old(store: ArtifactStore, data: bytes) -> str:
    sha = store.put_bytes(data)
    old(store, sha)
    return sha


def test_dry_run_reports_only_unreferenced_old_blobs_and_deletes_nothing(tmp_path):
    store = ArtifactStore(tmp_path)
    store.create_run("r1", "g" * 64, {})
    store.set_status("r1", "preparing")
    store.set_status("r1", "failed", "SYNTHETIC finished run")
    recorded = store.add_artifact("r1", "metrics", b"SYNTHETIC recorded", "final", None, {})["sha256"]
    old(store, recorded)
    orphan = put_old(store, b"SYNTHETIC orphan")
    young = store.put_bytes(b"SYNTHETIC young orphan")
    report = collect(tmp_path)
    assert report["applied"] is False
    assert report["list"] == [orphan] and report["candidates"] == 1 and report["withinGrace"] == 1
    assert report["referenced"] == 1 and report["blobs"] == 3 and "meta.db" in report["databases"]
    assert all(store.verify(s) for s in (recorded, orphan, young))


def test_apply_requires_offline_attestation(tmp_path):
    ArtifactStore(tmp_path)
    with pytest.raises(GcError) as error:
        collect(tmp_path, apply=True)
    assert error.value.code == "E_GC_OFFLINE"


def test_apply_deletes_orphans_and_keeps_every_kind_of_reference(tmp_path):
    store = ArtifactStore(tmp_path)
    # Transitive: a referenced JSON manifest names a child blob.
    child = put_old(store, b"SYNTHETIC child")
    manifest = put_old(store, json.dumps({"childSha256": child}).encode())
    (tmp_path / "projects").mkdir()
    (tmp_path / "projects" / "p.json").write_text(json.dumps({"release": {"manifest": manifest}}))
    # Name-only reference (tracker-style directory) and symlink target.
    named = put_old(store, b"SYNTHETIC named")
    (tmp_path / "trackers" / named).mkdir(parents=True)
    linked = put_old(store, b"SYNTHETIC linked")
    os.symlink(f"../artifacts/{linked}", tmp_path / "trackers" / "link")
    # SQLite text long enough to use overflow pages, with the reference past the first page.
    overflow = put_old(store, b"SYNTHETIC overflow")
    with closing(sqlite3.connect(tmp_path / "other.sqlite")) as db, db:
        db.execute("CREATE TABLE t (v TEXT, b BLOB)")
        db.execute("INSERT INTO t VALUES (?, ?)", ("x" * 9000 + overflow + "y" * 9000, None))
    blobbed = put_old(store, b"SYNTHETIC blob column")
    with closing(sqlite3.connect(tmp_path / "other.sqlite")) as db, db:
        db.execute("INSERT INTO t VALUES (?, ?)", (None, b"\x00\x01" + blobbed.encode() + b"\xff"))
    # Embedded in a longer hex run (e.g. a concatenated digest) is still kept.
    embedded = put_old(store, b"SYNTHETIC embedded")
    (tmp_path / "notes.txt").write_text("ab" + embedded + "cd")
    orphan = put_old(store, b"SYNTHETIC orphan")
    partial = store.artifact_dir / "stale.tmp"
    partial.write_bytes(b"partial")
    os.utime(partial, (time.time() - 7 * 86400,) * 2)
    report = collect(tmp_path, apply=True, offline=True)
    assert report["applied"] and report["list"] == [orphan, "stale.tmp"]
    assert not store.path_of(orphan).exists() and not partial.exists()
    for sha in (child, manifest, named, linked, overflow, blobbed, embedded):
        assert store.verify(sha), sha
    assert collect(tmp_path)["candidates"] == 0


def test_large_file_reference_across_chunk_boundary_is_kept(tmp_path, monkeypatch):
    import maintenance.cas_gc as gc
    monkeypatch.setattr(gc, "CHUNK", 1000)
    store = ArtifactStore(tmp_path)
    kept = put_old(store, b"SYNTHETIC boundary")
    (tmp_path / "big.bin").write_bytes(b"\x00" * 970 + kept.encode() + b"\x00" * 100)
    assert collect(tmp_path)["candidates"] == 0


def test_active_work_and_missing_reference_refuse(tmp_path):
    store = ArtifactStore(tmp_path)
    store.create_run("busy", "g" * 64, {})
    store.set_status("busy", "preparing")
    with pytest.raises(GcError) as error:
        collect(tmp_path, apply=True, offline=True)
    assert error.value.code == "E_GC_ACTIVE"
    store.set_status("busy", "failed", "SYNTHETIC")
    sha = store.add_artifact("busy", "metrics", b"SYNTHETIC", "final", None, {})["sha256"]
    store.path_of(sha).unlink()
    with pytest.raises(GcError) as error:
        collect(tmp_path)
    assert error.value.code == "E_GC_REFERENCE"


def test_unexpected_store_contents_refuse(tmp_path):
    store = ArtifactStore(tmp_path)
    (store.artifact_dir / "README").write_text("not a blob")
    with pytest.raises(GcError) as error:
        collect(tmp_path)
    assert error.value.code == "E_GC_FILE_TYPE"


def test_cli_dry_run_and_refusal(tmp_path, capsys):
    store = ArtifactStore(tmp_path)
    orphan = put_old(store, b"SYNTHETIC orphan")
    assert main(["--workbench", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["list"] == [orphan]
    assert main(["--workbench", str(tmp_path), "--apply"]) == 2
    assert "E_GC_OFFLINE" in capsys.readouterr().err
    assert store.verify(orphan)
