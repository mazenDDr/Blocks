"""Offline physical erasure of one serving user (ADR 0056) on real native conversation checkpoints; SYNTHETIC text only."""
import json
import os

import pytest

from maintenance.cas_gc import GcError
from maintenance.erase import erase_user, main
from production.runtime import ProductionRuntime
from test_production_conversation import head, req, setup

ALICE, BOB = b"SYNTHETIC-ERASE-ALICE-7f3a", b"SYNTHETIC-KEEP-BOB-9c1d"


def bytes_containing(root, marker):
    """Every workbench file whose raw bytes contain the marker (SQLite, WAL and CAS included)."""
    hits = []
    for parent, _, files in os.walk(root):
        for name in files:
            p = os.path.join(parent, name)
            with open(p, "rb") as f:
                if marker in f.read():
                    hits.append(os.path.relpath(p, root))
    return hits


@pytest.fixture()
def served(tmp_path):
    lab, rt, v, rel = setup(tmp_path)
    for i in range(3):
        assert rt.predict("local", "lab", req(f"a{i}", f"{ALICE.decode()} turn {i}", user="alice"))["status"] == 200
    for i in range(2):
        assert rt.predict("local", "lab", req(f"b{i}", f"{BOB.decode()} turn {i}", user="bob"))["status"] == 200
    rt.ps.add_labels("alice", "a0", [ALICE.decode() + " label"])
    return lab.store.root, rt, rel


def test_dry_run_reports_and_changes_nothing(served):
    root, rt, rel = served
    before = bytes_containing(root, ALICE)
    assert any(p.startswith("artifacts/") for p in before)
    report = erase_user(root, "alice")
    assert report["applied"] is False and report["rows"] == {"requests": 3, "labels": 1, "conversation_actions": 0, "release_memory": 0, "sessions": 0, "agent_sessions": 1}
    assert report["referencedBlobs"] > 0 and bytes_containing(root, ALICE) == before
    with pytest.raises(GcError) as e:
        erase_user(root, "alice", apply=True)
    assert e.value.code == "E_GC_OFFLINE"


def test_apply_removes_every_byte_of_the_user_and_keeps_others_working(served):
    root, rt, rel = served
    report = erase_user(root, "alice", apply=True, offline=True)
    assert report["applied"] and report["erasedBlobs"] and report["rows"]["requests"] == 3
    assert bytes_containing(root, ALICE) == []  # not in SQLite pages, WAL, CAS or anywhere else in the workbench
    assert bytes_containing(root, BOB)
    fresh = ProductionRuntime(rt.store)  # a new process view of the same workbench
    assert head(fresh, rel, user="alice") is None and head(fresh, rel, user="bob")["revision"] == 2
    assert {t["user"] for t in fresh.ps.traces(rel["id"])} == {"bob"}
    with pytest.raises(Exception):
        fresh.ps.trace("alice", "a0")
    t = fresh.predict("local", "lab", req("b9", f"{BOB.decode()} after erasure", user="bob"))
    assert t["status"] == 200 and t["result"]["predictions"][0].endswith(": 3/1")  # bob's thread continues from revision 2
    assert erase_user(root, "alice")["rows"]["requests"] == 0


def test_shared_blobs_are_retained_and_reported(served):
    root, rt, rel = served
    # Another record that names one of alice's blobs keeps it: erasure never breaks a different record.
    shared = sorted(erase_user.__globals__["blob_references"](root, json.dumps(rt.ps.query("SELECT trace FROM requests WHERE user='alice'")).encode()))[0]
    (root / "projects").mkdir(exist_ok=True)
    (root / "projects" / "other.json").write_text(json.dumps({"keeps": shared}))
    report = erase_user(root, "alice", apply=True, offline=True)
    assert shared in report["retainedBlobs"] and report["retainedReason"] and (root / "artifacts" / shared).exists()


def test_cli_and_bad_user(served, capsys):
    root, _, _ = served
    assert main(["--workbench", str(root), "--user", "alice"]) == 0
    assert json.loads(capsys.readouterr().out)["rows"]["requests"] == 3
    assert main(["--workbench", str(root), "--user", "../x"]) == 2
    assert "E_ERASE_USER" in capsys.readouterr().err
