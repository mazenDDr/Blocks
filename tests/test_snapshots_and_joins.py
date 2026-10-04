"""Snapshots and reproducibility of mutable sources (A41), DVC revisions (A42), cross-source joins (A43)."""
import json

import pandas as pd
import psycopg
import pytest

from connected_helpers import BUCKET, PLAIN_BUCKET, mk, node_summary, node_table, run_graph
from connectors import dvc as dvc_mod
from connectors import snapshots
from connectors import s3 as s3m
from connectors import synthetic
from connectors.errors import SourceError
from connectors.localtest import make_dvc_repo
from graph_core.validate import validate


def pg_node(sql="SELECT specimen_id, ph FROM specimens ORDER BY specimen_id", conn="lab", **kw):
    return ("db", "postgres.query", {"connection": conn, "mode": "sql", "sql": sql, "limit": 1000, **kw})


# ------------------------------------------------------------------------------------------------ A41: PostgreSQL
def test_a41_postgres_run_records_query_params_server_snapshot_token_and_content_hash(lab):
    g = mk([pg_node("SELECT specimen_id, ph FROM specimens WHERE site = %(site)s ORDER BY specimen_id", params=[{"name": "site", "type": "string", "value": "north"}])], [])
    store, status = run_graph(g, lab.wb, "r1")
    assert status == "completed"
    s = node_summary(store, "r1", "db")
    m = s["snapshot"]
    assert m["kind"] == "postgres" and m["query"]["text"].startswith("SELECT specimen_id") and m["query"]["params"] == [{"name": "site", "type": "string", "value": "north"}]
    assert m["server"]["version"].startswith("PostgreSQL 16") and ":" in m["snapshotToken"] and m["takenAt"] and "READ ONLY" in m["transaction"].upper() or "read only=True" in m["transaction"]
    assert m["result"]["rows"] == s["rows"] and len(m["result"]["sha256"]) == 64 and m["reproducibility"]["level"] == "materialized_extract"
    # the snapshot and its extract are artifacts of the run
    kinds = {a["kind"] for a in store.artifacts("r1")}
    assert {"source_snapshot", "source_extract"} <= kinds
    loaded, raw = snapshots.load(store, s["snapshotId"])
    assert loaded["result"]["sha256"] == m["result"]["sha256"] and raw is not None and len(raw) == m["result"]["bytes"]
    ev = [e for e in store.events("r1") if e["type"] == "source_snapshot_recorded"][0]["data"]
    assert ev["snapshotId"] == s["snapshotId"] and ev["mode"] == "live"


def test_a41_repeat_from_pinned_snapshot_survives_changes_in_the_live_database(lab, client_factory):
    c = client_factory(lab)
    with psycopg.connect(lab.pg.admin_uri, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS drift; CREATE TABLE drift (id int PRIMARY KEY, v double precision); INSERT INTO drift SELECT g, g * 1.5 FROM generate_series(1, 20) g")
    node = ("db", "postgres.query", {"connection": "lab", "mode": "sql", "sql": "SELECT id, v FROM drift ORDER BY id", "limit": 100})
    g = mk([node], [])
    store, st1 = run_graph(g, lab.wb, "first")
    assert st1 == "completed"
    s1 = node_summary(store, "first", "db")
    t1 = node_table(store, "first", "db")
    # the live source changes underneath
    with psycopg.connect(lab.pg.admin_uri, autocommit=True) as conn:
        conn.execute("UPDATE drift SET v = v + 100 WHERE id <= 3; DELETE FROM drift WHERE id = 20")
    # a plain re-run resolves a NEW snapshot with a different content hash: the change is visible, not hidden
    store, st2 = run_graph(g, lab.wb, "second")
    s2 = node_summary(store, "second", "db")
    assert st2 == "completed" and s2["mode"] == "live" and s2["resultSha256"] != s1["resultSha256"] and s2["snapshotId"] != s1["snapshotId"]
    # drift check against the live source says the recorded extract no longer matches
    chk = c.post(f"/api/snapshots/{s1['snapshotId']}/check").json()
    assert chk["identical"] is False and chk["recordedRows"] == 20 and chk["currentRows"] == 19
    # repeat from the pinned identity: same rows, same hash, no database access needed
    store, st3 = run_graph(g, lab.wb, "third", source_pins={"db": s1["snapshotId"]})
    s3 = node_summary(store, "third", "db")
    assert st3 == "completed" and s3["mode"] == "pinned" and s3["resultSha256"] == s1["resultSha256"] and s3["snapshotId"] == s1["snapshotId"]
    pd.testing.assert_frame_equal(node_table(store, "third", "db"), t1)
    assert [a["kind"] for a in store.artifacts("third") if a["kind"] == "source_snapshot"] == ["source_snapshot"]  # linked to the new run
    # the same through the node's own `pin` field, validated statically from the manifest without a connection
    pinned = mk([("db", "postgres.query", {**node[2], "pin": s1["snapshotId"]})], [])
    rep = validate(pinned)
    assert rep.ok and rep.output_types["db"]["table"].info["rows"] == 20
    lab.reg.delete("lab")
    store, st4 = run_graph(pinned, lab.wb, "offline")
    assert st4 == "completed" and node_summary(store, "offline", "db")["resultSha256"] == s1["resultSha256"]  # connection removed: pinned run still works
    # a pin from a different query is refused instead of substituting data silently
    other = mk([("db", "postgres.query", {**node[2], "sql": "SELECT id FROM drift ORDER BY id", "pin": s1["snapshotId"]})], [])
    assert "E_SRC_SNAPSHOT_MISMATCH" in [d.code for d in validate(other).errors]


def test_a41_snapshot_api_and_roundtrip_of_all_dtypes(lab, client_factory):
    c = client_factory(lab)
    sql = ("SELECT specimen_id, collected, ph, (ph > 7) AS basic, (row_number() OVER (ORDER BY specimen_id))::int AS n, "
           "NULL::double precision AS nothing, 'a,b\"c' AS tricky FROM specimens ORDER BY specimen_id")
    g = mk([pg_node(sql)], [])
    store, st = run_graph(g, lab.wb, "r1")
    assert st == "completed"
    s = node_summary(store, "r1", "db")
    df = node_table(store, "r1", "db")
    assert [c_["dtype"] for c_ in s["schema"]] == ["string", "datetime", "float", "bool", "int", "float", "string"]
    pinned, st2 = run_graph(mk([pg_node(sql, pin=s["snapshotId"])], []), lab.wb, "r2")
    assert st2 == "completed"
    pd.testing.assert_frame_equal(node_table(pinned, "r2", "db"), df)
    snap = c.get(f"/api/snapshots/{s['snapshotId']}").json()
    assert snap["manifest"]["result"]["sha256"] == s["resultSha256"] and "password" not in json.dumps(snap).lower()
    assert c.get("/api/snapshots/" + "0" * 64).json()["detail"]["code"] == "E_SRC_SNAPSHOT_UNAVAILABLE"


# ------------------------------------------------------------------------------------------------ A41: S3
def test_a41_s3_versioned_object_is_pinned_by_version_id_and_repeatable(lab):
    c = lab.s3.client()
    key = "work/versioned.csv"
    v1 = c.put_object(Bucket=BUCKET, Key=key, Body=b"id,x\n1,10\n2,20\n")["VersionId"]
    g = mk([("obj", "s3.csv_source", {"connection": "files", "key": key})], [])
    store, st = run_graph(g, lab.wb, "r1")
    assert st == "completed"
    s = node_summary(store, "r1", "obj")
    m = s["snapshot"]
    assert m["object"]["versionId"] == v1 and m["object"]["bucket"] == BUCKET and m["object"]["key"] == key and m["object"]["etag"]
    assert m["reproducibility"]["limited"] is False and m["reproducibility"]["level"] == "pinned_object_version" and "extract" not in m
    # a new version replaces the object: the key now holds different bytes
    v2 = c.put_object(Bucket=BUCKET, Key=key, Body=b"id,x\n1,10\n2,20\n3,30\n")["VersionId"]
    assert v2 != v1
    store, st2 = run_graph(g, lab.wb, "r2")
    assert node_summary(store, "r2", "obj")["rows"] == 3
    # repeating from the pinned snapshot reads exactly version v1
    store, st3 = run_graph(g, lab.wb, "r3", source_pins={"obj": s["snapshotId"]})
    s3 = node_summary(store, "r3", "obj")
    assert st3 == "completed" and s3["mode"] == "pinned" and s3["rows"] == 2 and s3["sha256"] == s["sha256"] and s3["object"]["versionId"] == v1
    # the node can also pin an explicit object version without any snapshot
    g2 = mk([("obj", "s3.csv_source", {"connection": "files", "key": key, "version_id": v1})], [])
    store, st4 = run_graph(g2, lab.wb, "r4")
    assert node_summary(store, "r4", "obj")["rows"] == 2
    # browsing shows both versions
    vs = s3m.versions(lab.reg, "files", key)["versions"]
    assert {v["versionId"] for v in vs} == {v1, v2} and [v["isLatest"] for v in vs].count(True) == 1


def test_a41_s3_unversioned_object_flags_limited_reproducibility_and_stores_a_copy(lab):
    c = lab.s3.client()
    key = "work/plain.csv"
    c.put_object(Bucket=PLAIN_BUCKET, Key=key, Body=b"id,x\n1,1\n2,2\n")
    g = mk([("obj", "s3.csv_source", {"connection": "plain", "key": key})], [])
    store, st = run_graph(g, lab.wb, "r1")
    assert st == "completed"
    s = node_summary(store, "r1", "obj")
    m = s["snapshot"]
    assert m["object"]["versionId"] is None and m["object"]["etag"]
    assert m["reproducibility"]["limited"] is True and "NOT VERSIONED" in m["reproducibility"]["statement"] and m["reproducibility"]["level"] == "materialized_copy_unversioned"
    assert "extract" in m
    c.put_object(Bucket=PLAIN_BUCKET, Key=key, Body=b"id,x\n1,1\n2,2\n3,3\n4,4\n")  # mutated in place
    store, st2 = run_graph(g, lab.wb, "r2")
    assert node_summary(store, "r2", "obj")["rows"] == 4
    store, st3 = run_graph(g, lab.wb, "r3", source_pins={"obj": s["snapshotId"]})
    assert node_summary(store, "r3", "obj")["rows"] == 2 and node_summary(store, "r3", "obj")["mode"] == "pinned"  # from the stored copy
    c.delete_object(Bucket=PLAIN_BUCKET, Key=key)
    store, st4 = run_graph(g, lab.wb, "r4", source_pins={"obj": s["snapshotId"]})
    assert st4 == "completed"  # the object is gone, the pinned run still works from the stored copy
    store, st5 = run_graph(g, lab.wb, "r5")
    assert st5 == "failed" and "E_SRC_NOT_FOUND" in store.get_run("r5")["error"]


# ------------------------------------------------------------------------------------------------ A42: DVC
@pytest.fixture
def dvc_lab(lab, tmp_path):
    info = make_dvc_repo(tmp_path / "dvcwork", [("v1", "id,x\n1,2\n2,3\n"), ("v2", "id,x\n1,2\n2,3\n3,4\n")])
    lab.reg.create("dvcdata", "DVC data repo", "dvc", {"repo": info["repo"], "default_rev": "main"}, {})
    lab.dvc = info
    return lab


def test_a42_dvc_revision_resolves_to_commit_md5_and_actual_content(dvc_lab):
    lab = dvc_lab
    info = lab.dvc
    d = dvc_mod.test_connection(lab.reg, "dvcdata")
    assert d["ok"] and d["head"] == info["commits"]["v2"]
    revs = dvc_mod.revisions(lab.reg, "dvcdata")["revisions"]
    assert [r["commit"] for r in revs][:2] == [info["commits"]["v2"], info["commits"]["v1"]] and "v1" in revs[1]["refs"]
    tree = dvc_mod.ls(lab.reg, "dvcdata", "v1", "")
    assert {"path": "data.csv", "type": "file", "size": 13, "dvcTracked": True} == {k: next(e for e in tree["entries"] if e["path"] == "data.csv")[k] for k in ("path", "type", "size", "dvcTracked")}
    g = mk([("d", "dvc.csv_source", {"connection": "dvcdata", "path": "data.csv", "rev": "v1"})], [])
    store, st = run_graph(g, lab.wb, "r1")
    assert st == "completed"
    s = node_summary(store, "r1", "d")
    f = s["snapshot"]["file"]
    assert f["commit"] == info["commits"]["v1"] and f["requestedRev"] == "v1" and f["path"] == "data.csv" and f["repo"] == info["repo"]
    assert f["dvcMd5"] == "ed4b04bf68b53fa729e32d00450f52f9" and f["dvcTracked"] is True  # md5 of "id,x\n1,2\n2,3\n"
    import hashlib

    assert f["sha256"] == hashlib.sha256(b"id,x\n1,2\n2,3\n").hexdigest() == s["sha256"] and s["rows"] == 2
    # the DVC cache was deleted, so the content really came from the DVC remote
    assert node_table(store, "r1", "d")["x"].tolist() == [2, 3]
    # a branch moves: 'main' resolves to the v2 commit (a new source revision)
    g2 = mk([("d", "dvc.csv_source", {"connection": "dvcdata", "path": "data.csv", "rev": "main"})], [])
    store, st2 = run_graph(g2, lab.wb, "r2")
    s2 = node_summary(store, "r2", "d")
    assert s2["snapshot"]["file"]["commit"] == info["commits"]["v2"] and s2["rows"] == 3 and s2["snapshot"]["file"]["dvcMd5"] != f["dvcMd5"]
    # repeat from the pinned identity: the branch name in the node is ignored in favour of the recorded commit
    store, st3 = run_graph(g2, lab.wb, "r3", source_pins={"d": s["snapshotId"]})
    s3 = node_summary(store, "r3", "d")
    assert st3 == "completed" and s3["rows"] == 2 and s3["mode"] == "pinned" and s3["snapshot"]["file"]["commit"] == info["commits"]["v1"]


def test_a42_dvc_missing_path_and_revision_are_source_errors(dvc_lab):
    lab = dvc_lab
    with pytest.raises(SourceError) as e:
        dvc_mod.read_file(lab.reg, "dvcdata", "v1", "nothing.csv")
    assert e.value.code == "E_SRC_NOT_FOUND"
    with pytest.raises(SourceError) as e:
        dvc_mod.read_file(lab.reg, "dvcdata", "no-such-tag", "data.csv")
    assert e.value.code == "E_SRC_NOT_FOUND"


def test_dvc_snapshot_drift_check(dvc_lab, client_factory):
    c = client_factory(dvc_lab)
    g = mk([("d", "dvc.csv_source", {"connection": "dvcdata", "path": "data.csv", "rev": "v1"})], [])
    store, _ = run_graph(g, dvc_lab.wb, "r1")
    sid = node_summary(store, "r1", "d")["snapshotId"]
    assert c.post(f"/api/snapshots/{sid}/check").json()["identical"] is True
    g2 = mk([("d", "dvc.csv_source", {"connection": "dvcdata", "path": "data.csv", "rev": "main"})], [])
    store, _ = run_graph(g2, dvc_lab.wb, "r2")
    sid2 = node_summary(store, "r2", "d")["snapshotId"]
    assert c.post(f"/api/snapshots/{sid2}/check").json()["identical"] is True  # main has not moved since


# ------------------------------------------------------------------------------------------------ A43
def journey_graph(**join_kw):
    q = {"base": {"name": "assays"}, "base_alias": "a",
         "columns": [{"table": "a", "column": "specimen_id"}, {"table": "a", "column": "response"}, {"table": "s", "column": "ph"}, {"table": "s", "column": "temp_c"}],
         "joins": [{"table": {"name": "specimens"}, "alias": "s", "type": "inner", "on": [{"left": {"table": "a", "column": "specimen_id"}, "right": {"table": "s", "column": "specimen_id"}}]}],
         "order_by": [{"by": "specimen_id"}, {"by": "response"}], "limit": 1000}
    return mk([("db", "postgres.query", {"connection": "lab", "mode": "visual", "query": q}),
               ("spectra", "s3.csv_source", {"connection": "files", "key": "features/spectra.csv"}),
               ("images", "s3.object_listing", {"connection": "files", "prefix": "images/", "key_regex": r"images/(?P<id>SP\d+)\.png"}),
               ("j1", "tabular.join", {"left_on": ["specimen_id"], "right_on": ["specimen_id"], "how": "left", **join_kw}),
               ("j2", "tabular.join", {"left_on": ["specimen_id"], "right_on": ["key_id"], "how": "left"})],
              [("db", "table", "j1", "left"), ("spectra", "table", "j1", "right"), ("j1", "table", "j2", "left"), ("images", "table", "j2", "right")])


def test_a43_join_database_rows_with_s3_csv_and_object_metadata(lab):
    g = journey_graph()
    rep = validate(g)
    assert rep.ok, [d.message for d in rep.errors]
    store, st = run_graph(g, lab.wb, "r1")
    assert st == "completed", store.get_run("r1")["error"]
    j1 = node_summary(store, "r1", "j1")
    # 63 assays (58 specimens; SP001..SP005 assayed twice) x spectra (55 specimens + stray SP999, unique ids)
    assert j1["cardinality"] == "many_to_one" and j1["leftKeys"] == ["specimen_id"]
    assert j1["left"]["rows"] == 63 and j1["left"]["distinctKeys"] == 58 and j1["left"]["duplicateKeyRows"] == 5
    assert j1["right"]["rows"] == 56 and j1["right"]["unique"] is True
    assert j1["left"]["unmatchedRows"] == 3 and {u["key"][0] for u in j1["left"]["unmatchedSample"]} == {"SP056", "SP057", "SP058"}
    assert j1["right"]["unmatchedRows"] == 1 and j1["right"]["unmatchedSample"][0]["key"] == ["SP999"]
    assert j1["resultRows"] == 63 and j1["rowMultiplication"] == 1.0 and j1["rowsMultiplied"] is False
    # where it ran and whether data moved
    ex = j1["execution"]
    assert "worker process" in ex["where"] and ex["pushdown"] is False
    assert {x["input"] for x in ex["dataMoved"]} == {"left", "right"} and ex["dataMoved"][0]["sources"][0]["kind"] == "postgres" and ex["dataMoved"][1]["sources"][0]["kind"] == "s3_object"
    # lineage: the joined table knows both sources and their snapshots
    j2 = node_summary(store, "r1", "j2")
    kinds = [s["kind"] for s in j2["lineage"]["left"]] + [s["kind"] for s in j2["lineage"]["right"]]
    assert kinds == ["postgres", "s3_object", "s3_listing"]
    assert all(s["snapshotId"] for s in j2["lineage"]["left"] + j2["lineage"]["right"])
    assert j2["right"]["rows"] == 55
    out = node_table(store, "r1", "j2")
    assert {"response", "ph", "absorbance_a", "size"} <= set(out.columns) and len(out) == 63
    assert out.loc[out["specimen_id"] == "SP057", "absorbance_a"].isna().all()  # unmatched left rows are kept by the left join, with missing features
    assert out.loc[out["specimen_id"] == "SP001", "size"].notna().all()


def test_a43_expectation_and_null_key_semantics(lab):
    # expecting one-to-one but the left key repeats: refused, naming the code
    g = journey_graph(expect="one_to_one")
    store, st = run_graph(g, lab.wb, "r1")
    assert st == "failed" and "E_JOIN_CARDINALITY" in store.get_run("r1")["error"]
    # NULL keys never match each other (unlike a raw pandas merge)
    left = "SELECT * FROM (VALUES ('a', 1), (NULL, 2), ('b', 3)) AS t(k, v)"
    right = "SELECT * FROM (VALUES ('a', 10), (NULL, 20)) AS t(k, w)"
    g2 = mk([("l", "postgres.query", {"connection": "lab", "mode": "sql", "sql": left}), ("r", "postgres.query", {"connection": "lab", "mode": "sql", "sql": right}),
             ("j", "tabular.join", {"left_on": ["k"], "right_on": ["k"], "how": "outer"})], [("l", "table", "j", "left"), ("r", "table", "j", "right")])
    store, st = run_graph(g2, lab.wb, "r2")
    assert st == "completed"
    s = node_summary(store, "r2", "j")
    assert s["left"]["nullKeyRows"] == 1 and s["right"]["nullKeyRows"] == 1
    out = node_table(store, "r2", "j")
    assert len(out) == 4  # a matched, b (left only), NULL-left, NULL-right: the two NULLs did not pair up
    assert sorted(out["v"].dropna().astype(int).tolist()) == [1, 2, 3] and sorted(out["w"].dropna().astype(int).tolist()) == [10, 20]


def test_join_key_type_mismatch_is_a_static_error(lab):
    g = mk([("l", "postgres.query", {"connection": "lab", "mode": "sql", "sql": "SELECT 1 AS k"}), ("r", "postgres.query", {"connection": "lab", "mode": "sql", "sql": "SELECT 'x' AS k"}),
            ("j", "tabular.join", {"left_on": ["k"], "right_on": ["k"]})], [("l", "table", "j", "left"), ("r", "table", "j", "right")])
    assert "E_JOIN_KEY_TYPE" in [d.code for d in validate(g).errors]
