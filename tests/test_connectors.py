"""Connections, secrets by reference, the PostgreSQL visual query builder, previews and source failures (A17, A39, A40).

These tests run against REAL local engines: PostgreSQL 16 (pgserver) and an S3-compatible server (moto). See connectors/localtest.py."""
import json
import os

import pandas as pd
import psycopg
import pytest

from connected_helpers import BUCKET, S3_KEY_ENV, S3_SECRET_ENV, mk, node_summary, node_table, run_graph
from connectors import postgres as pg
from connectors import s3 as s3m
from connectors import synthetic
from connectors.errors import SourceError
from connectors.pg_query import QuerySpec, compile_query
from connectors.registry import ConnectionRegistry


def spec(d):
    return QuerySpec.model_validate(d)


ASSAY_QUERY = {
    "base": {"schema": "public", "name": "assays"}, "base_alias": "a",
    "columns": [{"table": "a", "column": "specimen_id"}, {"table": "a", "column": "response"}, {"table": "s", "column": "ph"}, {"table": "s", "column": "site", "alias": "location"}],
    "joins": [{"table": {"schema": "public", "name": "specimens"}, "alias": "s", "type": "inner", "on": [{"left": {"table": "a", "column": "specimen_id"}, "right": {"table": "s", "column": "specimen_id"}}]}],
    "filters": [{"column": {"table": "s", "column": "ph"}, "op": ">=", "value": {"type": "float", "value": 7.0}},
                {"column": {"table": "s", "column": "site"}, "op": "in", "values": [{"type": "string", "value": "north"}, {"type": "string", "value": "south"}]}],
    "order_by": [{"by": "specimen_id", "direction": "asc"}, {"by": "response", "direction": "desc"}], "limit": 500}


# ------------------------------------------------------------------------------------------------ A39
def test_a39_visual_query_result_and_schema_match_the_declared_native_query(lab):
    comp = compile_query(spec(ASSAY_QUERY))
    # parameterized: values are bound, never interpolated into the text
    assert "7.0" not in comp.sql.replace("LIMIT", "") and "north" not in comp.sql and comp.sql.count("%s") == len(comp.params) == 3
    assert comp.params == [7.0, ["north", "south"], 500] and comp.param_types == ["double precision", "text[]", "integer"]
    r = pg.run_query(lab.reg, "lab", mode="visual", spec=spec(ASSAY_QUERY), sql=None, params=None, limit=500)
    native_sql = ('SELECT a.specimen_id AS specimen_id, a.response AS response, s.ph AS ph, s.site AS location FROM public.assays a '
                  'INNER JOIN public.specimens s ON a.specimen_id = s.specimen_id WHERE s.ph >= 7.0 AND s.site IN (\'north\', \'south\') '
                  'ORDER BY specimen_id ASC, response DESC LIMIT 500')
    with psycopg.connect(lab.pg.admin_uri) as c:
        cur = c.execute(native_sql)
        names = [d.name for d in cur.description]
        native = pd.DataFrame(cur.fetchall(), columns=names)
    got = r["df"].reset_index(drop=True)
    assert list(got.columns) == names == comp.columns
    assert len(got) == len(native) > 0
    pd.testing.assert_frame_equal(got, native.astype({"specimen_id": "str", "location": "str"}), check_dtype=False)
    assert [c["dtype"] for c in r["schema"]] == ["string", "float", "float", "string"]
    # static schema probing (used by graph validation) agrees with the executed result
    assert [c["name"] for c in pg.probe_schema(lab.reg, "lab", comp)] == names


def test_a39_visual_query_runs_in_a_graph_without_any_extraction_script(lab, tmp_path):
    g = mk([("q", "postgres.query", {"connection": "lab", "mode": "visual", "query": ASSAY_QUERY})], [])
    store, status = run_graph(g, lab.wb, "r1")
    assert status == "completed"
    s = node_summary(store, "r1", "q")
    assert s["mode"] == "live" and s["sql"].startswith("SELECT") and s["params"][0]["type"] == "double precision"
    df = node_table(store, "r1", "q")
    assert list(df.columns) == ["specimen_id", "response", "ph", "location"] and len(df) == s["rows"]


def test_values_are_bound_not_interpolated_injection_is_inert(lab):
    evil = "x'; DROP TABLE specimens; --"
    q = {**ASSAY_QUERY, "filters": [{"column": {"table": "s", "column": "site"}, "op": "=", "value": {"type": "string", "value": evil}}]}
    r = pg.run_query(lab.reg, "lab", mode="visual", spec=spec(q), sql=None, params=None, limit=10)
    assert len(r["df"]) == 0 and evil not in r["query"]["text"]
    with psycopg.connect(lab.pg.admin_uri) as c:
        assert c.execute("SELECT count(*) FROM specimens").fetchone()[0] == 60
    # identifiers are quoted: a hostile identifier cannot break out of the quotes
    q2 = {**ASSAY_QUERY, "columns": [{"table": "a", "column": 'r" FROM assays; DROP TABLE a;--'}], "order_by": []}
    with pytest.raises(SourceError) as e:
        pg.run_query(lab.reg, "lab", mode="visual", spec=spec(q2), sql=None, params=None, limit=10)
    assert e.value.code == "E_SRC_NOT_FOUND"
    with psycopg.connect(lab.pg.admin_uri) as c:
        assert c.execute("SELECT count(*) FROM assays").fetchone()[0] == 63


def test_typed_comparison_mismatch_is_reported_before_data_moves(lab):
    q = {**ASSAY_QUERY, "filters": [{"column": {"table": "s", "column": "site"}, "op": "=", "value": {"type": "int", "value": 5}}]}
    with pytest.raises(SourceError) as e:
        pg.run_query(lab.reg, "lab", mode="visual", spec=spec(q), sql=None, params=None, limit=10)
    assert e.value.code == "E_SRC_TYPE_MISMATCH"
    with pytest.raises(SourceError) as e2:
        compile_query(spec({**ASSAY_QUERY, "filters": [{"column": {"table": "s", "column": "ph"}, "op": ">", "value": {"type": "float", "value": "abc"}}]}))
    assert e2.value.code == "E_SRC_TYPE_MISMATCH"


def test_aggregates_group_order_and_limit(lab):
    q = {"base": {"name": "specimens"}, "base_alias": "s", "group_by": [{"table": "s", "column": "site"}],
         "aggregates": [{"fn": "count", "alias": "n"}, {"fn": "avg", "column": {"table": "s", "column": "ph"}, "alias": "mean_ph"}],
         "order_by": [{"by": "site"}], "limit": 10}
    r = pg.run_query(lab.reg, "lab", mode="visual", spec=spec(q), sql=None, params=None, limit=10)
    df = r["df"]
    assert list(df["site"]) == ["east", "north", "south"] and df["n"].sum() == 60
    sp = synthetic.specimens()
    assert df.set_index("site")["mean_ph"].round(6).to_dict() == sp.groupby("site")["ph"].mean().round(6).to_dict()


def test_in_database_join_reports_cardinality_and_unmatched_records(lab):
    q = {"base": {"name": "specimens"}, "base_alias": "s", "columns": [{"table": "s", "column": "specimen_id"}],
         "joins": [{"table": {"name": "assays"}, "alias": "a", "type": "left", "on": [{"left": {"table": "s", "column": "specimen_id"}, "right": {"table": "a", "column": "specimen_id"}}]}],
         "order_by": [{"by": "specimen_id"}], "limit": 1000}
    rep = pg.join_reports(lab.reg, "lab", spec(q))[0]
    assert rep["cardinality"] == "one_to_many" and rep["rowMultiplicationPossible"]
    assert (rep["leftRows"], rep["rightRows"]) == (60, 63) and rep["unmatchedLeftRows"] == 2 and rep["unmatchedRightRows"] == 0
    assert rep["rightDuplicateKeyRows"] == 63 - 58  # 58 distinct specimens have assays
    r = pg.run_query(lab.reg, "lab", mode="visual", spec=spec(q), sql=None, params=None, limit=1000)
    assert len(r["df"]) == 63 + 2  # the left join keeps the 2 specimens without assays


# ------------------------------------------------------------------------------------------------ raw SQL mode
def test_raw_sql_is_read_only_bounded_and_parameterized(lab):
    r = pg.run_query(lab.reg, "lab", mode="sql", spec=None, sql="SELECT specimen_id, ph FROM specimens WHERE site = %(site)s ORDER BY specimen_id",
                     params=[{"name": "site", "type": "string", "value": "north"}], limit=5)
    assert len(r["df"]) == 5 and r["truncated"] and r["ordered"] and r["query"]["params"][0]["value"] == "north"
    for bad in ("SELECT nextval('assays_assay_id_seq')", ):
        with pytest.raises(SourceError) as e:
            pg.run_query(lab.reg, "lab", mode="sql", spec=None, sql=bad, params=None, limit=5)
        assert e.value.code == "E_SRC_READONLY_VIOLATION"
    for bad in ("DELETE FROM assays", "SELECT 1; DROP TABLE assays", "INSERT INTO specimens VALUES ('x','x','2025-01-01',1,1)"):
        with pytest.raises(SourceError):
            pg.run_query(lab.reg, "lab", mode="sql", spec=None, sql=bad, params=None, limit=5)
    with psycopg.connect(lab.pg.admin_uri) as c:
        assert c.execute("SELECT count(*) FROM assays").fetchone()[0] == 63
    with pytest.raises(SourceError) as e:
        pg.run_query(lab.reg, "lab", mode="sql", spec=None, sql="SELECT pg_sleep(5)", params=None, limit=5, timeout_ms=300)
    assert e.value.code == "E_SRC_QUERY_TIMEOUT"


# ------------------------------------------------------------------------------------------------ A40
def test_a40_preview_row_and_time_bounds_are_enforced_at_the_source(lab):
    p = pg.preview(lab.reg, "lab", mode="sql", spec=None, sql="SELECT g FROM generate_series(1, 100000) g", params=None, rows=100000)
    assert p["rowCount"] == pg.MAX_PREVIEW_ROWS == 200 and p["truncated"] and p["bounds"] == {"rowLimit": 200, "timeoutMs": 5000, "readOnly": True, "isolation": "REPEATABLE READ"}
    p2 = pg.preview(lab.reg, "lab", mode="visual", spec=spec(ASSAY_QUERY), sql=None, params=None, rows=7)
    assert p2["rowCount"] <= 7 and "LIMIT" in p2["query"]["text"]
    with pytest.raises(SourceError) as e:
        pg.preview(lab.reg, "lab", mode="sql", spec=None, sql="SELECT pg_sleep(10)", params=None, rows=5, timeout_ms=200)
    assert e.value.code == "E_SRC_QUERY_TIMEOUT"
    # S3: ranged read with a byte bound
    sp = s3m.preview_object(lab.reg, "files", "features/spectra.csv", rows=1000, max_bytes=1024)
    assert sp["bounds"]["bytesRead"] <= 1024 and sp["truncated"] and sp["format"] == "csv" and len(sp["rows"]) < 56
    assert len(s3m.browse(lab.reg, "files", "images/", max_keys=10_000)["objects"]) <= s3m.MAX_LIST
    page = s3m.browse(lab.reg, "files", "images/", max_keys=10)
    assert len(page["objects"]) == 10 and page["nextToken"] and page["bounds"] == {"maxKeys": 10}
    page2 = s3m.browse(lab.reg, "files", "images/", token=page["nextToken"], max_keys=10)
    assert page2["objects"][0]["key"] > page["objects"][-1]["key"]


def test_a40_permission_auth_network_and_missing_resource_are_distinct_recoverable_errors(lab, client_factory):
    c = client_factory(lab)
    # permission: the reader role lacks SELECT on secret_notes; discovery shows it, preview is refused with a permission code
    d = c.get("/api/connections/reader/discover").json()
    assert {t["name"]: t["selectable"] for t in d["tables"]} == {"assays": True, "secret_notes": False, "specimens": True}
    r = c.post("/api/connections/reader/preview", json={"mode": "sql", "sql": "SELECT * FROM secret_notes"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "E_SRC_PERMISSION" and r.json()["detail"]["recoverable"] is True
    assert "hint" in r.json()["detail"]
    # authentication: a role that does not exist
    c.post("/api/connections", json={"id": "badrole", "name": "bad", "type": "postgres", "settings": lab.pg.settings("nobody")})
    t = c.post("/api/connections/badrole/test").json()
    assert t["ok"] is False and t["error"]["code"] == "E_SRC_AUTH"
    # network: nothing listens there
    c.post("/api/connections", json={"id": "down", "name": "down", "type": "postgres", "settings": {"host": "127.0.0.1", "port": 9, "dbname": "x", "user": "x", "connect_timeout_s": 2}})
    t = c.post("/api/connections/down/test").json()
    assert t["ok"] is False and t["error"]["code"] == "E_SRC_NETWORK"
    c.post("/api/connections", json={"id": "s3down", "name": "down", "type": "s3", "settings": {"bucket": "nobucket", "endpoint_url": "http://127.0.0.1:9"},
                                       "secrets": {"access_key_id": {"kind": "env", "name": S3_KEY_ENV}, "secret_access_key": {"kind": "env", "name": S3_SECRET_ENV}}})
    assert c.post("/api/connections/s3down/test").json()["error"]["code"] == "E_SRC_NETWORK"
    # missing resources
    r = c.get("/api/connections/lab/discover", params={"schema": "public", "table": "no_such_table"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "E_SRC_NOT_FOUND"
    c.post("/api/connections", json={"id": "nobkt", "name": "x", "type": "s3", "settings": {"bucket": "does-not-exist", "endpoint_url": lab.s3.endpoint},
                                     "secrets": {"access_key_id": {"kind": "env", "name": S3_KEY_ENV}, "secret_access_key": {"kind": "env", "name": S3_SECRET_ENV}}})
    assert c.post("/api/connections/nobkt/test").json()["error"]["code"] == "E_SRC_NOT_FOUND"
    r = c.post("/api/connections/files/preview", json={"key": "features/missing.csv"})
    assert r.status_code == 404 and r.json()["detail"]["code"] == "E_SRC_NOT_FOUND"
    # unresolved secret reference
    c.post("/api/connections", json={"id": "nosecret", "name": "x", "type": "s3", "settings": {"bucket": BUCKET, "endpoint_url": lab.s3.endpoint},
                                     "secrets": {"access_key_id": {"kind": "env", "name": "VOID_TEST_UNSET_VAR"}, "secret_access_key": {"kind": "env", "name": "VOID_TEST_UNSET_VAR"}}})
    t = c.post("/api/connections/nosecret/test").json()
    assert t["ok"] is False and t["error"]["code"] == "E_SRC_SECRET_UNRESOLVED"
    # the project (connection registry) survives every failure
    assert len(c.get("/api/connections").json()["connections"]) >= 8
    # failures are visible in the connection's last test
    assert c.get("/api/connections/down").json()["lastTest"]["error"]["code"] == "E_SRC_NETWORK"


def test_s3_error_classifier_maps_error_codes_without_a_real_iam_policy():
    """moto does not enforce IAM, so permission/credential failures are checked on the classifier with synthesized botocore errors."""
    from botocore.exceptions import ClientError

    def ce(code, status):
        return ClientError({"Error": {"Code": code, "Message": "m"}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject")

    row = {"id": "x"}
    assert s3m.classify(ce("AccessDenied", 403), row, "s3://b/k", []).code == "E_SRC_PERMISSION"
    assert s3m.classify(ce("InvalidAccessKeyId", 403), row, "s3://b/k", []).code == "E_SRC_AUTH"
    assert s3m.classify(ce("SignatureDoesNotMatch", 403), row, "s3://b/k", []).code == "E_SRC_AUTH"
    assert s3m.classify(ce("NoSuchKey", 404), row, "s3://b/k", []).code == "E_SRC_NOT_FOUND"


def test_failed_source_in_a_run_names_the_node_and_code_and_keeps_the_project(lab):
    g = mk([("q", "postgres.query", {"connection": "reader", "mode": "sql", "sql": "SELECT * FROM secret_notes"})], [])
    from graph_core.validate import validate

    rep = validate(g)  # static validation already probes the source and reports a permission problem
    assert [d.code for d in rep.errors] == ["E_SRC_PERMISSION"]
    assert "hint" not in rep.errors[0].message and "Ask for read access" in rep.errors[0].message


# ------------------------------------------------------------------------------------------------ A17 secrets
def test_a17_secrets_are_references_never_values(lab, client_factory, monkeypatch, tmp_path):
    c = client_factory(lab)
    secret = "pw-NEVER-LEAK-0xC0FFEE"
    monkeypatch.setenv("VOID_TEST_PGPW", secret)
    # a literal secret is rejected: settings are strict and secrets must be reference objects
    r = c.post("/api/connections", json={"id": "lit", "name": "x", "type": "postgres", "settings": {**lab.pg.settings(), "password": secret}})
    assert r.status_code == 422 and secret not in r.text and r.json()["detail"]["code"] == "E_SRC_SETTINGS_INVALID"
    r = c.post("/api/connections", json={"id": "lit", "name": "x", "type": "postgres", "settings": lab.pg.settings(), "secrets": {"password": secret}})
    assert r.status_code == 422 and secret not in r.text
    # a secrets file inside the project/workbench is rejected
    inside = lab.wb / "secrets.json"
    inside.write_text(json.dumps({"k": secret}))
    r = c.post("/api/connections", json={"id": "lit", "name": "x", "type": "postgres", "settings": lab.pg.settings(), "secrets": {"password": {"kind": "file", "path": str(inside), "key": "k"}}})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "E_SRC_SECRET_REF_INVALID"
    # accepted references: env var and a secrets file outside the project
    outside = tmp_path / "outside" / "secrets.json"
    outside.parent.mkdir()
    outside.write_text(json.dumps({"db": secret}))
    outside.chmod(0o600)
    for cid, ref in (("viaenv", {"kind": "env", "name": "VOID_TEST_PGPW"}), ("viafile", {"kind": "file", "path": str(outside), "key": "db"})):
        r = c.post("/api/connections", json={"id": cid, "name": cid, "type": "postgres", "settings": lab.pg.settings(), "secrets": {"password": ref}})
        assert r.status_code == 201, r.text
        assert r.json()["secrets"]["password"]["resolvable"] is True
        assert c.post(f"/api/connections/{cid}/test").json()["ok"] is True  # trust auth ignores the password, but the reference resolved and was used
    # a project that uses the connection; run it; export it
    g = mk([("q", "postgres.query", {"connection": "viafile", "mode": "sql", "sql": "SELECT specimen_id FROM specimens", "limit": 5})], [])
    c.put("/api/projects/p1", json={"graph": g.to_json()})
    rid = c.post("/api/runs", json={"projectId": "p1", "config": {}}, headers={"Idempotency-Key": "k1"}).json()["runId"]
    import time

    for _ in range(300):
        if c.get(f"/api/runs/{rid}").json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert c.get(f"/api/runs/{rid}").json()["status"] == "completed"
    bundle = c.get("/api/projects/p1/export/bundle")
    assert bundle.status_code == 200 and bundle.json()["requirements"]["connections"][0]["secretsRequired"] == ["password"]
    # the secret value, its env var name and the secrets-file path appear in NO API response, export, event, artifact or database file
    texts = [r.text for r in (c.get("/api/connections"), c.get("/api/connections/viaenv"), c.get("/api/connections/viafile"), bundle, c.get("/api/runs"), c.get(f"/api/runs/{rid}"),
                              c.get(f"/api/runs/{rid}/events", params={"after": -1}))]
    for t in texts:
        assert secret not in t and str(outside) not in t
    assert "VOID_TEST_PGPW" not in bundle.text and str(outside) not in bundle.text
    for root, _, files in os.walk(lab.wb):
        for f in files:
            if f == "secrets.json":
                continue
            data = open(os.path.join(root, f), "rb").read()
            assert secret.encode() not in data, f"secret value leaked into {f}"
            assert str(outside).encode() not in data or f == "connections.db", f
    assert b"VOID_TEST_UNUSED" not in open(lab.wb / "connections.db", "rb").read()
    # error messages never echo secrets either
    monkeypatch.setenv("VOID_TEST_PGPW", secret)
    c.post("/api/connections", json={"id": "badpw", "name": "x", "type": "postgres", "settings": lab.pg.settings("nobody"), "secrets": {"password": {"kind": "env", "name": "VOID_TEST_PGPW"}}})
    assert secret not in c.post("/api/connections/badpw/test").text


def test_connection_crud_validation(lab, client_factory):
    c = client_factory(lab)
    assert c.post("/api/connections", json={"id": "Bad Id", "name": "x", "type": "postgres", "settings": lab.pg.settings()}).status_code == 422
    assert c.post("/api/connections", json={"id": "lab", "name": "dup", "type": "postgres", "settings": lab.pg.settings()}).status_code == 422
    assert c.get("/api/connections/nope").json()["detail"]["code"] == "E_SRC_CONNECTION_NOT_FOUND"
    r = c.put("/api/connections/lab", json={"name": "Renamed"})
    assert r.json()["name"] == "Renamed" and r.json()["settings"]["dbname"] == "postgres"
    t = c.post("/api/connections/lab/test").json()
    assert t["ok"] and t["serverVersion"].startswith("PostgreSQL 16") and t["readOnlyEnforced"] is True and t["capabilities"]["writes"] is False
    assert c.post("/api/connections/files/test").json()["versioning"] == "Enabled"
    assert c.post("/api/connections/plain/test").json()["versioned"] is False
    assert c.delete("/api/connections/down").status_code == 404
