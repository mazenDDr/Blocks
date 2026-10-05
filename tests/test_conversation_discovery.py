"""Bounded PK pagination over real native serving heads, with no model execution."""
import json

import pytest
from fastapi.testclient import TestClient

from control.app import create_app
from production.conversations import action
from production.discovery import discover
from production.models import ReleaseCreate
from production.pipeline import ProductionError
from tabular.core import dumps
from test_conversation_actions import body
from test_production_conversation import setup, req, head


def test_native_sessions_keyset_prefix_user_release_isolation_reset_fork_restart(tmp_path):
    lab, rt, version, release = setup(tmp_path, capture=False)
    assert discover(rt.ps, release["id"], "alice")["sessions"] == []  # Warmup doesn't create a session.
    names = ["A", "A0", "A_1", "B", "a", "a-1", "a_1", "a_2"]
    for i, name in enumerate(reversed(names)):
        assert rt.predict("local", "lab", req(f"r{i}", "SYNTHETIC private text", session=name))["status"] == 200
    rt.predict("local", "lab", req("bob", user="bob", session="SECRET_other_user"))
    action(rt, release["id"], "reset", body(rt, release, "reset", session="B"))
    fork = action(rt, release["id"], "fork", body(rt, release, "fork", session="A", destination="branch"))
    other = rt.create_release(ReleaseCreate(versionId=version["id"], config={"maxBatch": 1, "sessionMode": "conversation", "concurrency": 1}))
    rt.activate(other["id"], release["id"])
    rt.predict("local", "lab", req("new-release", session="SECRET_other_release"))
    # Read exactly the canonical committed metadata; don't load checkpoint content.
    page = discover(rt.ps, release["id"], "alice", limit=3)
    found = []
    while True:
        assert page["user"] == "alice" and page["releaseId"] == release["id"] and page["readOnly"]
        found.extend(page["sessions"])
        if not page["nextAfter"]:
            break
        page = discover(rt.ps, release["id"], "alice", limit=3, after=page["nextAfter"])
    assert [x["session"] for x in found] == sorted(names + ["branch"])
    by_name = {x["session"]: x for x in found}
    assert by_name["B"] == {"session": "B", "status": "reset", "head": head(rt, release, session="B")}
    assert by_name["branch"]["head"] == fork["head"] and fork["head"]["lastRequestId"] is None
    assert "private text" not in json.dumps(found) and "SECRET" not in json.dumps(found)
    assert [x["session"] for x in discover(rt.ps, release["id"], "alice", prefix="a_")["sessions"]] == ["a_1", "a_2"]
    assert discover(rt.ps, release["id"], "alice", prefix="no_match")["sessions"] == []
    from production.runtime import ProductionRuntime
    restarted = ProductionRuntime(lab.store)
    assert discover(restarted.ps, release["id"], "alice", limit=100)["sessions"] == found


def test_discovery_uses_index_bound_and_never_verifies_or_executes_native_pipeline(tmp_path, monkeypatch):
    lab, rt, _, release = setup(tmp_path)
    rt.predict("local", "lab", req("one"))
    before = rt.ps.query("SELECT * FROM agent_sessions"), rt.ps.query("SELECT * FROM lifecycle"), set(lab.store.artifact_dir.iterdir())
    query = rt.ps.query
    seen = []
    def observed(sql, args=()):
        if "FROM agent_sessions" in sql:
            assert "scope>=?" in sql and "scope<?" in sql and "scope>?" in sql and "LIMIT ?" in sql
            plan = query("EXPLAIN QUERY PLAN " + sql, args)
            assert any("SEARCH agent_sessions USING INDEX" in x["detail"] for x in plan)
            seen.append(args[-1])
        return query(sql, args)
    monkeypatch.setattr(rt.ps, "query", observed)
    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not load state or invoke/verify providers.")
    monkeypatch.setattr(rt, "pipeline", forbidden)
    result = discover(rt.ps, release["id"], "alice", limit=1)
    assert len(result["sessions"]) == 1 and seen == [2]
    monkeypatch.setattr(rt.ps, "query", query)
    assert before == (rt.ps.query("SELECT * FROM agent_sessions"), rt.ps.query("SELECT * FROM lifecycle"), set(lab.store.artifact_dir.iterdir()))


@pytest.mark.parametrize("changes", [{"user": "../bob"}, {"limit": 0}, {"limit": 101}, {"limit": True}, {"prefix": "%"}, {"after": ""}, {"after": "../x"}])
def test_query_contract_refuses_invalid_pagination(tmp_path, changes):
    _, rt, _, release = setup(tmp_path)
    values = dict(user="alice", limit=25, prefix="", after=None)
    values.update(changes)
    with pytest.raises(ProductionError) as error:
        discover(rt.ps, release["id"], **values)
    assert error.value.code == "E_SESSION_QUERY"


def test_malformed_in_scope_row_refused_and_corrupt_checkpoint_not_loaded(tmp_path):
    lab, rt, _, release = setup(tmp_path)
    trace = rt.predict("local", "lab", req("one"))
    sha = trace["conversationState"]["checkpointSha256"]
    lab.store.path_of(sha).write_bytes(b"corrupt native checkpoint")
    result = discover(rt.ps, release["id"], "alice")
    assert result["sessions"][0]["head"]["checkpointSha256"] == sha  # Metadata is not a claim of integrity.
    scope = dumps([release["id"], "alice"])[:-1] + ', "invalid'
    rt.ps.query("INSERT INTO agent_sessions VALUES(?,1,NULL,NULL)", (scope,))
    with pytest.raises(ProductionError) as error:
        discover(rt.ps, release["id"], "alice")
    assert error.value.code == "E_SESSION_INTEGRITY"


def test_http_token_query_and_stateless_release_contracts(tmp_path, monkeypatch):
    lab, rt, _, release = setup(tmp_path, capture=False)
    rt.predict("local", "lab", req("one"))
    monkeypatch.setenv("VOID_API_TOKEN", "SYNTHETIC-discovery-test-token-123456")
    app = create_app(lab.wb)
    endpoint = f"/api/production/releases/{release['id']}/conversations"
    headers = {"Authorization": "Bearer SYNTHETIC-discovery-test-token-123456"}
    with TestClient(app) as client:
        assert client.get(endpoint + "?user=alice").status_code == 401
        result = client.get(endpoint + "?user=alice&limit=1", headers=headers)
        assert result.status_code == 200, result.text
        assert result.json()["sessions"][0]["session"] == "chat"
        assert client.get(endpoint + "?user=bob", headers=headers).json()["sessions"] == []
        for args in ("limit=0", "limit=101", "user=../x", "prefix=%25", "after="):
            assert client.get(endpoint + "?" + args, headers=headers).status_code == 422
        state = client.get(f"/api/production/releases/{release['id']}/conversation?user=alice&session=chat", headers=headers).json()
        assert state["state"]["n"] == 1 and state["head"] == result.json()["sessions"][0]["head"]
    # Real stateless release from the existing native agent test helper.
    from test_production_agent import setup as stateless_setup
    _, stateless, _, stateless_release = stateless_setup(tmp_path / "stateless")
    with pytest.raises(ProductionError) as error:
        discover(stateless.ps, stateless_release["id"], "alice")
    assert error.value.code == "E_RELEASE_CONFIG"
