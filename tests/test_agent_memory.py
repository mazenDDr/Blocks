"""Memory as a visible subsystem: stores, composable policy stages with per-record decisions, writes with audit, scope isolation."""
import time

import pytest

from agent import samples as sm
from agent.memory import MemoryStore
from agent.policy import run_policy, short_term_records, validate_stage
from agent.spec import Policy
from agent.validate import check_policy
from agent_helpers import Lab

NOW = 1_700_000_000.0
DAY = 86400.0


def rec(i, text, days=0, imp=0.5, ns="lab", scope="user:a", kind="semantic", meta=None, store="long_term", expires=None):
    return {"id": i, "store": store, "namespace": ns, "scope": scope, "kind": kind, "text": text, "importance": imp, "created_at": NOW - days * DAY, "metadata": meta or {},
            "generated": False, "expires_at": expires, "version": 1, "source": {}}


def policy(*stages, query="", dim=128):
    return Policy.model_validate({"id": "p", "query": query, "embeddings": {"provider": "local_hash", "dimension": dim}, "stages": [{"id": s[0], "op": s[1], "config": s[2]} for s in stages]})


def decisions(app, stage):
    return {d["id"]: d for s in app["stages"] if s["id"] == stage for d in s["decisions"]}


# ------------------------------------------------------------------------------------------------ long-term store
def test_long_term_records_have_metadata_scope_versions_and_an_audit_trail(tmp_path):
    m = MemoryStore(tmp_path)
    a = m.put_record({"id": "r1", "namespace": "lab", "scope": "user:a", "kind": "semantic", "text": "first", "importance": 0.7, "metadata": {"entity": "x"}, "generated": True},
                     run_id="run1", thread_id="t1", node="writer", evidence="from the question", validation={"ok": True})
    b = m.put_record({"id": "r1", "namespace": "lab", "scope": "user:a", "text": "second"}, run_id="run2", node="writer2")
    assert (a["version"], b["version"]) == (1, 2) and b["created_at"] == a["created_at"]
    w = m.writes(record_id="r1")
    assert [x["op"] for x in w] == ["insert", "update"] and w[0]["node"] == "writer" and w[0]["run_id"] == "run1" and w[0]["evidence"] == "from the question"
    assert w[0]["generated"] == 1 and w[1]["old"]["text"] == "first" and w[1]["new"]["text"] == "second"  # old and new values
    assert m.delete_record("r1", run_id="run3") and m.get_record("r1")["deleted_at"] and not m.list_records()
    assert m.writes(record_id="r1")[-1]["op"] == "delete" and m.list_records(include_deleted=True)
    assert MemoryStore(tmp_path).get_record("r1")["text"] == "second"  # persisted in the workbench database


# ------------------------------------------------------------------------------------------------ policy stages and their decisions
def test_retrieve_stage_reports_why_each_stored_record_is_ineligible():
    lt = [rec("ok", "kept", scope="user:a"), rec("other_user", "x", scope="user:b"), rec("other_ns", "x", ns="personal"), rec("other_kind", "x", kind="episodic"),
          rec("old", "x", expires=NOW - 1), rec("short", "x", store="short_term", ns="thread:t", scope="thread:t", kind="message")]
    p = policy(("sel", "retrieve", {"sources": ["long_term"], "namespaces": ["lab"], "scopes": ["user:{uid}"], "kinds": ["semantic"]}))
    app = run_policy(p, [r for r in lt if r["store"] == "long_term"], [r for r in lt if r["store"] == "short_term"], {"uid": "a"}, now=NOW)
    d = decisions(app, "sel")
    assert app["final"] == ["ok"]
    assert "scope filter" in d["other_user"]["reason"] and "'user:b'" in d["other_user"]["reason"]
    assert "namespace filter" in d["other_ns"]["reason"] and "kind filter" in d["other_kind"]["reason"] and "expired" in d["old"]["reason"] and "store filter" in d["short"]["reason"]
    # every record keeps its stage and reason; none of the excluded ones can show up as included
    assert {k for k, r in app["records"].items() if r["status"] == "included"} == {"ok"}
    assert all(r["excludedAt"]["stage"] == "sel" for k, r in app["records"].items() if k != "ok")


def test_retrieve_by_similarity_recency_and_limit_explain_the_cutoff():
    lt = [rec("a", "enzyme assay temperature limit", 5), rec("b", "pipette tips order", 1), rec("c", "enzyme storage freezer", 9), rec("d", "meeting thursday", 2)]
    app = run_policy(policy(("sel", "retrieve", {"method": "similarity", "k": 2}), query="enzyme assay temperature"), lt, [], {}, now=NOW)
    assert app["final"] == ["a", "c"]
    d = decisions(app, "sel")
    assert "ranked below the selected limit" in d["b"]["reason"] and "limit k=2" in d["b"]["reason"] and app["records"]["a"]["scores"]["relevance"] > app["records"]["c"]["scores"]["relevance"]
    app = run_policy(policy(("sel", "retrieve", {"method": "recency", "k": 2})), lt, [], {}, now=NOW)
    assert app["final"] == ["b", "d"] and "older than the selected window" in decisions(app, "sel")["c"]["reason"]
    app = run_policy(policy(("sel", "retrieve", {"method": "similarity", "min_score": 0.5}), query="enzyme assay temperature"), lt, [], {}, now=NOW)
    assert app["final"] == ["a"] and "below the minimum score" in decisions(app, "sel")["c"]["reason"]


def test_filter_by_metadata_and_age_with_a_visual_predicate():
    lt = [rec("a", "x", 1, meta={"entity": "kinase"}), rec("b", "x", 1, meta={"entity": "other"}), rec("c", "x", 40, meta={"entity": "kinase"})]
    w = {"all": [{"field": "metadata.entity", "op": "==", "value": "kinase"}, {"field": "age_days", "op": "<", "value": 30}]}
    app = run_policy(policy(("sel", "retrieve", {}), ("f", "filter", {"where": w})), lt, [], {}, now=NOW)
    assert app["final"] == ["a"]
    assert "excluded by filter" in decisions(app, "f")["b"]["reason"] and "metadata.entity == 'kinase'" in decisions(app, "f")["b"]["reason"]
    assert check_policy(policy(("sel", "retrieve", {}), ("f", "filter", {"where": {"field": "nonsense", "op": "==", "value": 1}})), __import__("agent.spec", fromlist=["AgentSpec"]).AgentSpec())[0][0] == "E_PREDICATE_FIELD"


def test_rank_weights_change_the_order_and_the_cutoff_reason_cites_the_scores():
    lt = [rec("old_important", "enzyme limit", 100, 0.95), rec("new_trivial", "enzyme chatter", 0, 0.1), rec("mid", "enzyme note", 20, 0.5)]
    by_recency = run_policy(policy(("s", "retrieve", {}), ("r", "rank", {"weights": {"recency": 1, "relevance": 0, "importance": 0}, "limit": 2}), query="enzyme"), lt, [], {}, now=NOW)
    assert by_recency["final"] == ["new_trivial", "mid"] and "ranked below the selected limit: rank 3 of 3, limit 2" in decisions(by_recency, "r")["old_important"]["reason"]
    by_importance = run_policy(policy(("s", "retrieve", {}), ("r", "rank", {"weights": {"recency": 0, "relevance": 0, "importance": 1}, "limit": 2}), query="enzyme"), lt, [], {}, now=NOW)
    assert by_importance["final"] == ["old_important", "mid"]
    sc = by_importance["records"]["old_important"]["scores"]
    assert set(sc) >= {"recency", "relevance", "importance", "final"} and sc["final"] == pytest.approx(0.95, abs=1e-3)
    # a half-life changes recency: with a very long half-life the old record's recency is no longer ~0
    short = run_policy(policy(("s", "retrieve", {}), ("r", "rank", {"half_life_days": 1})), lt, [], {}, now=NOW)["records"]["old_important"]["scores"]["recency"]
    long = run_policy(policy(("s", "retrieve", {}), ("r", "rank", {"half_life_days": 1000})), lt, [], {}, now=NOW)["records"]["old_important"]["scores"]["recency"]
    assert short < 1e-6 < 0.9 < long


def test_dedupe_keeps_the_record_that_comes_first_in_the_current_order():
    lt = [rec("a", "The Plate reader is booked  Monday", 1), rec("b", "the plate reader is booked monday", 2), rec("c", "plate reader booked on monday mornings", 3), rec("d", "unrelated", 4)]
    app = run_policy(policy(("s", "retrieve", {"method": "recency"}), ("d", "dedupe", {"method": "exact_text"})), lt, [], {}, now=NOW)
    assert "b" not in app["final"] and "duplicate of a" in decisions(app, "d")["b"]["reason"] and "c" in app["final"]
    app = run_policy(policy(("s", "retrieve", {"method": "recency"}), ("d", "dedupe", {"method": "similarity", "threshold": 0.6})), lt, [], {}, now=NOW)
    assert app["final"] == ["a", "d"] and "similarity" in decisions(app, "d")["c"]["reason"]


def test_budget_truncation_stop_vs_skip_and_estimate_label():
    lt = [rec("a", "x" * 40, 1), rec("b", "y" * 400, 2), rec("c", "z" * 40, 3)]  # 10, 100, 10 tokens (estimate)
    stop = run_policy(policy(("s", "retrieve", {"method": "recency"}), ("b", "budget", {"max_tokens": 50, "overflow": "stop"})), lt, [], {}, now=NOW)
    assert stop["final"] == ["a"] and "removed to meet the configured token budget" in decisions(stop, "b")["b"]["reason"] and "needs 100 tokens (estimate)" in decisions(stop, "b")["b"]["reason"]
    assert "overflow is 'stop'" in decisions(stop, "b")["c"]["reason"]
    skip = run_policy(policy(("s", "retrieve", {"method": "recency"}), ("b", "budget", {"max_tokens": 50, "overflow": "skip"})), lt, [], {}, now=NOW)
    assert skip["final"] == ["a", "c"] and skip["tokensEstimate"] == 20
    chrono = run_policy(policy(("s", "retrieve", {"method": "recency"}), ("b", "budget", {"max_tokens": 50, "overflow": "skip", "order": "chronological"})), lt, [], {}, now=NOW)
    assert chrono["final"] == ["c", "a"]  # oldest first


def test_summarize_replaces_old_records_with_a_generated_summary_that_links_its_sources():
    lt = [rec(f"r{i}", f"Fact number {i}. More detail about {i}.", i) for i in range(6)]
    app = run_policy(policy(("s", "retrieve", {}), ("m", "summarize", {"keep_recent": 2, "method": "extractive", "max_chars": 200})), lt, [], {}, now=NOW)
    summary = [r for r in app["records"].values() if r["kind"] == "summary"][0]
    assert summary["metadata"]["summaryOf"] == ["r5", "r4", "r3", "r2"] and summary["generated"] and "knownOmissions" in summary["metadata"]
    assert app["final"][0] == summary["id"] and set(app["final"][1:]) == {"r0", "r1"}
    d = decisions(app, "m")
    assert d["r5"]["action"] == "replaced" and app["records"]["r5"]["status"] == "summarized" and "replaced by summary" in app["records"]["r5"]["excludedAt"]["reason"]
    # a model summary is never produced when no summarizer is allowed (isolated preview): skipped and said so
    app = run_policy(policy(("s", "retrieve", {}), ("m", "summarize", {"keep_recent": 2, "method": "model"})), lt, [], {}, now=NOW, allow_model=False)
    assert app["stages"][1]["skipped"] == "needs_model_call" and len(app["final"]) == 6 and "no model was called" in app["stages"][1]["note"]


def test_policy_stage_configs_are_validated_and_ordering_is_enforced():
    assert validate_stage("rank", {"limit": 0}) and validate_stage("budget", {"max_tokens": 0}) and validate_stage("rank", {"nope": 1})
    from agent.spec import AgentSpec

    c = lambda *s, **k: {x[0] for x in check_policy(policy(*s, **k), AgentSpec())}  # noqa: E731
    assert "E_POLICY_FIRST_STAGE" in c(("r", "rank", {}), ("s", "retrieve", {}))
    assert "E_POLICY_QUERY" in c(("s", "retrieve", {"method": "similarity"}))
    assert "E_POLICY_QUERY" in c(("s", "retrieve", {}), ("r", "rank", {"weights": {"relevance": 1}}))
    assert "E_CONFIG" in c(("s", "retrieve", {}), ("r", "rank", {"weights": {"bogus": 1}}))
    assert c(("s", "retrieve", {}), ("r", "rank", {})) == set()


def test_policy_is_a_pure_function_of_its_inputs_so_decisions_reproduce():
    lt = [rec(f"r{i}", f"note {i} about enzymes", i, 0.1 * i) for i in range(8)]
    p = policy(("s", "retrieve", {}), ("r", "rank", {"limit": 4}), ("b", "budget", {"max_tokens": 30}), query="enzymes")
    a, b = run_policy(p, lt, [], {}, now=NOW), run_policy(p, lt, [], {}, now=NOW)
    assert a["final"] == b["final"] and a["stages"] == b["stages"] and a["records"] == b["records"]


# ------------------------------------------------------------------------------------------------ short-term thread memory
def test_short_term_memory_is_the_threads_own_checkpointed_messages(tmp_path):
    g = sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE)
    lab = Lab(tmp_path)
    MemoryStore(lab.wb)
    for t, q in (("ta", "First question A"), ("ta", "Second question A"), ("tb", "Only question B")):
        lab.run(g, thread=t, inp={"question": q})
    from agent.runtime import open_checkpointer

    saver = open_checkpointer(lab.wb)
    ha = saver.get_tuple({"configurable": {"thread_id": "ta"}}).checkpoint["channel_values"]["history"]
    hb = saver.get_tuple({"configurable": {"thread_id": "tb"}}).checkpoint["channel_values"]["history"]
    assert [m["role"] for m in ha] == ["user", "assistant", "user", "assistant"] and ha[2]["content"] == "Second question A"
    assert [m["content"] for m in hb if m["role"] == "user"] == ["Only question B"]  # no leakage between threads
    assert all(m["id"].startswith("msg:ta:") for m in ha) and ha[0]["ts"] <= ha[2]["ts"]
    # the policy's universe lists the thread's messages as short-term records (excluded here because the policy reads long-term only)
    app = MemoryStore(lab.wb).applications(thread_id="ta")[-1]
    st = [r for r in app["records"].values() if r["store"] == "short_term"]
    assert {r["namespace"] for r in st} == {"thread:ta"} and all("store filter" in r["excludedAt"]["reason"] for r in st)
    assert len(st) == 3  # two user messages + the first answer existed when the second turn selected memory


def test_short_term_window_and_conversation_in_the_prompt(tmp_path):
    msgs = [{"id": f"m{i}", "role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "w" * 20, "ts": NOW + i} for i in range(6)]
    p = policy(("sel", "retrieve", {"sources": ["short_term"], "method": "recency", "k": 3}), ("b", "budget", {"max_tokens": 100, "order": "chronological"}))
    app = run_policy(p, [], short_term_records(msgs, "t"), {}, now=NOW + 10)
    assert app["final"] == ["m3", "m4", "m5"] and "older than the selected window" in decisions(app, "sel")["m0"]["reason"]


# ------------------------------------------------------------------------------------------------ memory write block
def write_graph(**cfg):
    base = dict(target="long_term", text="{q}", namespace="lab", scope="user:{uid}", kind="episodic", importance=0.6, metadata={"topic": "{q}"}, generated=True, evidence="user said: {q}")
    return sm.make_graph([sm.N("w", "agent.memory_write", **{**base, **cfg})], [sm.E("START", "w"), sm.E("w", "END")], {"state": [sm.S("q"), sm.S("uid", default="a")], "limits": {"maxSteps": 5}})


def test_memory_write_records_validation_scope_evidence_and_marks_generated(tmp_path):
    lab = Lab(tmp_path)
    _, rid = lab.run(write_graph(), inp={"q": "the incubator door sticks"})
    mem = MemoryStore(lab.wb)
    r = mem.list_records()[0]
    assert r["text"] == "the incubator door sticks" and r["scope"] == "user:a" and r["generated"] is True and r["metadata"] == {"topic": "the incubator door sticks"}
    w = mem.writes(record_id=r["id"])[0]
    assert w["node"] == "w" and w["run_id"] == rid and w["evidence"] == "user said: the incubator door sticks" and w["validation"]["ok"] and w["thread_id"] == "t1"
    ev = lab.events(rid, "memory_write")[0]["data"]
    assert ev["status"] == "written" and ev["recordId"] == r["id"] and ev["generated"] is True
    _, rid2 = lab.run(write_graph(), inp={"q": "the incubator door sticks"}, thread="t2")  # duplicate text in the same scope is skipped, not stored twice
    assert lab.events(rid2, "memory_write")[0]["data"]["status"] == "skipped_duplicate" and len(mem.list_records()) == 1
    _, rid3 = lab.run(write_graph(max_chars=5), inp={"q": "far too long"}, thread="t3")
    assert lab.events(rid3, "memory_write")[0]["data"]["status"] == "invalid" and len(mem.list_records()) == 1


def test_memory_write_proposals_have_a_visible_accept_reject_edit_stage(tmp_path):
    lab = Lab(tmp_path)
    g = write_graph(mode="approve")
    st, rid = lab.run(g, inp={"q": "Kinase-7 is stable at 4 C"})
    assert st == "paused"
    p = lab.events(rid, "interrupt_raised")[0]["data"]["payload"]
    assert p["type"] == "memory_write_proposal" and p["proposal"]["generated"] is True and "does not establish its truth" in p["prompt"]
    assert MemoryStore(lab.wb).list_records() == []  # nothing stored before a decision
    st, _ = lab.run(g, run_id=rid, resume={"action": "edit", "value": "Kinase-7 is stable at 4 C for a week"})
    assert st == "completed" and MemoryStore(lab.wb).list_records()[0]["text"] == "Kinase-7 is stable at 4 C for a week"
    st, rid2 = lab.run(g, inp={"q": "rejected fact"}, thread="t9")
    lab.run(g, run_id=rid2, resume={"action": "reject"})
    assert len(MemoryStore(lab.wb).list_records()) == 1 and lab.events(rid2, "memory_write")[0]["data"]["status"] == "rejected"


def test_scopes_do_not_leak_between_users(tmp_path):
    lab = Lab(tmp_path)
    mem = MemoryStore(lab.wb)
    for r in sm.memory_seed():
        mem.put_record(r)
    g = sm.memory_debug_graph(sm.CONSTRAINT_FIXTURE, rank_limit=10, budget_tokens=1000)
    for uid in ("alice", "bob"):
        _, rid = lab.run(g, thread=f"u-{uid}", inp={"question": sm.CONSTRAINT_QUESTION, "user_id": uid})
        sel = lab.final(rid)["memory"]
        assert sel and all(r["scope"] == f"user:{uid}" for r in sel)
    bob = [r["id"] for r in lab.final(rid)["memory"]]
    assert bob == ["mem_bob"]
