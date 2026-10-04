"""VISION 12.3 retrieval with bounded revision. Control flow (transitions, retries, budget termination) is verified with a FIXTURE model;
the same graph with the real local model is exercised in tests/test_agent_live.py (pytest -m live)."""
import json

import pytest

from agent import samples as sm
from agent_helpers import Lab, copy_docs

Q = "What is the highest temperature the Kinase-7 water bath may be set to?"
GOOD = "The water bath must never be set higher than 40 degrees Celsius [enzyme_assay_protocol.txt#1]."


def graph(tmp_path, ans, grades, revs=("Kinase-7 water bath temperature never set higher", "Kinase-7 water bath temperature never set higher", "Kinase-7 water bath temperature never set higher"), n=2, grade_kw=None, **kw):
    return sm.retrieval_revision_graph(sm.fixture_model(**ans), sm.fixture_model(responses=list(grades), **(grade_kw or {})), sm.fixture_model(responses=list(revs)), max_revisions=n,
                                       docs_dir=copy_docs(tmp_path), **kw)


YES, NO = '{"supported": true, "reason": "stated in the context"}', '{"supported": false, "reason": "the context does not state it"}'


def walk(lab, rid):
    return [e["node_id"] for e in lab.events(rid, "node_started")]


def test_first_pass_is_accepted_without_revision(tmp_path):
    lab = Lab(tmp_path)
    g = graph(tmp_path, {"default": GOOD}, [YES])
    st, rid = lab.run(g, inp={"question": Q})
    f = lab.final(rid)
    assert st == "completed" and f["status"] == "answered" and f["result"] == GOOD and f["attempts"] == 0
    assert walk(lab, rid) == ["init", "retrieve", "draft_prompt", "draft", "cite", "grade_prompt", "grade", "accept"]
    r = lab.events(rid, "route_taken")[0]["data"]
    assert r["via"] == "case" and r["taken"] == "pass" and r["to"] == "accept" and r["label"] == "supported and citations valid"
    assert r["evaluated"][0]["values"] == {"grade.supported": True, "citation_check.allValid": True} and r["evaluated"][0]["result"] is True
    assert len(r["evaluated"]) == 1  # the first matching case wins: the revision case was not even evaluated


def test_one_revision_changes_the_query_and_loops_back_to_retrieval(tmp_path):
    lab = Lab(tmp_path)
    g = graph(tmp_path, {"default": GOOD}, [NO, YES])
    st, rid = lab.run(g, inp={"question": Q})
    f = lab.final(rid)
    assert f["status"] == "answered" and f["attempts"] == 1 and f["query"] == "Kinase-7 water bath temperature never set higher"
    assert walk(lab, rid) == (["init", "retrieve", "draft_prompt", "draft", "cite", "grade_prompt", "grade", "bump", "revise_prompt", "revise"]
                              + ["retrieve", "draft_prompt", "draft", "cite", "grade_prompt", "grade", "accept"])
    routes = lab.events(rid, "route_taken")
    assert [r["data"]["taken"] for r in routes] == ["retry", "pass"]
    retry = routes[0]["data"]["evaluated"]
    assert retry[0]["result"] is False and retry[1]["predicate"] == "attempts < max_revisions" and retry[1]["values"] == {"attempts": 0, "max_revisions": 2} and retry[1]["result"] is True
    # the second retrieval used the revised query and its recorded origin
    rets = lab.events(rid, "retrieval")
    assert [r["data"]["query"] for r in rets] == [Q, "Kinase-7 water bath temperature never set higher"]
    ups = [c for e in lab.events(rid, "state_update") if e["node_id"] == "bump" for c in e["data"]["changes"]]
    assert ups[0]["field"] == "attempts" and ups[0]["before"] == 0 and ups[0]["after"] == 1


def test_budget_exhaustion_after_n_revisions_returns_the_unresolved_result(tmp_path):
    lab = Lab(tmp_path)
    for n in (0, 1, 2):
        g = graph(tmp_path, {"default": GOOD}, [NO], n=n)
        st, rid = lab.run(g, inp={"question": Q}, thread=f"n{n}")
        f = lab.final(rid)
        assert st == "completed" and f["status"] == "unresolved" and f["attempts"] == n
        assert walk(lab, rid).count("revise") == n and walk(lab, rid).count("retrieve") == n + 1 and walk(lab, rid).count("grade") == n + 1  # exactly n revisions, never more
        last = lab.events(rid, "route_taken")[-1]["data"]
        assert last["via"] == "default" and last["label"] == "budget exhausted" and last["to"] == "unresolved" and last["evaluated"][1]["result"] is False
        assert f["result"].startswith(f"Unresolved after {n} revision(s).")
        assert lab.finished(rid)["stoppedBy"] == "end"  # a normal, in-graph termination (not the safety limit)


def test_invalid_citations_block_acceptance_even_when_the_grader_says_supported(tmp_path):
    lab = Lab(tmp_path)
    g = graph(tmp_path, {"default": "It is 40 degrees [made_up.txt#7]."}, [YES], n=1)
    st, rid = lab.run(g, inp={"question": Q})
    chk = lab.events(rid, "citations_checked")[0]["data"]
    assert chk["invalid"] == 1 and chk["allValid"] is False and chk["citations"][0]["valid"] is False and chk["citations"][0]["source_path"] is None
    assert lab.final(rid)["status"] == "unresolved" and lab.final(rid)["attempts"] == 1


def test_grader_output_is_validated_with_retries_inside_the_loop(tmp_path):
    lab = Lab(tmp_path)
    g = graph(tmp_path, {"default": GOOD}, [YES], grade_kw={"fail_first": 2})
    st, rid = lab.run(g, inp={"question": Q})
    att = lab.events(rid, "structured_attempt")
    assert [(a["data"]["attempt"], a["data"]["valid"]) for a in att] == [(1, False), (2, False), (3, True)]
    assert lab.final(rid)["status"] == "answered" and lab.final(rid)["grade"] == {"supported": True, "reason": "stated in the context"}
    # a grader that never produces valid JSON routes to a revision via the error branch instead of crashing
    g2 = graph(tmp_path, {"default": GOOD}, [YES], grade_kw={"fail_first": 99}, n=1)
    st, rid = lab.run(g2, inp={"question": Q}, thread="bad-grader")
    f = lab.final(rid)
    assert st == "completed" and f["grade"] == {} and "validation failed" in f["grade_error"] and f["status"] == "unresolved"


def test_trace_shows_passages_scores_prompt_response_parsed_fields_and_routing(tmp_path):
    lab = Lab(tmp_path)
    g = graph(tmp_path, {"default": GOOD}, [YES])
    st, rid = lab.run(g, inp={"question": Q})
    ret = lab.events(rid, "retrieval")[0]["data"]
    assert ret["included"][0]["chunk_id"] == "enzyme_assay_protocol.txt#1" and "cosine" in ret["scoreInterpretation"] and ret["embedding"].startswith("local_hash")
    ctx = lab.contexts(rid)
    draft = ctx[0]
    kinds = [s["source"]["kind"] for m in draft["messages"] for s in m["segments"]]
    assert kinds.count("retrieved_chunk") == 3 and kinds.count("prompt_template") >= 3
    # the rendered prompt shows each retrieved chunk with its id; the response and the parsed grade are in the record
    assert "[enzyme_assay_protocol.txt#1]" in draft["messages"][1]["content"] and draft["response"] == GOOD
    assert [e["id"] for e in draft["excluded"]][:1] and all(e["kind"] == "retrieved_chunk" and e["stage"] == "retrieve" for e in draft["excluded"])
    grade_ctx = ctx[1]
    assert grade_ctx["purpose"] == "structured_output" and json.loads(grade_ctx["response"])["supported"] is True
    # citation origin is the retrieval record, not generated text
    chk = lab.events(rid, "citations_checked")[0]["data"]["citations"][0]
    assert chk["chunk_id"] == "enzyme_assay_protocol.txt#1" and chk["source_path"].endswith("enzyme_assay_protocol.txt") and chk["score"] == ret["included"][0]["score"]
    # per-node latency and a clear fixture label (no tokens are claimed for a fixture)
    for e in lab.events(rid, "model_call"):
        assert e["data"]["fixture"] is True and e["data"]["latencyMs"] >= 0 and e["data"]["usage"]["source"] == "unavailable"
    assert all(e["data"]["durationMs"] >= 0 for e in lab.events(rid, "node_finished"))


def test_replay_of_recorded_output_is_distinct_from_a_new_model_call(tmp_path):
    """Reading a finished run shows recorded responses (no model call); a rerun creates new model calls under a new run id."""
    lab = Lab(tmp_path)
    g = graph(tmp_path, {"default": GOOD}, [YES])
    _, rid = lab.run(g, inp={"question": Q})
    n_before = len(lab.events(rid, "model_call"))
    from agent.trace import build_trace

    build_trace(lab.store, rid)
    lab.contexts(rid)
    assert len(lab.events(rid, "model_call")) == n_before == 2  # inspecting does not call anything
    _, rid2 = lab.run(g, inp={"question": Q}, thread="again")
    assert rid2 != rid and len(lab.events(rid2, "model_call")) == 2
