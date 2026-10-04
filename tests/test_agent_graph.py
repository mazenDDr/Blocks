"""Agent graph kind: typed state with reducers, visual predicates, validation codes, native LangGraph compilation, bounded loops (A14)."""
import copy
import json
from typing import Annotated, Any, TypedDict

import pytest

from agent import samples as sm
from agent.runtime import Runtime, compile_graph
from agent.spec import Reducer, apply_reducer, eval_predicate, reducer_fn, render_predicate
from agent.validate import validate_agent
from agent_helpers import Lab
from graph_core.hashing import semantic_hash
from graph_core.schema import Graph
from graph_core.validate import ExecutionBlocked, require_executable


def codes(g: Graph) -> set[str]:
    return {d.code for d in validate_agent(g).diagnostics}


def mutate(g: Graph, fn) -> Graph:
    d = copy.deepcopy(g.to_json())
    fn(d)
    return Graph.model_validate(d)


# ------------------------------------------------------------------------------------------------ predicates (visual, no Python)
@pytest.mark.parametrize("pred,ns,expected", [
    ({"field": "n", "op": ">=", "value": 3}, {"n": 3}, True),
    ({"field": "n", "op": "<", "value": 3}, {"n": 3}, False),
    ({"field": "n", "op": "<", "other": "limit"}, {"n": 2, "limit": 3}, True),
    ({"field": "docs", "op": ">", "value": 0, "fn": "len"}, {"docs": [1]}, True),
    ({"field": "text", "op": "contains", "value": "abc"}, {"text": "xxabcxx"}, True),
    ({"field": "text", "op": "startswith", "value": "xx"}, {"text": "xxabc"}, True),
    ({"field": "grade.supported", "op": "is_true"}, {"grade": {"supported": True}}, True),
    ({"field": "grade.supported", "op": "is_false"}, {"grade": {"supported": True}}, False),
    ({"field": "x", "op": "is_empty"}, {"x": ""}, True),
    ({"field": "x", "op": "not_empty"}, {"x": [1]}, True),
    ({"field": "k", "op": "in", "value": ["a", "b"]}, {"k": "b"}, True),
    ({"field": "missing", "op": "==", "value": 1}, {}, False),
    ({"all": [{"field": "a", "op": "==", "value": 1}, {"not": {"field": "b", "op": "==", "value": 2}}]}, {"a": 1, "b": 3}, True),
    ({"any": [{"field": "a", "op": "==", "value": 9}, {"field": "b", "op": "==", "value": 3}]}, {"a": 1, "b": 3}, True),
    ({"always": True}, {}, True),
])
def test_predicate_evaluation(pred, ns, expected):
    assert eval_predicate(pred, ns) is expected


def test_predicate_trace_and_rendering():
    p = {"all": [{"field": "attempts", "op": "<", "other": "max_revisions"}, {"not": {"field": "grade.supported", "op": "is_true"}}]}
    tr: dict = {}
    assert eval_predicate(p, {"attempts": 1, "max_revisions": 2, "grade": {"supported": False}}, tr)
    assert tr == {"attempts": 1, "max_revisions": 2, "grade.supported": False}
    assert render_predicate(p) == "(attempts < max_revisions AND NOT grade.supported is true)"


def test_typed_predicates_are_checked_at_validation():
    g = sm.counter_loop_graph()

    def bad(field, op, value):
        return mutate(g, lambda d: d["agent"]["routes"][0]["cases"][0].update(when={"field": field, "op": op, "value": value}))

    assert "E_PREDICATE_TYPE" in codes(bad("status", ">", 3))  # ordering on text
    assert "E_PREDICATE_TYPE" in codes(bad("n", "==", "five"))  # number compared with text
    assert "E_PREDICATE_TYPE" in codes(bad("n", "is_true", None))
    assert "E_PREDICATE_FIELD" in codes(bad("nope", "==", 1))
    assert "E_PREDICATE_OP" in codes(bad("n", "~=", 1))
    assert "E_PREDICATE_TYPE" not in codes(bad("n", ">", 3))
    assert "E_PREDICATE_TYPE" in codes(mutate(g, lambda d: d["agent"]["routes"][0]["cases"][0].update(when={"field": "log", "op": ">", "value": 1})))  # list without len()
    assert "E_PREDICATE_TYPE" not in codes(mutate(g, lambda d: d["agent"]["routes"][0]["cases"][0].update(when={"field": "log", "op": ">", "value": 1, "fn": "len"})))


# ------------------------------------------------------------------------------------------------ reducers
def test_reducer_functions():
    assert apply_reducer(Reducer(kind="append"), [1], [2, 3]) == [1, 2, 3]
    assert apply_reducer(Reducer(kind="append_unique"), [1, 2], [2, 3]) == [1, 2, 3]
    assert apply_reducer(Reducer(kind="add"), 2, 3) == 5
    assert apply_reducer(Reducer(kind="add"), "a", "b") == "ab"
    assert apply_reducer(Reducer(kind="max"), 2, 5) == 5 and apply_reducer(Reducer(kind="min"), 2, 5) == 2
    assert apply_reducer(Reducer(kind="merge"), {"a": 1, "b": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert apply_reducer(Reducer(kind="keep_last_n", n=2), [1, 2], [3]) == [2, 3]
    assert apply_reducer(Reducer(kind="replace"), 1, 2) == 2 and reducer_fn(Reducer(kind="replace")) is None


def test_reducer_choice_must_fit_the_field_type_and_keep_last_n_needs_n():
    g = sm.counter_loop_graph()
    assert "E_REDUCER" not in codes(mutate(g, lambda d: d["agent"]["state"][3].update(reducer={"kind": "add"})))  # add on text (concatenate) is allowed
    g2 = mutate(g, lambda d: d["agent"]["state"].append({"name": "flag", "type": "boolean", "reducer": {"kind": "add"}}))
    assert "E_REDUCER" in codes(g2)
    g3 = mutate(g, lambda d: d["agent"]["state"].append({"name": "h", "type": "list", "reducer": {"kind": "keep_last_n"}}))
    assert "E_REDUCER" in codes(g3)


def test_parallel_branches_use_their_reducers_and_a_join_waits(tmp_path):
    lab = Lab(tmp_path)
    st, rid = lab.run(sm.parallel_join_graph())
    assert st == "completed"
    f = lab.final(rid)
    assert sorted(f["scores"]) == [1, 2] and f["total"] == 15 and f["summary"].startswith("total=15")
    # both branches ran in the same superstep, and the join ran after both
    steps = {e["node_id"]: e["data"]["step"] for e in lab.events(rid, "node_started")}
    assert steps["branch_a"] == steps["branch_b"] < steps["join"]


def test_concurrent_writes_to_a_replace_field_are_rejected_before_running(tmp_path):
    g = sm.parallel_join_graph(conflict=True)
    r = validate_agent(g)
    d = [x for x in r.diagnostics if x.code == "E_CONCURRENT_WRITE"]
    assert d and "tag" in d[0].message and "reducer" in d[0].message
    with pytest.raises(ExecutionBlocked):
        require_executable(g)
    lab = Lab(tmp_path)
    st, rid = lab.run(g)
    assert st == "failed" and lab.events(rid, "validation_error")  # the worker refuses it too, with diagnostics


# ------------------------------------------------------------------------------------------------ structure / validation codes
def test_validation_codes_for_broken_graphs():
    g = sm.counter_loop_graph()
    assert codes(g) == set()
    assert "E_UNKNOWN_OP" in codes(mutate(g, lambda d: d["nodes"][0].update(type="agent.nope")))
    assert "E_OP_GRAPH_KIND" in codes(mutate(g, lambda d: d["nodes"][0].update(type="pytorch.nn.relu")))
    assert "E_BAD_ID" in codes(mutate(g, lambda d: d["nodes"][0].update(id="START")))
    assert "E_EDGE_KIND" in codes(mutate(g, lambda d: d["edges"][0].update(kind="tensor")))
    assert "E_NO_START" in codes(mutate(g, lambda d: d["edges"].pop(0)))
    assert "E_DEAD_END" in codes(mutate(g, lambda d: d["edges"].pop(1)))  # done -> END removed
    assert "E_ROUTE_TARGET" in codes(mutate(g, lambda d: d["agent"]["routes"][0].update(default="ghost")))
    assert "E_ROUTE_AND_EDGE" in codes(mutate(g, lambda d: d["edges"].append({"id": "x", "kind": "control", "from": {"node": "tick", "port": "out"}, "to": {"node": "done", "port": "in"}})))
    assert "E_UNREACHABLE_NODE" in codes(mutate(g, lambda d: d["nodes"].append({"id": "lonely", "type": "agent.set_state", "version": "1.0.0", "config": {"assignments": []}})))
    assert "E_STATE_FIELD_UNKNOWN" in codes(mutate(g, lambda d: d["nodes"][0]["config"]["assignments"][0].update(field="ghost")))
    assert "E_TEMPLATE_VAR" in codes(mutate(g, lambda d: d["nodes"][1]["config"]["assignments"].append({"field": "status", "kind": "template", "template": "{ghost}"})))
    assert "E_BACKEND" not in codes(g)
    bad = mutate(g, lambda d: d.update(backend="pytorch"))
    assert "E_UNSUPPORTED_BACKEND" in codes(bad)


def test_cycle_without_a_conditional_exit_is_a_warning_bounded_by_the_step_limit():
    g = sm.counter_loop_graph()
    ok = validate_agent(g)
    assert ok.analysis["loops"] == [{"nodes": ["tick"], "hasConditionalExit": True, "boundedBy": "route predicate and step limit"}]
    g2 = mutate(g, lambda d: (d["agent"].update(routes=[]), d["edges"].append({"id": "loop", "kind": "control", "from": {"node": "tick", "port": "out"}, "to": {"node": "tick", "port": "in"}}),
                              d["edges"].remove(next(e for e in d["edges"] if e["from"]["node"] == "done"))))
    r = validate_agent(g2)
    assert "W_LOOP_STEP_LIMIT_ONLY" in {d.code for d in r.diagnostics}
    assert r.analysis["loops"][0]["boundedBy"] == "step limit only"


def test_graph_json_roundtrip_and_hash_covers_agent_semantics():
    g = sm.counter_loop_graph()
    again = Graph.model_validate(json.loads(json.dumps(g.to_json())))
    assert semantic_hash(again) == semantic_hash(g)
    g2 = mutate(g, lambda d: d["agent"]["routes"][0]["cases"][0]["when"].update(op=">"))
    assert semantic_hash(g2) != semantic_hash(g)
    g3 = mutate(g, lambda d: d["agent"]["limits"].update(maxSteps=3))
    assert semantic_hash(g3) != semantic_hash(g)


# ------------------------------------------------------------------------------------------------ native LangGraph
def test_compiles_to_a_native_state_graph_with_native_channels(tmp_path):
    from langgraph.channels.binop import BinaryOperatorAggregate
    from langgraph.channels.last_value import LastValue
    from langgraph.graph.state import CompiledStateGraph

    from agent.spec import agent_spec
    from artifact_store import ArtifactStore

    g = sm.counter_loop_graph()
    rt = Runtime(g, agent_spec(g), ArtifactStore(tmp_path / "wb"), tmp_path / "wb", "r0", "t", "h")
    c = compile_graph(g, rt, None)
    assert isinstance(c, CompiledStateGraph)
    assert isinstance(c.channels["n"], BinaryOperatorAggregate) and isinstance(c.channels["log"], BinaryOperatorAggregate)
    assert isinstance(c.channels["status"], LastValue)
    nodes = c.get_graph().nodes
    assert {"tick", "done"} <= set(nodes)
    assert any(e.conditional for e in c.get_graph().edges)


def test_matches_a_handwritten_native_langgraph_reference(tmp_path):
    """The same counter workflow written directly against LangGraph gives the same final state as the compiled visual graph."""
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        n: Annotated[Any, lambda a, b: (a or 0) + b]
        log: Annotated[Any, lambda a, b: (a or []) + b]
        status: Any

    def tick(s):
        return {"n": 1, "log": ["tick"]}

    sg = StateGraph(S)
    sg.add_node("tick", tick)
    sg.add_node("done", lambda s: {"status": "finished"})
    sg.add_edge(START, "tick")
    sg.add_conditional_edges("tick", lambda s: "done" if s["n"] >= 5 else "tick", {"done": "done", "tick": "tick"})
    sg.add_edge("done", END)
    ref = sg.compile().invoke({"n": 0, "log": [], "status": ""})
    lab = Lab(tmp_path)
    _, rid = lab.run(sm.counter_loop_graph(stop_at=5))
    got = lab.final(rid)
    assert (got["n"], got["log"], got["status"]) == (ref["n"], ref["log"], ref["status"]) == (5, ["tick"] * 5, "finished")


# ------------------------------------------------------------------------------------------------ A14: bounded cycle
def test_a14_cycle_routes_state_updates_and_exit(tmp_path):
    lab = Lab(tmp_path)
    st, rid = lab.run(sm.counter_loop_graph(stop_at=5))
    assert st == "completed"
    routes = lab.events(rid, "route_taken")
    assert [r["data"]["via"] for r in routes] == ["default"] * 4 + ["case"]
    assert [r["data"]["evaluated"][0]["values"]["n"] for r in routes] == [1, 2, 3, 4, 5]  # predicate operand values are recorded
    assert routes[-1]["data"]["to"] == "done" and routes[0]["data"]["to"] == "tick"
    ups = [e for e in lab.events(rid, "state_update") if e["node_id"] == "tick"]
    assert [c["after"] for e in ups for c in e["data"]["changes"] if c["field"] == "n"] == [1, 2, 3, 4, 5]
    assert ups[2]["data"]["changes"][0] == {"field": "n", "reducer": "add", "written": 1, "before": 2, "after": 3, "changed": True}
    assert lab.finished(rid)["stoppedBy"] == "end"


def test_a14_terminates_at_the_configured_step_limit(tmp_path):
    lab = Lab(tmp_path)
    st, rid = lab.run(sm.counter_loop_graph(reachable_exit=False, max_steps=7))
    assert st == "completed"  # a bounded stop is a recorded outcome, not a crash
    fin = lab.finished(rid)
    assert fin["stoppedBy"] == "recursion_limit"
    b = lab.events(rid, "budget_exhausted")[0]["data"]
    assert b["kind"] == "maxSteps" and b["limit"] == 7
    n_ticks = len(lab.events(rid, "node_started"))
    assert n_ticks == 7  # exactly the configured number of supersteps ran
    assert lab.final(rid)["n"] == 7 and lab.final(rid)["status"] == ""  # state at the limit is preserved, 'done' never ran


def test_budget_termination_on_model_calls_and_seconds(tmp_path):
    ans = sm.fixture_model(responses=["no citation here"])
    grade = sm.fixture_model(responses=['{"supported": false, "reason": "never"}'])
    rev = sm.fixture_model(responses=["again"])
    g = sm.retrieval_revision_graph(ans, grade, rev, max_revisions=50, max_model_calls=4, docs_dir=__import__("agent_helpers").copy_docs(tmp_path))
    lab = Lab(tmp_path)
    st, rid = lab.run(g, inp={"question": "How hot may the water bath be?"})
    assert st == "completed"
    fin = lab.finished(rid)
    assert fin["stoppedBy"] == "budget:maxModelCalls" and fin["modelCalls"] == 4
    assert len(lab.events(rid, "model_call")) == 4
    b = lab.events(rid, "budget_exhausted")[0]["data"]
    assert (b["kind"], b["limit"], b["used"]) == ("maxModelCalls", 4, 4)
    g2 = mutate(sm.counter_loop_graph(), lambda d: d["agent"]["limits"].update(maxSeconds=1e-9))
    st, rid = lab.run(g2)
    assert lab.finished(rid)["stoppedBy"] == "budget:maxSeconds"


def test_turn_scoped_fields_reset_on_a_new_turn_on_the_same_thread(tmp_path):
    lab = Lab(tmp_path)
    g = sm.counter_loop_graph(stop_at=3)
    _, r1 = lab.run(g, thread="same")
    _, r2 = lab.run(g, thread="same")
    assert lab.final(r1)["n"] == 3 and lab.final(r2)["n"] == 3  # Overwrite reset the reduced field; it did not keep adding to 3
    assert lab.events(r2, "run_started")[0]["data"]["threadExisted"] is True


def test_cancel_is_cooperative(tmp_path):
    lab = Lab(tmp_path)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] >= 3

    lab.store.create_run("rc", "h", {"kind": "agent"})
    from worker.agent_run import AgentRunConfig, run_agent

    g = sm.counter_loop_graph(reachable_exit=False, max_steps=50)
    lab.store.create_run("rc2", semantic_hash(g), {})
    st = run_agent(g, AgentRunConfig(thread_id="tc"), lab.store, "rc2", cancel)
    assert st == "cancelled" and lab.store.get_run("rc2")["status"] == "cancelled"
    assert lab.events("rc2", "cancel_acknowledged")
