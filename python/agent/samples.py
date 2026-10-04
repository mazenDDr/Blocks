"""Builders for the Milestone 4 example graphs. The example project files are generated from these (examples/make_m4_examples.py), and
the tests build the same graphs with a FIXTURE model to verify control flow deterministically.

Documents and memory records used by the examples are SYNTHETIC (invented for this project) and labelled as such."""
from __future__ import annotations

import copy
import time
from typing import Any

from graph_core.schema import Graph

DAY = 86400.0


def S(name: str, type_: str = "text", reducer: str = "replace", default: Any = None, n: int | None = None, scope: str = "turn", properties: dict | None = None, description: str = "") -> dict:
    d: dict[str, Any] = {"name": name, "type": type_, "reducer": {"kind": reducer}, "scope": scope, "description": description}
    if n:
        d["reducer"]["n"] = n
    if default is not None:
        d["default"] = default
    if properties:
        d["properties"] = properties
    return d


def N(id_: str, type_: str, **config: Any) -> dict:
    return {"id": id_, "type": type_, "version": "1.0.0", "config": config}


def E(a: str, b: str) -> dict:
    return {"id": f"{a}__{b}", "kind": "control", "from": {"node": a, "port": "out"}, "to": {"node": b, "port": "in"}}


def chain(*ids: str) -> list[dict]:
    return [E(a, b) for a, b in zip(ids, ids[1:])]


def cmp(field: str, op: str, value: Any = None, other: str | None = None, fn: str | None = None) -> dict:
    p: dict[str, Any] = {"field": field, "op": op}
    if other is not None:
        p["other"] = other
    elif op not in ("is_empty", "not_empty", "is_true", "is_false"):
        p["value"] = value
    if fn:
        p["fn"] = fn
    return p


def make_graph(nodes: list[dict], edges: list[dict], agent: dict) -> Graph:
    return Graph.model_validate({"schemaVersion": "1.0.0", "graphKind": "agent", "backend": "langgraph", "nodes": nodes, "edges": edges, "agent": agent})


def layout(graph: Graph, dx: float = 260, dy: float = 150, per_row: int = 4) -> dict[str, dict[str, float]]:
    from agent.validate import validate_agent

    order = ["START"] + validate_agent(graph).order + ["END"]
    return {nid: {"x": 40 + (i % per_row) * dx, "y": 40 + (i // per_row) * dy} for i, nid in enumerate(order)}


def fixture_model(**fx: Any) -> dict:
    return {"provider": "fixture", "model": "fixture", "fixture": fx}


def ollama_model(model: str = "qwen3.5:2b", max_tokens: int = 300, seed: int | None = 7, temperature: float = 0.0) -> dict:
    return {"provider": "ollama", "model": model, "temperature": temperature, "max_tokens": max_tokens, "seed": seed, "think": False}


# ------------------------------------------------------------------------------------------------ 12.3 retrieval with bounded revision
def retrieval_revision_graph(answer_model: dict, grade_model: dict, revise_model: dict, max_revisions: int = 2, embeddings: dict | None = None,
                             docs_dir: str = "examples/agent_docs", max_model_calls: int | None = 20) -> Graph:
    state = [S("question"), S("query"), S("docs", "documents"), S("answer"), S("max_revisions", "integer", default=max_revisions), S("attempts", "integer", default=0),
             S("citation_check", "object", properties={"allValid": "boolean", "count": "integer", "invalid": "integer"}, default={}),
             S("grade", "object", properties={"supported": "boolean", "reason": "text"}, default={}), S("grade_error"), S("status"), S("result"),
             S("prompt_messages", "list"), S("grade_messages", "list"), S("revise_messages", "list")]
    cite_rule = "Cite every fact with the id of the chunk that states it, in square brackets, for example [enzyme_assay_protocol.txt#0]. If the context does not contain the answer, say that it does not."
    nodes = [
        N("init", "agent.set_state", assignments=[{"field": "query", "kind": "copy", "source": "question"}, {"field": "attempts", "kind": "literal", "value": 0},
                                                  {"field": "status", "kind": "literal", "value": "working"}]),
        N("retrieve", "agent.retrieve", index="lab_docs", query="{query}", k=3, score_threshold=None, output_field="docs"),
        N("draft_prompt", "agent.prompt", output_field="prompt_messages", items=[
            {"kind": "template", "role": "system", "template": "You answer questions using ONLY the context. " + cite_rule},
            {"kind": "documents", "role": "system", "field": "docs", "header": "Context:", "template": "[{chunk_id}] {text}"},
            {"kind": "template", "role": "user", "template": "{question}"}]),
        N("draft", "agent.chat_model", model=answer_model, messages_field="prompt_messages", output_field="answer"),
        N("cite", "agent.citations", text_field="answer", docs_field="docs", output_field="citation_check"),
        N("grade_prompt", "agent.prompt", output_field="grade_messages", items=[
            {"kind": "template", "role": "system", "template": "You are a strict grader. Decide whether the ANSWER is fully supported by the CONTEXT."},
            {"kind": "documents", "role": "system", "field": "docs", "header": "CONTEXT:", "template": "[{chunk_id}] {text}"},
            {"kind": "template", "role": "user", "template": "QUESTION: {question}\nANSWER: {answer}"}]),
        N("grade", "agent.structured_output", model=grade_model, messages_field="grade_messages", output_field="grade", error_field="grade_error",
          schema_fields=[{"name": "supported", "type": "boolean", "description": "true only if every claim in the answer is stated in the context"},
                         {"name": "reason", "type": "text", "description": "one short sentence"}], retry={"maxRetries": 2, "feedback": True}, on_failure="route"),
        N("bump", "agent.set_state", assignments=[{"field": "attempts", "kind": "increment", "by": 1}]),
        N("revise_prompt", "agent.prompt", output_field="revise_messages", items=[
            {"kind": "template", "role": "system", "template": "Rewrite the search query so a keyword search over lab documents finds the passage that answers the question. Reply with ONLY the new query."},
            {"kind": "template", "role": "user", "template": "Question: {question}\nPrevious query: {query}\nWhy the last answer failed: {grade.reason}{grade_error}"}]),
        N("revise", "agent.chat_model", model=revise_model, messages_field="revise_messages", output_field="query"),
        N("accept", "agent.set_state", assignments=[{"field": "status", "kind": "literal", "value": "answered"}, {"field": "result", "kind": "copy", "source": "answer"}]),
        N("unresolved", "agent.set_state", assignments=[{"field": "status", "kind": "literal", "value": "unresolved"},
                                                       {"field": "result", "kind": "template", "template": "Unresolved after {attempts} revision(s). Last answer (unverified): {answer}"}]),
    ]
    edges = [E("START", "init")] + chain("init", "retrieve", "draft_prompt", "draft", "cite", "grade_prompt", "grade") + chain("bump", "revise_prompt", "revise", "retrieve") + [E("accept", "END"), E("unresolved", "END")]
    agent = {"state": state, "limits": {"maxSteps": 60, "maxModelCalls": max_model_calls},
             "routes": [{"id": "after_grade", "from": "grade", "cases": [
                 {"id": "pass", "label": "supported and citations valid", "when": {"all": [cmp("grade.supported", "is_true"), cmp("citation_check.allValid", "is_true")]}, "to": "accept"},
                 {"id": "retry", "label": "retry: budget remains", "when": cmp("attempts", "<", other="max_revisions"), "to": "bump"}], "default": "unresolved", "defaultLabel": "budget exhausted"}],
             "indexes": [{"id": "lab_docs", "description": "SYNTHETIC lab documents (examples/agent_docs)", "loader": {"directory": docs_dir, "glob": "*.txt"},
                          "splitter": {"strategy": "recursive", "chunkSize": 420, "chunkOverlap": 40}, "embeddings": embeddings or {"provider": "local_hash", "dimension": 512}}]}
    return make_graph(nodes, edges, agent)


# ------------------------------------------------------------------------------------------------ 12.7 memory debugging journey
MEMORY_SEED_NOTE = "SYNTHETIC lab notes invented for this project, aged relative to the time they are seeded."


def memory_seed(now: float | None = None) -> list[dict]:
    now = now or time.time()

    def rec(i, text, days, imp, ns="lab", scope="user:alice", kind="semantic"):
        return {"id": i, "namespace": ns, "scope": scope, "kind": kind, "text": text, "importance": imp, "created_at": now - days * DAY, "generated": False,
                "metadata": {"entity": "kinase7" if "Kinase-7" in text else "general", "synthetic": True}, "source": {"seed": "memory_debugging", "note": MEMORY_SEED_NOTE}}
    return [
        rec("mem_constraint", "Constraint: the Kinase-7 assay must never be incubated above 40 degrees Celsius because the enzyme denatures.", 120, 0.9),
        rec("mem_units", "Alice prefers concentrations reported in micromolar units.", 10, 0.4),
        rec("mem_reader", "The plate reader in room 2 is booked on Mondays.", 3, 0.3),
        rec("mem_buffer", "The Tris-Lab buffer stock was remade last week.", 7, 0.3),
        rec("mem_meeting", "The team meeting moved to Thursday afternoon.", 2, 0.2),
        rec("mem_tips", "Order more pipette tips before the next assay.", 1, 0.3),
        rec("mem_bob", "Constraint for Bob: the Kinase-7 assay is incubated at 30 degrees Celsius.", 5, 0.9, scope="user:bob"),
        rec("mem_gym", "Alice goes to the gym on Tuesdays.", 4, 0.1, ns="personal"),
    ]


def memory_debug_graph(model: dict, rank_weights: dict | None = None, rank_limit: int = 3, budget_tokens: int = 120, relevance_weight_default: float = 0.2) -> Graph:
    state = [S("question"), S("user_id", default="alice"), S("memory", "records"), S("history", "messages", "append", scope="thread"), S("prompt_messages", "list"), S("answer")]
    nodes = [
        N("record_question", "agent.memory_write", target="short_term", field="history", role="user", text="{question}", mode="direct"),
        N("select", "agent.memory_select", policy="lab_memory", output_field="memory", short_term_field="history"),
        N("build_prompt", "agent.prompt", output_field="prompt_messages", items=[
            {"kind": "template", "role": "system", "template": "You are a laboratory assistant. Respect any recorded lab constraints in the notes."},
            {"kind": "memory", "role": "system", "field": "memory", "header": "Notes from long-term memory:", "template": "- ({id}) {text}"},
            {"kind": "template", "role": "user", "template": "{question}"}]),
        N("answer", "agent.chat_model", model=model, messages_field="prompt_messages", output_field="answer", append_to="history"),
    ]
    edges = [E("START", "record_question")] + chain("record_question", "select", "build_prompt", "answer", "END")
    weights = rank_weights or {"recency": 1.0, "relevance": relevance_weight_default, "importance": 0.1}
    policy = {"id": "lab_memory", "name": "Lab memory selection", "description": "Which stored notes reach the model: eligible records -> rank -> token budget.",
              "query": "{question}", "embeddings": {"provider": "local_hash", "dimension": 512},
              "stages": [{"id": "eligible", "op": "retrieve", "config": {"sources": ["long_term"], "namespaces": ["lab"], "scopes": ["user:{user_id}"], "method": "all"}},
                         {"id": "rank", "op": "rank", "config": {"weights": weights, "half_life_days": 14, "limit": rank_limit}},
                         {"id": "budget", "op": "budget", "config": {"max_tokens": budget_tokens, "overflow": "stop", "order": "rank"}}]}
    return make_graph(nodes, edges, {"state": state, "limits": {"maxSteps": 20, "maxModelCalls": 4}, "policies": [policy]})


CONSTRAINT_QUESTION = "What temperature should I use to incubate the Kinase-7 assay?"
CONSTRAINT_FIXTURE = fixture_model(rules=[{"when_contains": "40 degrees Celsius", "reply": "FIXTURE reply: incubate the Kinase-7 assay at 37 C, never above the recorded 40 C limit."}],
                                   default="FIXTURE reply: incubate the Kinase-7 assay at 45 C to finish faster.")


# ------------------------------------------------------------------------------------------------ tools, approval, human in the loop
def approval_tools_graph(outbox_dir: str = "") -> Graph:
    state = [S("expr", default="12 * (3 + 4)"), S("calc", "object", properties={"value": "number"}, default={}), S("note"), S("decision"), S("write_result", "object", default={}),
             S("trail", "list", "append", default=[], scope="turn")]
    nodes = [
        N("calc", "agent.tool_call", tool="calculator", args={"expression": "{expr}"}, output_field="calc"),
        N("draft_note", "agent.set_state", assignments=[{"field": "note", "kind": "template", "template": "calculation {expr} = {calc.result.value}"},
                                                         {"field": "trail", "kind": "append_item", "value": "drafted"}]),
        N("review", "agent.human_interrupt", prompt="Review the note before it is written to the outbox.", show_field="note", edit_field="note", decision_field="decision", actions=["approve", "reject", "edit"]),
        N("write", "agent.tool_call", tool="write_note", args={"path": "notes.txt", "text": "{note}"}, allowed_dir=outbox_dir, require_approval=True, output_field="write_result"),
        N("skipped", "agent.set_state", assignments=[{"field": "trail", "kind": "append_item", "value": "rejected"}]),
    ]
    edges = [E("START", "calc")] + chain("calc", "draft_note", "review") + [E("write", "END"), E("skipped", "END")]
    agent = {"state": state, "limits": {"maxSteps": 20}, "routes": [{"id": "after_review", "from": "review", "cases": [
        {"id": "rejected", "label": "rejected", "when": cmp("decision", "==", "reject"), "to": "skipped"}], "default": "write", "defaultLabel": "approved or edited"}]}
    return make_graph(nodes, edges, agent)


# ------------------------------------------------------------------------------------------------ pure control flow (no model)
def counter_loop_graph(stop_at: int = 5, max_steps: int = 25, reachable_exit: bool = True) -> Graph:
    """tick: n += 1 and log; route: n >= stop_at -> done, otherwise loop. With reachable_exit=False the exit predicate can never hold."""
    state = [S("n", "integer", "add", default=0), S("log", "list", "append", default=[]), S("limit", "integer", default=stop_at if reachable_exit else 10**9), S("status")]
    nodes = [N("tick", "agent.set_state", assignments=[{"field": "n", "kind": "increment", "by": 1}, {"field": "log", "kind": "append_item", "value": "tick"}]),
             N("done", "agent.set_state", assignments=[{"field": "status", "kind": "literal", "value": "finished"}])]
    edges = [E("START", "tick"), E("done", "END")]
    agent = {"state": state, "limits": {"maxSteps": max_steps},
             "routes": [{"id": "loop", "from": "tick", "cases": [{"id": "stop", "label": "n reached the limit", "when": cmp("n", ">=", other="limit"), "to": "done"}], "default": "tick", "defaultLabel": "keep looping"}]}
    return make_graph(nodes, edges, agent)


def parallel_join_graph(conflict: bool = False) -> Graph:
    """START fans out to two branches that both write `scores` (append reducer, so both survive) and a join node that waits for both."""
    state = [S("scores", "list", "append", default=[]), S("total", "integer", "add", default=0), S("tag", "text", "replace"), S("summary")]
    a_tag = [{"field": "tag", "kind": "literal", "value": "a"}] if conflict else []
    b_tag = [{"field": "tag", "kind": "literal", "value": "b"}] if conflict else []
    nodes = [N("branch_a", "agent.set_state", assignments=[{"field": "scores", "kind": "append_item", "value": 1}, {"field": "total", "kind": "increment", "by": 10}] + a_tag),
             N("branch_b", "agent.set_state", assignments=[{"field": "scores", "kind": "append_item", "value": 2}, {"field": "total", "kind": "increment", "by": 5}] + b_tag),
             N("join", "agent.set_state", assignments=[{"field": "summary", "kind": "template", "template": "total={total} scores={scores}"}])]
    edges = [E("START", "branch_a"), E("START", "branch_b"), E("branch_a", "join"), E("branch_b", "join"), E("join", "END")]
    return make_graph(nodes, edges, {"state": state, "limits": {"maxSteps": 10}, "joins": [{"node": "join", "waitFor": ["branch_a", "branch_b"]}]})
