"""Write the Milestone 4 example projects (agent graphs: graph JSON + separate UI JSON) into examples/. Deterministic.

    python examples/make_m4_examples.py

The graphs are built by python/agent/samples.py (the same builders the tests use, with a FIXTURE model there). The files written here use the
REAL local Ollama models. Documents (examples/agent_docs) and memory records (python/agent/samples.py:memory_seed) are SYNTHETIC."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "python")]

from agent import samples as sm  # noqa: E402
from agent.validate import validate_agent  # noqa: E402
from graph_core.hashing import semantic_hash  # noqa: E402
from graph_core.schema import Graph  # noqa: E402


def layout(graph: Graph, dx: int = 270, dy: int = 150) -> dict:
    succ: dict[str, list[str]] = {}
    for e in graph.edges:
        succ.setdefault(e.from_.node, []).append(e.to.node)
    for r in (graph.agent or {}).get("routes", []):
        succ.setdefault(r["from"], []).extend([c["to"] for c in r["cases"]] + [r["default"]])
    depth, order, q = {"START": 0}, ["START"], ["START"]
    while q:
        x = q.pop(0)
        for y in succ.get(x, []):
            if y not in depth and y != "END":
                depth[y] = depth[x] + 1
                order.append(y)
                q.append(y)
    for n in graph.nodes:
        depth.setdefault(n.id, max(depth.values()) + 1)
    depth["END"] = max(depth.values()) + 1
    rows: dict[int, int] = {}
    pos = {}
    for nid in ["START"] + [n.id for n in graph.nodes] + ["END"]:
        c = depth[nid]
        pos[nid] = {"x": 40 + c * dx, "y": 60 + rows.get(c, 0) * dy}
        rows[c] = rows.get(c, 0) + 1
    return pos


def write(name: str, graph: Graph, description: str, synthetic: bool = False, extra: dict | None = None) -> None:
    rep = validate_agent(graph)
    errs = [d for d in rep.diagnostics if d.severity == "error"]
    assert not errs, (name, [(d.code, d.message) for d in errs])
    (HERE / f"{name}.project.json").write_text(json.dumps(graph.to_json(), indent=2, ensure_ascii=False) + "\n")
    ui = {"schemaVersion": "1.0.0", "positions": layout(graph), "description": description, "synthetic": synthetic, **(extra or {})}
    (HERE / f"{name}.ui.json").write_text(json.dumps(ui, indent=2, ensure_ascii=False) + "\n")
    print(f"{name}: {len(graph.nodes)} nodes, hash {semantic_hash(graph)[:12]}")


def main() -> None:
    oll = sm.ollama_model("qwen3.5:2b", max_tokens=200)
    grade = sm.ollama_model("qwen3.5:2b", max_tokens=120)
    rev = sm.ollama_model("qwen3.5:2b", max_tokens=40)
    g = sm.retrieval_revision_graph(oll, grade, rev, max_revisions=2, embeddings={"provider": "ollama", "model": "nomic-embed-text", "dimension": 768, "normalize": True})
    write("agent_retrieval_revision", g, "VISION 12.3: retrieve -> draft a cited answer -> grade (structured output) -> revise the query up to max_revisions times -> answer or return unresolved. "
          "Uses the local Ollama model qwen3.5:2b and nomic-embed-text embeddings over SYNTHETIC lab documents (examples/agent_docs). Ask a question in the run panel, e.g. "
          "'What is the highest temperature the Kinase-7 water bath may be set to?'", synthetic=True,
          extra={"defaultInput": {"question": "What is the highest temperature the Kinase-7 water bath may be set to?"}})
    g = sm.memory_debug_graph(sm.ollama_model("qwen3.5:2b", max_tokens=150))
    write("agent_memory_debugging", g, "VISION 12.7: a stored lab constraint is missing from the answer. Seed the SYNTHETIC memory records (Memory tab), run, open the model call's context, trace the "
          "record to the stage that excluded it, edit the 'lab_memory' policy (rank weights / budget), preview the context without calling the model, then rerun.", synthetic=True,
          extra={"defaultInput": {"question": sm.CONSTRAINT_QUESTION, "user_id": "alice"}, "seedExample": "memory_debugging"})
    write("agent_approval_tools", sm.approval_tools_graph(""), "Tools with declared effects and human-in-the-loop: calculator (no effect) -> draft a note -> a person reviews/edits it (interrupt) -> write_note "
          "(external file_write effect: needs an approval interrupt and runs once). Pause, restart the service, resume.", extra={"defaultInput": {"expr": "12 * (3 + 4)"}})
    write("agent_bounded_loop", sm.counter_loop_graph(stop_at=5, max_steps=25), "A bounded cycle with no model: tick adds 1 to n (reducer: add) until the route predicate n >= limit holds. "
          "Lower 'Max steps' or raise 'limit' to see termination at the step limit instead.", extra={"defaultInput": {}})
    write("agent_parallel_join", sm.parallel_join_graph(), "Parallel branches and a join: two branches write the same list (append reducer) and a counter (add reducer) in one step; the join waits for both. "
          "Change a reducer to 'replace' on a shared field to see the concurrent-write validation error.", extra={"defaultInput": {}})


if __name__ == "__main__":
    main()
