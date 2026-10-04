"""Build the trace view (per-step state diff, route taken with predicate values, model calls with latency/tokens, tool calls) from the
recorded run events. Nothing here executes anything."""
from __future__ import annotations

from typing import Any


def build_trace(store, run_id: str) -> dict[str, Any]:
    evs = store.events(run_id)
    steps: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    run: dict[str, Any] = {"threadId": None, "events": len(evs)}
    totals = {"modelCalls": 0, "toolCalls": 0, "latencyMs": 0.0, "inputTokens": 0, "outputTokens": 0, "providerReportedCalls": 0, "unreportedCalls": 0, "fixtureCalls": 0}
    interrupts, budget, memory, failures = [], None, [], []
    for e in evs:
        t, d, n = e["type"], e["data"], e["node_id"]
        if t in ("run_started", "run_resumed"):
            run.update({"threadId": d.get("threadId"), "libraries": d.get("libraries"), "limits": d.get("limits"), "checkpointer": d.get("checkpointer"), "graphHash": e["graph_hash"]})
            if t == "run_resumed":
                run.setdefault("resumes", []).append({"seq": e["seq"], "value": d.get("resumeValue")})
        elif t == "node_started":
            cur = {"seq": e["seq"], "step": d.get("step"), "node": n, "type": d.get("type"), "replay": d.get("replay"), "status": "running", "changes": [], "reads": {}, "route": None,
                   "modelCalls": [], "toolCalls": [], "retrievals": [], "events": []}
            steps.append(cur)
        elif t == "state_update":
            tgt = next((s for s in reversed(steps) if s["node"] == n), None)
            if tgt:
                tgt["changes"], tgt["reads"], tgt["durationMs"] = d["changes"], d["reads"], d["durationMs"]
        elif t == "node_finished":
            tgt = next((s for s in reversed(steps) if s["node"] == n), None)
            if tgt:
                tgt["status"] = "finished"
        elif t == "route_taken":
            tgt = next((s for s in reversed(steps) if s["node"] == n), None)
            if tgt:
                tgt["route"] = {"route": d["route"], "evaluated": d["evaluated"], "taken": d["taken"], "label": d["label"], "to": d["to"], "via": d["via"], "seq": e["seq"]}
        elif t == "model_call":
            tgt = next((s for s in reversed(steps) if s["node"] == n), None)
            c = {k: d[k] for k in ("callId", "purpose", "attempt", "provider", "model", "fixture", "latencyMs", "usage", "cost", "resolved", "ignored", "tokensEstimate", "validationErrors", "response")}
            c["seq"] = e["seq"]
            if tgt:
                tgt["modelCalls"].append(c)
            totals["modelCalls"] += 1
            totals["latencyMs"] += d["latencyMs"]
            if d["fixture"]:
                totals["fixtureCalls"] += 1
            if d["usage"]["source"] == "provider":
                totals["providerReportedCalls"] += 1
                totals["inputTokens"] += d["usage"]["inputTokens"] or 0
                totals["outputTokens"] += d["usage"]["outputTokens"] or 0
            else:
                totals["unreportedCalls"] += 1
        elif t == "tool_call":
            tgt = next((s for s in reversed(steps) if s["node"] == n), None)
            if tgt:
                tgt["toolCalls"].append({**d, "seq": e["seq"]})
            totals["toolCalls"] += 1
        elif t in ("retrieval", "memory_selection", "memory_write", "structured_attempt", "citations_checked", "embedding", "prompt_rendered", "index_ready", "interrupt_resumed"):
            tgt = next((s for s in reversed(steps) if s["node"] == n), None)
            if t == "retrieval" and tgt:
                tgt["retrievals"].append({**d, "seq": e["seq"]})
            if tgt:
                tgt["events"].append({"type": t, "seq": e["seq"], **d})
            if t in ("memory_selection", "memory_write"):
                memory.append({"type": t, "node": n, "seq": e["seq"], **d})
        elif t == "interrupt_raised":
            interrupts.append({"seq": e["seq"], "node": n, "interruptId": d["interruptId"], "payload": d["payload"], "step": d.get("step"), "checkpointId": d.get("checkpointId")})
        elif t == "budget_exhausted":
            budget = d
        elif t in ("node_failed", "model_call_failed", "error"):
            failures.append({"seq": e["seq"], "type": t, "node": n, "message": d.get("message"), "code": d.get("code")})
        elif t == "run_finished":
            run.update({"finished": d.get("status"), "stoppedBy": d.get("stoppedBy"), "error": d.get("error")})
    row = store.get_run(run_id)
    pending = None
    if row and row["status"] == "paused" and interrupts:
        pending = interrupts[-1]
    totals["latencyMs"] = round(totals["latencyMs"], 1)
    return {"runId": run_id, "status": row["status"] if row else None, "run": run, "steps": steps, "totals": totals, "interrupts": interrupts, "pendingInterrupt": pending, "budget": budget,
            "memory": memory, "failures": failures,
            "provenance": {"runId": run_id, "source": "events recorded by the worker", "graphHash": run.get("graphHash")}}
