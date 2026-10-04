"""Isolated context preview (VISION 12.7 step 5, A32): recompute a model call's memory selection under an EDITED policy, on the
records that were recorded for that call, re-render the prompt, and diff it against what was actually sent.

It reads recorded data only. It never calls a chat model, never writes to a store, never appends an event or an artifact, and never
touches a checkpoint. (An `ollama` embeddings policy would call the embeddings endpoint; the response says so via `embeddingCalls`.)"""
from __future__ import annotations

import json
from typing import Any

from graph_core.schema import Graph

from .blocks import PromptConfig, render_prompt
from .memory import MemoryStore
from .models import Embedder, estimate_tokens
from .policy import run_policy, selection_records
from .spec import Policy, agent_spec


class PreviewError(Exception):
    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


COMPUTED = ("status", "scores", "trail", "excludedAt", "tokens")


def _universe(app: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    lt, st = [], []
    for r in app["records"].values():
        if r["store"] == "derived":
            continue
        clean = {k: v for k, v in r.items() if k not in COMPUTED}
        (st if r["store"] == "short_term" else lt).append(clean)
    return lt, st


def seg_key(s: dict[str, Any]) -> str:
    src = s.get("source", {})
    return json.dumps([src.get("kind"), src.get("recordId") or src.get("chunkId") or src.get("callId"), src.get("node"), src.get("item")], default=str)


def segments_of(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, m in enumerate(messages):
        for s in m.get("segments", []) or [{"start": 0, "end": len(m["content"]), "source": {"kind": "unlinked"}}]:
            t = m["content"][s["start"]:s["end"]]
            out.append({"key": seg_key(s), "message": i, "role": m["role"], "source": s.get("source", {}), "text": t, "tokensEstimate": estimate_tokens(t)})
    return out


def preview_policy_edit(store, memory: MemoryStore, graph: Graph, run_id: str, call_id: str, policy_override: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = agent_spec(graph)
    arts = [a for a in store.artifacts(run_id, "model_context") if a["meta"].get("callId") == call_id]
    if not arts:
        raise PreviewError("not_found", f"no recorded model call '{call_id}' in run '{run_id}'", 404)
    ctx = json.loads(store.read_artifact(arts[0]["sha256"]))
    call_ev = next((e for e in store.events(run_id, -1, ("model_call",)) if e["data"].get("callId") == call_id), None)
    if not ctx["applications"]:
        raise PreviewError("no_memory_selection", "this model call received no memory selection, so there is no policy to preview")
    new_apps, changed, embed_calls = [], [], False
    # the recorded prompt inputs of the prompt node(s) that fed this call, as of just before it
    prompt_nodes = []
    for m in ctx["messages"]:
        for s in m["segments"]:
            n = s["source"].get("node")
            if n and s["source"].get("kind") in ("prompt_template", "memory_record", "retrieved_chunk", "conversation_message", "tool_result") and n not in prompt_nodes:
                prompt_nodes.append(n)
    replaced: dict[str, list[dict[str, Any]]] = {}
    for aid, meta in ctx["applications"].items():
        app = memory.get_application(aid)
        if not app:
            raise PreviewError("not_found", f"the recorded application '{aid}' is missing")
        pol_json = policy_override if (policy_override and policy_override.get("id") == app["policyId"]) else (spec.policy(app["policyId"]).model_dump() if spec.policy(app["policyId"]) else None)
        if pol_json is None:
            raise PreviewError("policy_unknown", f"policy '{app['policyId']}' is not in the supplied graph")
        policy = Policy.model_validate(pol_json)
        embed_calls = embed_calls or policy.embeddings.provider == "ollama"
        lt, st = _universe(app)
        na = run_policy(policy, lt, st, app.get("inputs") or {}, now=app["asOf"], embedder=Embedder(policy.embeddings), allow_model=False, meta={"id": "preview", "preview": True})
        old_final, new_final = set(app["final"]), set(na["final"])
        for rid in sorted(set(app["records"]) | set(na["records"])):
            a, b = app["records"].get(rid), na["records"].get(rid)
            if a and b and a["store"] != "derived" and a["status"] != b["status"]:
                changed.append({"id": rid, "applicationId": aid, "before": a["status"], "after": b["status"], "beforeReason": (a["excludedAt"] or {}).get("reason"),
                                "afterReason": (b["excludedAt"] or {}).get("reason"), "afterScores": b["scores"]})
        replaced[app["outputField"]] = selection_records(na, "preview")
        new_apps.append({"applicationId": aid, "policy": policy.id, "stages": na["stages"], "final": na["final"], "records": na["records"], "tokensEstimate": na["tokensEstimate"],
                         "skippedStages": [s["id"] for s in na["stages"] if s.get("skipped")], "wasFinal": sorted(old_final), "nowFinal": sorted(new_final)})
    # re-render the prompt node(s) with the new selection
    new_messages = [dict(m) for m in ctx["messages"]]
    for pn in prompt_nodes:
        node = next((n for n in graph.nodes if n.id == pn and n.type == "agent.prompt"), None)
        if node is None:
            raise PreviewError("prompt_missing", f"prompt node '{pn}' is not in the supplied graph")
        ev = [e for e in store.events(run_id, -1, ("prompt_rendered",)) if e["node_id"] == pn and (call_ev is None or e["seq"] < call_ev["seq"])]
        if not ev:
            raise PreviewError("prompt_inputs_missing", f"no recorded inputs for prompt node '{pn}'")
        inputs = {**ev[-1]["data"]["inputs"], **replaced}
        cfg = PromptConfig.model_validate(node.config)
        rendered = render_prompt(cfg, inputs, pn)
        idx = [i for i, m in enumerate(new_messages) if any(s["source"].get("node") == pn for s in m["segments"])]
        if idx:
            new_messages = new_messages[:idx[0]] + rendered + new_messages[idx[-1] + 1:]
        else:
            new_messages = rendered + new_messages
    for i, m in enumerate(new_messages):
        m["index"], m["tokensEstimate"] = i, estimate_tokens(m["content"])
    before_segs, after_segs = segments_of(ctx["messages"]), segments_of(new_messages)
    bkeys, akeys = {s["key"]: s for s in before_segs}, {s["key"]: s for s in after_segs}
    diff = [{**s, "status": "unchanged" if s["key"] in bkeys and bkeys[s["key"]]["text"] == s["text"] else ("changed" if s["key"] in bkeys else "added")} for s in after_segs]
    diff += [{**s, "status": "removed"} for s in before_segs if s["key"] not in akeys]
    return {"noModelCall": True, "noStateMutation": True, "embeddingCalls": embed_calls, "callId": call_id, "runId": run_id,
            "before": {"messages": ctx["messages"], "tokensEstimate": ctx["tokens"]["estimateTotal"]},
            "after": {"messages": new_messages, "tokensEstimate": sum(m["tokensEstimate"] for m in new_messages)},
            "segments": diff, "recordChanges": changed, "applications": new_apps,
            "note": "Preview only: the policy was recomputed on the records recorded for this call and the prompt re-rendered. No model was called and nothing was stored."}
