"""Memory-selection policy engine (VISION 12.5): an ordered pipeline of typed stage blocks
retrieve -> filter -> rank -> dedupe -> budget -> summarize, each recording a per-record decision (kept / excluded / replaced) with
the stage and a specific reason. Pure function of (policy, record universe, state, clock): the same inputs give the same decisions,
so an edited policy can be previewed on a recorded universe without touching any store or calling a model."""
from __future__ import annotations

import copy
import hashlib
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import Embedder, cosine, estimate_tokens
from .spec import EmbeddingsSpec, Policy, eval_predicate, render_predicate, render_template

DAY = 86400.0
STAGE_OPS = ("retrieve", "filter", "rank", "dedupe", "budget", "summarize")


class RetrieveCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[Literal["long_term", "short_term"]] = ["long_term"]
    namespaces: list[str] = []  # empty = every namespace
    scopes: list[str] = []  # templates over the state, e.g. "user:{user_id}"; empty = every scope
    kinds: list[str] = []
    method: Literal["all", "similarity", "recency"] = "all"
    k: int | None = Field(None, ge=1)
    min_score: float | None = None  # similarity only
    include_expired: bool = False


class FilterCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    where: dict[str, Any] = {}  # predicate over: kind, namespace, scope, text, importance, age_days, generated, metadata.<key>


class RankCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weights: dict[str, float] = {"recency": 1.0, "relevance": 1.0, "importance": 1.0}
    half_life_days: float = Field(30.0, gt=0)
    limit: int | None = Field(None, ge=1)
    min_score: float | None = None


class DedupeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["exact_text", "similarity"] = "exact_text"
    threshold: float = Field(0.9, gt=0, le=1)


class BudgetCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_tokens: int = Field(500, ge=1)
    overflow: Literal["stop", "skip"] = "stop"  # stop = nothing after the first record that does not fit; skip = try the next, smaller ones
    order: Literal["rank", "chronological"] = "rank"  # order of the assembled result


class SummarizeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep_recent: int = Field(3, ge=0)
    method: Literal["extractive", "model"] = "extractive"
    max_chars: int = Field(400, ge=40)
    model: dict[str, Any] | None = None  # ModelSpec JSON for method=model


STAGE_CONFIG = {"retrieve": RetrieveCfg, "filter": FilterCfg, "rank": RankCfg, "dedupe": DedupeCfg, "budget": BudgetCfg, "summarize": SummarizeCfg}


def stage_defaults() -> dict[str, dict[str, Any]]:
    return {k: v().model_dump(mode="json") for k, v in STAGE_CONFIG.items()}


def stage_schemas() -> dict[str, dict[str, Any]]:
    return {k: v.model_json_schema() for k, v in STAGE_CONFIG.items()}


def validate_stage(op: str, cfg: dict[str, Any]) -> list[str]:
    try:
        STAGE_CONFIG[op].model_validate(cfg)
    except Exception as e:  # noqa: BLE001
        return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in getattr(e, "errors", lambda: [{"loc": [], "msg": str(e)}])()]
    return []


def short_term_records(messages: list[dict[str, Any]], thread_id: str) -> list[dict[str, Any]]:
    out = []
    for i, m in enumerate(messages or []):
        out.append({"id": m.get("id") or f"msg:{thread_id}:{i}", "store": "short_term", "namespace": f"thread:{thread_id}", "scope": f"thread:{thread_id}",
                    "kind": "message", "text": m.get("content", ""), "metadata": {"role": m.get("role", "user")}, "importance": 0.5,
                    "created_at": m.get("ts") or 0.0 + i, "generated": m.get("role") == "assistant", "expires_at": None, "version": 1, "source": m.get("source") or {}})
    return out


class _Cand(dict):
    pass


def _mk(rec: dict[str, Any]) -> _Cand:
    c = _Cand(copy.deepcopy(rec))
    c["tokens"] = estimate_tokens(c["text"])
    c["scores"] = {}
    c["status"] = "candidate"
    c["excludedAt"] = None
    c["trail"] = []
    return c


def run_policy(policy: Policy, long_term: list[dict[str, Any]], short_term: list[dict[str, Any]], state: dict[str, Any], *, now: float | None = None,
               embedder: Embedder | None = None, summarizer: Callable[[list[dict[str, Any]], int], str] | None = None,
               allow_model: bool = True, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the stages in order. Returns the application: stages with per-record decisions, the final ordered record list, every
    record's last status, and `asOf` (the clock used, so a preview reproduces the same recency scores)."""
    now = time.time() if now is None else now
    embedder = embedder or Embedder(policy.embeddings)
    query = render_template(policy.query, state).strip()
    qvec: list[float] | None = None
    vec_cache: dict[str, list[float]] = {}

    def vec(text: str) -> list[float]:
        if text not in vec_cache:
            vec_cache[text] = embedder.embed([text])[0]
        return vec_cache[text]

    def vecs(cands: list[_Cand]) -> None:
        need = [c["text"] for c in cands if c["text"] not in vec_cache]
        if need:
            for t, v in zip(need, embedder.embed(need)):
                vec_cache[t] = v

    def relevance(c: _Cand) -> float | None:
        nonlocal qvec
        if not query:
            return None
        if qvec is None:
            qvec = vec(query)
        return max(0.0, cosine(qvec, vec(c["text"])))

    universe = [_mk(r) for r in long_term + short_term]
    by_id = {c["id"]: c for c in universe}
    current: list[_Cand] = []
    stages_out: list[dict[str, Any]] = []
    uses_model = False

    def exclude(c: _Cand, stage: dict[str, Any], decisions: list, reason: str, detail: dict | None = None) -> None:
        c["status"], c["excludedAt"] = "excluded", {"stage": stage["id"], "op": stage["op"], "reason": reason}
        c["trail"].append({"stage": stage["id"], "op": stage["op"], "action": "excluded", "reason": reason})
        decisions.append({"id": c["id"], "action": "excluded", "reason": reason, **(detail or {})})

    def keep(c: _Cand, stage: dict[str, Any], decisions: list, note: str = "") -> None:
        c["trail"].append({"stage": stage["id"], "op": stage["op"], "action": "kept", "reason": note})
        decisions.append({"id": c["id"], "action": "kept", "reason": note})

    for i, st in enumerate(policy.stages):
        stage = {"id": st.id, "op": st.op}
        cfg = STAGE_CONFIG[st.op].model_validate(st.config)
        decisions: list[dict[str, Any]] = []
        n_in = len(current) if i else len(universe)
        note = ""
        if st.op == "retrieve":
            if i != 0:
                raise ValueError("retrieve must be the first stage")
            scopes = [render_template(s, state) for s in cfg.scopes]
            pool = []
            for c in universe:
                if c["store"] not in cfg.sources:
                    exclude(c, stage, decisions, f"excluded by store filter: '{c['store']}' is not among the selected stores {cfg.sources}")
                elif cfg.namespaces and c["namespace"] not in cfg.namespaces:
                    exclude(c, stage, decisions, f"excluded by namespace filter: namespace '{c['namespace']}' is not in {cfg.namespaces}")
                elif scopes and c["scope"] not in scopes:
                    exclude(c, stage, decisions, f"excluded by scope filter: scope '{c['scope']}' is not in {scopes}")
                elif cfg.kinds and c["kind"] not in cfg.kinds:
                    exclude(c, stage, decisions, f"excluded by kind filter: kind '{c['kind']}' is not in {cfg.kinds}")
                elif not cfg.include_expired and c.get("expires_at") and c["expires_at"] < now:
                    exclude(c, stage, decisions, "excluded as expired (expires_at is in the past)")
                else:
                    pool.append(c)
            if cfg.method == "similarity":
                if not query:
                    raise ValueError("similarity retrieval needs a policy query")
                vecs(pool)
                for c in pool:
                    c["scores"]["relevance"] = round(relevance(c) or 0.0, 4)
                ranked = sorted(pool, key=lambda c: (-c["scores"]["relevance"], c["id"]))
                note = f"similarity to the query ({embedder.identity})"
            elif cfg.method == "recency":
                ranked = sorted(pool, key=lambda c: (-c["created_at"], c["id"]))
                note = "most recent first"
            else:
                ranked = sorted(pool, key=lambda c: (c["created_at"], c["id"]))
                note = "all in-scope records, oldest first"
            cur: list[_Cand] = []
            for rank, c in enumerate(ranked, 1):
                if cfg.method == "similarity" and cfg.min_score is not None and c["scores"]["relevance"] < cfg.min_score:
                    exclude(c, stage, decisions, f"excluded: similarity {c['scores']['relevance']:.3f} is below the minimum score {cfg.min_score}")
                elif cfg.k is not None and rank > cfg.k:
                    why = ("older than the selected window" if cfg.method == "recency" else "ranked below the selected limit")
                    exclude(c, stage, decisions, f"{why}: position {rank} of {len(ranked)}, limit k={cfg.k}")
                else:
                    keep(c, stage, decisions, f"retrieved by {cfg.method}")
                    cur.append(c)
            current = cur
        elif st.op == "filter":
            cur = []
            for c in current:
                ns = {**c, "age_days": round((now - c["created_at"]) / DAY, 3)}
                if eval_predicate(cfg.where, ns):
                    keep(c, stage, decisions, "matched " + render_predicate(cfg.where))
                    cur.append(c)
                else:
                    exclude(c, stage, decisions, "excluded by filter: " + render_predicate(cfg.where) + " was false")
            current = cur
            note = render_predicate(cfg.where)
        elif st.op == "rank":
            wsum = sum(max(0.0, w) for w in cfg.weights.values()) or 1.0
            vecs(current) if (cfg.weights.get("relevance", 0) > 0 and query) else None
            for c in current:
                comps = {"recency": 0.5 ** (max(0.0, now - c["created_at"]) / DAY / cfg.half_life_days),
                         "relevance": (relevance(c) if cfg.weights.get("relevance", 0) > 0 else None) or c["scores"].get("relevance") or 0.0,
                         "importance": max(0.0, min(1.0, float(c.get("importance", 0.5))))}
                c["scores"].update({k: round(v, 4) for k, v in comps.items()})
                c["scores"]["final"] = round(sum(max(0.0, cfg.weights.get(k, 0.0)) / wsum * comps[k] for k in comps), 4)
            ordered = sorted(current, key=lambda c: (-c["scores"]["final"], c["id"]))
            cur = []
            for rank, c in enumerate(ordered, 1):
                if cfg.min_score is not None and c["scores"]["final"] < cfg.min_score:
                    exclude(c, stage, decisions, f"excluded: final score {c['scores']['final']:.3f} is below the minimum score {cfg.min_score}", {"rank": rank})
                elif cfg.limit is not None and rank > cfg.limit:
                    cutoff = ordered[cfg.limit - 1]["scores"]["final"]
                    exclude(c, stage, decisions, f"ranked below the selected limit: rank {rank} of {len(ordered)}, limit {cfg.limit} "
                            f"(score {c['scores']['final']:.3f} < cutoff {cutoff:.3f})", {"rank": rank})
                else:
                    keep(c, stage, decisions, f"rank {rank}, score {c['scores']['final']:.3f}")
                    decisions[-1]["rank"] = rank
                    cur.append(c)
            current = cur
            note = "weights " + ", ".join(f"{k}={cfg.weights.get(k, 0)}" for k in ("recency", "relevance", "importance")) + f"; half-life {cfg.half_life_days:g} d"
        elif st.op == "dedupe":
            cur, seen = [], []
            if cfg.method == "similarity":
                vecs(current)
            for c in current:
                dup, sim = None, None
                for s in seen:
                    if cfg.method == "exact_text":
                        if " ".join(s["text"].lower().split()) == " ".join(c["text"].lower().split()):
                            dup, sim = s, 1.0
                            break
                    else:
                        sim = cosine(vec(s["text"]), vec(c["text"]))
                        if sim >= cfg.threshold:
                            dup = s
                            break
                if dup is not None:
                    exclude(c, stage, decisions, f"duplicate of {dup['id']}" + (f" (similarity {sim:.3f} >= {cfg.threshold})" if cfg.method == "similarity" else " (identical text)"))
                else:
                    seen.append(c)
                    keep(c, stage, decisions)
                    cur.append(c)
            current = cur
            note = cfg.method
        elif st.op == "budget":
            remaining, cur, stopped = cfg.max_tokens, [], False
            for c in current:
                if stopped:
                    exclude(c, stage, decisions, "removed to meet the configured token budget: an earlier record did not fit and overflow is 'stop'")
                elif c["tokens"] <= remaining:
                    remaining -= c["tokens"]
                    keep(c, stage, decisions, f"{c['tokens']} tokens (estimate); {remaining} of {cfg.max_tokens} left")
                    cur.append(c)
                else:
                    exclude(c, stage, decisions, f"removed to meet the configured token budget: needs {c['tokens']} tokens (estimate), {remaining} of {cfg.max_tokens} remain")
                    stopped = cfg.overflow == "stop"
            if cfg.order == "chronological":
                cur.sort(key=lambda c: (c["created_at"], c["id"]))
            current = cur
            note = f"budget {cfg.max_tokens} tokens (estimate = ceil(chars/4)); used {cfg.max_tokens - remaining}"
        elif st.op == "summarize":
            recent = sorted(current, key=lambda c: (-c["created_at"], c["id"]))[:cfg.keep_recent]
            recent_ids = {c["id"] for c in recent}
            old = [c for c in current if c["id"] not in recent_ids]
            cur = [c for c in current if c["id"] in recent_ids]
            if not old:
                for c in current:
                    keep(c, stage, decisions, "kept verbatim (within keep_recent)")
                note = "nothing older than keep_recent to summarize"
            elif cfg.method == "model" and not (allow_model and summarizer):
                for c in current:
                    keep(c, stage, decisions, "kept verbatim: the model summary was not produced here (a preview never calls a model)")
                cur = list(current)
                note = "SKIPPED: this summarize stage needs a model call; no model was called"
                stage["skipped"] = "needs_model_call"
            else:
                old.sort(key=lambda c: (c["created_at"], c["id"]))
                if cfg.method == "model":
                    uses_model = True
                    text = summarizer(old, cfg.max_chars)  # type: ignore[misc]
                else:
                    text = " | ".join((c["text"].split(".")[0].strip() or c["text"])[:120] for c in old)[: cfg.max_chars]
                sid = "sum_" + hashlib.sha256("|".join(c["id"] for c in old).encode()).hexdigest()[:8]
                summ = _mk({"id": sid, "store": "derived", "namespace": "derived", "scope": "derived", "kind": "summary", "text": text, "metadata": {
                    "summaryOf": [c["id"] for c in old], "method": cfg.method, "generated": True,
                    "knownOmissions": "details beyond the first sentence of each source record" if cfg.method == "extractive" else "unknown (model summary)"},
                    "importance": max(c["importance"] for c in old), "created_at": max(c["created_at"] for c in old), "generated": True, "expires_at": None, "version": 1, "source": {}})
                summ["status"] = "candidate"
                by_id[sid] = summ
                for c in old:
                    c["status"] = "summarized"
                    c["excludedAt"] = {"stage": st.id, "op": "summarize", "reason": f"replaced by summary {sid} (verbatim text no longer reaches the model)"}
                    c["trail"].append({"stage": st.id, "op": "summarize", "action": "replaced", "reason": f"summarized into {sid}"})
                    decisions.append({"id": c["id"], "action": "replaced", "reason": f"summarized into {sid}"})
                for c in recent:
                    keep(c, stage, decisions, "kept verbatim (within keep_recent)")
                decisions.append({"id": sid, "action": "created", "reason": f"{cfg.method} summary of {len(old)} records"})
                cur = [summ] + sorted(cur, key=lambda c: (c["created_at"], c["id"]))
                note = f"{len(old)} records summarized ({cfg.method}); known omissions recorded on the summary"
            current = cur
        stages_out.append({**stage, "config": st.config, "in": n_in, "out": len(current), "note": note, "decisions": decisions})

    for c in current:
        c["status"] = "included"
    final = [c["id"] for c in current]
    return {"policyId": policy.id, "asOf": now, "query": query, "embeddings": embedder.identity, "stages": stages_out, "final": final,
            "records": {k: dict(v) for k, v in by_id.items()}, "tokensEstimate": sum(c["tokens"] for c in current), "usesModel": uses_model,
            **(meta or {})}


def selection_records(app: dict[str, Any], app_id: str) -> list[dict[str, Any]]:
    """The records handed to the workflow state (what the prompt block may use), each tagged with the application that chose it."""
    out = []
    for rid in app["final"]:
        r = app["records"][rid]
        out.append({"id": r["id"], "store": r["store"], "kind": r["kind"], "text": r["text"], "namespace": r["namespace"], "scope": r["scope"],
                    "created_at": r["created_at"], "metadata": r["metadata"], "scores": r["scores"], "tokens": r["tokens"], "generated": bool(r.get("generated")),
                    "applicationId": app_id})
    return out
