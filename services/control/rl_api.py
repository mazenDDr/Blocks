"""Milestone 5 routes for the RL laboratory: environment catalog and builder preview, run curves, captured rollouts and frames, replay-buffer browser,
transition-to-update trace, evaluation report and variant comparison. Inspection routes only read what the worker recorded."""
from __future__ import annotations

import base64
import io
import json
from collections import OrderedDict
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rl import gridworld as gw
from rl.compare import compare_variants
from rl.envs import CATALOG, GRID_ID, EnvSpec, catalog_entry, make_env
from rl.evaluate import frame_png
from rl.networks import mlp_graph
from graph_core.schema import Graph


def _err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


def rl_validation(graph: Graph, report) -> dict[str, Any]:
    """What the RL workspace shows beside the generic validation: the typed environment, network, learner equation, and the grid schematic."""
    def info(node_type: str, port: str):
        for n in graph.nodes:
            if n.type == node_type and n.id in report.output_types:
                return report.output_types[n.id][port].info
        return None
    env = info("rl.environment", "env")
    out: dict[str, Any] = {"environment": env, "network": info("rl.q_network", "network"), "learner": info("rl.dqn_learner", "learner"), "buffer": info("rl.replay_buffer", "buffer"),
                           "catalogIds": sorted(CATALOG), "ready": report.ok}
    for n in graph.nodes:
        if n.type == "rl.environment" and n.config.get("env_id") == GRID_ID:
            out["gridSchematic"] = gw.schematic(n.config.get("kwargs", {}))
    return out


class GridPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kwargs: dict[str, Any] = Field(default_factory=dict)


class MlpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    obsDim: int = Field(4, ge=1, le=4096)
    hidden: list[int] = Field(default_factory=lambda: [64, 64], max_length=8)
    nActions: int = Field(2, ge=1, le=1024)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variants: list[dict[str, Any]] = Field(..., min_length=1, max_length=20)


def register(app: FastAPI, sv) -> None:
    store = sv.store
    npz_cache: "OrderedDict[str, dict[str, np.ndarray]]" = OrderedDict()

    def rl_row(rid: str) -> dict[str, Any]:
        row = store.get_run(rid)
        if row is None:
            raise _err(404, "not_found", f"unknown run '{rid}'")
        if row["config"].get("kind") != "rl":
            raise _err(422, "not_rl_run", f"run '{rid}' is not an rl run")
        return row

    def last_art(rid: str, kind: str) -> dict[str, Any] | None:
        a = store.artifacts(rid, kind)
        return a[-1] if a else None

    def art_json(rid: str, kind: str) -> Any:
        a = last_art(rid, kind)
        return json.loads(store.read_artifact(a["sha256"])) if a else None

    # ------------------------------------------------------------------ catalog / builder / network helpers
    @app.get("/api/rl/catalog")
    def catalog():
        import gymnasium

        return {"gymnasium": gymnasium.__version__, "entries": [catalog_entry(k) for k in CATALOG],
                "note": "Facts come from the installed Gymnasium; only 'verified' entries have a tested learner pair in this workbench, the others are validated on their spaces only."}

    @app.post("/api/rl/grid/preview")
    def grid_preview(req: GridPreview):
        """Labelled schematic of the configured grid plus a REAL rendered frame of its reset state (rendering never steps the environment)."""
        try:
            env = make_env(EnvSpec(env_id=GRID_ID, kwargs=req.kwargs), render=True)
        except Exception as e:  # noqa: BLE001
            raise _err(422, "grid_invalid", f"{type(e).__name__}: {e}")
        try:
            env.reset(seed=0)
            png = frame_png(env.render(), 640)
            u = env.unwrapped
            return {"schematic": gw.schematic(req.kwargs), "legend": {"#": "wall", "S": "start", "G": "goal (terminates)", ".": "free"}, "framePng": base64.b64encode(png).decode(),
                    "frameSource": "env.render() of the reset state (a real render of the configured environment, not a drawing of the editor)", "actions": list(gw.ACTIONS),
                    "components": list(gw.COMPONENTS), "defaultWeights": gw.DEFAULT_WEIGHTS, "width": u.width, "height": u.height,
                    "observationSpace": str(env.observation_space), "actionSpace": str(env.action_space)}
        finally:
            env.close()

    @app.post("/api/rl/network/mlp")
    def network_mlp(req: MlpRequest):
        return {"network": mlp_graph(req.obsDim, tuple(req.hidden), req.nActions)}

    # ------------------------------------------------------------------ run data
    @app.get("/api/rl/runs/{rid}/curves")
    def curves(rid: str, after: int = Query(-1)):
        """Unsmoothed per-episode returns, learner updates and evaluations recorded so far (incremental by event sequence)."""
        row = rl_row(rid)
        evs = store.events(rid, after, ("episode_end", "train_update", "eval", "target_sync", "episode_captured", "rl_setup"))
        setup = None
        cursor = max([after] + [e["seq"] for e in evs])   # the last event actually read: the client resumes from here, so nothing recorded in between is skipped
        episodes, updates, evals, captured, syncs = [], [], [], [], 0
        for e in evs:
            d = e["data"]
            if e["type"] == "episode_end":
                episodes.append({"seq": e["seq"], **{k: d[k] for k in ("tick", "envIndex", "episodeId", "length", "return", "taskReturn", "components", "terminated", "truncated", "policyVersion", "epsilon")}})
            elif e["type"] == "train_update":
                updates.append({"seq": e["seq"], **{k: d[k] for k in ("update", "tick", "loss", "gradNorm", "meanQ", "epsilon", "bufferSize", "policyVersion", "targetVersion", "meanTarget")}})
            elif e["type"] == "eval":
                evals.append({"seq": e["seq"], **{k: d[k] for k in ("tick", "update", "policyVersion", "final", "return", "taskReturn", "length", "terminatedCount", "truncatedCount", "successRate", "episodeReturns")}})
            elif e["type"] == "target_sync":
                syncs += 1
            elif e["type"] == "episode_captured":
                captured.append({"seq": e["seq"], **d})
            elif e["type"] == "rl_setup":
                setup = d
        started = store.last_event(rid, "run_started")
        return {"runId": rid, "status": row["status"], "maxSeq": store.max_seq(rid), "cursor": cursor, "setup": setup, "episodes": episodes, "updates": updates, "evals": evals, "captured": captured,
                "targetSyncs": syncs, "totalSteps": started["data"]["totalSteps"] if started else None, "seed": row["config"].get("seed"),
                "provenance": {"runId": rid, "graphHash": row["graph_hash"], "source": "events recorded by the worker; episode returns are not smoothed"}}

    @app.get("/api/rl/runs/{rid}/setup")
    def setup(rid: str):
        rl_row(rid)
        s = store.last_event(rid, "rl_setup")
        st = store.last_event(rid, "run_started")
        return {"runId": rid, "setup": s["data"] if s else None, "started": st["data"] if st else None}

    @app.get("/api/rl/runs/{rid}/episodes")
    def captured_episodes(rid: str):
        rl_row(rid)
        out = []
        for e in store.events(rid, -1, ("episode_captured",)):
            out.append(e["data"])
        return {"runId": rid, "kind": "training", "episodes": out,
                "note": "BOUNDED CAPTURE: only environment 0's episodes that start at the declared points of the run are recorded, with real env.render() frames."}

    @app.get("/api/rl/runs/{rid}/episodes/{sha}")
    def captured_episode(rid: str, sha: str):
        rl_row(rid)
        if not any(a["sha256"] == sha for a in store.artifacts(rid, "rl_episode")):
            raise _err(404, "not_found", "no such captured episode in this run")
        return json.loads(store.read_artifact(sha))

    @app.get("/api/rl/runs/{rid}/frames/{sha}.png")
    def frame(rid: str, sha: str):
        rl_row(rid)
        if not any(a["sha256"] == sha for a in store.artifacts(rid, "rl_frame")):
            raise _err(404, "not_found", "no such frame in this run")
        return Response(store.read_artifact(sha), media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/api/rl/runs/{rid}/eval")
    def eval_report(rid: str):
        rl_row(rid)
        rep = art_json(rid, "rl_eval_report")
        if rep is None:
            raise _err(404, "not_available", "this run has no evaluation report (it did not finish)")
        return {"runId": rid, **rep}

    # ------------------------------------------------------------------ replay buffer browser
    def buffer_arrays(rid: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        a = last_art(rid, "rl_buffer")
        if a is None:
            raise _err(404, "not_available", "this run recorded no replay buffer (it did not finish)")
        if a["sha256"] not in npz_cache:
            with np.load(io.BytesIO(store.read_artifact(a["sha256"]))) as z:
                npz_cache[a["sha256"]] = {k: z[k] for k in z.files}
            while len(npz_cache) > 4:
                npz_cache.popitem(last=False)
        npz_cache.move_to_end(a["sha256"])
        return npz_cache[a["sha256"]], a["meta"]

    @app.get("/api/rl/runs/{rid}/buffer")
    def buffer_browser(rid: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), sort: str = Query("tid"), order: str = Query("asc"),
                       episode: int | None = None, flag: str = Query("any"), source: str = Query("any"), min_samples: int | None = None):
        rl_row(rid)
        z, meta = buffer_arrays(rid)
        names = meta["components"]
        n = len(z["tid"])
        tick = meta["tick"]
        mask = np.ones(n, bool)
        if episode is not None:
            mask &= z["episode_id"] == episode
        if flag == "terminated":
            mask &= z["terminated"]
        elif flag == "truncated":
            mask &= z["truncated"]
        elif flag == "ending":
            mask &= z["terminated"] | z["truncated"]
        if source in ("random", "greedy"):
            mask &= z["action_source"] == (1 if source == "random" else 0)
        if min_samples is not None:
            mask &= z["sample_count"] >= min_samples
        idx = np.nonzero(mask)[0]
        keymap = {"tid": z["tid"], "sampleCount": z["sample_count"], "age": -z["insert_tick"], "reward": z["reward"], "policyVersion": z["policy_version"], "episode": z["episode_id"]}
        if sort not in keymap:
            raise _err(422, "request_invalid", f"sort must be one of {sorted(keymap)}")
        idx = idx[np.argsort(keymap[sort][idx], kind="stable")]
        if order == "desc":
            idx = idx[::-1]
        page = idx[offset:offset + limit]
        rows = [{"tid": int(z["tid"][i]), "envIndex": int(z["env_index"][i]), "episodeId": int(z["episode_id"][i]), "episodeStep": int(z["episode_step"][i]),
                 "policyVersion": int(z["policy_version"][i]), "insertTick": int(z["insert_tick"][i]), "age": int(tick - z["insert_tick"][i]), "epsilon": float(z["eps"][i]),
                 "actionSource": "random" if z["action_source"][i] else "greedy", "action": int(z["action"][i]), "reward": float(z["reward"][i]),
                 "components": {k: float(v) for k, v in zip(names, z["components"][i])}, "terminated": bool(z["terminated"][i]), "truncated": bool(z["truncated"][i]),
                 "sampleCount": int(z["sample_count"][i]), "lastSampledUpdate": int(z["last_sampled_update"][i]), "obs": [round(float(x), 4) for x in z["obs"][i][:16]],
                 "nextObs": [round(float(x), 4) for x in z["next_obs"][i][:16]]} for i in page]
        sc = z["sample_count"]
        pv = z["policy_version"]
        return {"runId": rid, "rows": rows, "total": int(len(idx)), "offset": offset, "limit": limit,
                "buffer": {"size": meta["size"], "capacity": meta["capacity"], "transitionsInserted": meta["nextTid"], "evicted": max(0, meta["nextTid"] - meta["size"]), "tick": tick,
                           "finalPolicyVersion": meta["finalPolicyVersion"], "samplingWeight": "uniform: each stored transition has the same probability batch_size/size per update",
                           "sampleCounts": {"min": int(sc.min()), "max": int(sc.max()), "mean": float(sc.mean()), "neverSampled": int((sc == 0).sum())},
                           "policyVersionRange": [int(pv.min()), int(pv.max())], "terminated": int(z["terminated"].sum()), "truncated": int(z["truncated"].sum()),
                           "ageRange": [int(tick - z["insert_tick"].max()), int(tick - z["insert_tick"].min())], "sampleCountHistogram": np.bincount(np.minimum(sc, 20)).tolist()},
                "provenance": {"runId": rid, "source": "replay buffer contents at the END of the run (the newest transitions up to capacity)", "components": names}}

    # ------------------------------------------------------------------ transition -> update trace
    @app.get("/api/rl/runs/{rid}/trace")
    def trace_overview(rid: str):
        rl_row(rid)
        t = art_json(rid, "rl_trace")
        if t is None:
            raise _err(404, "not_available", "this run recorded no trace (it did not finish)")
        rows = [{"tid": v["tid"], "episodeId": v["episodeId"], "episodeStep": v["episodeStep"], "uses": len(v["uses"]), "evicted": v["evicted"] is not None, "policyVersion": v["policyVersion"],
                 "terminated": v["terminated"], "truncated": v["truncated"], "reward": v["reward"]} for v in t["transitions"].values()]
        return {"runId": rid, "capturePolicy": t["capturePolicy"], "dropped": t["dropped"], "tracked": rows, "updateRecords": len(t["updates"]), "finalPolicyVersion": t["finalPolicyVersion"],
                "envSteps": t["envSteps"]}

    @app.get("/api/rl/runs/{rid}/transitions/{tid}")
    def transition_trace(rid: str, tid: int):
        """Select a transition id -> its record, buffer insertion, every minibatch that sampled it with the TD target / loss contribution / gradient, and the resulting policy version."""
        rl_row(rid)
        t = art_json(rid, "rl_trace")
        if t is None:
            raise _err(404, "not_available", "this run recorded no trace (it did not finish)")
        v = t["transitions"].get(str(tid))
        if v is None:
            insp: dict[str, Any] = {"captured": False, "tid": tid, "capturePolicy": t["capturePolicy"]["statement"]}
            try:
                z, meta = buffer_arrays(rid)
                hit = np.nonzero(z["tid"] == tid)[0]
                if len(hit):
                    i = int(hit[0])
                    insp["reason"] = "This transition is in the final replay buffer but was not tracked (bounded capture), so its individual minibatch uses were not recorded."
                    insp["finalBuffer"] = {"sampleCount": int(z["sample_count"][i]), "lastSampledUpdate": int(z["last_sampled_update"][i]), "episodeId": int(z["episode_id"][i]),
                                           "episodeStep": int(z["episode_step"][i]), "policyVersion": int(z["policy_version"][i]), "terminated": bool(z["terminated"][i]), "truncated": bool(z["truncated"][i])}
                elif tid < meta["nextTid"]:
                    insp["reason"] = "This transition existed but was not tracked and has been overwritten in the ring buffer."
                else:
                    insp["reason"] = f"No transition with this id: the run inserted ids 0..{meta['nextTid'] - 1}."
            except HTTPException:
                insp["reason"] = "Not tracked (bounded capture)."
            return insp
        by_update = {u["update"]: u for u in t["updates"]}
        uses = []
        for u in v["uses"]:
            r = by_update.get(u["update"])
            uses.append({**u, "minibatch": ({"tracked": r["trackedInBatch"], "size": len(r["batchTids"]), "batchTids": r["batchTids"], "meanQ": r["meanQ"], "bufferSize": r["bufferSize"],
                                             "targetSynced": r["targetSynced"]} if r else None)})
        chain = [
            {"stage": "transition", "text": f"episode {v['episodeId']} step {v['episodeStep']}: acted with policy v{v['policyVersion']} ({v['actionSource']}, epsilon {v['epsilon']:.3f}) -> action {v['action']}, reward {v['reward']:.4g}, "
                                              f"terminated={v['terminated']}, truncated={v['truncated']}"},
            {"stage": "buffer insertion", "text": f"inserted as id {v['tid']} into slot {v['slot']} at environment step {v['insertTick']}" + (f"; overwritten by id {v['evicted']['byTid']} at step {v['evicted']['tick']}" if v["evicted"] else "; still in the buffer at the end of the run")},
            {"stage": "minibatches", "text": f"sampled in {len(uses)} recorded minibatch(es)" + (f" (first: update {uses[0]['update']})" if uses else "")},
            {"stage": "objective", "text": (f"update {uses[0]['update']}: target {uses[0]['target']:.5g} = {v['reward']:.4g} + gamma x mask {uses[0]['bootstrapMask']:.0f} x {uses[0]['nextValue']:.5g}; TD error {uses[0]['tdError']:.5g}; loss contribution {uses[0]['lossContribution']:.4g}" if uses else "not used by a recorded update")},
            {"stage": "policy version", "text": (f"the update produced policy v{uses[0]['policyVersionAfter']} (from v{uses[0]['policyVersionBefore']})" if uses else "no resulting version recorded")},
        ]
        return {"captured": True, "tid": tid, "transition": {k: v[k] for k in v if k not in ("uses",)}, "uses": uses, "chain": chain, "components": t["components"],
                "capturePolicy": t["capturePolicy"]["statement"], "usesDropped": t["dropped"]["uses"]}

    # ------------------------------------------------------------------ comparison (A54)
    @app.post("/api/rl/compare")
    def compare(req: CompareRequest):
        for v in req.variants:
            if not isinstance(v.get("label"), str) or not isinstance(v.get("runIds"), list):
                raise _err(422, "request_invalid", "each variant needs 'label' and 'runIds'")
            for r in v["runIds"]:
                rl_row(r)
        return compare_variants(store, req.variants)

    @app.get("/api/studies/{sid}/rl-comparison")
    def study_comparison(sid: str):
        st = sv.studies.get(sid)
        if st is None:
            raise _err(404, "not_found", f"unknown study '{sid}'")
        if st["spec"]["graphKind"] != "rl":
            raise _err(422, "not_rl_study", "this study is not on an rl graph")
        labels = st["spec"]["labels"]
        groups: dict[str, dict[str, Any]] = {}
        trials = sv.studies.trials(sid)
        attempts = sv.studies.attempts(sid)
        for t in trials:
            done = [a for a in attempts if a["trial_id"] == t["id"] and a["status"] == "completed" and a["run_id"]]
            g = groups.setdefault(t["group_key"], {"label": labels.get(t["group_key"]) or ("baseline" if t["is_baseline"] else t["group_key"]), "assignments": t["assignments"], "runIds": [],
                                                   "isBaseline": bool(t["is_baseline"])})
            if done:
                g["runIds"].append(done[-1]["run_id"])
        res = compare_variants(store, [g for g in groups.values()])
        for v, g in zip(res["variants"], groups.values()):
            v["isBaseline"] = g["isBaseline"]
        return {"studyId": sid, "name": st["spec"]["name"], **res}
