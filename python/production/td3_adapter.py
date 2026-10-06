"""Continuous-action TD3 policies in the local production registry (ADR 0069).

A version pins one completed TD3 run's FINAL deterministic actor (CPU state dict, weights_only), its environment spec and real Gymnasium
spaces, the evaluation report and up to 1024 observations the run actually collected (the frozen reference). Serving maps
observation vectors to the actor's action vectors, always inside the action bounds; there is no exploration noise. The DQN
policy adapter (ADR 0019) pins its own files, so this family lives in its own module.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import threading
import time
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabular.core import dumps
from .pipeline import ProductionError, read_verified

MAX_BATCH = 256
_PY = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = ("production/td3_adapter.py", "rl/td3.py", "rl/envs.py")


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__, "gymnasium": dist_version("gymnasium"), "numpy": dist_version("numpy")}


def implementation() -> dict[str, str]:
    return {f: hashlib.sha256((_PY / f).read_bytes()).hexdigest() for f in IMPLEMENTATION_FILES}


def _final_checkpoint(store, run_id):
    cks = [a for a in store.artifacts(run_id, "rl_checkpoint") if a["meta"].get("purpose") == "final" and a["meta"].get("algorithm") == "TD3" and a["status"] == "complete"]
    return cks[-1] if cks else None


def is_candidate(store, row) -> str | None:
    if row["status"] != "completed" or row["config"].get("kind") != "rl" or _final_checkpoint(store, row["id"]) is None:
        return None
    from .model_adapter import _graph
    try:
        g, _ = _graph(store, row["id"])
    except ProductionError:
        return None
    return next((n.id for n in g.nodes if n.type == "rl.td3_learner"), None)


def build_manifest(store, run_id: str, node: str) -> dict[str, Any]:
    from rl.envs import env_spaces, make_env
    from rl.validate import td3_spec_from_graph

    from .model_adapter import _graph

    row = store.get_run(run_id)
    if row is None or is_candidate(store, row) != node:
        raise ProductionError("E_PIPELINE_NOT_RECORDED", "Registration needs a completed TD3 run, its rl.td3_learner node and a final actor checkpoint.")
    g, graph_sha = _graph(store, run_id)
    env_spec, reward, tcfg, _ = td3_spec_from_graph(g)
    obs_sp, act_sp, extra = env_spaces(env_spec)
    env = make_env(env_spec, reward)
    try:
        lo, hi = np.ravel(env.observation_space.low).astype(float), np.ravel(env.observation_space.high).astype(float)
        alo, ahi = np.ravel(env.action_space.low).astype(float).tolist(), np.ravel(env.action_space.high).astype(float).tolist()
    finally:
        env.close()
    ck = _final_checkpoint(store, run_id)
    obs_art = store.artifacts(run_id, "rl_td3_observations")
    if not obs_art:
        raise ProductionError("E_REFERENCE_MISSING", "The run recorded no collected observations; rerun it to freeze a reference.", 409)
    with np.load(io.BytesIO(read_verified(store, obs_art[-1]["sha256"]))) as z:
        obs = z["obs"].astype(float)
    reference_sha = store.put_bytes(dumps({"observations": obs.tolist(), "source": "observations the TD3 run collected, oldest first",
                                           "artifactSha256": obs_art[-1]["sha256"]}).encode())
    reports = store.artifacts(run_id, "rl_eval_report")
    final = json.loads(read_verified(store, reports[-1]["sha256"]))["final"] if reports else None
    obs_dim, act_dim = int(np.prod(obs_sp["shape"])), int(np.prod(act_sp["shape"]))
    return {"adapter": "native-pytorch-td3-policy-local", "family": "rl_continuous_policy", "runId": run_id, "node": node, "graphHash": row["graph_hash"],
            "graphSha256": graph_sha, "modelSha256": ck["sha256"], "checkpointSha256": ck["sha256"], "policyVersion": ck["meta"].get("policyVersion"),
            "paramSha256": ck["meta"].get("paramSha256"), "envSpec": env_spec.model_dump(mode="json"), "envId": env_spec.env_id,
            "observationSpace": obs_sp, "actionSpace": act_sp, "observationDim": obs_dim, "actionDim": act_dim,
            "observationLow": [None if not math.isfinite(v) else v for v in lo], "observationHigh": [None if not math.isfinite(v) else v for v in hi],
            "actionLow": alo, "actionHigh": ahi, "timeLimit": extra.get("timeLimit"), "hidden": tcfg.hidden,
            "outputSchema": {"task": "continuous_policy", "classes": None, "target": None},
            "policy": "deterministic actor a = tanh(net(s)) * scale + bias, inside the action bounds; no exploration noise",
            "referenceSha256": reference_sha, "referenceRows": len(obs), "referencePolicy": "up to 1024 collected observations, frozen at registration",
            "source": {"envId": env_spec.env_id, "seed": row["config"].get("seed"), "observationsSha256": obs_art[-1]["sha256"]},
            "evaluationArtifacts": [r["sha256"] for r in reports],
            "evaluation": final and {k: final.get(k) for k in ("return", "taskReturn", "successRate", "episodes") if k in final},
            "environment": environment(), "implementation": implementation(), "fitArtifacts": {},
            "inputContract": f"records: [{{observation: {obs_dim} finite numbers within the environment's observation bounds}}]",
            "tokenizer": None, "customCode": "not supported by this adapter"}


def verify(store, manifest) -> None:
    if manifest["environment"] != environment() or manifest["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned native environment or implementation differs; serving is refused.", 409)
    ck = _final_checkpoint(store, manifest["runId"])
    if ck is None or ck["sha256"] != manifest["checkpointSha256"]:
        raise ProductionError("E_ARTIFACT_TRUST", "The pinned actor is not the run's recorded final checkpoint.", 409)
    for sha in (manifest["checkpointSha256"], manifest["graphSha256"], manifest["referenceSha256"]):
        read_verified(store, sha)


class TD3PolicyPipeline:
    def __init__(self, store, manifest: dict[str, Any]):
        from rl.td3 import Actor

        verify(store, manifest)
        self.store, self.manifest = store, manifest
        ck = torch.load(io.BytesIO(read_verified(store, manifest["checkpointSha256"])), map_location="cpu", weights_only=True)
        self.actor = Actor(ck["obsDim"], ck["actDim"], ck["hidden"], np.array(ck["low"], np.float32), np.array(ck["high"], np.float32))
        self.actor.load_state_dict(ck["actor"], strict=True)
        self.actor.eval()
        self.lock = threading.Lock()

    def validate_records(self, records, max_batch=MAX_BATCH):
        if not records or len(records) > min(max_batch, MAX_BATCH):
            raise ProductionError("E_REQUEST_BOUNDS", f"Policy requests take 1-{min(max_batch, MAX_BATCH)} observations.", 413)
        d, lo, hi = self.manifest["observationDim"], self.manifest["observationLow"], self.manifest["observationHigh"]
        for r in records:
            if not isinstance(r, dict) or set(r) != {"observation"} or not isinstance(r["observation"], list) or len(r["observation"]) != d:
                raise ProductionError("E_REQUEST_SCHEMA", f"Each record is exactly {{observation: [{d} numbers]}}.")
            for i, v in enumerate(r["observation"]):
                if type(v) not in (int, float) or not math.isfinite(v):
                    raise ProductionError("E_REQUEST_SCHEMA", f"observation[{i}] must be a finite number.")
                if (lo[i] is not None and v < lo[i]) or (hi[i] is not None and v > hi[i]):
                    raise ProductionError("E_REQUEST_SCHEMA", f"observation[{i}] = {v} is outside the environment's bounds [{lo[i]}, {hi[i]}].")

    def predict(self, records):
        t0 = time.perf_counter()
        x = torch.as_tensor([r["observation"] for r in records], dtype=torch.float32)
        t1 = time.perf_counter()
        with self.lock, torch.no_grad():
            a = self.actor(x).numpy().astype(float)
        t2 = time.perf_counter()
        return ({"predictions": a.tolist(), "actionLow": self.manifest["actionLow"], "actionHigh": self.manifest["actionHigh"],
                 "meaning": "deterministic actor actions (no exploration noise), inside the declared action bounds"},
                {"preprocessingMs": (t1 - t0) * 1000, "inferenceMs": (t2 - t1) * 1000, "postprocessingMs": None})

    def reference_observations(self) -> list[list[float]]:
        return json.loads(read_verified(self.store, self.manifest["referenceSha256"]))["observations"]

    def reference_records(self, n: int | None = None) -> list[dict[str, Any]]:
        return [{"observation": o} for o in self.reference_observations()[:n]]
