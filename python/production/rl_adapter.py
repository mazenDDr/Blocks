"""RL greedy-policy family for the local production registry (ADR 0019).

A version pins one completed DQN run's FINAL Q-network (native state dict, weights_only), its network model graph and environment spec,
the observation/action spaces described by the real Gymnasium environment, the evaluation report, and the native environment and
implementation hashes. Serving maps observation vectors to the greedy action argmax_a Q(s, a), with the Q-values. That is the
evaluation policy (epsilon = 0), not the epsilon-greedy behaviour policy used while training. Up to 256 replay-buffer observations are
frozen as the reference. Recorded behaviour actions are not treated as ground truth; agreement is measured only against actions a
person supplies."""
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
MAX_REFERENCE = 256
_PY = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = ("production/rl_adapter.py", "rl/networks.py", "rl/envs.py", "graph_core/lower.py", "graph_core/validate.py", "operations/layers.py",
                        "operations/core.py", "operations/_common.py")


def environment() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__, "gymnasium": dist_version("gymnasium"), "numpy": dist_version("numpy")}


def implementation() -> dict[str, str]:
    return {f: hashlib.sha256((_PY / f).read_bytes()).hexdigest() for f in IMPLEMENTATION_FILES}


def _final_checkpoint(store, run_id):
    cks = [a for a in store.artifacts(run_id, "rl_checkpoint") if a["meta"].get("purpose") == "final" and a["status"] == "complete"]
    return cks[-1] if cks else None


def is_candidate(store, row) -> str | None:
    """The q_network node of a completed RL run with a final Q-network checkpoint, else None."""
    if row["status"] != "completed" or row["config"].get("kind") != "rl" or _final_checkpoint(store, row["id"]) is None:
        return None
    from .model_adapter import _graph

    try:
        g, _ = _graph(store, row["id"])
    except ProductionError:
        return None
    return next((n.id for n in g.nodes if n.type == "rl.q_network"), None)


def build_manifest(store, run_id: str, node: str) -> dict[str, Any]:
    from rl.envs import CATALOG, env_spaces, make_env
    from rl.validate import spec_from_graph

    from .model_adapter import _graph

    row = store.get_run(run_id)
    if row is None or is_candidate(store, row) != node:
        raise ProductionError("E_PIPELINE_NOT_RECORDED", "Registration needs a completed RL run, its rl.q_network node and a final Q-network checkpoint.")
    g, graph_sha = _graph(store, run_id)
    spec = spec_from_graph(g)
    obs_sp, act_sp, extra = env_spaces(spec.env)
    if obs_sp["type"] != "Box" or act_sp["type"] != "Discrete":
        raise ProductionError("E_PIPELINE_UNSUPPORTED", "Only Box observations with Discrete actions are served.")
    obs_dim = int(np.prod(obs_sp["shape"]))
    env = make_env(spec.env, spec.reward)
    try:
        low, high = np.ravel(env.observation_space.low).astype(float), np.ravel(env.observation_space.high).astype(float)
    finally:
        env.close()
    ck = _final_checkpoint(store, run_id)
    bufs = store.artifacts(run_id, "rl_buffer")
    if not bufs:
        raise ProductionError("E_REFERENCE_MISSING", "The run has no recorded replay buffer; the reference cannot be frozen.", 409)
    with np.load(io.BytesIO(read_verified(store, bufs[-1]["sha256"]))) as z:
        obs = z["obs"][:MAX_REFERENCE].astype(float)
    reference_sha = store.put_bytes(dumps({"observations": obs.tolist(), "source": "replay buffer, oldest first", "bufferSha256": bufs[-1]["sha256"]}).encode())
    reports = store.artifacts(run_id, "rl_eval_report")
    final = json.loads(read_verified(store, reports[-1]["sha256"]))["final"] if reports else None
    return {"adapter": "native-pytorch-dqn-policy-local", "family": "rl_policy", "runId": run_id, "node": node, "graphHash": row["graph_hash"], "graphSha256": graph_sha,
            "modelSha256": ck["sha256"], "checkpointSha256": ck["sha256"], "policyVersion": ck["meta"].get("policyVersion"), "paramSha256": ck["meta"].get("paramSha256"),
            "network": spec.network, "envSpec": spec.env.model_dump(mode="json"), "envId": spec.env.env_id, "observationSpace": obs_sp, "actionSpace": act_sp,
            "observationDim": obs_dim, "observationLow": [None if not math.isfinite(v) else v for v in low], "observationHigh": [None if not math.isfinite(v) else v for v in high],
            "actionNames": CATALOG[spec.env.env_id].get("actions"), "timeLimit": extra.get("timeLimit"),
            "outputSchema": {"task": "policy", "classes": list(range(int(act_sp["n"]))), "target": None},
            "policy": "greedy: argmax_a Q(s, a) (evaluation policy, epsilon 0); not the epsilon-greedy behaviour policy",
            "referenceSha256": reference_sha, "referenceRows": len(obs), "referencePolicy": f"first {MAX_REFERENCE} replay-buffer observations (oldest first), frozen at registration",
            "source": {"envId": spec.env.env_id, "seed": row["config"].get("seed"), "bufferSha256": bufs[-1]["sha256"]},
            "evaluationArtifacts": [r["sha256"] for r in reports], "evaluation": final and {k: final.get(k) for k in ("return", "taskReturn", "successRate", "episodes") if k in final},
            "environment": environment(), "implementation": implementation(), "fitArtifacts": {},
            "inputContract": f"records: [{{observation: {obs_dim} finite numbers within the environment's observation bounds}}]",
            "tokenizer": None, "customCode": "not supported by this adapter"}


def verify(store, manifest) -> None:
    if manifest["environment"] != environment() or manifest["implementation"] != implementation():
        raise ProductionError("E_SERVING_ENVIRONMENT", "Pinned native environment or implementation differs; serving is refused.", 409)
    if _final_checkpoint(store, manifest["runId"]) is None or _final_checkpoint(store, manifest["runId"])["sha256"] != manifest["checkpointSha256"]:
        raise ProductionError("E_ARTIFACT_TRUST", "The pinned Q-network is not the run's recorded final checkpoint.", 409)
    for sha in (manifest["checkpointSha256"], manifest["graphSha256"], manifest["referenceSha256"]):
        read_verified(store, sha)


class PolicyPipeline:
    def __init__(self, store, manifest: dict[str, Any]):
        from rl.networks import build_network

        verify(store, manifest)
        self.store, self.manifest = store, manifest
        self.net = build_network(manifest["network"], manifest["observationDim"], int(manifest["actionSpace"]["n"]))
        state = torch.load(io.BytesIO(read_verified(store, manifest["checkpointSha256"])), map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
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
        x = torch.tensor([r["observation"] for r in records], dtype=torch.float32)
        with self.lock, torch.no_grad():
            q = self.net(x)
        t1 = time.perf_counter()
        actions = q.argmax(1).tolist()
        names = self.manifest.get("actionNames")
        return {"predictions": actions, "actionNames": [names[a] for a in actions] if names else None, "qValues": q.tolist(), "policy": "greedy"}, {
            "preprocessingMs": None, "inferenceMs": (t1 - t0) * 1000, "postprocessingMs": (time.perf_counter() - t1) * 1000}

    def reference_observations(self) -> list[list[float]]:
        return json.loads(read_verified(self.store, self.manifest["referenceSha256"]))["observations"]

    def reference_records(self, n: int | None = None) -> list[dict[str, Any]]:
        return [{"observation": o} for o in self.reference_observations()[:n]]
