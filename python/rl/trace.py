"""Bounded transition-to-update tracing (VISION 9.9 'Inspect a transition all the way into the update', A51).

Capturing every transition of a run is not affordable, so capture is BOUNDED and DECLARED in the run: the transitions of a few chosen episodes
(`TraceConfig`) are tracked; for each tracked transition the recorder keeps the transition, its buffer insertion (slot, tick, what it evicted,
when it was itself evicted), and every minibatch use (position in the batch, TD target, bootstrap mask, TD error, loss contribution, the gradient
of the loss with respect to its Q-value, policy/target versions). Untracked transitions are reported as not captured, never invented."""
from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .dqn import DQNConfig, UpdateResult


class TraceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    max_episodes: int = Field(4, ge=0, le=20, description="training episodes of environment 0 captured (frames, per-step Q-values) and fully tracked")
    max_steps_per_episode: int = Field(120, ge=1, le=1000, description="leading transitions of a captured episode that are tracked")
    max_tracked: int = Field(400, ge=0, le=5000, description="hard cap on tracked transitions over the run")
    frames: bool = True
    frame_width: int = Field(320, ge=64, le=800)
    max_update_records: int = Field(400, ge=0, le=5000, description="minibatch-level records kept (each lists the whole batch's transition ids)")
    max_uses_per_transition: int = Field(40, ge=1, le=500)
    eval_frames_episodes: int = Field(1, ge=0, le=5, description="evaluation episodes rendered at the final evaluation")


class TraceRecorder:
    def __init__(self, cfg: TraceConfig, dqn: DQNConfig, component_names: list[str]):
        self.cfg, self.dqn, self.components = cfg, dqn, component_names
        self.tracked: dict[int, dict[str, Any]] = {}
        self.updates: list[dict[str, Any]] = []
        self.update_records_dropped = 0
        self.uses_dropped = 0
        self.tracked_dropped = 0

    def is_tracked(self, tid: int) -> bool:
        return tid in self.tracked

    def track(self, record: dict[str, Any]) -> bool:
        if not self.cfg.enabled or len(self.tracked) >= self.cfg.max_tracked:
            self.tracked_dropped += 1
            return False
        record["uses"], record["evicted"] = [], None
        self.tracked[record["tid"]] = record
        return True

    def on_insert(self, tid: int, evicted_tid: int, tick: int, update_index: int) -> None:
        if evicted_tid in self.tracked:
            self.tracked[evicted_tid]["evicted"] = {"byTid": tid, "tick": tick, "afterUpdate": update_index}

    def on_update(self, res: UpdateResult, batch_tids: np.ndarray, slots: np.ndarray, tick: int, buffer_size: int, param_sha256: str | None = None) -> None:
        if not self.tracked:
            return
        hit = [i for i, t in enumerate(batch_tids) if int(t) in self.tracked]
        if not hit:
            return
        B = len(batch_tids)
        c = self.dqn
        for i in hit:
            t = self.tracked[int(batch_tids[i])]
            if len(t["uses"]) >= self.cfg.max_uses_per_transition:
                self.uses_dropped += 1
                continue
            e = float(res.td_error[i])
            dq = (np.clip(e, -c.huber_delta, c.huber_delta) if c.loss == "huber" else 2 * e) / B
            t["uses"].append({
                "update": res.update_index, "batchPosition": i, "batchSize": B, "slot": int(slots[i]), "samplingProbability": B / max(1, buffer_size), "tick": tick,
                "qSA": float(res.q_sa[i]), "nextValue": float(res.next_value[i]), "bootstrapMask": float(res.mask[i]), "target": float(res.target[i]), "tdError": e,
                "lossContribution": float(res.loss_contribution[i]), "dLossDQ": float(dq), "huberRegion": (None if c.loss != "huber" else "quadratic" if abs(e) <= c.huber_delta else "linear"),
                "batchLoss": res.loss, "gradNorm": res.grad_norm, "policyVersionBefore": res.version_before, "policyVersionAfter": res.version_after,
                "targetVersion": res.target_version, "paramSha256After": param_sha256})
        if len(self.updates) < self.cfg.max_update_records:
            self.updates.append({"update": res.update_index, "tick": tick, "loss": res.loss, "gradNorm": res.grad_norm, "meanQ": res.mean_q, "bufferSize": buffer_size,
                                 "policyVersionBefore": res.version_before, "policyVersionAfter": res.version_after, "targetVersion": res.target_version, "targetSynced": res.target_synced,
                                 "paramSha256After": param_sha256, "batchTids": [int(x) for x in batch_tids], "trackedInBatch": [int(batch_tids[i]) for i in hit]})
        else:
            self.update_records_dropped += 1

    def export(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"capturePolicy": {**self.cfg.model_dump(), "statement": (
            f"BOUNDED CAPTURE: only the first {self.cfg.max_steps_per_episode} transitions of up to {self.cfg.max_episodes} training episodes of environment 0 are tracked "
            f"(at most {self.cfg.max_tracked} transitions, {self.cfg.max_uses_per_transition} recorded uses each, {self.cfg.max_update_records} minibatch records). "
            "Every other transition is not captured.")},
                "dropped": {"trackedTransitions": self.tracked_dropped, "minibatchRecords": self.update_records_dropped, "uses": self.uses_dropped},
                "components": self.components, "transitions": {str(k): v for k, v in self.tracked.items()}, "updates": self.updates, **(extra or {})}
