"""The DQN training loop (collection -> replay -> learner -> evaluation), independent of storage: everything observable goes through a `Sink`."""
from __future__ import annotations

import io
import json
import time
from typing import Any, Callable, Protocol

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from .buffer import ReplayBuffer
from .collector import Collector
from .dqn import DQNConfig, DQNLearner, epsilon_at, explain, param_sha256
from .envs import CATALOG, EnvSpec, RewardSpec, env_spaces, wrapper_chain
from .evaluate import evaluate_policy, frame_png
from .networks import build_network
from .trace import TraceConfig, TraceRecorder


class Cancelled(Exception):
    pass


class BufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capacity: int = Field(10000, ge=1)
    sampling: str = Field("uniform", pattern="^uniform$", description="only uniform sampling is implemented")


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seeds: list[int] = Field(default_factory=lambda: list(range(1000, 1010)), min_length=1, max_length=100, description="one evaluation episode per seed, on a separate environment instance")
    interval_steps: int = Field(0, ge=0, description="evaluate every this many environment steps (0 = only before training and at the end)")
    initial: bool = True
    epsilon: float = Field(0.0, ge=0, le=1)


class RLSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    env: EnvSpec
    reward: RewardSpec = Field(default_factory=RewardSpec)
    network: dict[str, Any]
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    dqn: DQNConfig = Field(default_factory=DQNConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)


class Sink(Protocol):
    def emit(self, type_: str, **data: Any) -> None: ...
    def put_artifact(self, kind: str, data: bytes, meta: dict[str, Any]) -> str: ...


class MemorySink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.artifacts: list[tuple[str, bytes, dict[str, Any]]] = []

    def emit(self, type_: str, **data: Any) -> None:
        self.events.append((type_, data))

    def put_artifact(self, kind: str, data: bytes, meta: dict[str, Any]) -> str:
        import hashlib

        self.artifacts.append((kind, data, meta))
        return hashlib.sha256(data).hexdigest()

    def of(self, type_: str) -> list[dict[str, Any]]:
        return [d for t, d in self.events if t == type_]


class TrainResult:
    def __init__(self, **kw: Any):
        self.__dict__.update(kw)


def _npz(arrs: dict[str, np.ndarray]) -> bytes:
    b = io.BytesIO()
    np.savez_compressed(b, **arrs)
    return b.getvalue()


def _state_bytes(net: torch.nn.Module) -> bytes:
    b = io.BytesIO()
    torch.save(net.state_dict(), b)
    return b.getvalue()


def check_compatibility(spec: RLSpec) -> list[tuple[str, str]]:
    """(code, message) pairs for environment/algorithm incompatibilities, decided from the environment's real spaces BEFORE anything runs."""
    from .compat import dqn_compatibility

    return dqn_compatibility(spec.env)


def train(spec: RLSpec, seed: int, sink: Sink, should_cancel: Callable[[], bool] = lambda: False, hook: Callable[[str, dict], None] | None = None) -> TrainResult:
    """Run DQN. `hook(name, payload)` (tests only) sees every gradient step: ('update', {batch, result, learner}) and ('insert', {...})."""
    bad = check_compatibility(spec)
    if bad:
        raise ValueError(f"{bad[0][0]}: {bad[0][1]}")
    c, tc = spec.dqn, spec.trace
    obs_sp, act_sp, _ = env_spaces(spec.env)
    obs_dim, n_actions = int(np.prod(obs_sp["shape"])), act_sp["n"]
    torch.manual_seed(seed)
    net = build_network(spec.network, obs_dim, n_actions)
    learner = DQNLearner(net, c)
    names = list(CATALOG[spec.env.env_id]["rewardComponents"])
    buf = ReplayBuffer(max(spec.buffer.capacity, c.batch_size), (obs_dim,), len(names), seed + 1)
    coll = Collector(spec.env, spec.reward, c.num_envs, seed, render_env0=tc.enabled and tc.frames and tc.max_episodes > 0)
    rec = TraceRecorder(tc, c, names)
    rng = np.random.default_rng(seed + 7919)
    sink.emit("rl_setup", wrapperChain=wrapper_chain(coll.envs.envs[0]), autoresetMode=str(coll.mode.value), declaredAutoresetMode=spec.env.autoreset_mode,
              components=names, effectiveWeights=coll.envs.envs[0].get_wrapper_attr("effective_weights")(), obsDim=obs_dim, nActions=n_actions, equation=explain(c),
              networkParams=sum(p.numel() for p in net.parameters()))

    evals: list[dict[str, Any]] = []
    t0 = time.time()
    tick, vec_steps = 0, 0

    def run_eval(final: bool) -> None:
        r = evaluate_policy(learner.q, spec.env, spec.reward, spec.eval.seeds, epsilon=spec.eval.epsilon, frames_for=(tc.eval_frames_episodes if final and tc.enabled and tc.frames else 0),
                            frame_width=tc.frame_width)
        rep = r["report"]
        sha = sink.put_artifact("rl_checkpoint", _state_bytes(learner.q), {"policyVersion": learner.version, "tick": tick, "paramSha256": param_sha256(learner.q), "purpose": "evaluation"})
        entry = {"tick": tick, "update": learner.updates, "policyVersion": learner.version, "elapsedSec": time.time() - t0, "final": final, "checkpointSha256": sha, **rep}
        for ep in r["captured"]:
            shas = [sink.put_artifact("rl_frame", f, {"evalTick": tick, "seed": ep["seed"]}) for f in ep["frames"]]
            entry.setdefault("capturedEpisodes", []).append({**{k: v for k, v in ep.items() if k != "frames"}, "frames": shas})
        evals.append(entry)
        sink.emit("eval", **{k: v for k, v in entry.items() if k != "episodes"}, episodeReturns=[e["taskReturn"] for e in rep["episodes"]], episodes=rep["episodes"])

    if spec.eval.initial:
        run_eval(False)

    # ---------------------------------------------------------------- captured episodes (environment 0 only; frames from the real env.render())
    points = [int(round(c.total_steps * k / max(1, tc.max_episodes))) for k in range(tc.max_episodes)] if tc.enabled else []
    cap: dict[str, Any] | None = None
    n_captured = 0

    def start_capture(episode_id: int) -> None:
        nonlocal cap, n_captured
        if not tc.enabled or n_captured >= len(points) or tick < points[n_captured]:
            return
        n_captured += 1
        cap = {"episodeId": episode_id, "envIndex": 0, "kind": "training", "startTick": tick, "steps": [], "frames": [], "closed": False, "framesCapped": False}
        if tc.frames:
            cap["frames"].append(sink.put_artifact("rl_frame", frame_png(coll.render(0), tc.frame_width), {"episode": episode_id, "k": 0}))

    def finish_capture(summary) -> None:
        nonlocal cap
        if cap is None:
            return
        cap.update({"length": summary.length, "capturedSteps": len(cap["steps"]), "return": summary.ret, "taskReturn": summary.task_return, "terminated": summary.terminated,
                    "truncated": summary.truncated, "components": summary.components, "endTick": tick,
                    "boundedCapture": summary.length > len(cap["steps"]), "frameNote": ("the environment was already reset when the episode ended (SameStep autoreset): the final frame is not available"
                                                                                         if coll.mode.value == "SameStep" else "frame k is the state before step k; the last frame is the final state")})
        sha = sink.put_artifact("rl_episode", json.dumps(cap).encode(), {"episodeId": cap["episodeId"], "startTick": cap["startTick"], "length": summary.length})
        sink.emit("episode_captured", episodeId=cap["episodeId"], sha256=sha, length=summary.length, capturedSteps=len(cap["steps"]), startTick=cap["startTick"])
        cap = None

    start_capture(int(coll.episode_id[0]))
    updates_since_log = 0
    tracked_total = 0
    while tick < c.total_steps:
        if should_cancel():
            raise Cancelled()
        eps = epsilon_at(c, tick)
        q = learner.act_values(coll.obs)
        greedy = q.argmax(axis=1)
        explore = rng.random(c.num_envs) < eps
        actions = np.where(explore, rng.integers(n_actions, size=c.num_envs), greedy)
        version = learner.version
        out = coll.step(actions)
        vec_steps += 1
        for tr in out.transitions:
            i = tr.env_index
            comps = np.array([tr.components[k] for k in names], np.float32)
            raws = np.array([tr.raw_components[k] for k in names], np.float32)
            tid, slot, evicted = buf.add(obs=tr.obs, action=tr.action, reward=tr.reward, next_obs=tr.next_obs, terminated=tr.terminated, truncated=tr.truncated, env_index=i,
                                         episode_id=tr.episode_id, episode_step=tr.episode_step, policy_version=version, tick=tick, eps=eps, action_source=int(explore[i]),
                                         components=comps, raw_components=raws)
            tick += 1
            rec.on_insert(tid, evicted, tick, learner.updates)
            if hook:
                hook("insert", {"tid": tid, "slot": slot, "transition": tr, "evicted": evicted})
            if i == 0 and cap is not None and tr.episode_id == cap["episodeId"]:
                k = tr.episode_step
                if k < tc.max_steps_per_episode:
                    r_ = buf.record(slot, names)
                    r_.update({"frameIndex": k, "qValues": q[0].tolist(), "greedyAction": int(greedy[0]), "episodeStartFlag": tr.episode_start, "kind": "training"})
                    if rec.track(r_):
                        tracked_total += 1
                        cap["steps"].append({"k": k, "tid": tid, "qValues": q[0].tolist(), "action": tr.action, "actionSource": "random" if explore[0] else "greedy", "epsilon": eps,
                                             "policyVersion": version, "reward": tr.reward, "components": tr.components, "rawComponents": tr.raw_components, "terminated": tr.terminated,
                                             "truncated": tr.truncated, "obs": tr.obs.tolist(), "nextObs": tr.next_obs.tolist()})
                        if tc.frames:
                            if coll.mode.value == "NextStep" or not (tr.terminated or tr.truncated):
                                cap["frames"].append(sink.put_artifact("rl_frame", frame_png(coll.render(0), tc.frame_width), {"episode": cap["episodeId"], "k": k + 1}))
                else:
                    cap["framesCapped"] = True
        for ep in out.episodes:
            sink.emit("episode_end", tick=tick, envIndex=ep.env_index, episodeId=ep.episode_id, length=ep.length, **{"return": ep.ret}, taskReturn=ep.task_return, components=ep.components,
                      rawComponents=ep.raw_components, terminated=ep.terminated, truncated=ep.truncated, policyVersion=learner.version, epsilon=eps)
            if ep.env_index == 0 and cap is not None and ep.episode_id == cap["episodeId"]:
                finish_capture(ep)
        # a new episode of environment 0 began (NextStep: its reset step; SameStep: inside the ending step) -> maybe capture it
        if (0 in out.reset_steps) or (coll.mode.value == "SameStep" and any(e.env_index == 0 for e in out.episodes)):
            start_capture(int(coll.episode_id[0]))
        if len(buf) >= c.learning_starts and vec_steps % c.train_freq == 0:
            for _ in range(c.gradient_steps):
                slots = buf.sample_slots(c.batch_size, learner.updates + 1)
                batch = buf.batch(slots)
                res = learner.update(batch)
                sha = param_sha256(learner.q) if (rec.tracked and any(int(t) in rec.tracked for t in batch["tid"])) else None
                rec.on_update(res, batch["tid"], slots, tick, len(buf), sha)
                if hook:
                    hook("update", {"batch": batch, "slots": slots, "result": res, "learner": learner, "tick": tick})
                if res.target_synced:
                    sink.emit("target_sync", update=res.update_index, tick=tick, paramSha256=param_sha256(learner.q))
                updates_since_log += 1
                if res.update_index % c.log_interval == 0:
                    sink.emit("train_update", update=res.update_index, tick=tick, loss=res.loss, gradNorm=res.grad_norm, meanQ=res.mean_q, epsilon=eps, bufferSize=len(buf),
                              policyVersion=learner.version, targetVersion=learner.target_version, meanTarget=float(res.target.mean()), elapsedSec=time.time() - t0)
        if spec.eval.interval_steps and tick // spec.eval.interval_steps > (tick - len(out.transitions)) // spec.eval.interval_steps and tick < c.total_steps:
            run_eval(False)
    if cap is not None:  # still open at the end of training: record what was captured, labelled as an unfinished episode
        cap.update({"length": None, "capturedSteps": len(cap["steps"]), "endTick": tick, "unfinished": True, "boundedCapture": True,
                    "frameNote": "training ended before this episode did"})
        sha = sink.put_artifact("rl_episode", json.dumps(cap).encode(), {"episodeId": cap["episodeId"], "startTick": cap["startTick"], "unfinished": True})
        sink.emit("episode_captured", episodeId=cap["episodeId"], sha256=sha, length=None, capturedSteps=len(cap["steps"]), startTick=cap["startTick"], unfinished=True)
        cap = None
    run_eval(True)
    elapsed = time.time() - t0
    sink.put_artifact("rl_trace", json.dumps(rec.export({"finalPolicyVersion": learner.version, "envSteps": tick})).encode(), {"tracked": len(rec.tracked), "updates": len(rec.updates)})
    sink.put_artifact("rl_buffer", _npz(buf.to_npz()), {"size": len(buf), "capacity": buf.capacity, "components": names, "nextTid": buf.next_tid, "finalPolicyVersion": learner.version,
                                                       "tick": tick, "obsDim": obs_dim})
    sink.put_artifact("rl_checkpoint", _state_bytes(learner.q), {"policyVersion": learner.version, "tick": tick, "paramSha256": param_sha256(learner.q), "purpose": "final"})
    summary = {"envSteps": tick, "vectorSteps": vec_steps, "gradientUpdates": learner.updates, "targetSyncs": learner.target_syncs, "elapsedSec": elapsed,
               "stepsPerSec": tick / elapsed if elapsed else None, "updateToDataRatio": learner.updates / max(1, tick), "bufferSize": len(buf), "transitionsInserted": buf.next_tid,
               "trackedTransitions": len(rec.tracked), "finalPolicyVersion": learner.version, "seed": seed, "capturedEpisodes": n_captured}
    coll.close()
    return TrainResult(learner=learner, buffer=buf, tracker=rec, evals=evals, summary=summary, final_eval=evals[-1])
