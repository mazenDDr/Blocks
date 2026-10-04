"""DQN training end to end: transition-to-update tracing (A51), evaluation protocol, determinism, the worker run, and learning on CartPole."""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
import torch

from rl.dqn import param_sha256
from rl.envs import EnvSpec, RewardSpec
from rl.evaluate import evaluate_policy, mean_ci
from rl.networks import build_network, mlp_graph
from rl.samples import cartpole_graph, gridworld_graph
from rl.train import BufferConfig, DQNConfig, EvalConfig, MemorySink, RLSpec, TraceConfig, train
from rl.validate import spec_from_graph
from graph_core.schema import Graph


def small_spec(total=500, num_envs=1, mode="NextStep", **trace):
    return RLSpec(env=EnvSpec(env_id="CartPole-v1", autoreset_mode=mode), network=mlp_graph(4, (16,), 2),
                  buffer=BufferConfig(capacity=300),
                  dqn=DQNConfig(total_steps=total, num_envs=num_envs, learning_starts=60, batch_size=16, eps_decay_steps=200, target_update_interval=25, log_interval=10),
                  eval=EvalConfig(seeds=[1000, 1001, 1002]), trace=TraceConfig(frames=False, max_episodes=3, max_steps_per_episode=40, **trace))


# ------------------------------------------------------------------------------------------------ A51
def test_a51_trace_follows_a_transition_into_the_update_and_recomputes(tmp_path):
    spec = small_spec(500)
    snaps = {}      # update index -> the target network used by that update
    # capture the target network used by every update by wrapping the learner's update
    import rl.train as T
    orig = T.DQNLearner.update

    def spy(self, batch):
        tgt = copy.deepcopy(self.q_target)
        r = orig(self, batch)
        snaps[r.update_index] = tgt
        return r
    T.DQNLearner.update = spy
    try:
        sink = MemorySink()
        res = train(spec, 3, sink)
    finally:
        T.DQNLearner.update = orig
    tr = res.tracker.export()
    assert tr["capturePolicy"]["statement"].startswith("BOUNDED CAPTURE") and 0 < len(tr["transitions"]) <= res.tracker.cfg.max_episodes * 40
    used = [t for t in tr["transitions"].values() if t["uses"]]
    assert used, "at least one tracked transition must have been sampled into a minibatch"
    t = max(used, key=lambda t: len(t["uses"]))
    buf_art = [a for a in sink.artifacts if a[0] == "rl_buffer"][0]
    # (1) the transition: id, episode, step, acting policy version, reward components that sum to the reward, flags
    assert t["tid"] >= 0 and t["episodeId"] >= 0 and t["policyVersion"] >= 0 and t["kind"] == "training"
    assert sum(t["components"].values()) == pytest.approx(t["reward"])
    # (2) the buffer insertion: slot and tick are consistent with the id (ring buffer of 300)
    assert t["slot"] == t["tid"] % 300 and t["insertTick"] <= t["tid"] + 1
    # (3) every recorded use is a real minibatch occurrence whose target/loss recompute from the target network of that update
    for u in t["uses"]:
        tgt = snaps[u["update"]]
        with torch.no_grad():
            nv = tgt(torch.tensor(t["nextObs"], dtype=torch.float32)[None]).max(1).values.item()
        mask = 0.0 if t["terminated"] else 1.0
        assert u["bootstrapMask"] == mask and u["nextValue"] == pytest.approx(nv, abs=1e-5)
        assert u["target"] == pytest.approx(t["reward"] + 0.99 * mask * nv, abs=1e-5)
        assert u["tdError"] == pytest.approx(u["qSA"] - u["target"], abs=1e-5)
        e = u["tdError"]
        huber = 0.5 * e * e if abs(e) <= 1 else abs(e) - 0.5
        assert u["lossContribution"] == pytest.approx(huber / u["batchSize"], rel=1e-4, abs=1e-8)
        assert u["policyVersionAfter"] == u["policyVersionBefore"] + 1 == u["update"]
        assert 0 < u["samplingProbability"] <= 1 and u["samplingProbability"] == pytest.approx(u["batchSize"] / min(u["tick"], 300), rel=0.1)
    # (4) the minibatch record lists the whole batch and says which tracked ids were in it
    rec = {r["update"]: r for r in tr["updates"]}
    for u in t["uses"]:
        r = rec.get(u["update"])
        if r:
            assert t["tid"] in r["batchTids"] and t["tid"] in r["trackedInBatch"] and r["batchTids"][u["batchPosition"]] == t["tid"]
            assert r["policyVersionAfter"] == u["policyVersionAfter"] and len(r["paramSha256After"]) == 64
    # (5) the resulting network version: the checkpoint at the end carries the final version and the hash of the final parameters
    final = [a for a in sink.artifacts if a[0] == "rl_checkpoint" and a[2]["purpose"] == "final"][0]
    assert final[2]["policyVersion"] == res.learner.updates == res.summary["gradientUpdates"] and final[2]["paramSha256"] == param_sha256(res.learner.q)
    loaded = torch.load(io.BytesIO(final[1]))
    assert all(torch.equal(loaded[k], v) for k, v in res.learner.q.state_dict().items())


def test_a51_untracked_transitions_are_not_invented_and_caps_are_declared():
    spec = small_spec(400, max_tracked=25, max_update_records=3, max_uses_per_transition=2)
    res = train(spec, 0, MemorySink())
    tr = res.tracker.export()
    assert len(tr["transitions"]) == 25 and tr["dropped"]["trackedTransitions"] > 0
    assert len(tr["updates"]) <= 3 and all(len(t["uses"]) <= 2 for t in tr["transitions"].values())
    assert "25 transitions" in tr["capturePolicy"]["statement"] or "at most 25" in tr["capturePolicy"]["statement"]
    ids = {int(k) for k in tr["transitions"]}
    assert max(ids) < res.buffer.next_tid and not res.tracker.is_tracked(res.buffer.next_tid + 5)


def test_trace_disabled_records_nothing():
    spec = small_spec(300)
    spec.trace = TraceConfig(enabled=False, frames=False)
    res = train(spec, 0, MemorySink())
    assert res.tracker.export()["transitions"] == {} and res.tracker.export()["updates"] == []


def test_buffer_contents_are_correct_after_a_vector_run_in_both_modes():
    for mode in ("NextStep", "SameStep"):
        sink = MemorySink()
        res = train(small_spec(400, num_envs=3, mode=mode), 4, sink)
        b = res.buffer
        assert b.next_tid == res.summary["envSteps"] and 400 <= b.next_tid < 403         # one insertion per real transition (reset steps never count); overshoot < num_envs
        done = np.nonzero(b.terminated[:len(b)] | b.truncated[:len(b)])[0]
        assert len(done)
        # the stored next observation of every ending transition is a true final observation: CartPole terminal states exceed the thresholds, never a fresh reset state
        for i in done:
            if b.terminated[i]:
                x, th = b.next_obs[i][0], b.next_obs[i][2]
                assert abs(x) > 2.4 or abs(th) > 0.2095 - 1e-6
        # and a transition's next obs equals the following transition's obs of the same environment/episode
        order = np.argsort(b.tid[:len(b)])
        last = {}
        for i in order:
            key = (int(b.env_index[i]), int(b.episode_id[i]))
            if key in last:
                assert np.allclose(last[key][0], b.obs[i]) and last[key][1] + 1 == b.episode_step[i]
            last[key] = (b.next_obs[i], int(b.episode_step[i]))


# ------------------------------------------------------------------------------------------------ evaluation protocol
def test_evaluation_uses_separate_env_instances_fixed_seeds_and_reports_both_reward_bases():
    net = build_network(mlp_graph(4, (8,), 2), 4, 2)
    spec = EnvSpec(env_id="CartPole-v1", wrappers=[{"type": "ClipReward", "min": 0.0, "max": 0.25}])
    a = evaluate_policy(net, spec, RewardSpec(), [1, 2, 3])["report"]
    b = evaluate_policy(net, spec, RewardSpec(), [1, 2, 3])["report"]
    assert a["episodes"] == b["episodes"]                                           # same seeds -> identical initial conditions -> identical greedy rollouts
    assert a["seeds"] == [1, 2, 3] and a["return"]["n"] == 3
    ep = a["episodes"][0]
    assert ep["return"] == pytest.approx(ep["length"] * 0.25) and ep["taskReturn"] == ep["length"]    # trained-on (clipped) vs task (native) return
    assert ep["terminated"] != ep["truncated"] and a["terminatedCount"] + a["truncatedCount"] == 3
    assert set(a["componentReturns"]) == {"env_reward"} and a["successRule"]
    c = evaluate_policy(net, spec, RewardSpec(), [4, 5, 6])["report"]
    assert c["episodes"] != a["episodes"]


def test_mean_ci_is_a_student_t_interval():
    r = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert r["mean"] == 2.5 and r["ci"][0] < 2.5 < r["ci"][1] and r["ci"][1] - 2.5 == pytest.approx(3.182446 * np.std([1, 2, 3, 4], ddof=1) / 2, rel=1e-5)
    assert mean_ci([1.0])["ci"] is None and mean_ci([])["mean"] is None


def test_eval_does_not_touch_training_counters_or_buffer():
    spec = small_spec(300)
    spec.eval = EvalConfig(seeds=[7, 8], interval_steps=100)
    sink = MemorySink()
    res = train(spec, 0, sink)
    assert res.buffer.next_tid == 300 == res.summary["envSteps"]
    evals = sink.of("eval")
    assert [e["tick"] for e in evals] == [0, 100, 200, 300] and evals[0]["update"] == 0 and evals[-1]["final"]
    assert all(e["seeds"] == [7, 8] for e in evals)


def test_training_is_deterministic_for_a_seed():
    a = train(small_spec(300), 5, MemorySink())
    b = train(small_spec(300), 5, MemorySink())
    c = train(small_spec(300), 6, MemorySink())
    assert param_sha256(a.learner.q) == param_sha256(b.learner.q) != param_sha256(c.learner.q)
    assert a.final_eval["episodes"] == b.final_eval["episodes"]


def test_invalid_combination_is_rejected_before_training():
    spec = small_spec(100)
    spec.env = EnvSpec(env_id="Pendulum-v1")
    with pytest.raises(ValueError, match="E_RL_ALGO_ACTION_SPACE"):
        train(spec, 0, MemorySink())


def test_events_are_unsmoothed_and_complete():
    sink = MemorySink()
    res = train(small_spec(500), 1, sink)
    eps = sink.of("episode_end")
    assert sum(e["length"] for e in eps) <= 500 and all(isinstance(e["return"], float) for e in eps)
    assert all(e["terminated"] != e["truncated"] or e["length"] == 500 for e in eps)
    ups = sink.of("train_update")
    assert [u["update"] for u in ups] == list(range(10, res.summary["gradientUpdates"] + 1, 10))
    assert {"loss", "gradNorm", "meanQ", "epsilon", "bufferSize", "policyVersion"} <= set(ups[0])
    setup = sink.of("rl_setup")[0]
    assert setup["autoresetMode"] == "NextStep" and setup["equation"]["targetEquation"].startswith("y_i = r_i + 0.99 * (1 - terminated_i)")


# ------------------------------------------------------------------------------------------------ rollout frames
def test_captured_episode_frames_are_real_renders_and_align_with_steps():
    spec = RLSpec.model_validate({**small_spec(300).model_dump(), "trace": TraceConfig(frames=True, max_episodes=2, max_steps_per_episode=30, frame_width=160).model_dump()})
    sink = MemorySink()
    res = train(spec, 0, sink)
    eps = [json.loads(a[1]) for a in sink.artifacts if a[0] == "rl_episode"]
    assert len(eps) == 2
    frames = {__import__("hashlib").sha256(a[1]).hexdigest(): a[1] for a in sink.artifacts if a[0] == "rl_frame"}
    e = eps[0]
    assert len(e["frames"]) == len(e["steps"]) + 1 and all(f in frames for f in e["frames"]) and frames[e["frames"][0]][:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    im = Image.open(io.BytesIO(frames[e["frames"][0]]))
    assert im.width == 160
    assert len(set(e["frames"])) > 1                                                  # the cart actually moves between frames
    assert [s["k"] for s in e["steps"]] == list(range(len(e["steps"])))
    assert e["steps"][0]["tid"] in {int(k) for k in res.tracker.export()["transitions"]}


# ------------------------------------------------------------------------------------------------ the learner improves the policy
def test_dqn_improves_cartpole_return_over_the_untrained_policy():
    """Slower test (about 10 seconds): same fixed seeds before and after training; the greedy policy must be several times better."""
    g = Graph.model_validate(cartpole_graph(12000, evaluation={"seeds": list(range(1000, 1010)), "interval_steps": 4000}, trace={"enabled": False}))
    spec = spec_from_graph(g)
    sink = MemorySink()
    res = train(spec, 0, sink)
    rets = [e["taskReturn"]["mean"] for e in res.evals]
    assert rets[0] < 40, rets                              # untrained greedy policy
    assert max(rets[1:]) >= 100 and rets[-1] > 2 * rets[0], rets
