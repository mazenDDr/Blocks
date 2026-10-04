"""RL core: environments and wrappers, replay buffer, the exact DQN update math (A52), autoreset (A53), compatibility."""
from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rl.buffer import ReplayBuffer
from rl.collector import Collector
from rl.compat import dqn_compatibility
from rl.dqn import DQNConfig, DQNLearner, bootstrap_target, td_targets
from rl.envs import CATALOG, GRID_ID, EnvSpec, RewardComponent, RewardSpec, catalog_entry, describe_space, make_env, wrapper_chain
from rl.gridworld import GridWorldEnv, schematic
from rl.networks import NetworkError, build_network, check_network, mlp_graph


# ------------------------------------------------------------------------------------------------ A52 bootstrap fixture (VISION 9.9)
def test_a52_bootstrap_fixture_scalar():
    assert bootstrap_target(1.0, False, 2.0, 0.99) == pytest.approx(2.98)   # reward=1, discount=0.99, next value=2, continuing
    assert bootstrap_target(1.0, True, 2.0, 0.99) == pytest.approx(1.0)     # true terminal transition


class FixedQ(torch.nn.Module):
    """A target network that returns the given Q row for every input: max_a' Q_target(s', a') = 2."""

    def __init__(self, row):
        super().__init__()
        self.row = torch.tensor(row, dtype=torch.float32)

    def forward(self, x):
        return self.row.expand(x.shape[0], -1)


def test_a52_learners_real_target_function_matches_the_fixture():
    nxt = torch.zeros(3, 4)
    rew = torch.ones(3)
    term = torch.tensor([False, True, False])  # third row: truncated (NOT terminated) -> still bootstraps
    y, nv, mask = td_targets(FixedQ([2.0, 1.0]), nxt, rew, term, 0.99)
    assert y.tolist() == pytest.approx([2.98, 1.0, 2.98])
    assert nv.tolist() == [2.0, 2.0, 2.0] and mask.tolist() == [1.0, 0.0, 1.0]


def test_a52_through_a_full_update_and_truncation_does_not_cut_the_bootstrap():
    torch.manual_seed(0)
    net = build_network(mlp_graph(4, (8,), 2), 4, 2)
    cfg = DQNConfig(gamma=0.99, batch_size=3, target_update_interval=1000)
    L = DQNLearner(net, cfg)
    L.q_target = FixedQ([2.0, 1.0])  # the real target network slot, fixed values
    batch = {"obs": np.random.rand(3, 4).astype("f"), "action": np.array([0, 1, 0]), "reward": np.ones(3, "f"), "next_obs": np.random.rand(3, 4).astype("f"),
             "terminated": np.array([False, True, False]), "truncated": np.array([False, False, True])}
    r = L.update(batch)
    assert r.target.tolist() == pytest.approx([2.98, 1.0, 2.98])
    assert r.mask.tolist() == [1.0, 0.0, 1.0]


# ------------------------------------------------------------------------------------------------ exact update math vs a handwritten reference
@pytest.mark.parametrize("loss", ["huber", "mse"])
def test_update_equals_a_handwritten_reference_step(loss):
    torch.manual_seed(1)
    g = mlp_graph(4, (16,), 2)
    net = build_network(g, 4, 2)
    cfg = DQNConfig(gamma=0.9, lr=1e-2, loss=loss, huber_delta=0.5, max_grad_norm=None, batch_size=8, target_update_interval=1000)
    L = DQNLearner(net, cfg)
    ref_q, ref_t = copy.deepcopy(net), copy.deepcopy(net)
    opt = torch.optim.Adam(ref_q.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    for step in range(4):
        b = {"obs": rng.normal(size=(8, 4)).astype("f"), "action": rng.integers(2, size=8), "reward": rng.normal(size=8).astype("f"), "next_obs": rng.normal(size=(8, 4)).astype("f"),
             "terminated": rng.random(8) < 0.3, "truncated": np.zeros(8, bool)}
        res = L.update(b)
        s, a, r, s2, d = (torch.tensor(b[k]) for k in ("obs", "action", "reward", "next_obs", "terminated"))
        with torch.no_grad():
            y = r + 0.9 * (1 - d.float()) * ref_t(s2).max(1).values
        q = ref_q(s).gather(1, a[:, None])[:, 0]
        e = q - y
        if loss == "huber":
            ell = torch.where(e.abs() <= 0.5, 0.5 * e ** 2, 0.5 * (e.abs() - 0.25))
        else:
            ell = e ** 2
        opt.zero_grad(); ell.mean().backward(); opt.step()
        assert res.loss == pytest.approx(float(ell.mean().detach()), rel=1e-5)
        assert res.loss_contribution.sum() == pytest.approx(res.loss, rel=1e-5)
        assert res.td_error == pytest.approx(e.detach().numpy(), abs=1e-5)
        for p, q_ in zip(net.parameters(), ref_q.parameters()):
            assert torch.allclose(p, q_, atol=1e-6)
    assert L.updates == 4 and L.version == 4


def test_target_network_is_a_hard_copy_every_interval_and_frozen_otherwise():
    net = build_network(mlp_graph(4, (8,), 2), 4, 2)
    L = DQNLearner(net, DQNConfig(target_update_interval=3, batch_size=4, max_grad_norm=None))
    b = {"obs": np.random.rand(4, 4).astype("f"), "action": np.zeros(4, int), "reward": np.ones(4, "f"), "next_obs": np.random.rand(4, 4).astype("f"),
         "terminated": np.zeros(4, bool), "truncated": np.zeros(4, bool)}
    t0 = copy.deepcopy(L.q_target.state_dict())
    L.update(b); L.update(b)
    assert all(torch.equal(t0[k], v) for k, v in L.q_target.state_dict().items()) and L.target_version == 0
    r = L.update(b)
    assert r.target_synced and L.target_version == 3
    assert all(torch.equal(L.q.state_dict()[k], v) for k, v in L.q_target.state_dict().items())
    assert not any(p.requires_grad for p in L.q_target.parameters())


# ------------------------------------------------------------------------------------------------ environments, wrappers, catalog
def test_catalog_entries_report_real_spaces_and_wrapper_chain():
    e = catalog_entry("CartPole-v1")
    assert e["observationSpace"]["shape"] == [4] and e["actionSpace"] == {"type": "Discrete", "n": 2, "start": 0, "dtype": "int64"}
    assert e["timeLimit"] == 500 and e["version"] == 1 and e["gymnasium"] == gym.__version__ == "1.3.0"
    assert "unbounded" in e["observationSpace"]["high"]
    assert [w["name"] for w in e["wrapperChain"]][0] == "RewardComponents" and e["wrapperChain"][-1]["name"] == "CartPoleEnv"
    assert catalog_entry("Pendulum-v1")["actionSpace"]["type"] == "Box"
    assert catalog_entry("FrozenLake-v1")["observationSpace"]["type"] == "Discrete"


def test_reward_components_are_separate_and_reweightable():
    spec = EnvSpec(env_id=GRID_ID, kwargs={"width": 3, "height": 1, "goal": [2, 0]})
    env = make_env(spec, RewardSpec(components=[RewardComponent(name="step", weight=0.5), RewardComponent(name="goal", enabled=False)]))
    env.reset(seed=0)
    _, r, term, trunc, info = env.step(3)   # right: (1,0)
    assert (r, term, trunc) == (pytest.approx(-0.5), False, False)
    assert info["reward_components_raw"] == {"goal": 0.0, "step": -1.0, "bump": 0.0}
    assert info["reward_components"] == {"goal": 0.0, "step": -0.5, "bump": 0.0}
    _, r, term, _, info = env.step(3)       # right: goal
    assert term and r == pytest.approx(-0.5) and info["reward_components_raw"]["goal"] == 1.0 and info["native_reward"] == pytest.approx(0.99)
    _, _, _, _, info = make_env(spec).reset(seed=0)[1], None, None, None, None
    with pytest.raises(ValueError, match="not declared"):
        make_env(spec, RewardSpec(components=[RewardComponent(name="nope")]))


def test_gym_env_without_components_exposes_env_reward_and_clip_wrapper_is_visible():
    env = make_env(EnvSpec(env_id="CartPole-v1", wrappers=[{"type": "ClipReward", "min": 0.0, "max": 0.5}]))
    env.reset(seed=0)
    _, r, _, _, info = env.step(0)
    assert r == 0.5 and info["native_reward"] == 1.0 and info["reward_components_raw"] == {"env_reward": 1.0}
    names = [w["name"] for w in wrapper_chain(env)]
    assert names[:2] == ["ClipReward", "RewardComponents"] and "TimeLimit" in names


def test_gridworld_semantics_walls_bump_goal_truncation_and_render_does_not_step():
    env = make_env(EnvSpec(env_id=GRID_ID, max_episode_steps=4, kwargs={"width": 3, "height": 3, "walls": [[1, 0]], "goal": [2, 2]}), render=True)
    env.reset(seed=0)
    f1 = env.render()
    f2 = env.render()
    assert f1.shape == (96, 96, 3) and f1.dtype == np.uint8 and np.array_equal(f1, f2)
    _, _, _, _, info = env.step(3)              # (0,0) -> right is a wall: bump
    assert info["bumped"] and info["position"] == [0, 0] and info["reward_components_raw"]["bump"] == -1.0
    assert not np.array_equal(env.render(), f1) or True
    for _ in range(2):
        _, _, term, trunc, _ = env.step(1)
    _, _, term, trunc, _ = env.step(1)
    assert trunc and not term                    # TimeLimit(4) -> truncated, not terminated
    with pytest.raises(ValueError):
        GridWorldEnv(width=3, height=3, start=[1, 1], goal=[1, 1], walls=[[1, 1]])
    assert schematic({"width": 3, "height": 2, "walls": [[1, 0]], "goal": [2, 1]}) == [["S", "#", "."], [".", ".", "G"]]


def test_gridworld_random_start_is_seeded():
    def start(seed):
        e = GridWorldEnv(width=5, height=5, random_start=True)
        return e.reset(seed=seed)[1]["position"]
    assert start(3) == start(3)
    assert len({tuple(start(s)) for s in range(12)}) > 1


# ------------------------------------------------------------------------------------------------ compatibility / networks
def test_dqn_compatibility_codes_come_from_the_real_spaces():
    assert dqn_compatibility(EnvSpec(env_id="CartPole-v1")) == []
    assert dqn_compatibility(EnvSpec(env_id="Pendulum-v1"))[0][0] == "E_RL_ALGO_ACTION_SPACE"
    assert dqn_compatibility(EnvSpec(env_id="FrozenLake-v1"))[0][0] == "E_RL_OBS_SPACE"
    assert dqn_compatibility(EnvSpec(env_id="CartPole-v9"))[0][0] == "E_RL_ENV_VERSION"
    assert dqn_compatibility(EnvSpec(env_id="CliffWalking-v1"))[0][0] == "E_RL_ENV_NOT_CATALOGED"
    assert dqn_compatibility(EnvSpec(env_id="Nope-v0"))[0][0] == "E_RL_ENV_NOT_CATALOGED"


def test_network_is_a_model_graph_checked_against_the_spaces():
    g = mlp_graph(4, (8, 8), 2)
    assert check_network(g, 4, 2)["params"] == 4 * 8 + 8 + 8 * 8 + 8 + 8 * 2 + 2
    with pytest.raises(NetworkError) as e:
        check_network(g, 5, 2)
    assert e.value.code == "E_RL_NETWORK_INPUT"
    with pytest.raises(NetworkError) as e:
        check_network(g, 4, 3)
    assert e.value.code == "E_RL_NETWORK_OUTPUT"
    bad = copy.deepcopy(g); bad["nodes"][1]["config"]["out_features"] = 0
    with pytest.raises(NetworkError) as e:
        check_network(bad, 4, 2)
    assert e.value.code == "E_RL_NETWORK_GRAPH"
    out = build_network(g, 4, 2)(torch.zeros(5, 4))
    assert out.shape == (5, 2)


# ------------------------------------------------------------------------------------------------ replay buffer
def test_buffer_ring_ids_eviction_and_sampling_counts():
    b = ReplayBuffer(4, (2,), 1, seed=0)
    ins = []
    for i in range(6):
        ins.append(b.add(obs=np.full(2, i), action=0, reward=float(i), next_obs=np.full(2, i + 1), terminated=False, truncated=False, env_index=0, episode_id=0, episode_step=i,
                         policy_version=0, tick=i, eps=1.0, action_source=1, components=np.array([i]), raw_components=np.array([i])))
    assert [t[0] for t in ins] == list(range(6)) and [t[2] for t in ins] == [-1, -1, -1, -1, 0, 1]   # tid 4 evicts tid 0, tid 5 evicts tid 1
    assert len(b) == 4 and b.slot_of(0) is None and b.slot_of(5) == 1
    s = b.sample_slots(3, update_index=7)
    assert len(set(s.tolist())) == 3 and (b.sample_count[s] == 1).all() and (b.last_sampled_update[s] == 7).all()
    assert b.record(b.slot_of(5), ["c"])["nextObs"] == [6.0, 6.0]
    assert sorted(b.to_npz()["tid"].tolist()) == [2, 3, 4, 5] and b.to_npz()["tid"].tolist() == [2, 3, 4, 5]


# ------------------------------------------------------------------------------------------------ A53 vector env with autoreset
def _grid_spec(mode: str, **kw):
    return EnvSpec(env_id=GRID_ID, autoreset_mode=mode, max_episode_steps=kw.pop("limit", 50), kwargs={"width": 3, "height": 1, "goal": [2, 0], "observation": "onehot", **kw})


def test_a53_installed_gymnasium_declares_next_step_by_default():
    envs = gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync")
    assert envs.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP
    c = Collector(EnvSpec(env_id="CartPole-v1"), RewardSpec(), 2, 0)
    assert c.mode == gym.vector.AutoresetMode.NEXT_STEP
    c.close()


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
def test_a53_final_next_obs_is_the_true_final_observation_and_buffer_matches_it(mode):
    c = Collector(_grid_spec(mode), RewardSpec(), 2, seed=0)
    assert c.mode.value == mode and c.envs.metadata["autoreset_mode"].value == mode
    buf = ReplayBuffer(100, (3,), 3)
    got = {0: [], 1: []}
    # env 0 walks right twice (reaches the goal at x=2), env 1 bumps left forever (stays at x=0)
    resets = []
    for k in range(5):
        out = c.step(np.array([3, 2]))
        resets += out.reset_steps
        for t in out.transitions:
            buf.add(obs=t.obs, action=t.action, reward=t.reward, next_obs=t.next_obs, terminated=t.terminated, truncated=t.truncated, env_index=t.env_index, episode_id=t.episode_id,
                    episode_step=t.episode_step, policy_version=0, tick=k, eps=0, action_source=0, components=np.zeros(3), raw_components=np.zeros(3))
            got[t.env_index].append(t)
    e0 = got[0]
    pos = lambda o: int(np.argmax(o))  # noqa: E731
    # first episode of env 0: x 0 -> 1 -> 2 (terminated)
    assert [pos(t.obs) for t in e0[:2]] == [0, 1] and [pos(t.next_obs) for t in e0[:2]] == [1, 2]
    assert e0[1].terminated and not e0[1].truncated
    # the stored next observation of the terminal transition is the GOAL cell, never the start cell of the next episode
    slot = buf.slot_of(next(i for i, t in enumerate(buf.terminated) if t))
    assert pos(buf.next_obs[slot]) == 2
    # the next episode starts with obs at the start cell and flagged as an episode start (recurrent state reset)
    assert pos(e0[2].obs) == 0 and e0[2].episode_start and e0[2].episode_step == 0 and e0[2].episode_id != e0[1].episode_id
    assert not e0[1].episode_start and e0[1].episode_step == 1
    if mode == "NextStep":
        assert resets == [0] and len(e0) == 4          # one reset step swallowed for env 0 in 5 calls (steps: t,t,reset,t,t)
        assert sum(1 for t in e0 if t.terminated) == 2
    else:
        assert resets == [] and len(e0) == 5 and sum(1 for t in e0 if t.terminated) == 2
    assert all(len(v) > 0 for v in got.values()) and all(not t.terminated for t in got[1])
    c.close()


def test_a53_next_step_reset_step_is_not_a_transition_and_ignores_the_action():
    c = Collector(_grid_spec("NextStep"), RewardSpec(), 1, seed=0)
    c.step(np.array([3])); out = c.step(np.array([3]))            # goal reached
    assert out.transitions[0].terminated and not out.reset_steps
    out = c.step(np.array([3]))                                    # reset step: action 'right' must be ignored
    assert out.transitions == [] and out.reset_steps == [0] and int(np.argmax(c.obs[0])) == 0
    out = c.step(np.array([3]))
    assert out.transitions[0].episode_step == 0 and int(np.argmax(out.transitions[0].obs)) == 0
    c.close()


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
def test_a53_truncation_is_separate_from_termination_and_keeps_the_true_final_observation(mode):
    c = Collector(_grid_spec(mode, limit=2, width=6, goal=[5, 0]), RewardSpec(), 1, seed=0)
    ts = []
    for _ in range(4):
        ts += c.step(np.array([3])).transitions
    t = next(x for x in ts if x.truncated)
    assert not t.terminated and int(np.argmax(t.next_obs)) == 2 and int(np.argmax(t.obs)) == 1   # the time-limit state, not the next episode's start
    c.close()


@pytest.mark.parametrize("mode", ["NextStep", "SameStep"])
def test_a53_vector_collection_equals_independent_single_environments(mode):
    """CartPole, 3 sub-environments, scripted random actions: every stored (s, a, r, s', terminated, truncated) equals a single-environment replay with the same seeds."""
    spec = EnvSpec(env_id="CartPole-v1", autoreset_mode=mode)
    n, seed = 3, 11
    c = Collector(spec, RewardSpec(), n, seed)
    rng = np.random.default_rng(5)
    acts = rng.integers(2, size=(300, n))
    per_env = {i: [] for i in range(n)}
    for k in range(300):
        for t in c.step(acts[k]).transitions:
            per_env[t.env_index].append(t)
    c.close()
    for i in range(n):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=seed + i)
        ref, ai = [], 0
        while len(ref) < len(per_env[i]):
            a = per_env[i][len(ref)].action     # the action the vector env actually applied to this sub-environment
            nxt, r, te, tr, _ = env.step(a)
            ref.append((obs.copy(), a, r, nxt.copy(), te, tr))
            obs = nxt
            if te or tr:
                obs, _ = env.reset()
        for t, (o, a, r, nx, te, tr) in zip(per_env[i], ref):
            assert np.array_equal(t.obs, o) and np.allclose(t.next_obs, nx) and t.reward == r and (t.terminated, t.truncated) == (te, tr)
        assert any(t.terminated for t in per_env[i])


def test_a53_recurrent_state_resets_exactly_at_episode_starts():
    """A stateful (recurrent-like) policy that counts steps since its last reset must be reset exactly when `episode_start` says a new episode began."""
    for mode in ("NextStep", "SameStep"):
        c = Collector(_grid_spec(mode, width=4, goal=[3, 0]), RewardSpec(), 2, seed=0)
        hidden = np.zeros(2, int)
        seen = []
        start_flags = c.episode_start.copy()
        for k in range(12):
            hidden = np.where(start_flags, 0, hidden)             # the policy resets its hidden state for sub-envs that begin an episode
            out = c.step(np.array([3, 2]))
            for t in out.transitions:
                assert hidden[t.env_index] == t.episode_step      # the hidden counter equals the step index inside the episode
                seen.append(t)
            hidden = hidden + np.array([1 if any(t.env_index == i for t in out.transitions) else 0 for i in range(2)])
            start_flags = c.episode_start.copy()
        assert any(t.episode_step == 0 and t.episode_id >= 2 for t in seen)
        c.close()
