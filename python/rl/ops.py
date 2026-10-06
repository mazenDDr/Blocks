"""Operations of the `rl` graph kind. Wires carry typed values (reward_spec, env, network, buffer, learner, eval_report); a wire of one kind
never means another. Execution is the DQN loop in `train.py`; these classes declare the contracts, type the wires and validate compatibility."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graph_core.registry import register
from graph_core.types import Fix, OpError
from tabular.core import TabularOperation, VType

from .buffer import FIELDS  # noqa: F401
from .compat import dqn_compatibility, env_problems
from .dqn import DQNConfig, explain as dqn_explain
from .envs import CATALOG, GRID_ID, EnvSpec, RewardSpec, env_spaces
from .networks import NetworkError, check_network, mlp_graph
from .train import BufferConfig, EvalConfig
from .trace import TraceConfig
from .td3 import TD3Config, td3_compatibility

REWARD_SPEC, ENV, NETWORK, BUFFER, LEARNER, EVAL_REPORT = "reward_spec", "env", "network", "buffer", "learner", "eval_report"


class RLOperation(TabularOperation):
    graph_kind = "rl"
    backend = "gymnasium"
    stores_output = False

    def execute(self, cfg, ins, ctx):  # pragma: no cover - rl graphs run through rl.train, not node by node
        raise NotImplementedError("rl graphs execute through the DQN training loop")

    def param_count(self, cfg, ins):
        return 0


# ------------------------------------------------------------------------------------------------ reward
@register
class RewardOp(RLOperation):
    type = "rl.reward"
    inputs = ()
    outputs = ("reward",)
    in_kinds = {}
    out_kinds = {"reward": REWARD_SPEC}
    Config = RewardSpec
    summary_kind = "rl_reward"

    def infer(self, cfg, ins, node_id):
        names = [c.name for c in cfg.components]
        if len(set(names)) != len(names):
            raise OpError("E_RL_REWARD_COMPONENT", f"A reward component is listed twice: {sorted({n for n in names if names.count(n) > 1})}.")
        return {"reward": VType(REWARD_SPEC, {"components": [c.model_dump() for c in cfg.components]})}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "r_t = sum_c  weight_c * enabled_c * component_c(t)",
                "rule": "Components are declared by the environment and stay separate in every recorded transition; this node only re-weights or disables them. A component that is not listed keeps the environment's default weight.",
                "note": "Evaluation reports the return under this reward AND under the environment's default reward, so reward variants stay comparable."}


# ------------------------------------------------------------------------------------------------ environment
@register
class EnvironmentOp(RLOperation):
    type = "rl.environment"
    inputs = ("reward",)
    outputs = ("env",)
    in_kinds = {"reward": REWARD_SPEC}
    out_kinds = {"env": ENV}
    Config = EnvSpec
    summary_kind = "rl_environment"

    def infer(self, cfg, ins, node_id):
        probs = env_problems(cfg)
        if probs:
            raise OpError(probs[0][0], probs[0][1], None, [Fix(f"Use one of {sorted(CATALOG)}", node_id, "env_id", "CartPole-v1")])
        declared = CATALOG[cfg.env_id]["rewardComponents"]
        for c in ins["reward"].info["components"]:
            if c["name"] not in declared:
                raise OpError("E_RL_REWARD_COMPONENT", f"The reward node lists component '{c['name']}' but {cfg.env_id} declares {declared}.", "reward",
                              [Fix(f"Use one of {declared}")])
        try:
            obs, act, extra = env_spaces(cfg)
        except Exception as e:  # noqa: BLE001  (Gymnasium's own error text is the most precise)
            raise OpError("E_RL_ENV_CONFIG", f"{cfg.env_id} could not be built with these arguments: {type(e).__name__}: {e}") from e
        weights = dict(CATALOG[cfg.env_id]["defaultWeights"])
        for c in ins["reward"].info["components"]:
            weights[c["name"]] = c["weight"] if c["enabled"] else 0.0
        return {"env": VType(ENV, {"spec": cfg.model_dump(mode="json"), "reward": ins["reward"].info, "observationSpace": obs, "actionSpace": act, "timeLimit": extra["timeLimit"],
                                   "wrapperChain": extra["chain"], "components": declared, "weights": weights, "defaultWeights": CATALOG[cfg.env_id]["defaultWeights"],
                                   "rewardModified": weights != CATALOG[cfg.env_id]["defaultWeights"], "autoresetMode": cfg.autoreset_mode,
                                   "catalog": {k: CATALOG[cfg.env_id][k] for k in ("title", "status", "rewardDefinition", "termination", "truncation", "successText", "license", "dependencies",
                                                                                      "snapshot", "seeding", "resources", "renderModes")}})}

    def warnings(self, cfg, ins, node_id):
        out = []
        declared = CATALOG.get(cfg.env_id, {}).get("rewardComponents", [])
        w = {c["name"]: (c["weight"] if c["enabled"] else 0.0) for c in ins["reward"].info["components"]}
        d = CATALOG.get(cfg.env_id, {}).get("defaultWeights", {})
        if any(w.get(k, d.get(k)) != d.get(k) for k in declared):
            out.append(("W_RL_REWARD_MODIFIED", "The training reward differs from the environment's default reward. Returns under different reward definitions are not comparable; "
                        "the evaluation also reports the default-reward (task) return.", "reward"))
        if cfg.wrappers:
            out.append(("W_RL_WRAPPER_CHANGES_TASK", f"Wrappers {[w.type for w in cfg.wrappers]} change the reward the learner sees; compare policies on the task return, not on the wrapped reward.", None))
        return out

    def explain(self, cfg, inputs, outputs):
        e = CATALOG.get(cfg.env_id, {})
        return {"equation": "s_{t+1}, r_t, terminated_t, truncated_t, info_t = env.step(a_t)",
                "rule": f"{cfg.env_id} (Gymnasium {__import__('gymnasium').__version__}). terminated = {e.get('termination', '')}  truncated = {e.get('truncation', '')}",
                "note": "terminated and truncated are separate signals. Stepping and rendering are separate operations; rendering never advances the state."}


# ------------------------------------------------------------------------------------------------ q network
class QNetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    network: dict[str, Any] = Field(default_factory=lambda: mlp_graph(4, (64, 64), 2), description="a MODEL graph: tensor_input [N, obs_dim] -> ... -> [N, n_actions] Q-values")


@register
class QNetworkOp(RLOperation):
    type = "rl.q_network"
    inputs = ("env",)
    outputs = ("network",)
    in_kinds = {"env": ENV}
    out_kinds = {"network": NETWORK}
    Config = QNetworkConfig
    summary_kind = "rl_network"

    def infer(self, cfg, ins, node_id):
        e = ins["env"].info
        spec = EnvSpec.model_validate(e["spec"])
        probs = dqn_compatibility(spec)
        if probs:
            raise OpError(probs[0][0], probs[0][1], "env")
        obs_dim = int(__import__("numpy").prod(e["observationSpace"]["shape"]))
        n_actions = e["actionSpace"]["n"]
        try:
            facts = check_network(cfg.network, obs_dim, n_actions)
        except NetworkError as ex:
            raise OpError(ex.code, ex.message, "env", [Fix(f"Rebuild the network with an MLP for {obs_dim} inputs and {n_actions} actions", node_id, "network", mlp_graph(obs_dim, (64, 64), n_actions))]) from ex
        return {"network": VType(NETWORK, {"obsDim": obs_dim, "nActions": n_actions, "params": facts["params"], "outputNode": facts["outputNode"], "shapes": facts["shapes"],
                                           "env": e["spec"], "envInfo": {k: e[k] for k in ("actionSpace", "observationSpace", "components", "weights")}})}

    def param_count(self, cfg, ins):
        try:
            e = ins["env"].info
            return check_network(cfg.network, int(__import__("numpy").prod(e["observationSpace"]["shape"])), e["actionSpace"]["n"])["params"]
        except Exception:  # noqa: BLE001
            return 0

    def explain(self, cfg, inputs, outputs):
        return {"equation": "Q(s, .) = network(s)  in R^{n_actions}",
                "rule": "The Q-network is an ordinary model graph, lowered by the same code as every other model graph; its output has one value per discrete action.",
                "parameters": {"formula": "sum of the model graph's parameters", "terms": [], "total": outputs["network"].info["params"] if "network" in outputs else 0}}


# ------------------------------------------------------------------------------------------------ replay buffer
@register
class ReplayBufferOp(RLOperation):
    type = "rl.replay_buffer"
    inputs = ("env",)
    outputs = ("buffer",)
    in_kinds = {"env": ENV}
    out_kinds = {"buffer": BUFFER}
    Config = BufferConfig
    summary_kind = "rl_buffer"

    def infer(self, cfg, ins, node_id):
        return {"buffer": VType(BUFFER, {"capacity": cfg.capacity, "sampling": cfg.sampling, "env": ins["env"].info["spec"]})}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "B <- B + (s, a, r, s', terminated, truncated);  oldest transition overwritten when |B| = capacity;  minibatch ~ uniform without replacement",
                "rule": "Every transition gets a global id and keeps its episode, step, policy version at acting time, epsilon, action source (greedy/random), reward components and the TRUE next observation.",
                "note": "The first observation of the next episode is never stored as s' of a terminal or truncated transition."}


# ------------------------------------------------------------------------------------------------ learner
class LearnerConfig(DQNConfig):
    trace: TraceConfig = Field(default_factory=TraceConfig, description="bounded capture of transition-to-update traces and rollout frames")


@register
class DQNLearnerOp(RLOperation):
    type = "rl.dqn_learner"
    inputs = ("network", "buffer")
    outputs = ("learner",)
    in_kinds = {"network": NETWORK, "buffer": BUFFER}
    out_kinds = {"learner": LEARNER}
    Config = LearnerConfig
    summary_kind = "rl_learner"

    def infer(self, cfg, ins, node_id):
        net, buf = ins["network"].info, ins["buffer"].info
        if net["env"] != buf["env"]:
            raise OpError("E_RL_WIRING", "The network and the replay buffer come from different environment configurations.", "buffer")
        probs = dqn_compatibility(EnvSpec.model_validate(net["env"]))
        if probs:
            raise OpError(probs[0][0], probs[0][1], "network")
        if buf["capacity"] < max(cfg.batch_size, cfg.learning_starts):
            raise OpError("E_RL_BUFFER_CAPACITY", f"The buffer holds {buf['capacity']} transitions but learning needs at least max(batch_size={cfg.batch_size}, learning_starts={cfg.learning_starts}).", "buffer",
                          [Fix("Raise the buffer capacity", None, "capacity", max(cfg.batch_size, cfg.learning_starts))])
        if cfg.total_steps <= cfg.learning_starts:
            raise OpError("E_RL_BUDGET", f"total_steps={cfg.total_steps} does not exceed learning_starts={cfg.learning_starts}: no gradient step would ever happen.", None,
                          [Fix("Raise total_steps", node_id, "total_steps", cfg.learning_starts * 4)])
        return {"learner": VType(LEARNER, {"algorithm": "DQN", "env": net["env"], "envInfo": net["envInfo"], "config": cfg.model_dump(mode="json"), "equation": dqn_explain(cfg)})}

    def warnings(self, cfg, ins, node_id):
        out = []
        if cfg.num_envs > 1:
            out.append(("W_RL_VECTOR_ENV", f"{cfg.num_envs} environments are stepped together (SyncVectorEnv, declared autoreset mode); transitions of the reset step are excluded from the buffer.", None))
        if cfg.eps_decay_steps > cfg.total_steps:
            out.append(("W_RL_EPSILON_NEVER_DECAYS", "epsilon does not reach eps_end before training stops.", None))
        return out

    def explain(self, cfg, inputs, outputs):
        e = dqn_explain(cfg)
        return {"equation": e["targetEquation"], "rule": e["lossEquation"] + "  |  " + e["update"] + "  |  " + e["targetUpdate"], "dqn": e,
                "note": e["boundaries"]}


# ------------------------------------------------------------------------------------------------ TD3 learner (continuous actions)
@register
class TD3LearnerOp(RLOperation):
    """Continuous-action learner (ADR 0068). Owns its actor/critic MLPs and float-action replay; wired directly to the environment."""
    type = "rl.td3_learner"
    inputs = ("env",)
    outputs = ("learner",)
    in_kinds = {"env": ENV}
    out_kinds = {"learner": LEARNER}
    Config = TD3Config
    summary_kind = "rl_learner"

    def infer(self, cfg, ins, node_id):
        e = ins["env"].info
        probs = env_problems(EnvSpec.model_validate(e["spec"])) or td3_compatibility(EnvSpec.model_validate(e["spec"]))
        if probs:
            raise OpError(probs[0][0], probs[0][1], "env")
        if cfg.total_steps <= cfg.learning_starts:
            raise OpError("E_RL_BUDGET", f"total_steps={cfg.total_steps} does not exceed learning_starts={cfg.learning_starts}: no gradient step would ever happen.", None,
                          [Fix("Raise total_steps", node_id, "total_steps", cfg.learning_starts * 4)])
        return {"learner": VType(LEARNER, {"algorithm": "TD3", "env": e["spec"], "envInfo": {k: e[k] for k in ("actionSpace", "observationSpace", "components", "weights")},
                                           "config": cfg.model_dump(mode="json")})}

    def explain(self, cfg, inputs, outputs):
        return {"equation": "y = r + gamma (1 - terminated) min_i Q'_i(s', clip(mu'(s') + clip(eps, -c, c), low, high)),  eps ~ N(0, sigma)",
                "rule": f"Twin critics regress to y (MSE); the actor maximises Q_1(s, mu(s)) every {cfg.policy_delay} critic updates; targets track with tau={cfg.tau}.",
                "note": "Collection adds Gaussian noise; evaluation uses the deterministic actor. Truncation (time limit) does not stop bootstrapping; termination does."}


# ------------------------------------------------------------------------------------------------ evaluation
@register
class EvaluationOp(RLOperation):
    type = "rl.evaluation"
    inputs = ("learner",)
    outputs = ("report",)
    in_kinds = {"learner": LEARNER}
    out_kinds = {"report": EVAL_REPORT}
    Config = EvalConfig
    summary_kind = "rl_evaluation"

    def infer(self, cfg, ins, node_id):
        if len(set(cfg.seeds)) != len(cfg.seeds):
            raise OpError("E_RL_EVAL_SEEDS", "Evaluation seeds must be distinct.", None)
        return {"report": VType(EVAL_REPORT, {"seeds": cfg.seeds, "env": ins["learner"].info["env"]})}

    def warnings(self, cfg, ins, node_id):
        s = ins["learner"].info["env"]["seed"]
        if s in cfg.seeds:
            return [("W_RL_EVAL_SEED_OVERLAP", f"The training environment seed {s} is also an evaluation seed; evaluation should use initial conditions the learner was not trained on.", "learner")]
        return []

    def explain(self, cfg, inputs, outputs):
        return {"equation": "report = { return, task return, length, terminated/truncated counts, success } over fresh environment instances seeded by the evaluation seeds",
                "rule": f"{len(cfg.seeds)} episodes, greedy policy (epsilon={cfg.epsilon}), separate environment instances; mean and a Student t interval of the mean over the episodes.",
                "note": "Evaluation interactions are not training interactions: they never enter the replay buffer or the environment-step count."}
