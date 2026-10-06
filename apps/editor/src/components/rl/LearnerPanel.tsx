import { useState } from "react";
import { api } from "../../api";
import type { Graph, Validation } from "../../types";
import { Check, ErrorLine, Num, Row, Sel, useAction } from "../agent/common";
import { Collapsible, nodeOfType, patchConfig } from "./common";

export function LearnerPanel(props: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; validation: Validation | null }) {
  return nodeOfType(props.graph, "rl.td3_learner") ? <TD3Panel {...props} /> : <DQNPanel {...props} />;
}

const TD3_FIELDS: [string, string, { integer?: boolean; min?: number; max?: number }][] = [
  ["total_steps", "Environment steps", { integer: true, min: 100 }], ["learning_starts", "Random steps before learning", { integer: true, min: 1 }],
  ["buffer_capacity", "Replay capacity", { integer: true, min: 1000 }], ["batch_size", "Batch size", { integer: true, min: 8 }],
  ["gamma", "Discount gamma", { min: 0, max: 1 }], ["tau", "Target tracking tau", { min: 0, max: 1 }],
  ["actor_lr", "Actor learning rate", { min: 0 }], ["critic_lr", "Critic learning rate", { min: 0 }],
  ["exploration_noise", "Exploration noise (fraction of half-range)", { min: 0, max: 1 }], ["policy_noise", "Target smoothing noise", { min: 0, max: 1 }],
  ["noise_clip", "Smoothing noise clip", { min: 0, max: 1 }], ["policy_delay", "Critic updates per actor update", { integer: true, min: 1, max: 10 }],
  ["eval_every", "Evaluate every (steps)", { integer: true, min: 100 }],
];

// Mirrors python/rl/td3.py TD3Config defaults, so fields the saved node omits show the value the worker will actually use.
const TD3_DEFAULTS: Record<string, number> = { total_steps: 20000, learning_starts: 1000, buffer_capacity: 100000, batch_size: 256, gamma: 0.99, tau: 0.005, actor_lr: 3e-4,
  critic_lr: 3e-4, exploration_noise: 0.1, policy_noise: 0.2, noise_clip: 0.5, policy_delay: 2, eval_every: 5000 };

/** Continuous-action TD3 (ADR 0068): owns its actor/critic MLPs and float-action replay. */
function TD3Panel({ graph, setGraph, validation }: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; validation: Validation | null }) {
  const learner = nodeOfType(graph, "rl.td3_learner");
  const evalNode = nodeOfType(graph, "rl.evaluation");
  const c = { ...TD3_DEFAULTS, ...((learner?.config ?? {}) as Record<string, any>) };
  const [hidden, setHidden] = useState(((c.hidden as number[] | undefined) ?? [256, 256]).join(", "));
  const [seedText, setSeedText] = useState((((evalNode?.config ?? {}) as Record<string, any>).seeds ?? []).join(", "));
  const set = (patch: Record<string, unknown>) => patchConfig(setGraph, "rl.td3_learner", patch);
  const diags = (validation?.diagnostics ?? []).filter((d) => [learner?.id, evalNode?.id].includes(d.nodeId ?? ""));
  return (
    <div className="scroll pad rllearner">
      <h3>Learner <span className="badge">TD3</span> <span className="badge">continuous actions</span> <span className="badge">native torch</span></h3>
      <p className="small muted">Twin critics with clipped double-Q targets, target policy smoothing and delayed actor updates. Collection adds Gaussian noise; evaluation uses the deterministic actor. The time limit does not stop bootstrapping; termination does.</p>
      {diags.map((d, i) => <ErrorLine key={i} text={`${d.code}: ${d.message}`} />)}
      {TD3_FIELDS.map(([k, label, o]) => <Row key={k} label={label}><Num label={label} value={c[k]} integer={o.integer} min={o.min} max={o.max} onChange={(n) => n != null && set({ [k]: n })} /></Row>)}
      <Row label="Hidden layer sizes" hint="actor and both critics; 1–4 layers"><input aria-label="td3 hidden sizes" value={hidden} size={16}
        onChange={(e) => { setHidden(e.target.value); const h = e.target.value.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => Number.isFinite(x) && x > 0); if (h.length >= 1 && h.length <= 4) set({ hidden: h }); }} /></Row>
      <Row label="Evaluation seeds" hint="one deterministic episode per seed on a separate environment instance"><input aria-label="evaluation seeds" value={seedText} size={30}
        onChange={(e) => { setSeedText(e.target.value); const s = e.target.value.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => Number.isFinite(x)); if (s.length) patchConfig(setGraph, "rl.evaluation", { seeds: s }); }} /></Row>
    </div>
  );
}

function DQNPanel({ graph, setGraph, validation }: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; validation: Validation | null }) {
  const learner = nodeOfType(graph, "rl.dqn_learner");
  const buffer = nodeOfType(graph, "rl.replay_buffer");
  const evalNode = nodeOfType(graph, "rl.evaluation");
  const qnet = nodeOfType(graph, "rl.q_network");
  const c = (learner?.config ?? {}) as Record<string, any>;
  const tr = (c.trace ?? {}) as Record<string, any>;
  const ev = (evalNode?.config ?? {}) as Record<string, any>;
  const rl = validation?.rl;
  const eq = rl?.learner?.equation;
  const set = (patch: Record<string, unknown>) => patchConfig(setGraph, "rl.dqn_learner", patch);
  const setTrace = (patch: Record<string, unknown>) => set({ trace: { ...tr, ...patch } });
  const { busy, error, run } = useAction();
  const [hidden, setHidden] = useState("64, 64");
  const obsDim = rl?.network?.obsDim ?? (rl?.environment ? (rl.environment.observationSpace.shape ?? [4]).reduce((a, b) => a * b, 1) : 4);
  const nActions = rl?.network?.nActions ?? rl?.environment?.actionSpace.n ?? 2;
  const [seedText, setSeedText] = useState((ev.seeds ?? []).join(", "));
  const diags = (validation?.diagnostics ?? []).filter((d) => [learner?.id, buffer?.id, qnet?.id, evalNode?.id].includes(d.nodeId ?? ""));
  const net = (qnet?.config.network as any) ?? {};
  const shapes = rl?.network?.shapes ?? {};
  const num = (k: string, label: string, opts: { integer?: boolean; min?: number; max?: number } = {}) => <Num label={label} value={c[k]} integer={opts.integer} min={opts.min} max={opts.max} onChange={(n) => n != null && set({ [k]: n })} />;
  return (
    <div className="scroll pad rllearner">
      <h3>Learner <span className="badge">DQN</span> <span className="badge">native torch</span></h3>
      {diags.map((d, i) => <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}><b>{d.code}</b> {d.nodeId}: {d.message}{d.fixes.map((f, j) => <div key={j} className="small">Fix: {f.label}</div>)}</div>)}
      <div className="rlcols">
        <section>
          <h4>Update equation (with this configuration)</h4>
          {eq ? (
            <div className="equation" aria-label="update equation">
              <pre>{eq.targetEquation}{"\n"}{eq.lossEquation}{"\n"}{eq.update}{"\n"}{eq.targetUpdate}</pre>
              <div className="small"><b>Exploration.</b> {eq.exploration}</div>
              <div className="small"><b>Boundaries.</b> {eq.boundaries}</div>
              <div className="small muted">Update-to-data ratio {eq.updateToDataRatio.toFixed(3)} gradient steps per environment step · learning starts after {eq.learningStarts} stored transitions. Only DQN terms are shown: nothing here belongs to another algorithm.</div>
            </div>
          ) : <div className="muted">The equation appears once the graph validates.</div>}
          <h4>Collection and optimization</h4>
          <Row label="Budget"><span>environment steps</span> {num("total_steps", "total steps", { integer: true, min: 1 })} <span>environments</span> {num("num_envs", "number of environments", { integer: true, min: 1, max: 16 })}</Row>
          <Row label="Discount γ">{num("gamma", "gamma", { min: 0, max: 1 })}</Row>
          <Row label="Optimizer">Adam lr {num("lr", "learning rate", { min: 0 })} batch {num("batch_size", "batch size", { integer: true, min: 1 })}
            <span>clip grad norm</span><Num label="max grad norm" nullable value={(c.max_grad_norm as number | null | undefined) ?? null} onChange={(n) => set({ max_grad_norm: n })} /></Row>
          <Row label="TD loss"><Sel label="td loss" value={c.loss ?? "huber"} options={["huber", "mse"]} onChange={(v) => set({ loss: v })} />
            {(c.loss ?? "huber") === "huber" && <>δ {num("huber_delta", "huber delta", { min: 0 })}</>}</Row>
          <Row label="Schedule" hint="learning phases happen every train_freq vector steps">learning starts {num("learning_starts", "learning starts", { integer: true, min: 1 })} train freq {num("train_freq", "train freq", { integer: true, min: 1 })}
            gradient steps {num("gradient_steps", "gradient steps", { integer: true, min: 1 })}</Row>
          <Row label="Target network">hard copy every {num("target_update_interval", "target update interval", { integer: true, min: 1 })} gradient steps</Row>
          <Row label="ε-greedy">{num("eps_start", "epsilon start", { min: 0, max: 1 })} → {num("eps_end", "epsilon end", { min: 0, max: 1 })} over {num("eps_decay_steps", "epsilon decay steps", { integer: true, min: 1 })} steps</Row>
          <Row label="Log every">{num("log_interval", "log interval", { integer: true, min: 1 })} gradient steps (events are unsmoothed)</Row>
        </section>
        <section>
          <h4>Q-network (an ordinary model graph)</h4>
          <Row label="MLP hidden sizes" hint={`input ${obsDim} features (from the observation space) → ${nActions} Q-values (one per discrete action)`}>
            <input aria-label="hidden layer sizes" value={hidden} size={14} onChange={(e) => setHidden(e.target.value)} />
            <button disabled={busy} onClick={() => run(async () => {
              const h = hidden.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => Number.isFinite(x) && x > 0);
              const r = await api.post<{ network: unknown }>("/api/rl/network/mlp", { obsDim, hidden: h, nActions });
              patchConfig(setGraph, "rl.q_network", { network: r.network });
            })}>Rebuild network</button></Row>
          <ErrorLine text={error} />
          <table className="dtable" aria-label="network nodes"><thead><tr><th>node</th><th>type</th><th>config</th><th>output shape</th></tr></thead><tbody>
            {(net.nodes ?? []).map((n: any) => { const o = shapes[n.id] ? Object.values(shapes[n.id])[0] : undefined; return (
              <tr key={n.id}><td>{n.id}</td><td>{n.type}</td><td className="small">{JSON.stringify(n.config)}</td><td className="num">{o ? `[${o.shape.join(", ")}]` : "—"}</td></tr>); })}</tbody></table>
          <div className="small muted">{rl?.network ? `${rl.network.params} parameters. ` : ""}The network is validated and lowered by the same code as every model graph; its input and output shapes are checked against the environment's spaces before anything runs.</div>
          <Collapsible title="Model graph JSON"><pre className="small">{JSON.stringify(net, null, 1)}</pre></Collapsible>
          <h4>Replay buffer</h4>
          <Row label="Capacity"><Num label="buffer capacity" integer min={1} value={(buffer?.config.capacity as number | undefined) ?? 10000} onChange={(n) => n != null && patchConfig(setGraph, "rl.replay_buffer", { capacity: n })} /> transitions · uniform sampling without replacement within a batch</Row>
          <h4>Evaluation protocol</h4>
          <Row label="Seeds" hint="one greedy episode per seed on a separate environment instance"><input aria-label="evaluation seeds" value={seedText} size={30}
            onChange={(e) => { setSeedText(e.target.value); const s = e.target.value.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => Number.isFinite(x)); if (s.length) patchConfig(setGraph, "rl.evaluation", { seeds: s }); }} /></Row>
          <Row label="Schedule">every <Num label="evaluation interval" integer min={0} value={ev.interval_steps ?? 0} onChange={(n) => patchConfig(setGraph, "rl.evaluation", { interval_steps: n ?? 0 })} /> steps (0 = only at the end)
            <label className="small"><Check label="evaluate before training" value={ev.initial ?? true} onChange={(b) => patchConfig(setGraph, "rl.evaluation", { initial: b })} /> also before training</label></Row>
          <h4>Bounded capture (rollouts and transition tracing)</h4>
          <Row label="Enabled"><Check label="capture enabled" value={tr.enabled ?? true} onChange={(b) => setTrace({ enabled: b })} /></Row>
          <Row label="Episodes">{" "}<Num label="captured episodes" integer min={0} max={20} value={tr.max_episodes ?? 4} onChange={(n) => setTrace({ max_episodes: n ?? 0 })} /> episodes of environment 0, first
            <Num label="steps per captured episode" integer min={1} max={1000} value={tr.max_steps_per_episode ?? 120} onChange={(n) => setTrace({ max_steps_per_episode: n ?? 1 })} /> steps each</Row>
          <Row label="Caps">tracked transitions <Num label="max tracked" integer min={0} max={5000} value={tr.max_tracked ?? 400} onChange={(n) => setTrace({ max_tracked: n ?? 0 })} />
            minibatch records <Num label="max update records" integer min={0} max={5000} value={tr.max_update_records ?? 400} onChange={(n) => setTrace({ max_update_records: n ?? 0 })} />
            <label className="small"><Check label="render frames" value={tr.frames ?? true} onChange={(b) => setTrace({ frames: b })} /> real frames</label></Row>
          <div className="small muted">Capture is bounded on purpose: what is recorded is declared in the run, and everything else is reported as not captured.</div>
        </section>
      </div>
    </div>
  );
}
