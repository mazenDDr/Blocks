import { useState } from "react";
import { api, errorText } from "../../api";
import type { Graph, StudyView, Validation } from "../../types";
import { Check, ErrorLine, Num, Row, Sel } from "../agent/common";
import { fmt, nodeOfType, useGet } from "./common";
import type { Comparison, MeanCI, Variant } from "./types";

interface VariantDef { label: string; weights: Record<string, { weight: number; enabled: boolean }> }
const METRICS = [["eval_task_return_mean", "task return (default reward; comparable across reward variants)"], ["eval_success_rate", "task success rate"], ["eval_length_mean", "episode length"],
  ["eval_return_mean", "return under each variant's own reward (NOT comparable across reward variants)"], ["eval_truncated_fraction", "fraction of truncated evaluation episodes"]] as const;
const LEARNER_FIELDS = ["gamma", "lr", "target_update_interval", "eps_decay_steps", "batch_size", "train_freq", "gradient_steps"] as const;

const ci = (m: MeanCI) => (m.mean == null ? "—" : `${fmt(m.mean)}${m.ci ? ` [${fmt(m.ci[0])}, ${fmt(m.ci[1])}]` : ""}`);

function ComparisonTable({ c }: { c: Comparison }) {
  const comps = [...new Set(c.variants.flatMap((v) => Object.keys(v.components)))];
  return (
    <div className="comparison" aria-label="variant comparison">
      {c.notes.map((n, i) => <div key={i} className={n.startsWith("WARNING") ? "warn" : "small muted"}>{n}</div>)}
      <div className="tablewrap tall"><table className="dtable"><thead><tr><th>variant</th><th>runs</th><th>task return mean [95% CI over runs]</th><th>training-reward return</th><th>success</th><th>episode length</th>
        <th>terminated / truncated</th>{comps.map((k) => <th key={k}>{k} (weighted)</th>)}<th>reward weights</th><th>env steps</th></tr></thead>
        <tbody>{c.variants.map((v: Variant) => (
          <tr key={v.label}><td><b>{v.label}</b>{v.isBaseline && <span className="badge">baseline</span>}</td><td className="num">{v.nCompleted}/{v.nRuns}</td>
            <td className="num"><b>{ci(v.taskReturn)}</b>{v.taskReturn.warning && <div className="small warnb">{v.taskReturn.warning}</div>}</td><td className="num">{ci(v.trainReturn)}</td>
            <td className="num">{ci(v.successRate)}</td><td className="num">{ci(v.length)}</td><td className="num">{v.terminated} / {v.truncated}</td>
            {comps.map((k) => <td key={k} className="num">{v.components[k] ? ci(v.components[k]) : "—"}</td>)}
            <td className="small">{v.weights ? Object.entries(v.weights).map(([k, w]) => `${k} ${w}`).join(", ") : "—"}</td><td className="num">{v.envSteps != null ? Math.round(v.envSteps) : "—"}</td></tr>))}</tbody></table></div>
      <h4>Per-seed values (unsmoothed, each run is one point)</h4>
      <table className="dtable"><thead><tr><th>variant</th><th>run</th><th>seed</th><th>status</th><th>task return</th><th>success</th><th>length</th><th>terminated / truncated</th></tr></thead>
        <tbody>{c.variants.flatMap((v) => v.runs.map((r) => <tr key={r.runId}><td>{v.label}</td><td><code>{r.runId.slice(0, 8)}</code></td><td className="num">{r.seed}</td><td>{r.status}</td>
          <td className="num">{r.available ? fmt(r.taskReturn) : r.reason}</td><td className="num">{r.successRate == null ? "—" : `${(r.successRate * 100).toFixed(0)}%`}</td><td className="num">{r.available ? fmt(r.length) : ""}</td><td className="num">{r.available ? `${r.terminated} / ${r.truncated}` : ""}</td></tr>))}</tbody></table>
      <div className="small muted">Evaluation seed sets: {c.evaluationSeedSets.map((s) => `[${s.join(", ")}]`).join(" ")}. Each run's rollouts and traces stay inspectable from the run selector.</div>
    </div>
  );
}

export function VariantsTab({ projectId, graph, validation, ensureSaved, onOpenRun }: { projectId: string; graph: Graph; validation: Validation | null; ensureSaved: () => Promise<void>; onOpenRun: (id: string) => void }) {
  const env = validation?.rl?.environment;
  const comps = env?.components ?? [];
  const defaults = env?.defaultWeights ?? {};
  const [mode, setMode] = useState<"reward" | "learner">("reward");
  const [defs, setDefs] = useState<VariantDef[]>([]);
  const [field, setField] = useState<(typeof LEARNER_FIELDS)[number]>("gamma");
  const [values, setValues] = useState("0.9, 0.99");
  const [seeds, setSeeds] = useState("0, 1, 2");
  const [metric, setMetric] = useState<string>("eval_task_return_mean");
  const [baseline, setBaseline] = useState(true);
  const [name, setName] = useState("reward variants");
  const [hyp, setHyp] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sid, setSid] = useState<string | null>(null);
  const list = useGet<{ studies: { id: string; name: string; state: string; graphKind: string; trials: number; counts: Record<string, number> }[] }>("/api/studies", [sid, busy]);
  const live = sid && list.data?.studies.find((s) => s.id === sid)?.state === "running";
  const study = useGet<StudyView>(sid ? `/api/studies/${sid}` : null, [live ? Date.now() >> 11 : 0, list.data?.studies.length]);
  const cmp = useGet<Comparison>(sid ? `/api/studies/${sid}/rl-comparison` : null, [study.data?.state, study.data?.counts ? JSON.stringify(study.data.counts) : ""]);
  const learner = nodeOfType(graph, "rl.dqn_learner");
  const addDef = () => setDefs([...defs, { label: `variant ${defs.length + 1}`, weights: Object.fromEntries(comps.map((c) => [c, { weight: defaults[c] ?? 1, enabled: true }])) }]);

  async function create() {
    setBusy(true); setErr(null);
    try {
      await ensureSaved();
      const seedList = seeds.split(",").map((x) => parseInt(x.trim(), 10)).filter((x) => Number.isFinite(x));
      const variable = mode === "reward"
        ? { target: { scope: "node", node: nodeOfType(graph, "rl.reward")!.id, field: "components" }, labels: defs.map((d) => d.label),
          values: defs.map((d) => comps.filter((c) => d.weights[c].weight !== defaults[c] || !d.weights[c].enabled).map((c) => ({ name: c, weight: d.weights[c].weight, enabled: d.weights[c].enabled }))) }
        : { target: { scope: "node", node: learner!.id, field }, values: values.split(",").map((x) => Number(x.trim())).filter((x) => Number.isFinite(x)) };
      const body = { name, hypothesis: hyp, projectId, run_config: {}, objective: { metric: { name: metric }, direction: metric === "eval_length_mean" || metric === "eval_truncated_fraction" ? "minimize" : "maximize" },
        search: { method: "grid", variables: [variable] }, repeats: { seeds: seedList }, limits: { max_trials: 100 }, include_baseline: baseline, start: true };
      const r = await api.post<StudyView>("/api/studies", body);
      setSid(r.id);
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  const canCreate = !!validation?.ok && (mode === "learner" ? !!learner && values.trim() !== "" : defs.length > 0);
  return (
    <div className="scroll pad rlvar">
      <h3>Reward and policy variants <span className="badge">studies</span></h3>
      <div className="small muted">Each variant × seed becomes an ordinary study trial (its own worker process, run and seed). Variants are compared on the SAME evaluation seeds, separate from training. The unit of replication is the run, so the interval is across seeds.</div>
      <Row label="Name"><input aria-label="study name" value={name} onChange={(e) => setName(e.target.value)} /> <input aria-label="hypothesis" placeholder="hypothesis (optional)" value={hyp} size={40} onChange={(e) => setHyp(e.target.value)} /></Row>
      <Row label="Vary"><Sel label="variation kind" value={mode} options={[{ value: "reward", label: "reward components (named variants)" }, { value: "learner", label: "one learner field" }]} onChange={setMode} />
        <label className="small"><Check label="include baseline" value={baseline} onChange={setBaseline} /> include the graph as it is now as the baseline</label></Row>
      {mode === "reward" ? (
        <div className="variants">
          {defs.map((d, i) => (
            <div key={i} className="variantcard"><input aria-label={`variant label ${i}`} value={d.label} onChange={(e) => setDefs(defs.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))} />
              {comps.map((c) => <span key={c} className="arow2"><b>{c}</b> <Num label={`variant ${i} weight ${c}`} value={d.weights[c].weight} onChange={(n) => setDefs(defs.map((x, j) => (j === i ? { ...x, weights: { ...x.weights, [c]: { ...x.weights[c], weight: n ?? 0 } } } : x)))} />
                <Check label={`variant ${i} enabled ${c}`} value={d.weights[c].enabled} onChange={(b) => setDefs(defs.map((x, j) => (j === i ? { ...x, weights: { ...x.weights, [c]: { ...x.weights[c], enabled: b } } } : x)))} /></span>)}
              <button className="danger" onClick={() => setDefs(defs.filter((_, j) => j !== i))}>Remove</button></div>
          ))}
          <button onClick={addDef} disabled={!comps.length}>Add variant</button>
          <div className="small muted">Weights not changed from the environment default are left at the default; defaults: {JSON.stringify(defaults)}.</div>
        </div>
      ) : (
        <Row label="Field"><Sel label="learner field" value={field} options={[...LEARNER_FIELDS]} onChange={setField} /> values <input aria-label="learner values" value={values} size={24} onChange={(e) => setValues(e.target.value)} /></Row>
      )}
      <Row label="Seeds" hint="one independent training run per seed"><input aria-label="study seeds" value={seeds} onChange={(e) => setSeeds(e.target.value)} /></Row>
      <Row label="Objective"><Sel label="objective metric" value={metric} options={METRICS.map(([v, l]) => ({ value: v, label: l }))} onChange={setMetric} /></Row>
      <button className="primary" disabled={busy || !canCreate} onClick={create}>Run study</button>
      <ErrorLine text={err} />
      {!validation?.ok && <div className="warn">Fix the graph errors before planning a study.</div>}
      <h4>Studies</h4>
      <div className="small">{(list.data?.studies ?? []).filter((s) => s.graphKind === "rl").map((s) => <button key={s.id} className={s.id === sid ? "primary" : ""} onClick={() => setSid(s.id)}>{s.name} · {s.state} · {Object.entries(s.counts).map(([k, v]) => `${v} ${k}`).join(", ")}</button>)}</div>
      {study.data && (
        <div>
          <h4>{study.data.name} <span className={`status ${study.data.state}`}>{study.data.state}</span></h4>
          <div className="small muted">{study.data.objective.semantics}. Objective {study.data.objective.direction} {study.data.objective.metric.name}.</div>
          <table className="dtable"><thead><tr><th>trial</th><th>variant</th><th>seed</th><th>status</th><th>value</th><th>run</th></tr></thead>
            <tbody>{study.data.trials.map((t) => <tr key={t.trialId}><td>{t.trialId}</td><td>{t.label}</td><td className="num">{t.seed}</td><td><span className={`status ${t.status}`}>{t.status}</span></td><td className="num">{t.value == null ? "—" : fmt(t.value)}</td>
              <td>{t.runId ? <button className="link" onClick={() => onOpenRun(t.runId!)}>{t.runId.slice(0, 8)}</button> : "—"}</td></tr>)}</tbody></table>
        </div>
      )}
      {cmp.data && <ComparisonTable c={cmp.data} />}
      {cmp.error && <div className="error">{cmp.error}</div>}
    </div>
  );
}
