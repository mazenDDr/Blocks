import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../api";
import { usePolling } from "../hooks";
import type { Graph, OpInfo, RunDiff, StudyListItem, StudyTrial, StudyView, Validation } from "../types";
import { fmtNum, shortHash } from "../util";

const MODEL_METRICS = ["val_loss", "val_acc", "train_loss"];
const MODEL_RUN_FIELDS = ["epochs", "batch_size", "lr", "optimizer", "momentum", "seed", "val_fraction", "split_seed"];
const TABULAR_RUN_FIELDS = ["seed"];

interface VarForm { scope: "node" | "run"; node: string; field: string; values: string; labels: string; lo: string; hi: string; vtype: "float" | "int"; scale: "linear" | "log"; mode: "values" | "range" }
const newVar = (scope: "node" | "run", node: string, field: string): VarForm => ({ scope, node, field, values: "", labels: "", lo: "", hi: "", vtype: "float", scale: "linear", mode: "values" });

const parseValues = (t: string): unknown[] | string => {
  try { const v = JSON.parse(t.trim().startsWith("[") ? t : `[${t}]`); return Array.isArray(v) ? v : "values must be a list"; } catch { return "values must be valid JSON, e.g. [8, 16] or [true, false] or [\"mean\", \"median\"]"; }
};

export function Experiments({ graph, validation, projectId, ops, ensureSaved, onOpenRun }: {
  graph: Graph; validation: Validation | null; projectId: string; ops: Record<string, OpInfo>; ensureSaved: () => Promise<void>; onOpenRun: (runId: string) => void;
}) {
  const list = usePolling<{ studies: StudyListItem[] }>("/api/studies", 3000);
  const [sel, setSel] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const studies = list.data?.studies ?? [];
  return (
    <div className="workspace expws">
      <aside className="wsleft">
        <h3>Studies</h3>
        <div className="muted small">A study holds a plan of trials: configurations (grid or random) × repeat identities (seed, fold), each trial a run. Failed trials stay in the record.</div>
        <button className="primary" onClick={() => { setCreating(true); setSel(null); }}>New sweep</button>
        {studies.length === 0 && <div className="muted">No studies yet.</div>}
        {[...studies].reverse().map((s) => (
          <button key={s.id} className={`connitem ${sel === s.id && !creating ? "on" : ""}`} onClick={() => { setSel(s.id); setCreating(false); }}>
            <b>{s.name}</b> <span className={`status ${s.state}`}>{s.state.replace(/_/g, " ")}</span>
            <small>{s.graphKind} · {s.trials} trials · {Object.entries(s.counts).map(([k, v]) => `${v} ${k}`).join(", ")} · {s.objective.direction} {s.objective.metric.name}</small>
          </button>
        ))}
      </aside>
      <main className="wsmain">
        {creating && <NewSweep graph={graph} validation={validation} projectId={projectId} ops={ops} ensureSaved={ensureSaved} onCreated={(id) => { setCreating(false); setSel(id); list.reload(); }} />}
        {!creating && sel && <StudyDetail key={sel} id={sel} onOpenRun={onOpenRun} onChanged={list.reload} />}
        {!creating && !sel && <div className="empty pad">Select a study, or create a sweep from the graph that is open now ({graph.graphKind} graph, {graph.nodes.length} nodes). Sweeps change node-config and run-config fields of that graph; nothing is copied by hand.</div>}
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------------------- new sweep
function NewSweep({ graph, validation, projectId, ops, ensureSaved, onCreated }: {
  graph: Graph; validation: Validation | null; projectId: string; ops: Record<string, OpInfo>; ensureSaved: () => Promise<void>; onCreated: (id: string) => void;
}) {
  const tabular = graph.graphKind === "tabular";
  const metricNodes = graph.nodes.filter((n) => n.type === "sklearn.metrics");
  const [name, setName] = useState("");
  const [hyp, setHyp] = useState("");
  const [metricNode, setMetricNode] = useState(metricNodes[0]?.id ?? "");
  const nodeMetrics: string[] = (validation?.nodes[metricNode]?.outputShapes?.metrics as any)?.metrics ?? [];
  const [metric, setMetric] = useState(tabular ? "rmse" : "val_loss");
  const [select, setSelect] = useState<"last" | "min" | "max">("last");
  const [direction, setDirection] = useState<"minimize" | "maximize">(tabular ? "minimize" : "minimize");
  const [method, setMethod] = useState<"single" | "grid" | "random">("grid");
  const [vars, setVars] = useState<VarForm[]>([]);
  const [randomTrials, setRandomTrials] = useState(4);
  const [samplerSeed, setSamplerSeed] = useState(0);
  const [seeds, setSeeds] = useState("");
  const [folds, setFolds] = useState("");
  const splitNodes = graph.nodes.filter((n) => n.type === "tabular.train_validation_split");
  const [foldNode, setFoldNode] = useState(splitNodes[0]?.id ?? "");
  const [maxTrials, setMaxTrials] = useState(12);
  const [maxAttempts, setMaxAttempts] = useState(3);
  const [baseline, setBaseline] = useState(true);
  const [data, setData] = useState("examples/data/shapes10");
  const [epochs, setEpochs] = useState(1);
  const [batch, setBatch] = useState(16);
  const [plan, setPlan] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (tabular && nodeMetrics.length && !nodeMetrics.includes(metric)) setMetric(nodeMetrics.includes("rmse") ? "rmse" : nodeMetrics[0]); }, [metricNode, nodeMetrics.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (tabular && metric) setDirection(["r2", "accuracy", "roc_auc"].includes(metric) ? "maximize" : "minimize"); if (!tabular) setDirection(metric === "val_acc" ? "maximize" : "minimize"); }, [metric, tabular]);

  const nodeFields = (id: string) => Object.keys(ops[graph.nodes.find((n) => n.id === id)?.type ?? ""]?.configSchema.properties ?? {});
  const runFields = tabular ? TABULAR_RUN_FIELDS : MODEL_RUN_FIELDS;
  const setVar = (i: number, p: Partial<VarForm>) => setVars((v) => v.map((x, k) => (k === i ? { ...x, ...p } : x)));

  function body(start: boolean) {
    const variables = vars.map((v) => {
      const target = v.scope === "node" ? { scope: "node", node: v.node, field: v.field } : { scope: "run", field: v.field };
      if (v.mode === "range") return { target, range: { low: Number(v.lo), high: Number(v.hi), type: v.vtype, scale: v.scale } };
      const vals = parseValues(v.values);
      if (typeof vals === "string") throw new Error(`${v.scope === "node" ? v.node + "." : ""}${v.field}: ${vals}`);
      const labels = v.labels.trim() ? v.labels.split(",").map((s) => s.trim()) : undefined;
      return { target, values: vals, ...(labels ? { labels } : {}) };
    });
    const seedList = seeds.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
    if (seedList.some((n) => !Number.isInteger(n))) throw new Error("Seeds must be integers, comma separated.");
    return {
      name: name || "sweep", hypothesis: hyp, projectId, start,
      run_config: tabular ? {} : { data, epochs, batch_size: batch },
      objective: { metric: tabular ? { name: metric, node: metricNode } : { name: metric, select }, direction },
      search: { method, variables, random_trials: method === "random" ? randomTrials : 0, sampler_seed: samplerSeed },
      repeats: { seeds: seedList, ...(folds ? { folds: Number(folds), fold_node: foldNode } : {}) },
      limits: { max_trials: maxTrials, max_concurrency: 1, max_attempts: maxAttempts }, include_baseline: baseline,
    };
  }
  async function doPlan() {
    setErr(null); setPlan(null); setBusy(true);
    try { await ensureSaved(); setPlan(await api.post("/api/studies/plan", body(false))); } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  async function create() {
    setErr(null); setBusy(true);
    try { await ensureSaved(); const r = await api.post<StudyView>("/api/studies", body(true)); onCreated(r.id); } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  return (
    <div className="newsweep">
      <h3>New sweep on “{projectId}” <span className="badge">{graph.graphKind}</span></h3>
      {!validation?.ok && <div className="warn">The open graph has errors; a study needs a valid base graph.</div>}
      <div className="row"><label className="lbl">Name</label><input aria-label="study name" value={name} onChange={(e) => setName(e.target.value)} /></div>
      <div className="row"><label className="lbl">Hypothesis</label><input aria-label="hypothesis" value={hyp} onChange={(e) => setHyp(e.target.value)} /></div>

      <h4>Objective</h4>
      {tabular ? (
        <div className="row"><label className="lbl">Metric node</label>
          <select aria-label="metric node" value={metricNode} onChange={(e) => setMetricNode(e.target.value)}>{metricNodes.map((n) => <option key={n.id}>{n.id}</option>)}</select>
          <select aria-label="metric" value={metric} onChange={(e) => setMetric(e.target.value)}>{(nodeMetrics.length ? nodeMetrics : ["rmse"]).map((m) => <option key={m}>{m}</option>)}</select>
          {metricNodes.length === 0 && <span className="warn small">add a Validation metrics node</span>}</div>
      ) : (
        <div className="row"><label className="lbl">Metric</label>
          <select aria-label="metric" value={metric} onChange={(e) => setMetric(e.target.value)}>{MODEL_METRICS.map((m) => <option key={m}>{m}</option>)}</select>
          <select aria-label="epoch selection" value={select} onChange={(e) => setSelect(e.target.value as any)}><option value="last">last epoch</option><option value="min">best (minimum) epoch</option><option value="max">best (maximum) epoch</option></select></div>
      )}
      <div className="row"><label className="lbl">Direction</label>
        <select aria-label="direction" value={direction} onChange={(e) => setDirection(e.target.value as any)}><option>minimize</option><option>maximize</option></select></div>
      <div className="muted small">{tabular ? "Value: one evaluation of the fitted model on the validation partition; there are no training steps." : "Value: from the per-epoch evaluation events; the step and epoch it came from are recorded with each trial. Different metrics are never ranked together."}</div>

      {!tabular && <>
        <h4>Run configuration (shared)</h4>
        <div className="row"><label className="lbl">Dataset folder</label><input aria-label="dataset folder" value={data} onChange={(e) => setData(e.target.value)} /></div>
        <div className="row"><label className="lbl">Epochs</label><input type="number" aria-label="epochs" min={1} value={epochs} onChange={(e) => setEpochs(Math.max(1, Number(e.target.value) || 1))} />
          <label className="lbl">Batch size</label><input type="number" aria-label="batch size" min={1} value={batch} onChange={(e) => setBatch(Math.max(1, Number(e.target.value) || 1))} /></div></>}

      <h4>Search</h4>
      <div className="row"><label className="lbl">Method</label>
        <select aria-label="search method" value={method} onChange={(e) => setMethod(e.target.value as any)}>
          <option value="single">single configuration (repeats only)</option><option value="grid">grid</option><option value="random">random</option></select>
        {method === "random" && <><label className="lbl">Trials</label><input type="number" aria-label="random trials" min={1} value={randomTrials} onChange={(e) => setRandomTrials(Math.max(1, Number(e.target.value) || 1))} />
          <label className="lbl">Sampler seed</label><input type="number" aria-label="sampler seed" value={samplerSeed} onChange={(e) => setSamplerSeed(Number(e.target.value) || 0)} /></>}</div>
      {method !== "single" && vars.map((v, i) => (
        <div className="qbbox" key={i}>
          <div className="row">
            <select aria-label={`variable ${i + 1} scope`} value={v.scope} onChange={(e) => setVar(i, newVar(e.target.value as any, graph.nodes[0]?.id ?? "", e.target.value === "run" ? runFields[0] : nodeFields(graph.nodes[0]?.id ?? "")[0] ?? ""))}>
              <option value="node">node config</option><option value="run">run config</option></select>
            {v.scope === "node" && <select aria-label={`variable ${i + 1} node`} value={v.node} onChange={(e) => setVar(i, { node: e.target.value, field: nodeFields(e.target.value)[0] ?? "" })}>{graph.nodes.map((n) => <option key={n.id}>{n.id}</option>)}</select>}
            <select aria-label={`variable ${i + 1} field`} value={v.field} onChange={(e) => setVar(i, { field: e.target.value })}>{(v.scope === "node" ? nodeFields(v.node) : runFields).map((f) => <option key={f}>{f}</option>)}</select>
            <button className="danger" aria-label={`remove variable ${i + 1}`} onClick={() => setVars(vars.filter((_, k) => k !== i))}>×</button>
          </div>
          <div className="row">
            <select aria-label={`variable ${i + 1} kind`} value={v.mode} onChange={(e) => setVar(i, { mode: e.target.value as any })}><option value="values">discrete values</option>{method === "random" && <option value="range">interval</option>}</select>
            {v.mode === "values" ? <>
              <input aria-label={`variable ${i + 1} values`} placeholder='JSON list: 8, 16   or   ["mean","median"]' value={v.values} onChange={(e) => setVar(i, { values: e.target.value })} />
              <input aria-label={`variable ${i + 1} labels`} placeholder="optional names, comma separated" value={v.labels} onChange={(e) => setVar(i, { labels: e.target.value })} /></> : <>
              <input aria-label={`variable ${i + 1} low`} size={6} placeholder="low" value={v.lo} onChange={(e) => setVar(i, { lo: e.target.value })} />
              <input aria-label={`variable ${i + 1} high`} size={6} placeholder="high" value={v.hi} onChange={(e) => setVar(i, { hi: e.target.value })} />
              <select aria-label={`variable ${i + 1} type`} value={v.vtype} onChange={(e) => setVar(i, { vtype: e.target.value as any })}><option>float</option><option>int</option></select>
              <select aria-label={`variable ${i + 1} scale`} value={v.scale} onChange={(e) => setVar(i, { scale: e.target.value as any })}><option>linear</option><option>log</option></select></>}
          </div>
        </div>
      ))}
      {method !== "single" && <button onClick={() => { const n = graph.nodes[0]; setVars([...vars, newVar("node", n?.id ?? "", nodeFields(n?.id ?? "")[0] ?? "")]); }} disabled={!graph.nodes.length}>Add variable</button>}

      <h4>Repeats (explicit dimensions)</h4>
      <div className="row"><label className="lbl">Seeds</label><input aria-label="seeds" placeholder="e.g. 1, 2, 3 (empty = none)" value={seeds} onChange={(e) => setSeeds(e.target.value)} /></div>
      <div className="muted small">A seed replaces the run-level <code>seed</code> ({tabular ? "applied to every node that has a seed, e.g. the split" : "training and split seed"}). Each seed is its own trial; they are never merged into one curve.</div>
      {tabular && <div className="row"><label className="lbl">Folds (k)</label><input type="number" aria-label="folds" min={2} placeholder="none" value={folds} onChange={(e) => setFolds(e.target.value)} />
        <select aria-label="fold node" value={foldNode} onChange={(e) => setFoldNode(e.target.value)}>{splitNodes.map((n) => <option key={n.id}>{n.id}</option>)}</select></div>}

      <h4>Limits</h4>
      <div className="row"><label className="lbl">Max trials</label><input type="number" aria-label="max trials" min={1} max={200} value={maxTrials} onChange={(e) => setMaxTrials(Math.max(1, Math.min(200, Number(e.target.value) || 1)))} />
        <label className="lbl">Max attempts per trial</label><input type="number" aria-label="max attempts" min={1} max={5} value={maxAttempts} onChange={(e) => setMaxAttempts(Math.max(1, Math.min(5, Number(e.target.value) || 1)))} />
        <span className="muted small">concurrency 1 (trials run one at a time)</span></div>
      <div className="row"><label><input type="checkbox" checked={baseline} onChange={(e) => setBaseline(e.target.checked)} /> include the unchanged graph as the baseline configuration</label></div>
      <div className="muted small">The plan is checked against “max trials” before anything is stored; it is never truncated silently.</div>

      {err && <div className="error pre">{err}</div>}
      <div className="actions"><button onClick={doPlan} disabled={busy}>Check plan</button><button className="primary" onClick={create} disabled={busy || !validation?.ok}>Create and run</button></div>
      {plan && (
        <div className="plan">
          <h4>Plan: {plan.counts.total} trials = {plan.counts.groups} configurations × {plan.counts.seeds} seed(s) × {plan.counts.folds} fold(s){plan.counts.invalid ? ` · ${plan.counts.invalid} invalid (will not run)` : ""}</h4>
          <div className="tablewrap short"><table className="dtable"><thead><tr><th>trial</th><th>configuration</th><th>seed</th><th>fold</th><th>status</th></tr></thead><tbody>
            {plan.trials.map((t: any) => <tr key={t.id}><td>{t.id}</td><td>{t.label}</td><td>{t.seed ?? "—"}</td><td>{t.fold ?? "—"}</td>
              <td>{t.status === "invalid" ? <span className="errbadge">invalid: {t.diagnostics.map((d: any) => `${d.code} ${d.message}`).join("; ")}</span> : t.status}</td></tr>)}</tbody></table></div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- study detail
const fmtV = (v: number | null | undefined) => (v == null ? "—" : fmtNum(v, 5));
function Delta({ d, better }: { d: number | null; better: boolean | null }) {
  if (d == null) return <span className="muted">—</span>;
  return <span className={better == null ? "" : better ? "good" : "bad"}>{d > 0 ? "+" : ""}{fmtNum(d, 4)}{better == null ? "" : better ? " better" : " worse"}</span>;
}

function StudyDetail({ id, onOpenRun, onChanged }: { id: string; onOpenRun: (runId: string) => void; onChanged: () => void }) {
  const st = usePolling<StudyView>(`/api/studies/${id}`, 1500);
  const [picked, setPicked] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const s = st.data;
  const byId = useMemo(() => Object.fromEntries((s?.trials ?? []).map((t) => [t.trialId, t])), [s]);
  const cmp = picked.length === 2 ? [byId[picked[0]], byId[picked[1]]] : null;
  async function act(path: string, method: "post" | "put" = "post", body: unknown = {}) {
    setErr(null);
    try { await (method === "post" ? api.post(path, body) : api.put(path, body)); st.reload(); onChanged(); } catch (e) { setErr(errorText(e)); }
  }
  if (!s) return <div className="muted pad">{st.error ?? "loading…"}</div>;
  const running = s.state === "running";
  const hasPlanned = s.trials.some((t) => t.status === "planned");
  return (
    <div className="studydetail">
      <div className="ni-head"><div><h3>{s.name} <span className={`status ${s.state}`}>{s.state.replace(/_/g, " ")}</span></h3>
        <div className="small muted">{s.graphKind} graph {shortHash(s.graphHash)} · {s.limits.max_trials} trials max · concurrency {s.limits.max_concurrency} · up to {s.limits.max_attempts} attempts per trial</div></div>
        <div>{hasPlanned && !running && <button className="primary" onClick={() => act(`/api/studies/${id}/start`)}>Start / resume</button>} {running && <button onClick={() => act(`/api/studies/${id}/cancel`)}>Cancel study</button>}</div></div>
      {err && <div className="error pre">{err}</div>}
      {s.schedulerError && <div className="errbadge">Scheduler error: {s.schedulerError}</div>}
      {s.hypothesis && <p><b>Hypothesis:</b> {s.hypothesis}</p>}
      <div className="small"><b>Objective:</b> {s.objective.direction} <code>{s.objective.metric.name}</code>{s.objective.metric.node ? ` of ${s.objective.metric.node}` : ` (${s.objective.metric.select} epoch)`} — {s.objective.semantics}</div>
      <div className="small"><b>Baseline:</b> {s.baseline.mode === "pinned_run" ? <>pinned run <code>{s.baseline.runId}</code> = {fmtV(s.baseline.value)} <button onClick={() => act(`/api/studies/${id}/baseline`, "put", { runId: null })}>Unpin</button></>
        : s.baseline.mode ? <>baseline configuration mean of {s.baseline.n} completed repeat(s) = {fmtV(s.baseline.value)}</> : "no completed baseline trial yet"}
        {s.baseline.note && <span className="muted"> · {s.baseline.note}</span>}</div>
      <div className="small">{Object.entries(s.counts).map(([k, v]) => <span key={k} className={`status ${k}`}>{v} {k}</span>)}</div>

      <h4>Configurations (repeats aggregated)</h4>
      <div className="tablewrap"><table className="dtable"><thead><tr><th>rank</th><th>configuration</th><th>trials</th><th>completed / failed / cancelled / invalid</th><th>n</th><th>mean</th><th>std</th><th>min</th><th>max</th><th>Δ vs baseline</th></tr></thead><tbody>
        {s.groups.map((g) => <tr key={g.group} className={g.isBaseline ? "sel" : ""}>
          <td>{g.rank ?? "—"}</td><td>{g.label}</td><td className="num">{g.trials}</td><td>{g.completed} / {g.failed} / {g.cancelled} / {g.invalid}{g.retried ? ` · ${g.retried} retried` : ""}</td>
          <td className="num">{g.n}</td><td className="num">{fmtV(g.mean)}</td><td className="num">{fmtV(g.std)}</td><td className="num">{fmtV(g.min)}</td><td className="num">{fmtV(g.max)}</td><td><Delta d={g.delta} better={g.better} /></td></tr>)}
      </tbody></table></div>
      <div className="muted small">{s.notice}</div>

      <h4>Trials</h4>
      <div className="tablewrap"><table className="dtable trialtable"><thead><tr><th></th><th>trial</th><th>configuration</th><th>seed</th><th>fold</th><th>attempts</th><th>status</th><th>{s.objective.metric.name}</th><th>Δ</th><th>run</th><th></th></tr></thead><tbody>
        {s.trials.map((t) => <TrialRow key={t.trialId} t={t} picked={picked.includes(t.trialId)} direction={s.objective.direction}
          onPick={(on) => setPicked((p) => (on ? [...p, t.trialId].slice(-2) : p.filter((x) => x !== t.trialId)))} onOpenRun={onOpenRun}
          onRetry={() => act(`/api/studies/${id}/trials/${t.trialId}/retry`)} onPin={() => t.runId && act(`/api/studies/${id}/baseline`, "put", { runId: t.runId })} pinned={t.runId === s.baseline.runId} />)}
      </tbody></table></div>
      <div className="muted small">Select two trials to compare them. {picked.length === 1 ? "Select one more." : ""}</div>
      {cmp && cmp[0] && cmp[1] && <Compare a={cmp[0]} b={cmp[1]} objective={s.objective.metric.name} />}
    </div>
  );
}

function TrialRow({ t, picked, onPick, onOpenRun, onRetry, onPin, pinned, direction }: {
  t: StudyTrial; picked: boolean; onPick: (on: boolean) => void; onOpenRun: (id: string) => void; onRetry: () => void; onPin: () => void; pinned: boolean; direction: string;
}) {
  const [open, setOpen] = useState(false);
  const m = t.metric;
  return (
    <>
      <tr className={t.isBaseline ? "baseline" : ""}>
        <td><input type="checkbox" aria-label={`select trial ${t.trialId}`} disabled={!t.runId} checked={picked} onChange={(e) => onPick(e.target.checked)} /></td>
        <td>{t.trialId}{t.isBaseline && <span className="badge">baseline</span>}</td><td>{t.label}{t.attributionWarning && <span className="badge old" title={t.attributionWarning}>multi-field change</span>}</td>
        <td className="num">{t.seed ?? "—"}</td><td className="num">{t.fold ?? "—"}</td>
        <td className="num"><button className="link" onClick={() => setOpen(!open)} aria-label={`attempts of ${t.trialId}`}>{t.attemptCount}{t.retried ? " (retried)" : ""}</button></td>
        <td><span className={`status ${t.status}`}>{t.status}</span></td>
        <td className="num" title={m?.available ? `${m.aggregation}${m.step != null ? ` · step ${m.step}, epoch ${m.epoch}` : ""} · ${m.partition ?? ""} · run ${m.provenance?.runId}` : m?.reason ?? ""}>{fmtV(t.value)}</td>
        <td><Delta d={t.delta} better={t.better} /></td>
        <td>{t.runId ? <button className="link" onClick={() => onOpenRun(t.runId!)}>{t.runId}</button> : "—"}</td>
        <td>{(t.status === "failed" || t.status === "cancelled") && <button onClick={onRetry}>Retry</button>}
          {t.status === "completed" && <button onClick={onPin} disabled={pinned} title="Compare every trial with this single run">{pinned ? "Baseline" : "Pin as baseline"}</button>}</td>
      </tr>
      {(open || t.status === "invalid" || t.status === "failed") && (
        <tr><td></td><td colSpan={10} className="small">
          {t.status === "invalid" && t.diagnostics.map((d, i) => <div key={i} className="errbadge"><b>{d.code}</b> {d.message} <span className="muted">(not scheduled)</span></div>)}
          {t.attempts.map((a) => <div key={a.attempt}>attempt {a.attempt} ({a.kind}) · <span className={`status ${a.status}`}>{a.status}</span> · run {a.runId ?? "—"}{a.error ? <span className="bad"> · {a.error}</span> : ""}</div>)}
          {m?.available && <div className="muted">metric: {m.aggregation}{m.step != null ? ` · step ${m.step} (epoch ${m.epoch})` : ""} · partition {m.partition} · {direction}</div>}
        </td></tr>
      )}
    </>
  );
}

function Compare({ a, b, objective }: { a: StudyTrial; b: StudyTrial; objective: string }) {
  const [d, setD] = useState<RunDiff | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setD(null); setErr(null); if (a.runId && b.runId) api.get<RunDiff>(`/api/runs/${a.runId}/diff/${b.runId}`).then(setD).catch((e) => setErr(errorText(e))); }, [a.runId, b.runId]);
  const ident = (t: StudyTrial) => `${t.trialId} · seed ${t.seed ?? "—"} · fold ${t.fold ?? "—"} · attempt ${t.attempts.at(-1)?.attempt ?? "—"}`;
  return (
    <div className="compare">
      <h4>Comparison</h4>
      <table className="dtable"><thead><tr><th></th><th>A: {a.label}</th><th>B: {b.label}</th></tr></thead><tbody>
        <tr><td>identity</td><td>{ident(a)}</td><td>{ident(b)}</td></tr>
        <tr><td>{objective}</td><td className="num">{fmtV(a.value)}</td><td className="num">{fmtV(b.value)}</td></tr>
        <tr><td>semantics</td><td className="small">{a.metric?.aggregation}</td><td className="small">{b.metric?.aggregation}</td></tr>
      </tbody></table>
      {err && <div className="error">{err}</div>}
      {d && (
        <>
          {d.warning && <div className="warn">{d.warning}</div>}
          <table className="dtable"><thead><tr><th>what differs</th><th>field</th><th>A</th><th>B</th></tr></thead><tbody>
            {d.graph.changes.map((c, i) => <tr key={"g" + i}><td>{c.kind}{c.node ? ` · ${c.node}` : ""}</td><td>{c.field ?? c.edge ?? c.type ?? ""}</td><td>{c.from !== undefined ? JSON.stringify(c.from) : ""}</td><td>{c.to !== undefined ? JSON.stringify(c.to) : ""}</td></tr>)}
            {d.runConfig.map((c, i) => <tr key={"r" + i}><td>run config</td><td>{c.field}</td><td>{JSON.stringify(c.from)}</td><td>{JSON.stringify(c.to)}</td></tr>)}
            {d.data.map((c, i) => <tr key={"d" + i}><td>source identity · {c.node}</td><td>snapshot / hash</td><td><code>{String(c.from ?? "").slice(0, 12)}</code></td><td><code>{String(c.to ?? "").slice(0, 12)}</code></td></tr>)}
            {d.changeCount === 0 && <tr><td colSpan={4} className="muted">The graphs, run configurations and recorded source identities are identical (only the trial identity differs).</td></tr>}
          </tbody></table>
        </>
      )}
    </div>
  );
}
