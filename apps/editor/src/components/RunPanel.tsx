import { useEffect, useRef, useState } from "react";
import { api, errorText } from "../api";
import { useInspect, usePolling, useRunStream } from "../hooks";
import type { ConfusionResult, Graph, InferResult, RunConfig, RunMetrics, RunSummary, SampleLossResult, Unavailable, Validation } from "../types";
import { fmtInt, fmtNum, shortHash, uid } from "../util";
import { LineChart, type Series } from "./LineChart";
import { NotRecorded, ProvLine } from "./Provenance";
import type { Ctx } from "./InspectorTabs";

const ACTIVE = ["queued", "preparing", "running", "cancelling"];
const COLORS = { a: "#1f6feb", b: "#d9730d" };

export interface RunPanelProps {
  projectId: string; graph: Graph; validation: Validation | null; runs: RunSummary[]; reloadRuns: () => void;
  ctx: Ctx; setCtx: (c: Ctx) => void; baseline: string | null; setBaseline: (id: string | null) => void;
  ensureSaved: () => Promise<void>;
}

// ---------------------------------------------------------------------------------------- train
const DEFAULT_CFG = { data: "examples/data/shapes10", epochs: 2, batch_size: 16, lr: 0.05, optimizer: "sgd" as const, device: "cpu" as "cpu" | "cuda", backend: "pytorch" as "pytorch" | "keras" | "jax", workers: 1, momentum: 0, seed: 0, val_fraction: 0.2, split_seed: "" as number | "" };

function TrainTab({ p }: { p: RunPanelProps }) {
  const [cfg, setCfg] = useState(DEFAULT_CFG);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const key = useRef<{ sig: string; id: string }>({ sig: "", id: "" });
  const run = p.runs.find((r) => r.id === p.ctx.runId) ?? null;
  const stream = useRunStream(p.ctx.runId, p.reloadRuns);
  const status = run?.status ?? stream.status;
  const active = !!status && ACTIVE.includes(status);
  const set = <K extends keyof typeof cfg>(k: K, v: (typeof cfg)[K]) => setCfg({ ...cfg, [k]: v });
  const num = (k: "epochs" | "batch_size" | "lr" | "momentum" | "seed" | "val_fraction" | "workers", label: string, step?: string) => (
    <label>{label}<input type="number" step={step ?? "any"} value={cfg[k]} onChange={(e) => e.target.value !== "" && set(k, Number(e.target.value))} /></label>
  );

  async function start() {
    setBusy(true); setErr(null);
    try {
      await p.ensureSaved();
      const body = { projectId: p.projectId, config: { ...cfg, split_seed: cfg.split_seed === "" ? null : cfg.split_seed } as Partial<RunConfig> };
      const sig = JSON.stringify(body) + p.validation?.graphHash;
      if (key.current.sig !== sig) key.current = { sig, id: uid() };
      const r = await api.post<{ runId: string; idempotentReplay: boolean }>("/api/runs", body, { "Idempotency-Key": key.current.id });
      key.current = { sig: "", id: "" }; // a later deliberate click is a new submission
      p.setCtx({ runId: r.runId, step: null, sample: null });
      p.reloadRuns();
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  async function cancel() {
    if (!p.ctx.runId) return;
    try { await api.post(`/api/runs/${p.ctx.runId}/cancel`, {}); p.reloadRuns(); } catch (e) { setErr(errorText(e)); }
  }

  const valSeries: Series[] = [
    { id: "train", label: "train loss (per step)", color: COLORS.a, points: stream.trainLoss },
    { id: "val", label: "val loss (per epoch, at its step)", color: COLORS.b, points: stream.epochs.map((e) => [e.step, e.val_loss]) },
  ];
  const blocked = !p.validation?.ok;
  return (
    <div className="train">
      <div className="trainform">
        <label>Dataset folder (relative to the server's working dir)<input value={cfg.data} onChange={(e) => set("data", e.target.value)} /></label>
        {num("epochs", "Epochs", "1")}{num("batch_size", "Batch size", "1")}{num("lr", "Learning rate")}
        <label>Optimizer<select value={cfg.optimizer} onChange={(e) => set("optimizer", e.target.value as "sgd")}><option value="sgd">sgd</option><option value="adam">adam</option></select></label>
        <label title="Keras/JAX train with plain SGD (momentum 0) on CPU from the same seeded initialization; checkpoints are saved in PyTorch format.">Backend<select aria-label="training backend" value={cfg.backend} onChange={(e) => set("backend", e.target.value as "pytorch" | "keras" | "jax")}><option value="pytorch">pytorch</option><option value="keras">keras (plain SGD)</option><option value="jax">jax (plain SGD)</option></select></label>
        <label title="CUDA runs only where the worker has a usable NVIDIA GPU; otherwise the run fails with E_DEVICE_UNAVAILABLE. CUDA results are not bitwise reproducible.">Device<select aria-label="training device" value={cfg.device} onChange={(e) => set("device", e.target.value as "cpu" | "cuda")}><option value="cpu">cpu</option><option value="cuda">cuda (NVIDIA GPU)</option></select></label>
        {num("momentum", "Momentum")}{num("seed", "Seed", "1")}{num("val_fraction", "Val fraction")}<span title="Data-parallel processes on the worker machine (PyTorch, CPU); 1–8. Same result as one process up to float summation order.">{num("workers", "Workers", "1")}</span>
        <label>Split seed (blank = seed)<input type="number" value={cfg.split_seed} onChange={(e) => set("split_seed", e.target.value === "" ? "" : Number(e.target.value))} /></label>
        <div className="actions">
          <button className="primary" disabled={busy || blocked} onClick={start} title={blocked ? "Fix the graph errors first" : "Save the project and train a new run"}>Run</button>
          <button disabled={!active || status === "cancelling"} onClick={cancel}>Cancel</button>
        </div>
        {blocked && <div className="warn">The draft graph has errors; Run is disabled until validation passes.</div>}
        {err && <div className="error pre">{err}</div>}
      </div>
      <div className="trainview">
        {!p.ctx.runId ? <NotRecorded message="No run selected. Press Run to train; curves appear here from the live event stream." /> : (
          <>
            <div className="statusline">
              <b>run {p.ctx.runId}</b> <span className={`status ${status}`}>{status ?? "..."}</span>
              {run && <span> step {run.progress.step}{run.progress.stepsPerEpoch ? ` / ${run.progress.stepsPerEpoch * run.progress.epochs}` : ""} · epoch {run.progress.epochsDone}/{run.progress.epochs}</span>}
              <span className="muted"> · stream {stream.connected ? "connected" : "closed"} · {stream.events} events (last seq {stream.lastSeq})</span>
              {run?.error && <div className="error">{run.error}</div>}
            </div>
            <div className="charts">
              <div><LineChart series={valSeries} xLabel="step" yLabel="cross-entropy loss" /></div>
              <div><LineChart series={[{ id: "acc", label: "val accuracy (per epoch)", color: "#2a9d4b", points: stream.epochs.map((e) => [e.step, e.val_acc]) }]} xLabel="step" yLabel="val accuracy" yMin={0} /></div>
            </div>
            <div className="prov">Provenance: run {p.ctx.runId} · graph {shortHash(run?.graphHash)} · raw points from the worker's event log (no smoothing)</div>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------- compare
type Flat = Record<string, string>;
function flatten(o: unknown, prefix = "", out: Flat = {}): Flat {
  if (o && typeof o === "object" && !Array.isArray(o)) for (const [k, v] of Object.entries(o)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  else out[prefix] = JSON.stringify(o);
  return out;
}
function diff(a: Flat, b: Flat): { key: string; a: string; b: string }[] {
  return [...new Set([...Object.keys(a), ...Object.keys(b)])].sort().filter((k) => a[k] !== b[k]).map((k) => ({ key: k, a: a[k] ?? "(absent)", b: b[k] ?? "(absent)" }));
}
const graphFlat = (g: Graph | null): Flat => {
  const o: Flat = {};
  for (const n of g?.nodes ?? []) { o[`node ${n.id}`] = n.type; flatten(n.config, `node ${n.id}`, o); }
  for (const e of g?.edges ?? []) o[`edge ${e.from.node}.${e.from.port}->${e.to.node}.${e.to.port}`] = "present";
  return o;
};

function Compare({ aId, bId }: { aId: string; bId: string }) {
  const a = usePolling<RunMetrics>(`/api/runs/${aId}/metrics`, 3000).data;
  const b = usePolling<RunMetrics>(`/api/runs/${bId}/metrics`, 3000).data;
  if (!a || !b) return <div className="muted">loading…</div>;
  const cfgDiff = diff(flatten({ ...a.summary.config, project_id: undefined }), flatten({ ...b.summary.config, project_id: undefined }));
  const gDiff = diff(graphFlat(a.graph), graphFlat(b.graph));
  const dataDiff = a.summary.split?.datasetSha256 !== b.summary.split?.datasetSha256;
  const total = cfgDiff.length + gDiff.length + (dataDiff ? 1 : 0);
  const mk = (id: string, m: RunMetrics, color: string, dashed: boolean) => ({ id, label: `${id}${dashed ? " (baseline)" : ""}`, color, dashed, m });
  const A = mk(aId, a, COLORS.a, true), B = mk(bId, b, COLORS.b, false);
  const fin = (m: RunMetrics) => m.epochs[m.epochs.length - 1];
  return (
    <div className="compare">
      <div className="charts">
        <LineChart xLabel="optimizer step" yLabel="train loss" series={[A, B].map((s) => ({ id: s.id, label: s.label, color: s.color, dashed: s.dashed, points: s.m.trainLoss }))} />
        <LineChart xLabel="optimizer step" yLabel="val loss" series={[A, B].map((s) => ({ id: s.id, label: s.label, color: s.color, dashed: s.dashed, points: s.m.epochs.map((e) => [e.step, e.val_loss] as [number, number]) }))} />
        <LineChart xLabel="optimizer step" yLabel="val accuracy" yMin={0} series={[A, B].map((s) => ({ id: s.id, label: s.label, color: s.color, dashed: s.dashed, points: s.m.epochs.map((e) => [e.step, e.val_acc] as [number, number]) }))} />
      </div>
      <table className="cmp">
        <thead><tr><th>Final metric</th><th>{aId} (baseline)</th><th>{bId}</th></tr></thead>
        <tbody>
          <tr><td>status</td><td>{a.summary.status}</td><td>{b.summary.status}</td></tr>
          <tr><td>graph hash</td><td>{shortHash(a.summary.graphHash)}</td><td>{shortHash(b.summary.graphHash)}</td></tr>
          <tr><td>parameters</td><td>{a.summary.totalParams != null ? fmtInt(a.summary.totalParams) : "n/a"}</td><td>{b.summary.totalParams != null ? fmtInt(b.summary.totalParams) : "n/a"}</td></tr>
          <tr><td>epochs finished</td><td>{a.epochs.length}</td><td>{b.epochs.length}</td></tr>
          <tr><td>train loss (epoch mean)</td><td>{fin(a) ? fmtNum(fin(a).train_loss) : "n/a"}</td><td>{fin(b) ? fmtNum(fin(b).train_loss) : "n/a"}</td></tr>
          <tr><td>val loss</td><td>{fin(a) ? fmtNum(fin(a).val_loss) : "n/a"}</td><td>{fin(b) ? fmtNum(fin(b).val_loss) : "n/a"}</td></tr>
          <tr><td>val accuracy</td><td>{fin(a) ? `${(fin(a).val_acc * 100).toFixed(1)}%` : "n/a"}</td><td>{fin(b) ? `${(fin(b).val_acc * 100).toFixed(1)}%` : "n/a"}</td></tr>
        </tbody>
      </table>
      <h4>Differences ({total})</h4>
      {total === 0 && <div className="muted">Same graph, configuration and dataset.</div>}
      {total > 1 && <div className="warn">More than one thing differs, so a metric difference cannot be attributed to a single change.</div>}
      {a.summary.config.batch_size !== b.summary.config.batch_size && <div className="warn">Batch sizes differ, so a step is not the same amount of data in both runs.</div>}
      {dataDiff && <div className="warn">The dataset contents differ (sha256 {shortHash(a.summary.split?.datasetSha256)} vs {shortHash(b.summary.split?.datasetSha256)}).</div>}
      <table className="cmp">
        <tbody>
          {cfgDiff.map((d) => <tr key={"c" + d.key}><td>run config: {d.key}</td><td>{d.a}</td><td>{d.b}</td></tr>)}
          {gDiff.map((d) => <tr key={"g" + d.key}><td>graph: {d.key}</td><td>{d.a}</td><td>{d.b}</td></tr>)}
        </tbody>
      </table>
      <div className="prov">Provenance: runs {aId} and {bId} · raw per-step and per-epoch points from their event logs · graph configs from each run's stored graph</div>
    </div>
  );
}

function RunsTab({ p }: { p: RunPanelProps }) {
  const [other, setOther] = useState<string | null>(null);
  const cmpB = other ?? (p.ctx.runId && p.ctx.runId !== p.baseline ? p.ctx.runId : null);
  const cur = p.validation?.graphHash;
  return (
    <div className="runs">
      <table className="cmp">
        <thead><tr><th></th><th>run</th><th>status</th><th>graph</th><th>params</th><th>epochs</th><th>val acc</th><th>val loss</th><th>seed</th><th>lr</th><th></th></tr></thead>
        <tbody>
          {[...p.runs].reverse().map((r) => (
            <tr key={r.id} className={r.id === p.ctx.runId ? "sel" : ""}>
              <td>{r.id === p.baseline ? "★" : ""}</td>
              <td><button className="link" onClick={() => p.setCtx({ runId: r.id, step: null, sample: p.ctx.sample })}>{r.id}</button></td>
              <td><span className={`status ${r.status}`}>{r.status}</span></td>
              <td>{shortHash(r.graphHash)} {cur && (r.graphHash === cur ? <span className="badge ok">current</span> : <span className="badge old">older architecture</span>)}</td>
              <td>{r.totalParams != null ? fmtInt(r.totalParams) : "n/a"}</td>
              <td>{r.progress.epochsDone}/{r.progress.epochs}</td>
              <td>{r.final ? `${(r.final.val_acc * 100).toFixed(1)}%` : "n/a"}</td>
              <td>{r.final ? fmtNum(r.final.val_loss) : "n/a"}</td>
              <td>{r.config.seed}</td><td>{r.config.lr}</td>
              <td>
                <button onClick={() => p.setBaseline(r.id === p.baseline ? null : r.id)}>{r.id === p.baseline ? "Unpin" : "Pin baseline"}</button>{" "}
                <button disabled={!p.baseline || r.id === p.baseline} onClick={() => setOther(r.id)}>Compare to baseline</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {p.runs.length === 0 && <NotRecorded message="No runs yet." />}
      {!p.baseline && p.runs.length > 0 && <div className="muted small">Pin a baseline, then choose another run to compare against it. The baseline is stored in the project's UI document (save to keep it).</div>}
      {p.baseline && cmpB && p.baseline !== cmpB && <Compare aId={p.baseline} bId={cmpB} />}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- results
function ConfusionView({ d }: { d: ConfusionResult }) {
  const mx = Math.max(1, ...d.matrix.flat());
  return (
    <div className="confusion">
      <table>
        <thead><tr><th>true \ pred</th>{d.classes.map((c) => <th key={c} className="rot"><span>{c}</span></th>)}</tr></thead>
        <tbody>
          {d.matrix.map((row, i) => (
            <tr key={i}><th>{d.classes[i]}</th>{row.map((v, j) => <td key={j} className={i === j ? "diag" : ""} style={{ background: v ? `rgba(${i === j ? "42,157,75" : "178,24,43"},${0.15 + 0.85 * (v / mx)})` : undefined }}>{v || ""}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <div className="muted small">{d.axes}. Counts of validation samples.</div>
      <ProvLine p={d.provenance} />
    </div>
  );
}

function ResultsTab({ p }: { p: RunPanelProps }) {
  const url = p.ctx.runId ? `/api/runs/${p.ctx.runId}/inspect` : null;
  const run0 = p.runs.find((r) => r.id === p.ctx.runId);
  const conf = useInspect<ConfusionResult | Unavailable>(url, { kind: "confusion", _rev: run0?.progress.epochsDone });
  const worst = useInspect<SampleLossResult | Unavailable>(url, { kind: "sample_loss", limit: 12, _rev: run0?.progress.epochsDone });
  if (!p.ctx.runId) return <NotRecorded message="No run selected. Confusion matrix and per-sample losses come from a recorded run's validation pass." />;
  return (
    <div className="results">
      <div>
        <h4>Confusion matrix</h4>
        {conf.error ? <div className="error">{conf.error}</div> : conf.data && (conf.data.available ? <ConfusionView d={conf.data as ConfusionResult} /> : <><NotRecorded message={(conf.data as Unavailable).message} /><ProvLine p={(conf.data as Unavailable).provenance} /></>)}
      </div>
      <div>
        <h4>Worst validation samples (by loss)</h4>
        {worst.error ? <div className="error">{worst.error}</div> : worst.data && (worst.data.available ? (
          <>
            <table className="cmp"><thead><tr><th></th><th>sample</th><th>true</th><th>predicted</th><th>loss</th></tr></thead>
              <tbody>
                {(worst.data as SampleLossResult).worst.map((w) => (
                  <tr key={w.index} className={p.ctx.sample === w.index ? "sel" : ""} onClick={() => p.setCtx({ ...p.ctx, sample: w.index })} style={{ cursor: "pointer" }}>
                    <td><img className="thumb" src={`/api/runs/${p.ctx.runId}/samples/${w.index}/image`} alt={w.id} /></td>
                    <td>#{w.index} {w.id}</td><td>{w.labelName}</td><td className={w.pred === w.label ? "" : "bad"}>{w.predName}</td><td className="num">{fmtNum(w.loss)}</td>
                  </tr>
                ))}
              </tbody></table>
            <div className="muted small">Click a row to use it as the inspection sample (activations, wires).</div>
            <ProvLine p={(worst.data as SampleLossResult).provenance} />
          </>
        ) : <><NotRecorded message={(worst.data as Unavailable).message} /><ProvLine p={(worst.data as Unavailable).provenance} /></>)}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------- infer
function InferTab({ p }: { p: RunPanelProps }) {
  const [result, setResult] = useState<InferResult | Unavailable | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  useEffect(() => { setResult(null); setErr(null); setPreview(null); }, [p.ctx.runId, p.ctx.step]);
  async function go(body: Record<string, unknown>) {
    setErr(null);
    try { setResult(await api.post("/api/infer", { runId: p.ctx.runId, checkpointStep: p.ctx.step, ...body })); } catch (e) { setErr(errorText(e)); }
  }
  async function upload(f: File | undefined) {
    if (!f) return;
    const url = await new Promise<string>((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = () => rej(r.error); r.readAsDataURL(f); });
    setPreview(url);
    await go({ imageBase64: url });
  }
  if (!p.ctx.runId) return <NotRecorded message="No run selected. Inference needs a checkpoint from a run." />;
  const r = result && result.available ? (result as InferResult) : null;
  const imgSrc = preview ?? (r?.provenance.sampleIndex != null ? `/api/runs/${p.ctx.runId}/samples/${r.provenance.sampleIndex}/image` : null);
  return (
    <div className="infer-tab">
      <div className="actions">
        <button disabled={p.ctx.sample == null} onClick={() => { setPreview(null); go({ sample: p.ctx.sample }); }}>Infer on val sample {p.ctx.sample != null ? `#${p.ctx.sample}` : "(pick one in the inspection context)"}</button>
        <label className="file">or upload an image: <input type="file" accept="image/*" onChange={(e) => upload(e.target.files?.[0])} /></label>
      </div>
      {err && <div className="error">{err}</div>}
      {result && !result.available && <NotRecorded message={(result as Unavailable).message} />}
      {r && (
        <div className="inferout">
          {imgSrc && <img className="inferimg" src={imgSrc} alt="input" />}
          <div>
            <div><b>Predicted: {r.predictedName}</b> ({(r.probabilities[r.predicted] * 100).toFixed(1)}%){r.provenance.trueLabel ? ` · true label: ${r.provenance.trueLabel}` : ""}</div>
            <table className="cmp"><thead><tr><th>class</th><th>logit</th><th>probability</th><th></th></tr></thead>
              <tbody>{r.classes.map((c, i) => (
                <tr key={c} className={i === r.predicted ? "sel" : ""}><td>{c}</td><td className="num">{fmtNum(r.logits[i])}</td><td className="num">{(r.probabilities[i] * 100).toFixed(2)}%</td>
                  <td><div className="bar" style={{ width: `${r.probabilities[i] * 100}%` }} /></td></tr>))}</tbody></table>
            <ProvLine p={r.provenance} extra={r.provenance.preprocessing} />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- shell
const TABS = ["Train", "Runs & compare", "Results", "Inference"] as const;
export function RunPanel(p: RunPanelProps) {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Train");
  const content = tab;
  return (
    <div className="runpanel">
      <div className="tabs" role="tablist">{TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}</div>
      <div className="tabbody">
        {content === "Train" && <TrainTab p={p} />}
        {content === "Runs & compare" && <RunsTab p={p} />}
        {content === "Results" && <ResultsTab p={p} />}
        {content === "Inference" && <InferTab p={p} />}
      </div>
    </div>
  );
}
