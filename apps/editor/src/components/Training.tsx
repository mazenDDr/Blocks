import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, errorText } from "../api";
import { usePolling } from "../hooks";
import type { Graph, ProcedureCheck, ProcedureRunSummary } from "../types";
import { fmtNum, uid } from "../util";
import { LineChart, type Series } from "./LineChart";
import { NotRecorded } from "./Provenance";

type P = Record<string, any>;
const STAGE_HELP: Record<string, (p: P, i: number) => string> = {
  zero_grad: (p, i) => (i < p.stages.indexOf("forward") ? "clears gradients at the START of each accumulation window" : "clears gradients at the END of each accumulation window (after the step)"),
  forward: () => "runs the model on each micro-batch (train mode)",
  loss: () => "evaluates the declared loss on each micro-batch",
  backward: (p) => `back-propagates each micro-batch's loss${(p.accumulation?.steps ?? 1) > 1 ? ` scaled by 1/window length (accumulating over ${p.accumulation.steps} micro-batches)` : ""}`,
  clip: (p) => (p.clip?.kind === "none" ? "no clipping configured (does nothing)" : `once per optimizer step, on the accumulated gradients: ${p.clip.kind === "norm" ? `global norm to ${p.clip.max_norm}` : `each value to ±${p.clip.value}`}`),
  optimizer_step: (p) => `${p.optimizer?.kind ?? "sgd"} update, once per accumulation window`,
  scheduler_step: (p) => (p.scheduler?.kind === "none" ? "no scheduler configured" : p.scheduler?.timing === "optimizer_step" ? "steps the scheduler after each optimizer step" : `ignored: the scheduler steps on '${p.scheduler?.timing}'`),
};
const REQUIRED = ["forward", "loss", "backward", "optimizer_step"];
const OPTIONAL = ["zero_grad", "clip", "scheduler_step"];

function Row({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <div className="row"><label className="lbl">{label}</label><span>{children}{hint && <small className="muted"> {hint}</small>}</span></div>;
}
function Num({ v, set, label, step, min }: { v: number; set: (n: number) => void; label: string; step?: number | string; min?: number }) {
  return <input type="number" aria-label={label} value={v} step={step ?? "any"} min={min} onChange={(e) => { const n = Number(e.target.value); if (e.target.value !== "" && Number.isFinite(n)) set(n); }} />;
}
function Sel({ v, set, opts, label }: { v: string; set: (s: string) => void; opts: string[]; label: string }) {
  return <select aria-label={label} value={v} onChange={(e) => set(e.target.value)}>{opts.map((o) => <option key={o}>{o}</option>)}</select>;
}
function Section({ title, children, open }: { title: string; children: ReactNode; open?: boolean }) {
  return <details open={open} className="procsec"><summary>{title}</summary>{children}</details>;
}

function useProcCheck(proc: P | null) {
  const [c, setC] = useState<ProcedureCheck | null>(null);
  useEffect(() => {
    if (!proc) { setC(null); return; }
    const ctl = new AbortController();
    const t = setTimeout(() => { api.post<ProcedureCheck>("/api/procedure/check", { procedure: proc }, undefined, ctl.signal).then(setC).catch(() => {}); }, 250);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [JSON.stringify(proc)]); // eslint-disable-line react-hooks/exhaustive-deps
  return c;
}

export function TrainingWorkspace({ projectId, graph, setGraph, runs, reloadRuns, ensureSaved, valid, onOpenDebug, setMessage }: {
  projectId: string; graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; runs: ProcedureRunSummary[]; reloadRuns: () => void;
  ensureSaved: () => Promise<void>; valid: boolean; onOpenDebug: (runId: string) => void; setMessage: (m: string) => void;
}) {
  const proc = (graph.training ?? null) as P | null;
  const check = useProcCheck(proc);
  const [runId, setRunId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const key = useRef<{ sig: string; id: string }>({ sig: "", id: "" });
  useEffect(() => { if (!runId && runs.length) setRunId(runs[runs.length - 1].id); }, [runs, runId]);
  const run = runs.find((r) => r.id === runId) ?? null;
  const active = !!run && ["queued", "preparing", "running", "cancelling"].includes(run.status);
  const view = usePolling<any>(runId ? `/api/runs/${runId}/procedure` : null, active ? 1500 : 0, [run?.status]);
  const [ckStep, setCkStep] = useState<number | null>(null);
  const opt = usePolling<any>(runId && view.data ? `/api/runs/${runId}/optimizer${ckStep ? `?step=${ckStep}` : ""}` : null, 0, [view.data?.checkpoints?.length, ckStep]);

  const set = (patch: P) => setGraph((g) => ({ ...g, training: { ...(g.training ?? {}), ...patch } }));
  const sub = (k: string, patch: P) => set({ [k]: { ...(proc?.[k] ?? {}), ...patch } });

  async function createDefault() {
    try {
      const d = await api.post<{ procedure: P; note: string }>("/api/procedure/default", { graph });
      setGraph((g) => ({ ...g, training: d.procedure }));
      setMessage(d.note);
    } catch (e) { setMessage(errorText(e)); }
  }
  async function start(resume?: { run_id: string; step: number }) {
    setBusy(true); setErr(null);
    try {
      await ensureSaved();
      const body = { projectId, config: { kind: "procedure", procedure: proc, ...(resume ? { resume_from: resume } : {}) } };
      const sig = JSON.stringify(body);
      if (key.current.sig !== sig) key.current = { sig, id: uid() };
      const r = await api.post<{ runId: string }>("/api/runs", body, { "Idempotency-Key": key.current.id });
      key.current = { sig: "", id: "" };
      setRunId(r.runId);
      reloadRuns();
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  async function cancel() { if (runId) { try { await api.post(`/api/runs/${runId}/cancel`, {}); reloadRuns(); } catch (e) { setErr(errorText(e)); } } }

  if (!proc) {
    return (
      <div className="pad">
        <h3>Training procedure</h3>
        <p className="muted">This model graph has no declared training procedure yet. The procedure is an ordered list of stages (zero_grad, forward, loss, backward, clip, optimizer step, scheduler step) plus the configuration of each stage, saved with the project.</p>
        <button className="primary" onClick={createDefault}>Create a default procedure for this graph</button>
      </div>
    );
  }
  const move = (i: number, d: number) => { const s = [...proc.stages]; const j = i + d; if (j < 0 || j >= s.length) return; [s[i], s[j]] = [s[j], s[i]]; set({ stages: s }); };
  const missing = OPTIONAL.filter((o) => !proc.stages.includes(o));
  const diags = check?.diagnostics ?? [];
  const errs = diags.filter((d) => d.severity === "error");
  const fields = proc.data?.kind === "synthetic_regression" ? ["x", "y", "mask", "weights"] : ["x", "y"];
  const lossMods = (graph.modules ?? []).filter((m) => m.outputs.some((o) => o.name === (proc.loss?.output ?? "loss")));
  const steps = view.data?.steps ?? [];
  const series = (id: string, label: string, color: string, pts: [number, number][]): Series => ({ id, label, color, points: pts });
  const lossSeries: Series[] = [
    series("train", "train loss (per optimizer step)", "#1f6feb", steps.map((s: any) => [s.step, s.loss])),
    series("val", "validation loss", "#c2410c", (view.data?.validations ?? []).map((v: any) => [v.step, v.val_loss])),
  ];
  const lrSeries: Series[] = [series("lr", "learning rate", "#2a9d4b", steps.map((s: any) => [s.step, s.lr]))];
  const gnSeries: Series[] = [series("gn", "gradient norm (before clipping)", "#7c3aed", steps.filter((s: any) => s.grad_norm != null).map((s: any) => [s.step, s.grad_norm]))];

  return (
    <div className="training">
      <div className="procedit">
        <h3>Training procedure <small className="muted">saved in the project; part of its semantic identity</small></h3>
        <div className="notice synthetic">{proc.data?.kind === "image_folder" ? `Data: image folder ${proc.data.path}.` : <><b>SYNTHETIC data.</b> {proc.data?.kind} generated from data_seed; it stands for no real measurement.</>}</div>
        {diags.map((d, i) => <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}><b>{d.code}</b> {d.message}</div>)}
        <Section title="Order of operations (executed exactly in this order)" open>
          <ol className="stages">
            {proc.stages.map((s: string, i: number) => (
              <li key={s}>
                <span className="stagebtns">
                  <button aria-label={`move ${s} up`} disabled={i === 0} onClick={() => move(i, -1)}>▲</button>
                  <button aria-label={`move ${s} down`} disabled={i === proc.stages.length - 1} onClick={() => move(i, 1)}>▼</button>
                  {!REQUIRED.includes(s) && <button className="danger" aria-label={`remove ${s}`} onClick={() => set({ stages: proc.stages.filter((x: string) => x !== s) })}>×</button>}
                </span>
                <b>{s}</b>
                <span className="muted small"> {STAGE_HELP[s]?.(proc, i)}</span>
              </li>
            ))}
          </ol>
          {missing.length > 0 && <div className="row">{missing.map((m) => <button key={m} onClick={() => set({ stages: [...proc.stages, m] })}>Add {m}</button>)}</div>}
          <div className="muted small">Gradients are cleared at the boundary where zero_grad is placed; clip must sit between backward and optimizer_step; the scheduler steps where its timing says. Wrong orders are rejected before anything runs.</div>
        </Section>
        <Section title="Data and batching" open>
          <Row label="Data"><Sel label="data kind" v={proc.data.kind} opts={["synthetic_sequence", "synthetic_regression", "image_folder"]} set={(v) => sub("data", { kind: v })} /></Row>
          {proc.data.kind === "image_folder" && <Row label="Folder"><input aria-label="image folder" value={proc.data.path} onChange={(e) => sub("data", { path: e.target.value })} /></Row>}
          <Row label="Batch size"><Num label="batch size" v={proc.data.batch_size} min={1} step={1} set={(n) => sub("data", { batch_size: n })} /></Row>
          <Row label="Shuffle"><input type="checkbox" aria-label="shuffle" checked={proc.data.shuffle} onChange={(e) => sub("data", { shuffle: e.target.checked })} /></Row>
          <Row label="Data seed"><Num label="data seed" v={proc.data.data_seed} step={1} set={(n) => sub("data", { data_seed: n })} /></Row>
          <Row label="Epochs"><Num label="epochs" v={proc.epochs} min={1} step={1} set={(n) => set({ epochs: n })} /></Row>
          <Row label="Seed"><Num label="seed" v={proc.seed} step={1} set={(n) => set({ seed: n })} />
            <label> <input type="checkbox" aria-label="deterministic" checked={proc.deterministic} onChange={(e) => set({ deterministic: e.target.checked })} /> deterministic (CPU, single thread: exact resume)</label></Row>
        </Section>
        <Section title="Loss">
          <Row label="Loss"><Sel label="loss kind" v={proc.loss.kind} opts={["cross_entropy", "mse", "module"]} set={(v) => sub("loss", { kind: v })} /></Row>
          {proc.loss.kind === "module" && (
            <>
              <Row label="Module"><select aria-label="loss module" value={`${proc.loss.module}@${proc.loss.version}`} onChange={(e) => { const [m, v] = e.target.value.split("@"); sub("loss", { module: m, version: v }); }}>
                <option value="@1.0.0">choose a module…</option>
                {lossMods.map((m) => <option key={`${m.id}@${m.version}`} value={`${m.id}@${m.version}`}>{m.id} v{m.version}</option>)}</select></Row>
              {(() => { const m = (graph.modules ?? []).find((x) => x.id === proc.loss.module && x.version === proc.loss.version); return m ? (
                <>
                  {m.reduction && <div className="muted small">Declared reduction: {m.reduction.description ?? m.reduction.divisor}. {m.reduction.empty_policy}</div>}
                  {m.inputs.map((pt) => (
                    <Row key={pt.name} label={`input ${pt.name}`}><select aria-label={`loss port ${pt.name}`} value={proc.loss.ports?.[pt.name] ?? ""} onChange={(e) => sub("loss", { ports: { ...proc.loss.ports, [pt.name]: e.target.value } })}>
                      <option value="">(not wired)</option><option value="output">the model output</option>{fields.map((f) => <option key={f} value={f}>dataset field {f}</option>)}</select></Row>
                  ))}
                </>) : null; })()}
            </>
          )}
        </Section>
        <Section title="Optimizer">
          <Row label="Optimizer"><Sel label="optimizer" v={proc.optimizer.kind} opts={["sgd", "adam", "adamw"]} set={(v) => sub("optimizer", { kind: v })} /></Row>
          <Row label="Learning rate"><Num label="learning rate" v={proc.optimizer.lr} set={(n) => sub("optimizer", { lr: n })} /></Row>
          <Row label="Weight decay"><Num label="weight decay" v={proc.optimizer.weight_decay} min={0} set={(n) => sub("optimizer", { weight_decay: n })} />
            <small className="muted"> {proc.optimizer.kind === "adamw" ? "decoupled (w ← w(1−lr·wd))" : "L2 added to the gradient"}</small></Row>
          {proc.optimizer.kind === "sgd" ? (
            <>
              <Row label="Momentum"><Num label="momentum" v={proc.optimizer.momentum} min={0} set={(n) => sub("optimizer", { momentum: n })} /></Row>
              <Row label="Nesterov"><input type="checkbox" aria-label="nesterov" checked={proc.optimizer.nesterov} onChange={(e) => sub("optimizer", { nesterov: e.target.checked })} /></Row>
            </>
          ) : (
            <>
              <Row label="Betas"><Num label="beta1" v={proc.optimizer.betas[0]} set={(n) => sub("optimizer", { betas: [n, proc.optimizer.betas[1]] })} /> <Num label="beta2" v={proc.optimizer.betas[1]} set={(n) => sub("optimizer", { betas: [proc.optimizer.betas[0], n] })} /></Row>
              <Row label="Epsilon"><Num label="epsilon" v={proc.optimizer.eps} set={(n) => sub("optimizer", { eps: n })} /></Row>
              <Row label="AMSGrad"><input type="checkbox" aria-label="amsgrad" checked={proc.optimizer.amsgrad} onChange={(e) => sub("optimizer", { amsgrad: e.target.checked })} /></Row>
            </>
          )}
        </Section>
        <Section title="Gradient accumulation and clipping">
          <Row label="Micro-batches per step"><Num label="accumulation steps" v={proc.accumulation.steps} min={1} step={1} set={(n) => sub("accumulation", { steps: n })} /></Row>
          <Row label="Loss scaling"><Sel label="accumulation normalize" v={proc.accumulation.normalize} opts={["mean", "sum"]} set={(v) => sub("accumulation", { normalize: v })} /></Row>
          <Row label="Short last window"><Sel label="partial window" v={proc.accumulation.partial_window} opts={["step", "drop"]} set={(v) => sub("accumulation", { partial_window: v })} /></Row>
          <Row label="Clipping"><Sel label="clip kind" v={proc.clip.kind} opts={["none", "norm", "value"]} set={(v) => sub("clip", { kind: v })} /></Row>
          {proc.clip.kind === "norm" && <Row label="Max norm"><Num label="max norm" v={proc.clip.max_norm} min={0} set={(n) => sub("clip", { max_norm: n })} /> <Num label="norm type" v={proc.clip.norm_type} set={(n) => sub("clip", { norm_type: n })} /></Row>}
          {proc.clip.kind === "value" && <Row label="Clip value"><Num label="clip value" v={proc.clip.value} min={0} set={(n) => sub("clip", { value: n })} /></Row>}
        </Section>
        <Section title="Scheduler">
          <Row label="Scheduler"><Sel label="scheduler kind" v={proc.scheduler.kind} opts={["none", "step", "exponential", "cosine", "linear_warmup", "reduce_on_plateau"]} set={(v) => sub("scheduler", { kind: v, timing: v === "reduce_on_plateau" ? "validation" : proc.scheduler.timing === "validation" ? "optimizer_step" : proc.scheduler.timing })} /></Row>
          {proc.scheduler.kind !== "none" && <Row label="Steps on"><Sel label="scheduler timing" v={proc.scheduler.timing} opts={["optimizer_step", "epoch", "validation"]} set={(v) => sub("scheduler", { timing: v })} /></Row>}
          {proc.scheduler.kind === "step" && <><Row label="Step size"><Num label="step size" v={proc.scheduler.step_size} min={1} step={1} set={(n) => sub("scheduler", { step_size: n })} /></Row><Row label="Gamma"><Num label="gamma" v={proc.scheduler.gamma} set={(n) => sub("scheduler", { gamma: n })} /></Row></>}
          {proc.scheduler.kind === "exponential" && <Row label="Gamma"><Num label="gamma" v={proc.scheduler.gamma} set={(n) => sub("scheduler", { gamma: n })} /></Row>}
          {proc.scheduler.kind === "cosine" && <Row label="T max"><Num label="t max" v={proc.scheduler.t_max} min={1} step={1} set={(n) => sub("scheduler", { t_max: n })} /></Row>}
          {proc.scheduler.kind === "linear_warmup" && <Row label="Warmup steps"><Num label="warmup steps" v={proc.scheduler.warmup_steps} min={0} step={1} set={(n) => sub("scheduler", { warmup_steps: n })} /></Row>}
          {proc.scheduler.kind === "reduce_on_plateau" && <><Row label="Factor"><Num label="factor" v={proc.scheduler.factor} set={(n) => sub("scheduler", { factor: n })} /></Row><Row label="Patience"><Num label="patience" v={proc.scheduler.patience} min={0} step={1} set={(n) => sub("scheduler", { patience: n })} /></Row></>}
        </Section>
        <Section title="Validation, checkpoints, early stopping">
          <Row label="Validate every"><Num label="validation every" v={proc.validation.every.n} min={1} step={1} set={(n) => sub("validation", { every: { ...proc.validation.every, n } })} /> <Sel label="validation unit" v={proc.validation.every.unit} opts={["epoch", "optimizer_step"]} set={(u) => sub("validation", { every: { ...proc.validation.every, unit: u } })} /></Row>
          <Row label="Evaluation"><label><input type="checkbox" aria-label="eval mode" checked={proc.validation.eval_mode} onChange={(e) => sub("validation", { eval_mode: e.target.checked })} /> eval mode</label> <label><input type="checkbox" aria-label="no grad" checked={proc.validation.no_grad} onChange={(e) => sub("validation", { no_grad: e.target.checked })} /> no gradient recording</label><small className="muted"> separate switches</small></Row>
          <Row label="Checkpoint every">
            <input type="checkbox" aria-label="checkpointing" checked={!!proc.checkpoint.every} onChange={(e) => sub("checkpoint", { every: e.target.checked ? { unit: "epoch", n: 1 } : null })} />
            {proc.checkpoint.every && <><Num label="checkpoint every" v={proc.checkpoint.every.n} min={1} step={1} set={(n) => sub("checkpoint", { every: { ...proc.checkpoint.every, n } })} /> <Sel label="checkpoint unit" v={proc.checkpoint.every.unit} opts={["epoch", "optimizer_step"]} set={(u) => sub("checkpoint", { every: { ...proc.checkpoint.every, unit: u } })} /></>}
          </Row>
          <Row label="Keep last"><input aria-label="keep last" type="number" min={1} value={proc.checkpoint.keep_last ?? ""} placeholder="all" onChange={(e) => sub("checkpoint", { keep_last: e.target.value === "" ? null : Number(e.target.value) })} /><small className="muted"> older ones are marked pruned</small></Row>
          <Row label="Best checkpoint"><input type="checkbox" aria-label="best checkpoint" checked={!!proc.checkpoint.best} onChange={(e) => sub("checkpoint", { best: e.target.checked ? { metric: "val_loss", mode: "min" } : null })} />{proc.checkpoint.best && <> on {proc.checkpoint.best.metric} <Sel label="best mode" v={proc.checkpoint.best.mode} opts={["min", "max"]} set={(m) => sub("checkpoint", { best: { ...proc.checkpoint.best, mode: m } })} /></>}</Row>
          <Row label="Early stopping"><input type="checkbox" aria-label="early stopping" checked={!!proc.early_stopping} onChange={(e) => set({ early_stopping: e.target.checked ? { metric: "val_loss", mode: "min", patience: 3, min_delta: 0 } : null })} />
            {proc.early_stopping && <> patience <Num label="patience validations" v={proc.early_stopping.patience} min={1} step={1} set={(n) => set({ early_stopping: { ...proc.early_stopping, patience: n } })} /> min delta <Num label="min delta" v={proc.early_stopping.min_delta} min={0} set={(n) => set({ early_stopping: { ...proc.early_stopping, min_delta: n } })} /></>}</Row>
        </Section>
        <Section title="Frozen nodes, watched element, optimizer inspection">
          <Row label="Frozen nodes"><input aria-label="frozen nodes" placeholder="node ids, comma separated" value={(proc.frozen ?? []).join(", ")} onChange={(e) => set({ frozen: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} /></Row>
          <Row label="Watch element"><input aria-label="watch parameter" placeholder="parameter, e.g. head.weight" value={proc.watch?.param ?? ""} onChange={(e) => set({ watch: e.target.value ? { param: e.target.value, index: proc.watch?.index ?? [0] } : null })} />
            {proc.watch && <input aria-label="watch index" size={8} value={(proc.watch.index ?? [0]).join(",")} onChange={(e) => set({ watch: { ...proc.watch, index: e.target.value.split(",").map((s) => Number(s.trim())).filter(Number.isFinite) } })} />}</Row>
        </Section>
        <Section title="Instrumentation (debugger)" open>
          <Row label="Capture steps"><input aria-label="capture steps" placeholder="optimizer steps, e.g. 5, 20" value={(proc.capture?.steps ?? []).join(", ")} onChange={(e) => { const s = e.target.value.split(",").map((x) => Number(x.trim())).filter((n) => Number.isInteger(n) && n > 0); set({ capture: s.length ? { steps: s, nodes: proc.capture?.nodes ?? null, max_elements: 400000 } : null }); }} />
            <small className="muted"> records every node's output and gradient for the first micro-batch of those steps</small></Row>
          <Row label="Assertions"><label><input type="checkbox" aria-label="enable assertions" checked={!!proc.assertions} onChange={(e) => set({ assertions: e.target.checked })} /> enable Assert blocks <b className="exec">execution-changing</b></label></Row>
          <h4>Conditional breakpoints <span className="exec">execution-changing: they stop the run</span></h4>
          {(proc.breakpoints ?? []).map((b: P, i: number) => (
            <div className="row" key={i}>
              <input aria-label={`breakpoint ${i + 1} id`} value={b.id} size={8} onChange={(e) => set({ breakpoints: proc.breakpoints.map((x: P, j: number) => (j === i ? { ...x, id: e.target.value } : x)) })} />
              <span>
                <Sel label={`breakpoint ${i + 1} kind`} v={b.kind} opts={["nonfinite_loss", "nonfinite_node", "grad_norm_above", "loss_above", "step_reached"]} set={(k) => set({ breakpoints: proc.breakpoints.map((x: P, j: number) => (j === i ? { ...x, kind: k } : x)) })} />
                {b.kind === "nonfinite_node" && <input aria-label={`breakpoint ${i + 1} node`} placeholder="node id" size={10} value={b.node ?? ""} onChange={(e) => set({ breakpoints: proc.breakpoints.map((x: P, j: number) => (j === i ? { ...x, node: e.target.value } : x)) })} />}
                {(b.kind === "grad_norm_above" || b.kind === "loss_above") && <Num label={`breakpoint ${i + 1} threshold`} v={b.threshold ?? 0} set={(n) => set({ breakpoints: proc.breakpoints.map((x: P, j: number) => (j === i ? { ...x, threshold: n } : x)) })} />}
                {b.kind === "step_reached" && <Num label={`breakpoint ${i + 1} step`} v={b.step ?? 1} step={1} min={1} set={(n) => set({ breakpoints: proc.breakpoints.map((x: P, j: number) => (j === i ? { ...x, step: n } : x)) })} />}
                <button className="danger" aria-label={`remove breakpoint ${i + 1}`} onClick={() => set({ breakpoints: proc.breakpoints.filter((_: P, j: number) => j !== i) })}>×</button>
              </span>
            </div>
          ))}
          <button onClick={() => set({ breakpoints: [...(proc.breakpoints ?? []), { id: `bp${(proc.breakpoints ?? []).length + 1}`, kind: "nonfinite_loss" }] })}>Add breakpoint</button>
        </Section>
      </div>

      <div className="procrun">
        <div className="actions">
          <button className="primary" disabled={busy || !valid || errs.length > 0} onClick={() => start()} title={!valid ? "Fix the graph errors first" : errs.length ? "Fix the procedure errors first" : "Save the project and train"}>Run procedure</button>
          <button disabled={!active || run?.status === "cancelling"} onClick={cancel}>Cancel</button>
          <select aria-label="run" value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)}>
            <option value="">(no run)</option>{[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status} · step {r.progress.step}{r.rerunOf ? " · rerun" : ""}</option>)}
          </select>
          {runId && <button onClick={() => onOpenDebug(runId)}>Open in debugger</button>}
        </div>
        {!valid && <div className="warn">The graph has errors; Run is disabled until validation passes.</div>}
        {err && <div className="error pre">{err}</div>}
        {!run ? <NotRecorded message="No run selected. Press Run procedure; everything below is read from the recorded events and checkpoints of the run." /> : (
          <>
            <div className="statusline">
              <b>run {run.id}</b> <span className={`status ${run.status}`}>{run.status}</span> step {run.progress.step}{run.progress.stepsPerEpoch && run.progress.epochs ? ` / ${run.progress.stepsPerEpoch * run.progress.epochs}` : ""} · epoch {run.progress.epochsDone}/{run.progress.epochs}
              {run.stoppedBy && <> · stopped by <b>{run.stoppedBy}</b></>}
              {run.resumedFrom && <> · resumed from {run.resumedFrom.run_id} step {run.resumedFrom.step}</>}
              {run.rerunOf && <> · rerun of {run.rerunOf}</>}
              {run.dataNote && <div className="muted small">{run.synthetic ? "SYNTHETIC. " : ""}{run.dataNote}</div>}
            </div>
            {run.error && <div className="error pre">{run.error}</div>}
            <LineChart series={lossSeries} xLabel="optimizer step" yLabel="loss" height={170} />
            <div className="chartrow"><LineChart series={lrSeries} xLabel="optimizer step" yLabel="lr" height={120} /><LineChart series={gnSeries} xLabel="optimizer step" yLabel="‖g‖" height={120} /></div>
            {view.data?.breakpoints?.length > 0 && <div className="warn">{view.data.breakpoints.map((b: any) => <div key={b.seq}><b>execution-changing stop</b> at step {b.step}: {b.detail} {b.breakpoint ? `(breakpoint ${b.breakpoint})` : ""}</div>)}</div>}
            <h4>Validation</h4>
            <table><thead><tr><th>step</th><th>epoch</th><th>val loss</th><th>val acc</th><th>eval mode</th><th>no grad</th></tr></thead><tbody>
              {(view.data?.validations ?? []).slice(-8).map((v: any) => <tr key={v.seq}><td>{v.step}</td><td>{v.epoch}</td><td className="num">{fmtNum(v.val_loss)}</td><td className="num">{v.val_acc != null ? `${(v.val_acc * 100).toFixed(1)}%` : "n/a"}</td><td>{String(v.eval_mode)}</td><td>{String(v.no_grad)}</td></tr>)}
            </tbody></table>
            {(view.data?.earlyStopping?.length ?? 0) > 0 && <div className="muted small">Early stopping: {view.data.earlyStopping.slice(-1).map((e: any) => `${e.metric} ${fmtNum(e.value)}, best ${fmtNum(e.best)}, ${e.bad}/${e.patience} validations without improvement`)}</div>}
            {(view.data?.schedulerSteps?.length ?? 0) > 0 && <div className="muted small">Scheduler steps recorded: {view.data.schedulerSteps.length} (timing {view.data.schedulerSteps[0].timing}); last lr {fmtNum(view.data.schedulerSteps.slice(-1)[0].lr_after)}</div>}
            <h4>Checkpoints <small className="muted">resume continues exactly on CPU in deterministic mode</small></h4>
            <table><thead><tr><th>step</th><th>tag</th><th>status</th><th>exact resume</th><th /></tr></thead><tbody>
              {(view.data?.checkpoints ?? []).map((c: any) => (
                <tr key={c.sha256 + c.step + c.tag} className={ckStep === c.step ? "sel" : ""}>
                  <td>{c.step}</td><td>{c.tag}</td><td>{c.status}</td><td>{c.exact_resume ? "yes (bitwise)" : "approximate"}</td>
                  <td>{c.status !== "pruned" && <><button onClick={() => setCkStep(c.step)}>Inspect optimizer state</button> <button disabled={busy || active} onClick={() => start({ run_id: run.id, step: c.step })}>Resume in a new run</button></>}</td>
                </tr>))}
            </tbody></table>
            <h4>Optimizer state (VISION 10.2)</h4>
            {opt.data?.available ? (
              <>
                <div className="muted small">{opt.data.optimizer.kind} stored in the checkpoint at step {opt.data.step}: groups {opt.data.paramGroups.map((g: any) => `lr ${fmtNum(g.lr)}${g.betas ? `, betas ${g.betas.join("/")}` : ""}${g.momentum ? `, momentum ${g.momentum}` : ""}, wd ${g.weight_decay}`).join(" | ")}</div>
                <table><thead><tr><th>parameter</th><th>step count</th><th>exp_avg ‖·‖</th><th>exp_avg_sq mean</th><th>momentum_buffer ‖·‖</th></tr></thead><tbody>
                  {opt.data.state.map((s: any) => <tr key={s.index}><td>{s.name}</td><td className="num">{s.step?.value ?? "n/a"}</td><td className="num">{s.exp_avg ? fmtNum(s.exp_avg.norm ?? s.exp_avg.value) : "n/a"}</td><td className="num">{s.exp_avg_sq ? fmtNum(s.exp_avg_sq.mean ?? s.exp_avg_sq.value) : "n/a"}</td><td className="num">{s.momentum_buffer ? fmtNum(s.momentum_buffer.norm ?? s.momentum_buffer.value) : "n/a"}</td></tr>)}
                </tbody></table>
              </>
            ) : <NotRecorded message={opt.data?.message ?? "Pick a checkpoint above to read the optimizer state stored in it."} />}
            {(view.data?.optimizerTrace?.length ?? 0) > 0 && (
              <>
                <h4>Watched element {view.data.optimizerTrace[0].param}[{view.data.optimizerTrace[0].index.join(",")}]: the first updates</h4>
                <table><thead><tr><th>step</th><th>w before</th><th>gradient{view.data.optimizerTrace[0].grad_is_after_clipping ? " (after clipping)" : ""}</th><th>w after (actual)</th><th>formula predicts</th><th>|diff|</th></tr></thead><tbody>
                  {view.data.optimizerTrace.slice(0, 6).map((t: any) => <tr key={t.seq} title={(t.formula ?? []).join("\n")}><td>{t.step}</td><td className="num">{fmtNum(t.w_before, 7)}</td><td className="num">{fmtNum(t.grad, 5)}</td><td className="num">{fmtNum(t.w_after, 7)}</td><td className="num">{fmtNum(t.w_predicted, 7)}</td><td className="num">{t.abs_diff.toExponential(1)}</td></tr>)}
                </tbody></table>
                <pre>{(view.data.optimizerTrace[0].formula ?? []).join("\n")}</pre>
                <div className="muted small">The formula is a hand-written mirror of the {view.data.optimizerTrace[0].optimizer} update; |diff| is its distance from the value torch produced (hover a row for its steps).</div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
