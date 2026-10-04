import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../api";
import { useInspect, usePolling } from "../hooks";
import type { Graph, ProcedureRunSummary, Validation } from "../types";
import { fmtNum } from "../util";
import { NotRecorded } from "./Provenance";

interface StepInfo { step: number; epoch: number; loss: number; lr: number | null; gradNorm: number | null; captured: boolean; checkpoint: boolean; captureNodes?: number }
interface Steps { runId: string; status: string; steps: StepInfo[]; captureSteps: number[]; breakpointsHit: any[] }

function Scrubber({ steps, step, setStep }: { steps: StepInfo[]; step: number; setStep: (s: number) => void }) {
  if (!steps.length) return <NotRecorded message="This run has recorded no optimizer steps yet." />;
  const cur = steps.find((s) => s.step === step);
  const W = Math.max(300, Math.min(860, steps.length * 8)), bw = W / steps.length;
  return (
    <div className="scrubber">
      <div className="row" style={{ gridTemplateColumns: "1fr 150px" }}>
        <input type="range" aria-label="step scrubber" min={steps[0].step} max={steps[steps.length - 1].step} value={step} onChange={(e) => setStep(Number(e.target.value))} />
        <b>run step {step} / {steps[steps.length - 1].step}</b>
      </div>
      <svg width={W} height={34} role="img" aria-label="captured steps">
        {steps.map((s, i) => (
          <g key={s.step} onClick={() => setStep(s.step)} style={{ cursor: "pointer" }}>
            <rect x={i * bw} y={s.captured ? 2 : 14} width={Math.max(2, bw - 1)} height={s.captured ? 28 : 16} fill={s.step === step ? "#c2410c" : s.captured ? "#1f6feb" : "#c8ccd4"}><title>{`step ${s.step}: loss ${fmtNum(s.loss)}${s.captured ? " · intermediate values recorded" : " · intermediates not recorded"}${s.checkpoint ? " · checkpoint" : ""}`}</title></rect>
            {s.checkpoint && <polygon points={`${i * bw},34 ${i * bw + bw / 2 + 2},27 ${i * bw + bw},34`} fill="#2a9d4b" />}
          </g>
        ))}
      </svg>
      <div className="muted small">Tall blue bars: intermediate values were captured at that step. Grey: loss and learning rate only; anything else there is <b>not recorded</b>. Green marks: checkpoints.</div>
      {cur && <div className="small">Selected: step {cur.step} · epoch {cur.epoch} · loss {fmtNum(cur.loss)} · lr {cur.lr != null ? fmtNum(cur.lr) : "n/a"} · gradient norm {cur.gradNorm != null ? fmtNum(cur.gradNorm) : "n/a"} · {cur.captured ? `${cur.captureNodes} nodes captured` : "no captured values"}</div>}
    </div>
  );
}

function ValueTable({ s }: { s: any }) {
  return (
    <table className="valtable"><tbody>
      <tr><td>shape / dtype</td><td>[{s.shape.join(", ")}] {s.dtype}</td></tr>
      <tr><td>elements</td><td>{s.count}{s.nonFinite ? <b className="exec"> · {s.nonFinite} non-finite</b> : ""}</td></tr>
      {s.min !== undefined && <><tr><td>min / max</td><td>{fmtNum(s.min)} / {fmtNum(s.max)}</td></tr><tr><td>mean / std / norm</td><td>{fmtNum(s.mean)} / {fmtNum(s.std)} / {fmtNum(s.norm)}</td></tr></>}
      <tr><td>first values</td><td><code>{s.values.map((v: number | null) => (v === null ? "nan" : fmtNum(v, 4))).join("  ")}{s.truncatedValues ? " …" : ""}</code></td></tr>
    </tbody></table>
  );
}

function CaptureButton({ runId, step, nodes, setMessage, onStarted }: { runId: string; step: number; nodes: string[] | null; setMessage: (m: string) => void; onStarted: (id: string) => void }) {
  const [busy, setBusy] = useState(false);
  return (
    <button className="primary" disabled={busy} onClick={async () => {
      setBusy(true);
      try {
        const r = await api.post<{ runId: string; resumeFrom: any }>(`/api/runs/${runId}/debug/capture`, { step, nodes });
        setMessage(`Started run ${r.runId}: a deterministic rerun of ${runId} (${r.resumeFrom ? `from its checkpoint at step ${r.resumeFrom.max_step} or earlier` : "from the seeded initialisation"}) capturing step ${step}. It is verified against the original when it finishes.`);
        onStarted(r.runId);
      } catch (e) { setMessage(errorText(e)); } finally { setBusy(false); }
    }}>Capture and rerun</button>
  );
}

function WireTab({ runId, step, node, setNode, graph, nodes, setMessage, setRunId }: { runId: string; step: number; node: string; setNode: (n: string) => void; graph: Graph; nodes: string[]; setMessage: (m: string) => void; setRunId: (id: string) => void }) {
  const r = useInspect<any>(node ? `/api/runs/${runId}/debug/wire` : null, node ? { step, node } : null);
  const top = graph.nodes.find((n) => n.id === node);
  const producer = top ? graph.edges.find((e) => e.to.node === node)?.from.node : undefined;
  const consumers = top ? graph.edges.filter((e) => e.from.node === node).map((e) => e.to.node) : [];
  return (
    <div>
      <div className="row"><label className="lbl">Node / wire</label><select aria-label="debug node" value={node} onChange={(e) => setNode(e.target.value)}><option value="">choose a node…</option>{nodes.map((n) => <option key={n}>{n}</option>)}</select></div>
      {top && <div className="small">trace source: {producer ? <button className="link" onClick={() => setNode(producer)}>{producer}</button> : "none"} · follow consumers: {consumers.length ? consumers.map((c) => <button key={c} className="link" onClick={() => setNode(c)}>{c}</button>) : "none"}</div>}
      {!node && <NotRecorded message="Choose a node: its output is the value crossing its outgoing wire." />}
      {r.loading && <div className="muted">loading…</div>}
      {r.error && <div className="error">{r.error}</div>}
      {r.data && !r.data.available && (
        <div className="notrec">
          <b>{r.data.reason === "not_recorded" ? "Not recorded" : "Unavailable"}</b> — {r.data.message}
          {r.data.action && <div style={{ marginTop: 6 }}><div className="muted small">{r.data.action.description}</div><CaptureButton runId={runId} step={step} nodes={[node]} setMessage={setMessage} onStarted={setRunId} /></div>}
        </div>
      )}
      {r.data?.available && (
        <>
          <ValueTable s={r.data.summary} />
          <div className="muted small">Captured value of {r.data.node} · run {r.data.provenance.runId} · optimizer step {r.data.provenance.step} · micro-batch {r.data.provenance.micro_batch} · graph {r.data.provenance.graphHash.slice(0, 8)}<br />{r.data.provenance.note}</div>
        </>
      )}
    </div>
  );
}

function GradTab({ runId, step, node, setNode, nodes }: { runId: string; step: number; node: string; setNode: (n: string) => void; nodes: string[] }) {
  const ov = useInspect<any>(`/api/runs/${runId}/debug/gradients`, { step });
  const nd = useInspect<any>(node ? `/api/runs/${runId}/debug/gradients` : null, node ? { step, node } : null);
  return (
    <div>
      {ov.data && !ov.data.available && <NotRecorded message={ov.data.message} />}
      {ov.data?.available && (
        <>
          <div className="muted small">{ov.data.provenance.note}. Captured at step {ov.data.provenance.step} in run {runId}.</div>
          <table><thead><tr><th>node output</th><th>‖∂L/∂out‖</th><th>min</th><th>max</th><th /></tr></thead><tbody>
            {Object.entries(ov.data.nodes).map(([n, s]: [string, any]) => <tr key={n} className={n === node ? "sel" : ""}><td>{n}</td><td className="num">{s.norm !== undefined ? fmtNum(s.norm) : "n/a"}</td><td className="num">{s.min !== undefined ? fmtNum(s.min) : ""}</td><td className="num">{s.max !== undefined ? fmtNum(s.max) : ""}</td><td><button onClick={() => setNode(n)}>open</button></td></tr>)}
          </tbody></table>
          <table><thead><tr><th>parameter</th><th>‖grad‖</th><th>min</th><th>max</th></tr></thead><tbody>
            {Object.entries(ov.data.parameters).map(([n, s]: [string, any]) => <tr key={n}><td>{n}</td><td className="num">{fmtNum(s.norm)}</td><td className="num">{fmtNum(s.min)}</td><td className="num">{fmtNum(s.max)}</td></tr>)}
          </tbody></table>
        </>
      )}
      <div className="row"><label className="lbl">Node</label><select aria-label="gradient node" value={node} onChange={(e) => setNode(e.target.value)}><option value="">(overview above)</option>{nodes.map((n) => <option key={n}>{n}</option>)}</select></div>
      {nd.data && !nd.data.available && <div className="notrec">{nd.data.message}</div>}
      {nd.data?.available && nd.data.output && <><h4>∂loss/∂(output of {node})</h4><ValueTable s={nd.data.output} /></>}
      {nd.data?.available && Object.keys(nd.data.parameters ?? {}).length > 0 && <><h4>Parameter gradients of {node}</h4>{Object.entries(nd.data.parameters).map(([n, s]: [string, any]) => <div key={n}><b>{n}</b><ValueTable s={s} /></div>)}</>}
    </div>
  );
}

function ProbesTab({ graph, nodes, setMessage }: { graph: Graph; nodes: string[]; setMessage: (m: string) => void }) {
  const [text, setText] = useState("");
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const verify = async () => {
    setBusy(true);
    try { setRes(await api.post<any>("/api/debug/probes/verify", { graph, nodes: text.split(",").map((s) => s.trim()).filter(Boolean).length ? text.split(",").map((s) => s.trim()).filter(Boolean) : null, batch: 8, repeats: 20 })); }
    catch (e) { setMessage(errorText(e)); setRes(null); } finally { setBusy(false); }
  };
  return (
    <div>
      <div className="muted small">Probes are observation-only: forward hooks that read a detached copy. Press the button to verify on this model that they change nothing and to measure what they cost. Pin probes into a training run in the Training tab (Instrumentation).</div>
      <div className="row"><label className="lbl">Probed nodes</label><input aria-label="probed nodes" placeholder={`blank = every node · e.g. ${nodes.slice(0, 2).join(", ")}`} value={text} onChange={(e) => setText(e.target.value)} /></div>
      <button className="primary" disabled={busy} onClick={verify}>Verify invariants and measure overhead</button>
      {res && (
        <>
          <table><thead><tr><th>invariant (probes on vs off, same seed)</th><th>holds</th></tr></thead><tbody>
            {Object.entries(res.invariants).filter(([k]) => k.endsWith("identical")).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{v ? "✓ yes" : <b className="exec">✗ NO</b>}</td></tr>)}
          </tbody></table>
          <div>Overhead on {res.invariants.probed_nodes} probed node(s), forward+backward, batch {res.batch}: {fmtNum(res.overhead.baselineMs)} ms → {fmtNum(res.overhead.probedMs)} ms (<b>{res.overhead.overheadPercent >= 0 ? "+" : ""}{res.overhead.overheadPercent.toFixed(1)}%</b>, median of {res.overhead.repeats} runs)</div>
          <div className="muted small">{res.overhead.note}. Mode: {res.mode}. Inputs: {res.inputs}.</div>
        </>
      )}
    </div>
  );
}

function SandboxTab({ runId, step, steps, nodes, setMessage }: { runId: string; step: number; steps: StepInfo[]; nodes: string[]; setMessage: (m: string) => void }) {
  const [kind, setKind] = useState<"hyper" | "activation" | "parameter">("hyper");
  const [path, setPath] = useState("optimizer.lr");
  const [value, setValue] = useState(0.5);
  const [node, setNode] = useState("");
  const [name, setName] = useState("");
  const [op, setOp] = useState("zero");
  const [index, setIndex] = useState("");
  const [fwd, setFwd] = useState(2);
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const list = usePolling<{ sandboxes: any[] }>(`/api/runs/${runId}/debug/sandboxes`, 0, [res?.sandboxRunId]);
  const iv = () => {
    const idx = index.trim() ? index.split(",").map((s) => Number(s.trim())) : null;
    if (kind === "hyper") return { kind, path, value };
    if (kind === "activation") return { kind, node, op, index: idx, value };
    return { kind, name, op, index: idx, value };
  };
  const run = async () => {
    setBusy(true);
    try { setRes(await api.post<any>(`/api/runs/${runId}/debug/sandbox`, { step, interventions: [iv()], stepsForward: fwd, captureNodes: node ? [node] : null, label: `${kind} at step ${step}` })); }
    catch (e) { setMessage(errorText(e)); } finally { setBusy(false); }
  };
  return (
    <div>
      <div className="muted small">Branch from the state at the end of step {step - 1}, change something, rerun in isolation. The original run, its checkpoints and its events are never modified (they are hashed before and after); the comparison is stored as a new sandbox record.</div>
      <div className="row"><label className="lbl">Change</label><select aria-label="intervention kind" value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}><option value="hyper">a setting (learning rate, clip, momentum, decay)</option><option value="activation">a node's value (execution-changing, sandbox only)</option><option value="parameter">a parameter element</option></select></div>
      {kind === "hyper" && <div className="row"><label className="lbl">Setting</label><span><select aria-label="setting" value={path} onChange={(e) => setPath(e.target.value)}>{["optimizer.lr", "optimizer.weight_decay", "optimizer.momentum", "clip.max_norm", "clip.value"].map((p) => <option key={p}>{p}</option>)}</select> = <input type="number" aria-label="new value" value={value} step="any" onChange={(e) => setValue(Number(e.target.value))} /></span></div>}
      {kind === "activation" && <div className="row"><label className="lbl">Node</label><span><select aria-label="intervention node" value={node} onChange={(e) => setNode(e.target.value)}><option value="">choose…</option>{nodes.map((n) => <option key={n}>{n}</option>)}</select></span></div>}
      {kind === "parameter" && <div className="row"><label className="lbl">Parameter</label><input aria-label="parameter name" placeholder="e.g. head.weight" value={name} onChange={(e) => setName(e.target.value)} /></div>}
      {kind !== "hyper" && <div className="row"><label className="lbl">Operation</label><span><select aria-label="operation" value={op} onChange={(e) => setOp(e.target.value)}>{["zero", "set", "scale", "add"].map((o) => <option key={o}>{o}</option>)}</select> {op !== "zero" && <input type="number" aria-label="operand" value={value} step="any" onChange={(e) => setValue(Number(e.target.value))} />} element <input aria-label="element index" size={8} placeholder="blank = all" value={index} onChange={(e) => setIndex(e.target.value)} /></span></div>}
      <div className="row"><label className="lbl">Steps to run</label><input type="number" aria-label="steps forward" min={1} max={50} value={fwd} onChange={(e) => setFwd(Math.max(1, Math.min(50, Number(e.target.value) || 1)))} /></div>
      <button className="primary" disabled={busy || step < 1 || step > (steps[steps.length - 1]?.step ?? 0) || (kind === "activation" && !node) || (kind === "parameter" && !name)} onClick={run}>Branch at step {step} and run</button>
      {res && (
        <div className="sandboxres">
          <h4>Result {res.sandboxRunId}</h4>
          <div>{res.outcome}</div>
          <h4>What changed</h4>
          <table><thead><tr><th>kind</th><th>target</th><th>before</th><th>after</th></tr></thead><tbody>{res.changed.map((c: any, i: number) => <tr key={i}><td>{c.kind}{c.op ? ` (${c.op})` : ""}</td><td>{c.target}{c.index ? `[${c.index.join(",")}]` : ""}</td><td className="num">{c.before != null ? fmtNum(c.before) : "n/a"}</td><td className="num">{c.after != null ? fmtNum(c.after) : "n/a"}</td></tr>)}</tbody></table>
          <h4>Held fixed</h4><ul className="small">{res.heldFixed.map((h: string) => <li key={h}>{h}</li>)}</ul>
          <h4>Outcome per step</h4>
          <table><thead><tr><th>step</th><th>original</th><th>branch</th><th>Δ</th></tr></thead><tbody>{res.diff.lossPerStep.map((s: any) => <tr key={s.step}><td>{s.step}</td><td className="num">{fmtNum(s.baseline, 6)}</td><td className="num">{fmtNum(s.branch, 6)}</td><td className="num">{s.delta >= 0 ? "+" : ""}{fmtNum(s.delta, 3)}</td></tr>)}</tbody></table>
          <div className="small">parameters differ by L2 {fmtNum(res.diff.paramsL2DistanceBranchVsBaseline)} · gradient norm at the step {fmtNum(res.diff.gradNormAtStep.baseline)} → {fmtNum(res.diff.gradNormAtStep.branch)}</div>
          {Object.keys(res.diff.activationMaxAbsDiff).length > 0 && <div className="small">max |Δ activation|: {Object.entries(res.diff.activationMaxAbsDiff).map(([n, d]: [string, any]) => `${n} ${fmtNum(d)}`).join(" · ")}</div>}
          <div className={res.immutability.originalRunUnchanged ? "badge ok" : "errbadge"}>{res.immutability.originalRunUnchanged ? "original run unchanged" : "ORIGINAL CHANGED"}: evidence hash {res.immutability.fingerprintBefore.sha256.slice(0, 12)} before = {res.immutability.fingerprintAfter.sha256.slice(0, 12)} after ({res.immutability.fingerprintAfter.events} events, {res.immutability.fingerprintAfter.artifacts} artifacts)</div>
          <div className="muted small">Original state reconstructed from {res.parent.reconstruction.startedFrom}{res.parent.reconstruction.checkpoint ? ` (checkpoint at step ${res.parent.reconstruction.checkpoint.step})` : ""}, replaying {res.parent.reconstruction.replayedSteps} step(s); replay identical to the recorded losses: {String(res.parent.reconstruction.replayIdenticalToRecorded)}.</div>
        </div>
      )}
      <h4>Earlier sandboxes of this run</h4>
      {(list.data?.sandboxes ?? []).length === 0 ? <div className="muted">None yet.</div> : (
        <table><tbody>{list.data!.sandboxes.map((s) => <tr key={s.sandboxRunId}><td>{s.sandboxRunId}</td><td>step {s.step}</td><td>{s.label}</td><td className="num">Δloss {fmtNum(s.lossDelta, 3)}</td><td><button onClick={async () => setRes(await api.get<any>(`/api/runs/${runId}/debug/sandboxes/${s.sandboxRunId}`))}>show</button></td></tr>)}</tbody></table>
      )}
    </div>
  );
}

export function DebuggerWorkspace({ graph, validation, runs, runId, setRunId, setMessage, onEditProcedure }: {
  graph: Graph; validation: Validation | null; runs: ProcedureRunSummary[]; runId: string | null; setRunId: (id: string | null) => void; setMessage: (m: string) => void; onEditProcedure: () => void;
}) {
  const run = runs.find((r) => r.id === runId) ?? null;
  const steps = usePolling<Steps>(runId ? `/api/runs/${runId}/debug/steps` : null, run && ["running", "preparing", "queued"].includes(run.status) ? 1500 : 0, [run?.status, run?.progress.step]);
  const [step, setStep] = useState(1);
  const [node, setNode] = useState("");
  const [tab, setTab] = useState<"Wire" | "Gradients" | "Breakpoints" | "Probes" | "Sandbox">("Wire");
  const list = steps.data?.steps ?? [];
  // start on the first captured step of a run (else its last step), once per run
  useEffect(() => { if (list.length) setStep(steps.data?.captureSteps?.[0] ?? list[list.length - 1].step); }, [runId, list.length > 0]); // eslint-disable-line react-hooks/exhaustive-deps
  const nodes = useMemo(() => {
    const flat = Object.entries(validation?.flat ?? {}).filter(([, v]) => v.typed).map(([k]) => k);
    const top = Object.entries(validation?.nodes ?? {}).filter(([k, v]) => v.typed && !v.structural && graph.nodes.some((n) => n.id === k)).map(([k]) => k);
    return [...top, ...flat];
  }, [validation, graph.nodes]);
  const rerunOf = run?.rerunOf;
  const ver = usePolling<any>(rerunOf && runId ? `/api/runs/${runId}/procedure` : null, 0, [run?.status]);
  return (
    <div className="debugger">
      <div className="actions">
        <label>Run <select aria-label="debug run" value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)}><option value="">(none)</option>{[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status} · step {r.progress.step}{r.rerunOf ? ` · rerun of ${r.rerunOf}` : ""}</option>)}</select></label>
        <button onClick={onEditProcedure}>Edit capture / breakpoints in the procedure</button>
        <span className="muted small">Mode: <b>Inspect</b> — captured steps only; a capture-and-rerun is a separate, labelled run.</span>
      </div>
      {!run ? <NotRecorded message="Select a training-procedure run. The debugger reads what the run actually recorded; it never invents history." /> : (
        <>
          {rerunOf && ver.data?.rerunVerification?.[0] && <div className={ver.data.rerunVerification[0].identical ? "badge ok" : "errbadge"}>Rerun of {rerunOf}: {ver.data.rerunVerification[0].identical ? `losses at steps ${ver.data.rerunVerification[0].compared_steps.join(", ")} are bit-identical to the original` : "DID NOT reproduce the original: values below are not evidence about it"}</div>}
          <Scrubber steps={list} step={step} setStep={setStep} />
          <div className="tabs" role="tablist">{(["Wire", "Gradients", "Breakpoints", "Probes", "Sandbox"] as const).map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}</div>
          <div className="tabbody">
            {tab === "Wire" && runId && <WireTab runId={runId} step={step} node={node} setNode={setNode} graph={graph} nodes={nodes} setMessage={setMessage} setRunId={setRunId} />}
            {tab === "Gradients" && runId && <GradTab runId={runId} step={step} node={node} setNode={setNode} nodes={nodes} />}
            {tab === "Breakpoints" && (
              <div>
                <div className="muted small">Conditional breakpoints are defined in the procedure and are <b className="exec">execution-changing</b>: when one fires the run stops at a safe boundary and keeps a checkpoint of the state before the offending step.</div>
                {(steps.data?.breakpointsHit ?? []).length === 0 ? <div className="muted">This run hit no breakpoint.</div> : steps.data!.breakpointsHit.map((b, i) => <div key={i} className="warn"><b>{b.breakpoint ?? "assertion"}</b> at step {b.step}: {b.detail}</div>)}
                <button onClick={onEditProcedure}>Edit breakpoints</button>
              </div>
            )}
            {tab === "Probes" && <ProbesTab graph={graph} nodes={nodes} setMessage={setMessage} />}
            {tab === "Sandbox" && runId && <SandboxTab runId={runId} step={step} steps={list} nodes={nodes} setMessage={setMessage} />}
          </div>
        </>
      )}
    </div>
  );
}
