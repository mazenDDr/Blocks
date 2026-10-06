import { useEffect, useMemo, useRef, useState } from "react";
import { api, errorText } from "../../api";
import type { AnyRun, Graph, OpInfo, UiDoc, Validation } from "../../types";
import { isRLRun } from "../../types";
import { runLabel, uid } from "../../util";
import { Num } from "../agent/common";
import { BufferTab } from "./BufferBrowser";
import { useCurves, useGet } from "./common";
import { EnvPanel } from "./EnvPanel";
import { EvalTab } from "./EvalTab";
import { LearnerPanel } from "./LearnerPanel";
import { RolloutsTab } from "./Rollouts";
import { RunTab } from "./RunTab";
import { TraceTab } from "./TraceView";
import type { EvalReport } from "./types";
import { VariantsTab } from "./Variants";
import { RLOutline } from "./RLOutline";

type Tab = "structure" | "env" | "learner" | "run" | "rollouts" | "buffer" | "trace" | "eval" | "variants";
const TABS: [Tab, string][] = [["structure", "Structure"], ["env", "Environment"], ["learner", "Learner"], ["run", "Run & curves"], ["rollouts", "Rollouts"], ["buffer", "Replay buffer"], ["trace", "Transition trace"], ["eval", "Evaluation"], ["variants", "Variants"]];
const ACTIVE = ["queued", "preparing", "running", "cancelling"];

/** The reinforcement-learning workspace (VISION 9.9). RL agents are distinct from language-model workflow agents: observation / action / reward semantics live here. */
export function RLWorkspace({ projectId, graph, setGraph, ui, validation, validationPending, validationError, ops, allRuns, reloadRuns, ensureSaved, initialTab }: {
  projectId: string; graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; ui: UiDoc; validation: Validation | null; allRuns: AnyRun[]; reloadRuns: () => void;
  ensureSaved: () => Promise<void>; initialTab?: Tab;
  validationPending: boolean; validationError: string | null; ops: OpInfo[];
}) {
  const [tab, setTab] = useState<Tab>(initialTab ?? "env");
  const [runId, setRunId] = useState<string | null>(null);
  const [tid, setTid] = useState<number | null>(null);
  const [device, setDevice] = useState<"cpu" | "cuda">("cpu");
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const key = useRef<{ sig: string; id: string }>({ sig: "", id: "" });
  const runs = useMemo(() => allRuns.filter(isRLRun), [allRuns]);
  const opsByType = useMemo(() => Object.fromEntries(ops.filter(op=>op.graphKind === "rl").map(op=>[op.type,op])), [ops]);
  useEffect(() => { setRunId(null); setTid(null); }, [projectId]);
  useEffect(() => { if (!runId && runs.length) setRunId(runs[runs.length - 1].id); }, [runs, runId]);
  const run = runs.find((r) => r.id === runId);
  const live = !!run && ACTIVE.includes(run.status);
  const curves = useCurves(runId, live);
  const evalReport = useGet<EvalReport>(runId && run?.status === "completed" ? `/api/rl/runs/${runId}/eval` : null, [run?.status]);
  const errCount = validation?.diagnostics.filter((d) => d.severity === "error").length ?? 0;
  const blocked = !validation?.ok;

  const td3 = graph.nodes.some((n) => n.type === "rl.td3_learner");
  const algorithm = run?.algorithm ?? (td3 ? "TD3" : "DQN");
  const tabs = algorithm === "TD3" ? TABS.filter(([k]) => !["rollouts", "buffer", "trace"].includes(k)) : TABS;  // TD3 records none of these (ADR 0068)
  useEffect(() => { if (!tabs.some(([k]) => k === tab)) setTab("run"); }, [tabs, tab]);
  async function start() {
    setBusy(true); setErr(null);
    try {
      await ensureSaved();
      const body = { projectId, config: td3 ? { seed, device } : { seed } };
      const sig = JSON.stringify(body) + validation?.graphHash;
      if (key.current.sig !== sig) key.current = { sig, id: uid() };
      const r = await api.post<{ runId: string }>("/api/runs", body, { "Idempotency-Key": key.current.id });
      key.current = { sig: "", id: "" };
      setRunId(r.runId); reloadRuns(); setTab("run");
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  async function cancel() {
    if (!runId) return;
    try { await api.post(`/api/runs/${runId}/cancel`, {}); reloadRuns(); } catch (e) { setErr(errorText(e)); }
  }
  const openTrace = (t: number) => { setTid(t); setTab("trace"); };
  return (
    <div className="aworkspace fullws rlworkspace">
      <div className="tabs atabs" role="tablist" aria-label="rl workspace">
        {tabs.map(([k, l]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}{k === "env" && errCount ? ` (${errCount} errors)` : ""}</button>)}
      </div>
      <div className="rlbar">
        <span className="badge kind" title="RL agents are not language-model agents">RL · {td3 ? "TD3" : "DQN"}</span>
        {td3 && <label className="small" title="CUDA needs a usable NVIDIA GPU on the worker machine; otherwise the run fails with E_DEVICE_UNAVAILABLE">device <select aria-label="rl device" value={device} onChange={(e) => setDevice(e.target.value as "cpu" | "cuda")}><option value="cpu">cpu</option><option value="cuda">cuda</option></select></label>}
        <label className="small">seed <Num label="run seed" integer value={seed} onChange={(n) => setSeed(n ?? 0)} /></label>
        <button className="primary" disabled={busy || blocked || live} onClick={start} title={blocked ? "Fix the graph errors first" : "Save the project and train in a worker process"}>Run</button>
        <button disabled={!live || run?.status === "cancelling"} onClick={cancel}>Cancel</button>
        <label className="small">run <select aria-label="rl run" value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)}>
          <option value="">(none)</option>{[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}</select></label>
        {run && <span className={`status ${run.status}`}>{run.status}</span>}
        {run?.error && <span className="error small">{run.error}</span>}
        {err && <span className="error small pre">{err}</span>}
        {run && validation && run.graphHash !== validation.graphHash && <span className="badge old" title="the draft graph differs from the graph this run used">older graph</span>}
      </div>
      {ui.description && <div className={`notice-inline ${ui.synthetic ? "synthetic" : ""}`}>{ui.synthetic && <b>Synthetic data. </b>}{ui.description}</div>}
      <div className="atabbody">
        {tab === "structure" && <RLOutline key={`rl-outline:${projectId}`} graph={graph} ops={opsByType} validation={validation} pending={validationPending} error={validationError} onOpenPanel={setTab} />}
        {tab === "env" && <EnvPanel graph={graph} setGraph={setGraph} validation={validation} ui={ui} />}
        {tab === "learner" && <LearnerPanel graph={graph} setGraph={setGraph} validation={validation} />}
        {tab === "run" && <RunTab algorithm={algorithm} curves={curves.data} status={run?.status} summary={run ? { envSteps: run.envSteps, updates: run.updates, totalSteps: run.totalSteps } : undefined} />}
        {tab === "rollouts" && <RolloutsTab runId={runId} captured={curves.data?.captured ?? []} setup={curves.data?.setup ?? null} evalReport={evalReport.data} onTrace={openTrace} />}
        {tab === "buffer" && <BufferTab runId={runId} onTrace={openTrace} />}
        {tab === "trace" && <TraceTab runId={runId} tid={tid} setTid={setTid} />}
        {tab === "eval" && <EvalTab runId={runId} status={run?.status} />}
        {tab === "variants" && <VariantsTab projectId={projectId} graph={graph} validation={validation} ensureSaved={ensureSaved} onOpenRun={(id) => { setRunId(id); reloadRuns(); setTab("run"); }} />}
      </div>
      {curves.error && <div className="error small pad">{curves.error}</div>}
    </div>
  );
}
