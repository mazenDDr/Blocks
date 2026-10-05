import { useEffect, useMemo, useState } from "react";
import type { Graph, OpInfo, UiDoc, Validation } from "../../types";
import { isAgentRun } from "../../types";
import type { AnyRun } from "../../types";
import { AgentCanvas } from "./AgentCanvas";
import { ContextTab } from "./ContextInspector";
import { IndexesTab } from "./IndexesPanel";
import { MemoryTab } from "./MemoryPanel";
import { AgentRunTab } from "./RunTrace";
import { StateTab } from "./StatePanel";
import { usePolling } from "../../hooks";
import type { AgentRunSummary, Trace } from "./types";

type Tab = "canvas" | "state" | "run" | "context" | "memory" | "indexes";

/** The agent graph workspace (VISION 4, 12): canvas, state schema and routes, run + trace, context inspector, memory, indexes. */
export function AgentWorkspace({ projectId, graph, setGraph, ui, setUi, validation, ops, allRuns, reloadRuns, setMessage, requestedRunId }: {
  projectId: string; graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; ui: UiDoc; setUi: (f: (u: UiDoc) => UiDoc, dragging?: boolean) => void; validation: Validation | null; ops: OpInfo[];
  allRuns: AnyRun[]; reloadRuns: () => void; setMessage: (m: string) => void; requestedRunId?: string | null;
}) {
  const [tab, setTab] = useState<Tab>("canvas");
  const [selNode, setSelNode] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [callId, setCallId] = useState<string | null>(null);
  const [focusRecord, setFocusRecord] = useState<string | null>(null);
  const runs = useMemo(() => allRuns.filter(isAgentRun) as AgentRunSummary[], [allRuns]);
  const agentOps = useMemo(() => ops.filter((o) => o.graphKind === "agent"), [ops]);
  useEffect(() => { setRunId(null); setCallId(null); setSelNode(null); }, [projectId]);
  useEffect(() => {
    if (requestedRunId) { setRunId(requestedRunId); setCallId(null); setTab("run"); }
  }, [requestedRunId]);
  useEffect(() => { if (!runId && !requestedRunId && runs.length) setRunId(runs[runs.length - 1].id); }, [runs, runId, requestedRunId]);
  const run = runs.find((r) => r.id === runId);
  const live = !!run && ["queued", "preparing", "running", "paused", "cancelling"].includes(run.status);
  const trace = usePolling<Trace>(runId ? `/api/agent/runs/${runId}/trace` : null, live ? 1200 : 0, [run?.status, run?.maxSeq]);
  const writes = useMemo(() => {
    const w: Record<string, string[]> = {};
    for (const [id, v] of Object.entries(validation?.nodes ?? {})) for (const f of ((v as unknown as { writes?: string[] }).writes ?? [])) (w[f] ??= []).push(id);
    return w;
  }, [validation]);
  const errCount = validation?.diagnostics.filter((d) => d.severity === "error").length ?? 0;
  const tabs: [Tab, string][] = [["canvas", "Canvas"], ["state", "State & routes"], ["run", "Run & trace"], ["context", "Context inspector"], ["memory", "Memory"], ["indexes", "Indexes"]];
  const runValues = (nodeId: string) => {
    const step = trace.data ? [...trace.data.steps].reverse().find((s) => s.node === nodeId) : undefined;
    return step?.reads as Record<string, any> | undefined;
  };
  return (
    <div className="aworkspace fullws">
      <div className="tabs atabs" role="tablist" aria-label="agent workspace">
        {tabs.map(([k, l]) => <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}{k === "run" && run?.status === "paused" ? " ●" : ""}{k === "canvas" && errCount ? ` (${errCount} errors)` : ""}</button>)}
        {run && <span className="small muted atabrun">selected run {run.id.slice(0, 8)} · {run.status} · thread {run.threadId}</span>}
      </div>
      {ui.description && <div className={`notice-inline ${ui.synthetic ? "synthetic" : ""}`}>{ui.synthetic && <b>Synthetic data. </b>}{ui.description}</div>}
      <div className="atabbody">
        {tab === "canvas" && <AgentCanvas graph={graph} setGraph={setGraph} ui={ui} setUi={setUi} validation={validation} ops={agentOps} selNode={selNode} setSelNode={setSelNode} trace={trace.data} setMessage={setMessage} runValues={runValues} />}
        {tab === "state" && <div className="scroll pad"><StateTab graph={graph} setGraph={setGraph} analysis={validation?.agent} writes={writes} /></div>}
        {tab === "run" && <div className="scroll pad"><AgentRunTab projectId={projectId} graph={graph} ui={ui} validation={validation} runs={runs} reloadRuns={reloadRuns} runId={runId} setRunId={setRunId}
          selectNode={(id) => { setSelNode(id); setTab("canvas"); }} openCall={(r, c) => { setRunId(r); setCallId(c); setTab("context"); }} setMessage={setMessage} /></div>}
        {tab === "context" && <ContextTab runs={runs} runId={runId} setRunId={setRunId} callId={callId} setCallId={setCallId} onSelectNode={(id) => { setSelNode(id); setTab("canvas"); }}
          openMemoryRecord={(id) => { setFocusRecord(id); setTab("memory"); }} />}
        {tab === "memory" && <div className="scroll pad"><MemoryTab graph={graph} setGraph={setGraph} ui={ui} projectId={projectId} runs={runs} runId={runId} setRunId={setRunId} callId={callId} setCallId={setCallId}
          focusRecord={focusRecord} onRerun={(id) => { reloadRuns(); setRunId(id); setTab("run"); }} /></div>}
        {tab === "indexes" && <div className="scroll pad"><IndexesTab graph={graph} setGraph={setGraph} /></div>}
      </div>
    </div>
  );
}
