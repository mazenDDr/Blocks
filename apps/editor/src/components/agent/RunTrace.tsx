import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../../api";
import { usePolling } from "../../hooks";
import type { Graph, UiDoc, Validation } from "../../types";
import { Check, ErrorLine, FixtureBadge, JsonBox, Num, Sel, fmtMs, fmtTime, j, uid, useAction } from "./common";
import { specOf, type AgentRunSummary, type StateField, type Trace, type TraceStep } from "./types";

interface ThreadInfo { threadId: string; projectId: string | null; runs: { runId: string; status: string; createdAt: number }[]; status: string | null; hasCheckpoint: boolean; step: number | null; pendingInterrupt: boolean }

const STOP_TEXT: Record<string, string> = {
  end: "The run reached END.", recursion_limit: "Stopped at the step limit (LangGraph recursion limit): the loop did not reach END within the configured number of steps.",
  "budget:maxModelCalls": "Stopped by the model-call budget.", "budget:maxTokens": "Stopped by the token budget.", "budget:maxSeconds": "Stopped by the time budget.", "budget:maxToolCalls": "Stopped by the tool-call budget.",
};

// ------------------------------------------------------------------------------------------------ interrupt form
function InterruptPanel({ run, reload }: { run: AgentRunSummary; reload: () => void }) {
  const p = run.pendingInterrupt!.payload as Record<string, any>;
  const { busy, error, run: act } = useAction();
  const [text, setText] = useState<string>(typeof p.proposed === "string" ? p.proposed : p.proposal?.text ?? "");
  const [args, setArgs] = useState<Record<string, string>>(p.args ?? {});
  const send = (action: string, value?: unknown) => act(async () => { await api.post(`/api/runs/${run.id}/resume`, { value: value === undefined ? { action } : { action, value } }); reload(); });
  const actions: string[] = p.actions ?? ["approve", "reject"];
  return (
    <div className="interrupt" role="form" aria-label="pending input">
      <h4>Waiting for you — run paused at <b>{run.pendingInterrupt!.node}</b></h4>
      <div className="small muted">Resumable identity: run {run.id} · thread {run.threadId}. The state is in the thread's checkpoint; you can restart the service and resume later.</div>
      {p.type === "approval" && (
        <>
          <p><b>{p.prompt}</b></p>
          <div className="warn">Tool <b>{p.tool}</b> has the external effect <b>{(p.effects ?? []).join(", ")}</b>, bounded to <code>{p.boundedTo}</code>. It will not run before you approve, and it runs at most once.</div>
          <table><tbody>{Object.entries(args).map(([k, v]) => <tr key={k}><td>{k}</td><td><input aria-label={`argument ${k}`} value={v} size={40} onChange={(e) => setArgs({ ...args, [k]: e.target.value })} /></td></tr>)}</tbody></table>
          <button className="primary" disabled={busy} onClick={() => send("approve")}>Approve (run the effect)</button>{" "}
          {actions.includes("edit") && <button disabled={busy} onClick={() => send("edit", args)}>Edit arguments and approve</button>}{" "}
          <button className="danger" disabled={busy} onClick={() => send("reject")}>Reject (no effect)</button>
        </>
      )}
      {p.type === "human_input" && (
        <>
          <p><b>{p.prompt}</b></p>
          {p.proposed != null && (actions.includes("edit") && p.editField ? <textarea className="atext" aria-label="value to review" rows={3} value={text} onChange={(e) => setText(e.target.value)} /> : <pre>{j(p.proposed, 2000)}</pre>)}
          {actions.includes("approve") && <button className="primary" disabled={busy} onClick={() => send("approve")}>Approve</button>}{" "}
          {actions.includes("edit") && p.editField && <button disabled={busy} onClick={() => send("edit", text)}>Save edit and continue</button>}{" "}
          {actions.includes("reject") && <button className="danger" disabled={busy} onClick={() => send("reject")}>Reject</button>}
        </>
      )}
      {p.type === "memory_write_proposal" && (
        <>
          <p><b>{p.prompt}</b></p>
          <div className="small">Scope <code>{p.proposal.scope}</code> · namespace <code>{p.proposal.namespace}</code> · kind {p.proposal.kind}{p.proposal.generated ? " · GENERATED assertion" : ""}{p.proposal.evidence ? ` · evidence: ${p.proposal.evidence}` : ""}</div>
          <textarea className="atext" aria-label="proposed record text" rows={3} value={text} onChange={(e) => setText(e.target.value)} />
          <button className="primary" disabled={busy} onClick={() => send("approve")}>Accept</button>{" "}
          <button disabled={busy} onClick={() => send("edit", text)}>Accept edited text</button>{" "}
          <button className="danger" disabled={busy} onClick={() => send("reject")}>Reject</button>
        </>
      )}
      <ErrorLine text={error} />
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ trace
function Diff({ step }: { step: TraceStep }) {
  const ch = step.changes;
  if (!ch.length) return <div className="small muted">no state change</div>;
  return (
    <table className="adiff"><thead><tr><th>field</th><th>reducer</th><th>before</th><th>after</th></tr></thead>
      <tbody>{ch.map((c) => <tr key={c.field} className={c.changed ? "" : "same"}><td><b>{c.field}</b></td><td>{c.reducer}</td><td className="mono">{j(c.before, 70)}</td><td className="mono">{c.changed ? j(c.after, 140) : <i className="muted">unchanged</i>}</td></tr>)}</tbody></table>
  );
}

function StepCard({ s, onNode, onCall }: { s: TraceStep; onNode: (id: string) => void; onCall: (callId: string) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`tstep ${s.status} ${s.replay ? "replay" : ""}`}>
      <div className="thead" onClick={() => setOpen(!open)}>
        <span className="tnum">{s.step}</span> <button className="link" onClick={(e) => { e.stopPropagation(); onNode(s.node); }}>{s.node}</button>
        <span className="badge">{s.type.replace("agent.", "")}</span>{s.replay && <span className="badge warnb" title="This node ran again from its start after a resume; effects before an interrupt may repeat unless protected">replayed</span>}
        <span className="muted small"> {s.durationMs != null ? fmtMs(s.durationMs) : ""}</span>
        {s.changes.filter((c) => c.changed).length > 0 && <span className="small muted"> · changed {s.changes.filter((c) => c.changed).map((c) => c.field).join(", ")}</span>}
        {s.route && <span className="small"> → <b>{s.route.label}</b> ({s.route.to})</span>}
        <span className="chev">{open ? "▾" : "▸"}</span>
      </div>
      {open && (
        <div className="tbody">
          <h5>State read</h5><div className="small mono">{Object.entries(s.reads).map(([k, v]) => `${k}=${j(v, 60)}`).join("  ") || "—"}</div>
          <h5>State diff</h5><Diff step={s} />
          {s.route && (
            <>
              <h5>Route {s.route.route}: taken <b>{s.route.label}</b> → {s.route.to} ({s.route.via})</h5>
              <table><tbody>{s.route.evaluated.map((e) => <tr key={e.case}><td>{e.result ? "✔" : "✖"}</td><td>{e.label}</td><td className="mono">{e.predicate}</td><td className="mono small">{Object.entries(e.values).map(([k, v]) => `${k}=${j(v, 30)}`).join(", ")}</td><td>→ {e.to}</td></tr>)}</tbody></table>
              <div className="small muted">Cases after the first match are not evaluated.</div>
            </>
          )}
          {s.retrievals.map((r) => (
            <div key={r.retrievalId}><h5>Retrieval · index {r.index} · k={r.k}</h5><div className="small">query: <i>{r.query}</i> · {r.embedding}</div><div className="small muted">{r.scoreInterpretation}</div>
              <table><tbody>{r.included.map((d: any) => <tr key={d.chunk_id}><td>{d.rank}</td><td>{d.chunk_id}</td><td className="num">{d.score}</td><td>included</td></tr>)}{r.excluded.slice(0, 4).map((d: any) => <tr key={d.chunk_id} className="muted"><td>{d.rank}</td><td>{d.chunk_id}</td><td className="num">{d.score}</td><td>excluded: {d.reason}</td></tr>)}</tbody></table></div>
          ))}
          {s.modelCalls.map((m) => (
            <div key={m.callId} className="mcall">
              <h5>Model call <FixtureBadge fixture={m.fixture} /> {m.purpose}{m.attempt > 1 ? ` · attempt ${m.attempt}` : ""} <button className="link" onClick={() => onCall(m.callId)}>open context</button></h5>
              <div className="small">{m.provider}/{m.model} · latency {fmtMs(m.latencyMs)} · tokens: {m.usage.source === "provider" ? <>provider-reported in {m.usage.inputTokens} / out {m.usage.outputTokens}</> : <i>not reported by the provider (request estimated at ≈{m.tokensEstimate} tokens)</i>} · cost: {m.cost.amount == null ? "unknown" : `${m.cost.amount} (${m.cost.basis})`}</div>
              {Object.keys(m.ignored).length > 0 && <div className="small muted">ignored settings: {Object.entries(m.ignored).map(([k, v]) => `${k} (${v})`).join("; ")}</div>}
              {m.validationErrors && m.validationErrors.length > 0 && <div className="errbadge">validation failed: {m.validationErrors.join("; ")}</div>}
              <pre className="resp">{m.response}</pre>
            </div>
          ))}
          {s.toolCalls.map((t) => (
            <div key={t.callId}><h5>Tool call · {t.tool} · {t.status}</h5><div className="small mono">{j(t.args, 200)} → {j(t.result, 200)}</div>
              <div className="small">effects: {t.effects.length ? t.effects.join(", ") : "none"}{t.approval ? ` · approval: ${t.approval.action}` : ""}{t.replaySkipped ? " · REPLAY: the effect was already performed and was not repeated" : ""}</div></div>
          ))}
          {s.events.filter((e) => ["structured_attempt", "memory_selection", "memory_write", "citations_checked", "index_ready", "embedding"].includes(e.type)).map((e) => (
            <div key={e.seq} className="small"><b>{e.type}</b> {e.type === "structured_attempt" ? `attempt ${e.attempt}: ${e.valid ? "valid" : "rejected: " + (e.errors ?? []).join("; ")}` : e.type === "memory_selection" ? `${e.selected} of ${e.universe} records selected by ${e.policy} (${e.excluded} excluded)` : e.type === "memory_write" ? `${e.status} ${e.recordId ?? ""}` : e.type === "citations_checked" ? `${e.count} cited, ${e.invalid} invalid` : e.type === "index_ready" ? `${e.index} ${e.action}: ${e.chunks} chunks (${e.embeddingsReused} reused / ${e.embeddingsComputed} computed embeddings)` : `${e.identity} d=${e.dimension}`}</div>
          ))}
        </div>
      )}
    </li>
  );
}

function Totals({ t, graph }: { t: Trace; graph: Graph }) {
  const lim = specOf(graph).limits;
  const steps = t.steps.length;
  const bar = (label: string, used: number, max?: number | null) => max ? <span className={`bud ${used >= max ? "full" : ""}`}>{label} {used}/{max}</span> : <span className="bud">{label} {used}</span>;
  return (
    <div className="totals">
      {bar("steps", steps, lim.maxSteps)} {bar("model calls", t.totals.modelCalls, lim.maxModelCalls)} {bar("tool calls", t.totals.toolCalls, lim.maxToolCalls)}
      <span className="bud">model latency {fmtMs(t.totals.latencyMs)}</span>
      <span className="bud">tokens: {t.totals.providerReportedCalls ? `${t.totals.inputTokens} in / ${t.totals.outputTokens} out (provider-reported, ${t.totals.providerReportedCalls} calls)` : "not reported"}{t.totals.unreportedCalls ? ` · ${t.totals.unreportedCalls} call(s) without usage` : ""}</span>
      {t.totals.fixtureCalls > 0 && <span className="badge fixture">{t.totals.fixtureCalls} FIXTURE call(s): not real model output</span>}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ thread inspection
function ThreadPanel({ threadId, reload, onFork }: { threadId: string; reload: () => void; onFork: (id: string) => void }) {
  const [isOpen, setIsOpen] = useState(false);  // fetched only when expanded (a thread that has not checkpointed yet has no state)
  const hist = usePolling<{ checkpoints: { checkpointId: string; step: number; source: string; ts: string; writes: string[] | null }[] }>(isOpen ? `/api/agent/threads/${encodeURIComponent(threadId)}/history` : null, 0);
  const [cp, setCp] = useState<string>("");
  const state = usePolling<{ values: Record<string, unknown>; step: number; checkpointId: string; pendingInterrupt: boolean }>(isOpen ? `/api/agent/threads/${encodeURIComponent(threadId)}/state${cp ? `?checkpointId=${cp}` : ""}` : null, 0, [cp]);
  const [field, setField] = useState("");
  const [val, setVal] = useState<unknown>(null);
  const { busy, error, run } = useAction();
  const keys = Object.keys(state.data?.values ?? {});
  return (
    <details className="threadpanel" onToggle={(e) => setIsOpen((e.currentTarget as HTMLDetailsElement).open)}><summary>Thread {threadId}: checkpoints and state{hist.data ? ` (${hist.data.checkpoints.length} checkpoints)` : ""}</summary>
      <div className="small muted">Rewinding the viewer reads captured state; it does not undo external actions. Editing creates a new branch on a new thread.</div>
      <Sel label="checkpoint" value={cp || "latest"} options={["latest", ...(hist.data?.checkpoints ?? []).map((c) => c.checkpointId)]} onChange={(v) => setCp(v === "latest" ? "" : v)} />
      {(hist.data?.checkpoints ?? []).filter((c) => !cp || c.checkpointId === cp).slice(0, 1).map((c) => <span key={c.checkpointId} className="small"> step {c.step} · {c.source}{c.writes ? ` · wrote ${c.writes.join(", ")}` : ""}</span>)}
      <pre className="mono">{JSON.stringify(state.data?.values ?? {}, null, 1).slice(0, 3000)}</pre>
      <div className="rcasehead">Fork with an edit: <Sel label="field to edit" value={field || (keys[0] ?? "")} options={keys.length ? keys : [""]} onChange={(f) => { setField(f); setVal(state.data?.values[f] ?? null); }} />
        <JsonBox label="edited value" rows={2} value={val ?? state.data?.values[field || keys[0]] ?? null} onChange={setVal} />
        <button disabled={busy || !keys.length} onClick={() => run(async () => {
          const f = field || keys[0];
          const r = await api.post<{ threadId: string }>(`/api/agent/threads/${encodeURIComponent(threadId)}/fork`, { checkpointId: cp || undefined, edits: { [f]: val ?? state.data?.values[f] } });
          reload(); onFork(r.threadId);
        })}>Fork to a new thread</button></div>
      <ErrorLine text={error} />
    </details>
  );
}

// ------------------------------------------------------------------------------------------------ the tab
export function AgentRunTab({ projectId, graph, ui, validation, runs, reloadRuns, runId, setRunId, selectNode, openCall, setMessage }: {
  projectId: string; graph: Graph; ui: UiDoc; validation: Validation | null; runs: AgentRunSummary[]; reloadRuns: () => void; runId: string | null; setRunId: (id: string | null) => void;
  selectNode: (id: string) => void; openCall: (runId: string, callId: string) => void; setMessage: (m: string) => void;
}) {
  const spec = specOf(graph);
  const threads = usePolling<{ threads: ThreadInfo[] }>(`/api/agent/threads?project=${encodeURIComponent(projectId)}`, 3000);
  const run = runs.find((r) => r.id === runId);
  const live = !!run && ["queued", "preparing", "running", "cancelling"].includes(run.status);
  const trace = usePolling<Trace>(runId ? `/api/agent/runs/${runId}/trace` : null, live ? 1000 : 0, [run?.status, run?.maxSeq]);
  const inputNames = useMemo(() => validation?.agent?.inputFields ?? [], [validation]);
  const inputFields: StateField[] = spec.state.filter((f) => inputNames.includes(f.name) && ["text", "integer", "number", "boolean"].includes(f.type));
  const [values, setValues] = useState<Record<string, unknown>>({});
  useEffect(() => {
    const d: Record<string, unknown> = {};
    for (const f of inputFields) d[f.name] = (ui.defaultInput as Record<string, unknown> | undefined)?.[f.name] ?? f.default ?? (f.type === "text" ? "" : f.type === "boolean" ? false : 0);
    setValues(d);
  }, [projectId, JSON.stringify(inputFields.map((f) => f.name)), ui.defaultInput]); // eslint-disable-line react-hooks/exhaustive-deps
  const [thread, setThread] = useState<string>("new");
  const [custom, setCustom] = useState("");
  const { busy, error, run: act } = useAction();
  const threadId = thread === "new" ? custom.trim() || undefined : thread;
  const errs = validation?.diagnostics.filter((d) => d.severity === "error") ?? [];

  const start = () => act(async () => {
    const r = await api.post<{ runId: string; threadId: string }>("/api/runs", { graph, config: { project_id: projectId, thread_id: threadId, input: values } }, { "Idempotency-Key": uid() });
    reloadRuns(); threads.reload(); setRunId(r.runId); setThread(r.threadId); setCustom("");
  });
  const rerun = () => act(async () => { const r = await api.post<{ runId: string; threadId: string }>(`/api/runs/${runId}/rerun`, {}); reloadRuns(); setRunId(r.runId); setThread(r.threadId); setMessage("Rerun started: new model calls on a new thread."); });
  const cancel = () => act(async () => { await api.post(`/api/runs/${runId}/cancel`, {}); reloadRuns(); });
  const t = trace.data;
  const fixture = validation?.agent?.usesFixtureModel;
  return (
    <div className="arun">
      <section className="runctl" aria-label="start a run">
        <h3>Run</h3>
        {fixture && <div className="warn"><b>This graph contains a FIXTURE model.</b> Its replies are scripted for control-flow tests and are not real model output.</div>}
        {errs.length > 0 && <div className="errbadge">{errs.length} validation error(s): fix them before running ({errs[0].code}: {errs[0].message})</div>}
        <div className="inputs">
          {inputFields.map((f) => (
            <label key={f.name} className="arow"><span className="alabel">{f.name}<small className="muted"> ({f.type})</small></span>
              <span className="actl">{f.type === "text" ? <textarea className="atext" aria-label={`input ${f.name}`} rows={2} value={String(values[f.name] ?? "")} onChange={(e) => setValues({ ...values, [f.name]: e.target.value })} />
                : f.type === "boolean" ? <Check label={`input ${f.name}`} value={values[f.name] === true} onChange={(b) => setValues({ ...values, [f.name]: b })} />
                  : <Num label={`input ${f.name}`} value={Number(values[f.name] ?? 0)} integer={f.type === "integer"} onChange={(n) => setValues({ ...values, [f.name]: n ?? 0 })} />}</span></label>
          ))}
          {inputFields.length === 0 && <div className="muted small">This graph takes no input fields; it runs from its state defaults.</div>}
        </div>
        <div className="rcasehead">
          Thread <select aria-label="thread" value={thread} onChange={(e) => setThread(e.target.value)}>
            <option value="new">new thread</option>
            {(threads.data?.threads ?? []).map((th) => <option key={th.threadId} value={th.threadId}>{th.threadId} · {th.status}{th.pendingInterrupt ? " · waiting" : ""} · step {th.step ?? "?"}</option>)}
          </select>
          {thread === "new" && <input aria-label="new thread id" placeholder="id (optional)" value={custom} size={14} onChange={(e) => setCustom(e.target.value)} />}
          <button className="primary" disabled={busy || errs.length > 0} onClick={start}>Run graph</button>
        </div>
        <div className="hint">A run on an existing thread continues from its checkpoint: thread-scoped fields (conversation) persist, turn-scoped fields reset. {projectId ? `Project: ${projectId}.` : ""}</div>
        <ErrorLine text={error} />
        <h4>Runs</h4>
        <div className="runlist">{runs.length === 0 && <span className="muted">No agent runs yet.</span>}
          {[...runs].reverse().map((r) => (
            <button key={r.id} className={`runchip ${r.id === runId ? "on" : ""}`} onClick={() => setRunId(r.id)}>
              {r.id.slice(0, 8)} · <span className={`status ${r.status}`}>{r.status}</span> · {r.threadId}{r.fixtureCalls ? " · FIXTURE" : ""}{r.rerunOf ? " · rerun" : ""}
            </button>
          ))}</div>
      </section>
      {run && (
        <section className="rundetail" aria-label="run detail">
          <h3>Run {run.id} <span className={`status ${run.status}`}>{run.status}</span> <small className="muted">thread {run.threadId} · graph {run.graphHash.slice(0, 8)} · started {fmtTime(run.createdAt)}</small></h3>
          {run.error && <div className="errbadge pre">{run.error}</div>}
          {run.status === "completed" && run.stoppedBy && <div className={run.stoppedBy === "end" ? "notrec" : "warn"}>{STOP_TEXT[run.stoppedBy] ?? run.stoppedBy}{t?.budget ? ` (${t.budget.kind}: ${t.budget.used} of ${t.budget.limit})` : ""}</div>}
          {run.status === "paused" && run.pendingInterrupt && <InterruptPanel run={run} reload={() => { reloadRuns(); trace.reload(); }} />}
          <div className="rcasehead">
            {live && <button onClick={cancel}>Cancel</button>}
            <button onClick={rerun} disabled={busy} title="Makes new model calls on a new thread. A stochastic or hosted model is not guaranteed to repeat its earlier response.">Rerun (new model calls)</button>
            <span className="small muted">The timeline below shows RECORDED events; viewing it makes no model call.</span>
          </div>
          <ThreadPanel threadId={run.threadId} reload={threads.reload} onFork={(id) => { setThread(id); setMessage(`Forked to thread ${id}. Run the graph on it to continue from the edited state.`); }} />
          {t && <Totals t={t} graph={graph} />}
          {t?.failures.map((f) => <div key={f.seq} className="errbadge">{f.type} at {f.node ?? "run"}: {f.message}</div>)}
          <h4>Trace timeline <small className="muted">provenance: {t?.provenance.source}, run {run.id}, graph {run.graphHash.slice(0, 8)}</small></h4>
          <ol className="timeline">{t?.steps.map((s) => <StepCard key={s.seq} s={s} onNode={selectNode} onCall={(c) => openCall(run.id, c)} />)}</ol>
          {t && t.steps.length === 0 && <div className="empty">No steps recorded yet.</div>}
          {trace.error && <div className="error">{trace.error}</div>}
        </section>
      )}
    </div>
  );
}
export { errorText };
