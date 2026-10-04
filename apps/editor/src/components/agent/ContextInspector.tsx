import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { usePolling } from "../../hooks";
import { ErrorLine, FixtureBadge, SOURCE_LABEL, fmtMs, j, sourceText, useAction } from "./common";
import type { AgentRunSummary, CallContext, CtxMessage, ExcludedItem, Segment, TraceRecord } from "./types";

interface CallRow { callId: string; node: string; purpose: string; attempt: number; provider: string; model: string; fixture: boolean; latencyMs: number; usage: { inputTokens: number | null; outputTokens: number | null; source: string }; tokensEstimate: number; failed: boolean; code?: string; message?: string }

const KIND_CLASS: Record<string, string> = { prompt_template: "k-template", memory_record: "k-memory", retrieved_chunk: "k-chunk", tool_result: "k-tool", conversation_message: "k-conv" };

export function TraceRecordView({ trace }: { trace: TraceRecord }) {
  return (
    <div className="tracerec" aria-label={`trace of ${trace.recordId}`}>
      <h4>Trace of record <code>{trace.recordId}</code> — {trace.stored ? "stored" : "not stored (deleted or never written)"}</h4>
      {trace.storedRecord && <div className="small muted">“{trace.storedRecord.text.slice(0, 140)}” · {trace.storedRecord.namespace}/{trace.storedRecord.scope} · {trace.storedRecord.kind}{trace.storedRecord.generated ? " · generated" : ""}</div>}
      {trace.applications.map((a) => (
        <div key={a.applicationId} className={`verdict ${a.usedInContext ? "inc" : "exc"}`}>
          <b>{a.policyId}</b> <span className="muted small">selection {a.applicationId} at {a.node}</span>
          <div>{a.verdict}</div>
          {a.trail && (
            <ol className="trail">{a.trail.map((t, i) => <li key={i}><span className={`pill ${t.action}`}>{t.action}</span> <b>{t.stage}</b> <span className="muted">({t.op})</span> {t.reason}</li>)}</ol>
          )}
          {a.scores && Object.keys(a.scores).length > 0 && <div className="small mono">scores: {Object.entries(a.scores).map(([k, v]) => `${k}=${v}`).join("  ")} · ≈{a.tokensEstimate} tokens (estimate)</div>}
        </div>
      ))}
      <div className="small muted">{trace.caution}</div>
    </div>
  );
}

function MessageBlock({ m, query, onSource }: { m: CtxMessage; query: string; onSource: (s: Segment["source"]) => void }) {
  const q = query.trim().toLowerCase();
  return (
    <div className={`cmsg role-${m.role}`}>
      <div className="cmeta"><span className="pill">{m.role}</span> <span className="muted small">message {m.index + 1} · ≈{m.tokensEstimate} tokens (estimate)</span></div>
      <div className="cbody">
        {m.segments.map((s, i) => {
          const text = m.content.slice(s.start, s.end);
          const hit = q && text.toLowerCase().includes(q);
          return (
            <span key={i} className={`seg ${KIND_CLASS[s.source.kind] ?? "k-other"} ${hit ? "hit" : ""}`} title={sourceText(s.source)} onClick={() => onSource(s.source)} role="button" tabIndex={0}
              aria-label={`segment from ${SOURCE_LABEL[s.source.kind] ?? s.source.kind}`}>
              <span className="segtag">{SOURCE_LABEL[s.source.kind] ?? s.source.kind}{s.source.recordId ? ` ${s.source.recordId}` : s.source.chunkId ? ` ${s.source.chunkId}` : ""} · ≈{s.tokensEstimate}t</span>
              {text}
            </span>
          );
        })}
      </div>
    </div>
  );
}

export function ContextView({ ctx, onSelectNode, onTrace, traceResult, openMemoryRecord }: {
  ctx: CallContext; onSelectNode: (id: string) => void; onTrace: (recordId: string) => void; traceResult: TraceRecord | null; openMemoryRecord?: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [raw, setRaw] = useState(false);
  const [src, setSrc] = useState<Segment["source"] | null>(null);
  const q = query.trim().toLowerCase();
  const found = q ? ctx.messages.flatMap((m) => m.segments.filter((s) => m.content.slice(s.start, s.end).toLowerCase().includes(q)).map((s) => ({ m, s }))) : [];
  const exclHits = q ? ctx.excluded.filter((e) => e.text.toLowerCase().includes(q)) : [];
  const tk = ctx.tokens;
  const limitPct = tk.modelLimit ? Math.min(100, ((tk.providerInput ?? tk.estimateTotal) / tk.modelLimit) * 100) : null;
  return (
    <div className="ctxview">
      <div className="ctxhead">
        <b>{ctx.provider}/{ctx.model}</b> <FixtureBadge fixture={ctx.fixture} /> <span className="muted small">node {ctx.node} · {ctx.purpose}{ctx.attempt > 1 ? ` attempt ${ctx.attempt}` : ""} · {fmtMs(ctx.latencyMs)} · run {ctx.runId.slice(0, 8)} · graph {ctx.provenance.graphHash.slice(0, 8)}</span>
        <button className="link" onClick={() => onSelectNode(ctx.node)}>show node</button>
      </div>
      <div className="tokacct" aria-label="token accounting">
        <div><b>Tokens</b>: request ≈{tk.estimateTotal} <i>(estimate: {tk.estimateBasis})</i>{" · "}
          {tk.providerReported ? <>provider-reported: <b>{tk.providerInput}</b> in / <b>{tk.providerOutput}</b> out</> : <i>the provider reported no token counts</i>}</div>
        <div>Model limit: {tk.modelLimit ? <>{tk.modelLimit.toLocaleString()} <span className="muted">({tk.modelLimitSource})</span></> : <i>unknown ({tk.modelLimitSource})</i>} · reserved output allowance: {tk.reservedOutput ?? "none set"}</div>
        {limitPct != null && <div className="meter"><div style={{ width: `${Math.max(1, limitPct)}%` }} /></div>}
        <div className="small muted">Resolved settings: {j(ctx.resolved, 200)}{Object.keys(ctx.ignored).length ? ` · ignored: ${Object.entries(ctx.ignored).map(([k, v]) => `${k} (${v})`).join("; ")}` : ""}</div>
      </div>
      <div className="searchbar"><input aria-label="search this call's context" placeholder="Search what this call received (e.g. a constraint)…" value={query} onChange={(e) => setQuery(e.target.value)} />
        {q && <span className={found.length ? "ok" : "error"}> {found.length ? `found in ${found.length} segment(s) of the request that was sent` : "NOT found in the request that was sent"}{exclHits.length ? ` — but ${exclHits.length} stored record(s) containing it were excluded (see below)` : ""}</span>}</div>
      <label className="small"><input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} /> show the provider-visible request only (role + content, no application metadata)</label>
      <h4>Included in the request, in order</h4>
      {raw ? <pre className="mono" aria-label="provider request">{JSON.stringify(ctx.providerRequest, null, 1)}</pre>
        : ctx.messages.map((m) => <MessageBlock key={m.index} m={m} query={query} onSource={(s) => setSrc(s)} />)}
      {src && (
        <div className="srcdetail" aria-label="segment source"><b>Source</b>: {sourceText(src)} <button className="link" onClick={() => setSrc(null)}>close</button>
          <div className="small mono">{j(src, 600)}</div>
          {src.node && <button className="link" onClick={() => onSelectNode(src.node)}>open node {src.node}</button>}{" "}
          {src.kind === "memory_record" && <><button className="link" onClick={() => onTrace(src.recordId)}>trace this record</button>{" "}{openMemoryRecord && <button className="link" onClick={() => openMemoryRecord(src.recordId)}>open in memory browser</button>}</>}
        </div>
      )}
      <h4>Stored or retrieved, but NOT in this request ({ctx.excluded.length})</h4>
      {ctx.excluded.length === 0 ? <div className="muted small">Nothing was excluded from the selections that fed this call.</div> : (
        <table className="excl"><thead><tr><th>item</th><th>kind</th><th>excluded at</th><th>reason</th><th>scores</th><th>≈tokens</th><th /></tr></thead>
          <tbody>{ctx.excluded.map((e: ExcludedItem) => (
            <tr key={`${e.kind}:${e.id}`} className={q && e.text.toLowerCase().includes(q) ? "hit" : ""}>
              <td title={e.text}><code>{e.id}</code><div className="small muted">{e.text.slice(0, 70)}</div></td><td>{e.kind === "memory_record" ? "memory record" : "retrieved chunk"}</td>
              <td><b>{e.stage}</b> <span className="muted">({e.op})</span></td><td>{e.reason}</td><td className="small mono">{Object.entries(e.scores).map(([k, v]) => `${k}=${v}`).join(" ")}</td><td className="num">{e.tokensEstimate}</td>
              <td>{e.kind === "memory_record" && <button className="small" onClick={() => onTrace(e.id)}>trace</button>}</td></tr>))}</tbody></table>
      )}
      {traceResult && <TraceRecordView trace={traceResult} />}
      <h4>Response</h4><pre className="resp">{ctx.response}</pre>
      <div className="small muted">{ctx.boundary}</div>
    </div>
  );
}

export function ContextTab({ runs, runId, setRunId, callId, setCallId, onSelectNode, openMemoryRecord }: {
  runs: AgentRunSummary[]; runId: string | null; setRunId: (id: string | null) => void; callId: string | null; setCallId: (id: string | null) => void; onSelectNode: (id: string) => void; openMemoryRecord?: (id: string) => void;
}) {
  const calls = usePolling<{ calls: CallRow[] }>(runId ? `/api/agent/runs/${runId}/model-calls` : null, 0, [runs.find((r) => r.id === runId)?.status]);
  const rows = calls.data?.calls ?? [];
  useEffect(() => { if (rows.length && (!callId || !rows.some((c) => c.callId === callId))) setCallId(rows[rows.length - 1].callId); }, [rows, callId, setCallId]);
  const ctx = usePolling<CallContext>(runId && callId && rows.some((c) => c.callId === callId) ? `/api/agent/runs/${runId}/model-calls/${callId}` : null, 0, [callId, rows.length]);
  const [trace, setTrace] = useState<TraceRecord | null>(null);
  const { error, run } = useAction();
  useEffect(() => setTrace(null), [callId]);
  const doTrace = (rid: string) => run(async () => setTrace(await api.get<TraceRecord>(`/api/agent/memory/trace?recordId=${encodeURIComponent(rid)}&runId=${runId}&callId=${callId}`)));
  const fixture = useMemo(() => rows.some((r) => r.fixture), [rows]);
  return (
    <div className="actx">
      <section className="ctxlist">
        <h3>Model calls</h3>
        <label className="small">Run <select aria-label="run" value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)}><option value="">(choose)</option>{[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{r.id.slice(0, 8)} · {r.status} · {r.threadId}</option>)}</select></label>
        {fixture && <div className="warn small">This run used a FIXTURE model; its replies are scripted.</div>}
        {rows.length === 0 && <div className="empty">{runId ? "This run made no model calls." : "Choose a run."}</div>}
        <ol className="calls">{rows.map((c, i) => (
          <li key={c.callId}><button className={`callbtn ${c.callId === callId ? "on" : ""}`} onClick={() => setCallId(c.callId)}>
            #{i + 1} {c.node} <span className="muted small">{c.purpose}{c.attempt > 1 ? ` #${c.attempt}` : ""}</span> <FixtureBadge fixture={c.fixture} />
            <div className="small muted">{c.provider}/{c.model} · {fmtMs(c.latencyMs)} · {c.usage.source === "provider" ? `${c.usage.inputTokens}→${c.usage.outputTokens} tok` : `≈${c.tokensEstimate} tok (est.)`}</div></button></li>
        ))}</ol>
      </section>
      <section className="ctxmain">
        <ErrorLine text={error ?? ctx.error} />
        {ctx.data ? <ContextView ctx={ctx.data} onSelectNode={onSelectNode} onTrace={doTrace} traceResult={trace} openMemoryRecord={openMemoryRecord} /> : <div className="empty pad">Select a model call to see exactly what it received.</div>}
      </section>
    </div>
  );
}
