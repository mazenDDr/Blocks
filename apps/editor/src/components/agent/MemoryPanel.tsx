import { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import { usePolling } from "../../hooks";
import type { Graph, UiDoc } from "../../types";
import { Check, ErrorLine, JsonBox, Num, Row, Sel, fmtTime, j, uid, useAction, useCatalog } from "./common";
import { ModelForm } from "./NodeForms";
import { PredicateBuilder, renderPredicate, type PathInfo } from "./PredicateBuilder";
import { specOf, type AgentRunSummary, type AppRecord, type AppStage, type MemRecord, type Policy, type PolicyStage, type PreviewResult } from "./types";

// ------------------------------------------------------------------------------------------------ pipeline view (recorded or previewed)
export function ApplicationView({ stages, records, final }: { stages: AppStage[]; records: Record<string, AppRecord>; final: string[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="appview" aria-label="policy decisions">
      <ol className="funnel">{stages.map((s) => (
        <li key={s.id}>
          <button className="stagebtn" onClick={() => setOpen(open === s.id ? null : s.id)}><b>{s.id}</b> <span className="muted">({s.op})</span> {s.in} → <b>{s.out}</b>{s.skipped && <span className="badge warnb">skipped: {s.skipped}</span>}</button>
          <div className="small muted">{s.note}</div>
          {open === s.id && (
            <table className="decisions"><tbody>{s.decisions.map((d, i) => (
              <tr key={i} className={d.action}><td><code>{d.id}</code><div className="small muted">{(records[d.id]?.text ?? "").slice(0, 60)}</div></td><td><span className={`pill ${d.action}`}>{d.action}</span></td><td>{d.reason}</td></tr>
            ))}</tbody></table>
          )}
        </li>
      ))}</ol>
      <div className="small"><b>Result</b> ({final.length} records): {final.map((id) => <code key={id} className="chip">{id}</code>)}</div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ stores
function RecordForm({ rec, onSaved, onCancel }: { rec: Partial<MemRecord>; onSaved: () => void; onCancel: () => void }) {
  const cat = useCatalog();
  const [r, setR] = useState<Partial<MemRecord>>({ namespace: "default", scope: "global", kind: "semantic", importance: 0.5, metadata: {}, generated: false, ...rec });
  const { busy, error, run } = useAction();
  const save = () => run(async () => {
    const body = { text: r.text ?? "", namespace: r.namespace, scope: r.scope, kind: r.kind, importance: r.importance, metadata: r.metadata ?? {}, generated: !!r.generated, expires_at: r.expires_at ?? null, created_at: r.created_at };
    if (rec.id) await api.put(`/api/agent/memory/records/${encodeURIComponent(rec.id)}`, body); else await api.post("/api/agent/memory/records", body);
    onSaved();
  });
  return (
    <div className="recform" role="form" aria-label="memory record">
      <h4>{rec.id ? `Edit ${rec.id}` : "New long-term record"}</h4>
      <Row label="Text"><textarea className="atext" aria-label="record text" rows={3} value={r.text ?? ""} onChange={(e) => setR({ ...r, text: e.target.value })} /></Row>
      <Row label="Namespace"><input aria-label="record namespace" value={r.namespace} onChange={(e) => setR({ ...r, namespace: e.target.value })} /></Row>
      <Row label="Scope" hint="e.g. user:alice, project:p1, global"><input aria-label="record scope" value={r.scope} onChange={(e) => setR({ ...r, scope: e.target.value })} /></Row>
      <Row label="Kind"><Sel label="record kind" value={r.kind ?? "semantic"} options={cat?.memoryKinds ?? ["semantic"]} onChange={(k) => setR({ ...r, kind: k })} /></Row>
      <Row label="Importance (0-1)"><Num label="record importance" min={0} max={1} value={r.importance ?? 0.5} onChange={(n) => setR({ ...r, importance: n ?? 0.5 })} /></Row>
      <Row label="Generated"><Check label="record generated" value={!!r.generated} onChange={(b) => setR({ ...r, generated: b })} /></Row>
      <Row label="Metadata"><JsonBox label="record metadata" rows={2} value={r.metadata ?? {}} onChange={(m) => setR({ ...r, metadata: m })} /></Row>
      <button className="primary" disabled={busy || !(r.text ?? "").trim()} onClick={save}>Save (audited write)</button> <button onClick={onCancel}>Cancel</button>
      <ErrorLine text={error} />
    </div>
  );
}

export function StoresView({ graph, ui, runs, focusRecord }: { graph: Graph; ui: UiDoc; runs: AgentRunSummary[]; focusRecord: string | null }) {
  const [ns, setNs] = useState(""); const [scope, setScope] = useState(""); const [kind, setKind] = useState(""); const [deleted, setDeleted] = useState(false);
  const q = new URLSearchParams(); if (ns) q.set("namespace", ns); if (scope) q.set("scope", scope); if (kind) q.set("kind", kind); if (deleted) q.set("deleted", "true");
  const recs = usePolling<{ records: MemRecord[]; namespaces: string[]; scopes: string[]; path: string; note: string }>(`/api/agent/memory/records?${q}`, 0, [ns, scope, kind, deleted]);
  const [editing, setEditing] = useState<Partial<MemRecord> | null>(null);
  const [histId, setHistId] = useState<string | null>(focusRecord);
  useEffect(() => { if (focusRecord) setHistId(focusRecord); }, [focusRecord]);
  const hist = usePolling<{ writes: any[]; retainedSnapshots: any[] }>(histId ? `/api/agent/memory/records/${encodeURIComponent(histId)}/history` : null, 0, [histId]);
  const { error, run, busy } = useAction();
  const [note, setNote] = useState<string | null>(null);
  const threads = usePolling<{ threads: { threadId: string }[] }>("/api/agent/threads", 4000);
  const [tid, setTid] = useState("");
  const spec = specOf(graph);
  const convFields = spec.state.filter((f) => f.type === "messages").map((f) => f.name);
  const tstate = usePolling<{ values: Record<string, any[]> }>(tid ? `/api/agent/threads/${encodeURIComponent(tid)}/state` : null, 0, [tid, runs.length]);
  const msgs = convFields.flatMap((f) => (tstate.data?.values?.[f] ?? []).map((m: any) => ({ ...m, field: f })));
  return (
    <div className="stores">
      <section aria-label="long-term store">
        <h3>Long-term records <small className="muted">{recs.data?.path} · shared across threads · scope and namespace filter what a policy may read</small></h3>
        <div className="small muted">{recs.data?.note}</div>
        <div className="rcasehead">
          namespace <Sel label="namespace filter" value={ns} options={["", ...(recs.data?.namespaces ?? [])]} onChange={setNs} />
          scope <Sel label="scope filter" value={scope} options={["", ...(recs.data?.scopes ?? [])]} onChange={setScope} />
          kind <Sel label="kind filter" value={kind} options={["", "semantic", "episodic", "procedural", "summary"]} onChange={setKind} />
          <label className="small"><input type="checkbox" checked={deleted} onChange={(e) => setDeleted(e.target.checked)} /> show deleted</label>
          <button onClick={() => setEditing({})}>New record</button>
          {ui.seedExample && <button disabled={busy} title="Writes SYNTHETIC records invented for this example" onClick={() => run(async () => { const r = await api.post<{ seeded: number }>("/api/agent/memory/seed", { example: ui.seedExample }); setNote(`Seeded ${r.seeded} SYNTHETIC records.`); recs.reload(); })}>Seed example records (SYNTHETIC)</button>}
        </div>
        {note && <div className="notrec">{note}</div>}
        <ErrorLine text={error} />
        <div className="tablewrap"><table className="mem">
          <thead><tr><th>id</th><th>text</th><th>namespace / scope</th><th>kind</th><th>imp.</th><th>created</th><th>v</th><th /></tr></thead>
          <tbody>{(recs.data?.records ?? []).map((r) => (
            <tr key={r.id} className={`${r.deleted_at ? "deleted" : ""} ${r.id === histId ? "sel" : ""}`}>
              <td><code>{r.id}</code></td><td title={r.text}>{r.text.slice(0, 110)}{r.generated && <span className="badge">generated</span>}{r.metadata?.synthetic && <span className="badge">synthetic</span>}</td>
              <td>{r.namespace} / {r.scope}</td><td>{r.kind}</td><td className="num">{r.importance}</td><td className="small">{fmtTime(r.created_at)}</td><td>{r.version}</td>
              <td><button className="small" onClick={() => setHistId(r.id)}>history</button> {!r.deleted_at && <><button className="small" onClick={() => setEditing(r)}>edit</button> <button className="danger small" onClick={() => run(async () => {
                const d = await api.del<{ retainedSnapshots: { runId: string; usedInModelContext: boolean }[]; note: string }>(`/api/agent/memory/records/${encodeURIComponent(r.id)}`);
                setNote(`Deleted ${r.id}. Future selections cannot retrieve it. ${d.retainedSnapshots.length} recorded selection(s) still hold a snapshot of it (${d.retainedSnapshots.filter((x) => x.usedInModelContext).length} were sent to a model). ${d.note}`); recs.reload();
              })}>delete</button></>}</td>
            </tr>))}</tbody></table></div>
        {editing && <RecordForm key={editing.id ?? "new"} rec={editing} onCancel={() => setEditing(null)} onSaved={() => { setEditing(null); recs.reload(); setHistId(editing.id ?? null); }} />}
        {histId && hist.data && (
          <div className="audit" aria-label="record history">
            <h4>Write history of {histId}</h4>
            {hist.data.writes.map((w, i) => <div key={i} className="small"><b>{w.op}</b> {fmtTime(w.ts)} · node {w.node ?? "—"} · run {w.run_id ?? "manual"} · scope {w.scope} · {w.generated ? "GENERATED · " : ""}evidence: {j(w.evidence, 80)} · validation ok={String(w.validation?.ok)}{w.old && <> · old “{j(w.old.text, 60)}”</>}{w.new && <> · new “{j(w.new.text, 60)}”</>}</div>)}
            <div className="small muted">Retained snapshots in recorded selections: {hist.data.retainedSnapshots.length ? hist.data.retainedSnapshots.map((s) => `${s.runId?.slice(0, 8) ?? "?"}:${s.status}`).join(", ") : "none"}</div>
          </div>
        )}
      </section>
      <section aria-label="short-term store">
        <h3>Short-term: a thread's messages <small className="muted">stored in the thread's LangGraph checkpoint</small></h3>
        <div className="rcasehead">thread <Sel label="short-term thread" value={tid} options={["", ...(threads.data?.threads ?? []).map((t) => t.threadId)]} onChange={setTid} />{convFields.length === 0 && <span className="muted small">This graph has no messages state field.</span>}</div>
        {tid && msgs.length === 0 && <div className="empty">No messages stored in this thread (fields: {convFields.join(", ") || "none"}).</div>}
        <table className="mem"><thead><tr><th>#</th><th>role</th><th>content</th><th>id</th><th>time</th></tr></thead>
          <tbody>{msgs.map((m, i) => <tr key={i}><td>{i + 1}</td><td>{m.role}</td><td>{String(m.content ?? "").slice(0, 160)}</td><td className="small"><code>{m.id}</code></td><td className="small">{m.ts ? fmtTime(m.ts) : ""}</td></tr>)}</tbody></table>
      </section>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ policy editor
const RECORD_PATHS: PathInfo[] = [{ path: "kind", type: "text" }, { path: "namespace", type: "text" }, { path: "scope", type: "text" }, { path: "text", type: "text" }, { path: "importance", type: "number" },
  { path: "age_days", type: "number" }, { path: "generated", type: "boolean" }, { path: "store", type: "text" }, { path: "id", type: "text" }];

function csv(v: string[] | undefined) { return (v ?? []).join(", "); }
const unCsv = (t: string) => t.split(",").map((s) => s.trim()).filter(Boolean);

function StageForm({ st, onChange }: { st: PolicyStage; onChange: (c: Record<string, any>) => void }) {
  const cat = useCatalog();
  const c = st.config;
  const [mkeys, setMkeys] = useState<string[]>([]);
  const set = (p: Record<string, any>) => onChange({ ...c, ...p });
  switch (st.op) {
    case "retrieve": return (
      <div>
        <Row label="Stores">{(["long_term", "short_term"] as const).map((s) => <label key={s} className="small"><input type="checkbox" aria-label={`store ${s}`} checked={(c.sources ?? []).includes(s)} onChange={(e) => set({ sources: e.target.checked ? [...(c.sources ?? []), s] : (c.sources ?? []).filter((x: string) => x !== s) })} /> {s.replace("_", "-")} </label>)}</Row>
        <Row label="Namespaces" hint="empty = all"><input aria-label="namespaces" value={csv(c.namespaces)} onChange={(e) => set({ namespaces: unCsv(e.target.value) })} /></Row>
        <Row label="Scopes" hint="templates over the state, e.g. user:{user_id}; empty = all"><input aria-label="scopes" value={csv(c.scopes)} onChange={(e) => set({ scopes: unCsv(e.target.value) })} /></Row>
        <Row label="Kinds"><input aria-label="kinds" value={csv(c.kinds)} placeholder="semantic, episodic…" onChange={(e) => set({ kinds: unCsv(e.target.value) })} /></Row>
        <Row label="Read by" hint="exact scope filters always apply first"><Sel label="retrieve method" value={c.method} options={[{ value: "all", label: "all eligible (oldest first)" }, { value: "similarity", label: "similarity to the query" }, { value: "recency", label: "most recent first" }]} onChange={(m) => set({ method: m })} /></Row>
        <Row label="Limit k"><Num label="retrieve k" integer min={1} nullable value={c.k ?? null} onChange={(n) => set({ k: n })} /></Row>
        {c.method === "similarity" && <Row label="Min similarity"><Num label="min score" nullable value={c.min_score ?? null} onChange={(n) => set({ min_score: n })} /></Row>}
        <Row label="Include expired"><Check label="include expired" value={!!c.include_expired} onChange={(b) => set({ include_expired: b })} /></Row>
      </div>);
    case "filter": {
      const paths = [...RECORD_PATHS, ...mkeys.map((k) => ({ path: `metadata.${k}`, type: "text" as const }))];
      return (
        <div>
          <PredicateBuilder value={c.where && Object.keys(c.where).length ? c.where : { always: true }} paths={paths} catalog={cat} onChange={(w) => set({ where: w })} />
          <div className="small">metadata key: <input aria-label="add metadata key" size={10} placeholder="entity" onKeyDown={(e) => { if (e.key === "Enter" && e.currentTarget.value) { setMkeys([...mkeys, e.currentTarget.value]); e.currentTarget.value = ""; } }} /> (Enter adds it to the field list)</div>
          <div className="small muted">keeps records where {renderPredicate(c.where ?? { always: true })}</div>
        </div>);
    }
    case "rank": return (
      <div>
        {(["recency", "relevance", "importance"] as const).map((k) => (
          <Row key={k} label={`weight: ${k}`} hint={k === "recency" ? "0.5^(age / half-life)" : k === "relevance" ? "cosine similarity to the policy query" : "the record's importance"}>
            <input type="range" min={0} max={2} step={0.05} aria-label={`${k} weight slider`} value={c.weights?.[k] ?? 0} onChange={(e) => set({ weights: { ...c.weights, [k]: Number(e.target.value) } })} />
            <Num label={`${k} weight`} min={0} value={c.weights?.[k] ?? 0} onChange={(n) => set({ weights: { ...c.weights, [k]: n ?? 0 } })} width={64} /></Row>
        ))}
        <Row label="Recency half-life (days)"><Num label="half life days" min={0.01} value={c.half_life_days} onChange={(n) => set({ half_life_days: n ?? 30 })} /></Row>
        <Row label="Keep top"><Num label="rank limit" integer min={1} nullable value={c.limit ?? null} onChange={(n) => set({ limit: n })} /></Row>
        <Row label="Min final score"><Num label="rank min score" nullable value={c.min_score ?? null} onChange={(n) => set({ min_score: n })} /></Row>
      </div>);
    case "dedupe": return (
      <div>
        <Row label="Method"><Sel label="dedupe method" value={c.method} options={[{ value: "exact_text", label: "identical text" }, { value: "similarity", label: "embedding similarity" }]} onChange={(m) => set({ method: m })} /></Row>
        {c.method === "similarity" && <Row label="Threshold"><Num label="dedupe threshold" min={0.01} max={1} value={c.threshold} onChange={(n) => set({ threshold: n ?? 0.9 })} /></Row>}
        <div className="small muted">Keeps the record that comes first in the current order (the higher ranked one after a rank stage).</div>
      </div>);
    case "budget": return (
      <div>
        <Row label="Token budget" hint="estimate: ceil(characters / 4)"><Num label="budget tokens" integer min={1} value={c.max_tokens} onChange={(n) => set({ max_tokens: n ?? 100 })} /></Row>
        <Row label="When a record does not fit"><Sel label="overflow" value={c.overflow} options={[{ value: "stop", label: "stop: drop it and everything after" }, { value: "skip", label: "skip it, try smaller ones" }]} onChange={(o) => set({ overflow: o })} /></Row>
        <Row label="Result order"><Sel label="result order" value={c.order} options={[{ value: "rank", label: "by rank" }, { value: "chronological", label: "chronological" }]} onChange={(o) => set({ order: o })} /></Row>
      </div>);
    case "summarize": return (
      <div>
        <Row label="Keep verbatim"><Num label="keep recent" integer min={0} value={c.keep_recent} onChange={(n) => set({ keep_recent: n ?? 0 })} /> most recent records</Row>
        <Row label="Method"><Sel label="summary method" value={c.method} options={[{ value: "extractive", label: "extractive (first sentences; no model)" }, { value: "model", label: "model summary (a model call; skipped in previews)" }]} onChange={(m) => set({ method: m, model: m === "model" ? c.model ?? { provider: "ollama", model: "qwen3.5:2b", temperature: 0, max_tokens: 200, seed: null, timeout_s: 120, think: false, fixture: {} } : null })} /></Row>
        <Row label="Max characters"><Num label="summary chars" integer min={40} value={c.max_chars} onChange={(n) => set({ max_chars: n ?? 400 })} /></Row>
        {c.method === "model" && c.model && <ModelForm model={c.model} onChange={(m) => set({ model: m })} />}
        <div className="small muted">Replaced records are marked “summarized”; the summary lists its sources and known omissions and is marked generated.</div>
      </div>);
  }
}

export function PolicyEditor({ graph, setGraph, policyId, setPolicyId }: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; policyId: string; setPolicyId: (id: string) => void }) {
  const cat = useCatalog();
  const spec = specOf(graph);
  const pol = spec.policies.find((p) => p.id === policyId) ?? spec.policies[0];
  const setPol = (fn: (p: Policy) => Policy) => setGraph((g) => ({ ...g, agent: { ...(g.agent ?? {}), policies: specOf(g).policies.map((p) => (p.id === pol?.id ? fn(p) : p)) } }));
  const [add, setAdd] = useState("budget");
  if (!pol) return (
    <div className="pad"><div className="empty">This graph has no memory policy.</div>
      <button onClick={() => { setGraph((g) => ({ ...g, agent: { ...(g.agent ?? {}), policies: [...specOf(g).policies, { id: "policy_1", name: "", query: "{question}", embeddings: { provider: "local_hash", model: "nomic-embed-text", dimension: 256, normalize: true }, stages: [{ id: "eligible", op: "retrieve", config: cat?.stages.defaults.retrieve ?? {} }] }] } })); setPolicyId("policy_1"); }}>Create a policy</button></div>
  );
  const move = (i: number, d: number) => setPol((p) => { const s = [...p.stages]; const [x] = s.splice(i, 1); s.splice(i + d, 0, x); return { ...p, stages: s }; });
  return (
    <div className="policyed">
      <div className="rcasehead">Policy <Sel label="policy" value={pol.id} options={spec.policies.map((p) => p.id)} onChange={setPolicyId} />
        <span className="small muted">An ordered pipeline of memory-operation blocks. Each stage records a decision for every record; edit it here, then preview.</span></div>
      <Row label="Query" hint="template over the state; its embedding gives each record's relevance"><input aria-label="policy query" value={pol.query} onChange={(e) => setPol((p) => ({ ...p, query: e.target.value }))} /></Row>
      <Row label="Embeddings" hint="relevance scoring only"><Sel label="policy embedding provider" value={pol.embeddings.provider} options={[{ value: "local_hash", label: "local_hash (lexical, NOT semantic)" }, { value: "ollama", label: "Ollama embeddings" }]} onChange={(p) => setPol((x) => ({ ...x, embeddings: { ...x.embeddings, provider: p } }))} />
        {pol.embeddings.provider === "ollama" && <input aria-label="policy embedding model" value={pol.embeddings.model} onChange={(e) => setPol((x) => ({ ...x, embeddings: { ...x.embeddings, model: e.target.value } }))} />}</Row>
      <ol className="stages">{pol.stages.map((st, i) => (
        <li key={st.id} className={`stage op-${st.op}`}>
          <div className="stagehead"><b>{i + 1}. {st.op}</b> <input aria-label={`stage id ${i + 1}`} value={st.id} size={10} onChange={(e) => setPol((p) => ({ ...p, stages: p.stages.map((x, j) => (j === i ? { ...x, id: e.target.value } : x)) }))} />
            <button className="small" disabled={i === 0 || st.op === "retrieve"} aria-label={`move stage ${i + 1} up`} onClick={() => move(i, -1)}>↑</button>
            <button className="small" disabled={i === pol.stages.length - 1 || pol.stages[i + 1]?.op === "retrieve"} aria-label={`move stage ${i + 1} down`} onClick={() => move(i, 1)}>↓</button>
            <button className="danger small" disabled={st.op === "retrieve"} aria-label={`remove stage ${i + 1}`} onClick={() => setPol((p) => ({ ...p, stages: p.stages.filter((_, j) => j !== i) }))}>×</button></div>
          <StageForm st={st} onChange={(config) => setPol((p) => ({ ...p, stages: p.stages.map((x, j) => (j === i ? { ...x, config } : x)) }))} />
        </li>))}</ol>
      <div className="rcasehead">Add stage <Sel label="new stage op" value={add} options={["filter", "rank", "dedupe", "budget", "summarize"]} onChange={setAdd} />
        <button onClick={() => setPol((p) => ({ ...p, stages: [...p.stages, { id: `${add}_${p.stages.length + 1}`, op: add as PolicyStage["op"], config: JSON.parse(JSON.stringify(cat?.stages.defaults[add] ?? {})) }] }))}>Add</button></div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ isolated preview
function PreviewTab({ graph, projectId, runs, runId, setRunId, callId, setCallId, onRerun }: {
  graph: Graph; projectId: string; runs: AgentRunSummary[]; runId: string | null; setRunId: (id: string | null) => void; callId: string | null; setCallId: (id: string | null) => void; onRerun: (runId: string) => void;
}) {
  const calls = usePolling<{ calls: { callId: string; node: string; purpose: string; fixture: boolean }[] }>(runId ? `/api/agent/runs/${runId}/model-calls` : null, 0, [runId]);
  const [res, setRes] = useState<PreviewResult | null>(null);
  const { busy, error, run } = useAction();
  const run0 = runs.find((r) => r.id === runId);
  useEffect(() => setRes(null), [runId, callId]);
  const doPreview = () => run(async () => setRes(await api.post<PreviewResult>("/api/agent/policy/preview", { graph, runId, callId })));
  const rerun = () => run(async () => {
    const r = await api.post<{ runId: string }>("/api/runs", { graph, config: { project_id: projectId, thread_id: `th-${uid().slice(0, 8)}`, input: run0?.input ?? {} } }, { "Idempotency-Key": uid() });
    onRerun(r.runId);
  });
  return (
    <div className="previewtab">
      <div className="rcasehead">
        Recorded call: run <select aria-label="preview run" value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)}><option value="">(choose)</option>{[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{r.id.slice(0, 8)} · {r.threadId}</option>)}</select>
        call <select aria-label="preview call" value={callId ?? ""} onChange={(e) => setCallId(e.target.value || null)}><option value="">(choose)</option>{(calls.data?.calls ?? []).map((c, i) => <option key={c.callId} value={c.callId}>#{i + 1} {c.node} {c.purpose}{c.fixture ? " (FIXTURE)" : ""}</option>)}</select>
        <button className="primary" disabled={busy || !runId || !callId} onClick={doPreview}>Preview context with the edited policy</button>
        <button disabled={busy || !runId} onClick={rerun} title="Starts a NEW run with the current draft graph: this one makes real model calls">Rerun with the edited policy (new model calls)</button>
      </div>
      <div className="notrec"><b>Preview is isolated:</b> it recomputes the policy on the records recorded for that call and re-renders the prompt. No model is called and no store, state or run is changed.</div>
      <ErrorLine text={error} />
      {res && (
        <div className="preview" aria-label="context preview">
          <div className="notrec">{res.note} {res.embeddingCalls && "(This policy uses Ollama embeddings, which were requested.)"} Tokens (estimate): {res.before.tokensEstimate} → <b>{res.after.tokensEstimate}</b></div>
          <h4>Record changes caused by the edit</h4>
          {res.recordChanges.length === 0 ? <div className="muted">The edited policy selects exactly the same records as the recorded one.</div> : (
            <table><thead><tr><th>record</th><th>before</th><th>after</th><th>why it changed</th></tr></thead><tbody>{res.recordChanges.map((c) => (
              <tr key={c.id}><td><code>{c.id}</code></td><td><span className={`pill ${c.before}`}>{c.before}</span></td><td><span className={`pill ${c.after}`}>{c.after}</span></td>
                <td className="small">{c.before !== "included" ? `was: ${c.beforeReason}` : ""}{c.after !== "included" ? ` now: ${c.afterReason}` : ` now scores ${j(c.afterScores, 120)}`}</td></tr>))}</tbody></table>
          )}
          <h4>Context diff (what the model would receive)</h4>
          <div className="diffview">{res.segments.map((s, i) => (
            <div key={i} className={`dseg ${s.status}`}><span className="pill">{s.status}</span> <span className="small muted">{s.role} · {s.source.kind}{s.source.recordId ? ` ${s.source.recordId}` : s.source.chunkId ? ` ${s.source.chunkId}` : ""} · ≈{s.tokensEstimate}t</span><div className="dtext">{s.text}</div></div>
          ))}</div>
          {res.applications.map((a) => (<div key={a.applicationId}><h4>Decisions under the edited policy “{a.policy}”</h4><ApplicationView stages={a.stages} records={a.records} final={a.final} />{a.skippedStages.length > 0 && <div className="warn">Skipped (needs a model call): {a.skippedStages.join(", ")}</div>}</div>))}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ tab
export function MemoryTab({ graph, setGraph, ui, projectId, runs, runId, setRunId, callId, setCallId, focusRecord, onRerun }: {
  graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; ui: UiDoc; projectId: string; runs: AgentRunSummary[]; runId: string | null; setRunId: (id: string | null) => void;
  callId: string | null; setCallId: (id: string | null) => void; focusRecord: string | null; onRerun: (runId: string) => void;
}) {
  const [sub, setSub] = useState<"stores" | "policy" | "preview" | "decisions">(focusRecord ? "stores" : "stores");
  const spec = specOf(graph);
  const [policyId, setPolicyId] = useState(spec.policies[0]?.id ?? "");
  const apps = usePolling<{ applications: { id: string; policyId: string; runId: string; node: string; universe: number; selected: number }[] }>(runId ? `/api/agent/memory/applications?runId=${runId}` : null, 0, [runId, sub]);
  const [appId, setAppId] = useState("");
  const app = usePolling<{ stages: AppStage[]; records: Record<string, AppRecord>; final: string[]; asOf: number; query: string; embeddings: string }>(appId ? `/api/agent/memory/applications/${appId}` : null, 0, [appId]);
  useEffect(() => { if (apps.data?.applications.length && !apps.data.applications.some((a) => a.id === appId)) setAppId(apps.data.applications[apps.data.applications.length - 1].id); }, [apps.data, appId]);
  const subs = useMemo(() => [["stores", "Stores"], ["policy", "Policy editor"], ["preview", "Isolated preview"], ["decisions", "Recorded decisions"]] as const, []);
  return (
    <div className="amemory">
      <div className="tabs subtabs" role="tablist">{subs.map(([k, l]) => <button key={k} role="tab" aria-selected={sub === k} className={sub === k ? "on" : ""} onClick={() => setSub(k)}>{l}</button>)}</div>
      {sub === "stores" && <StoresView graph={graph} ui={ui} runs={runs} focusRecord={focusRecord} />}
      {sub === "policy" && <PolicyEditor graph={graph} setGraph={setGraph} policyId={policyId} setPolicyId={setPolicyId} />}
      {sub === "preview" && <PreviewTab graph={graph} projectId={projectId} runs={runs} runId={runId} setRunId={setRunId} callId={callId} setCallId={setCallId} onRerun={onRerun} />}
      {sub === "decisions" && (
        <div>
          <div className="rcasehead">Run <select aria-label="decisions run" value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)}><option value="">(choose)</option>{[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{r.id.slice(0, 8)} · {r.threadId}</option>)}</select>
            selection <select aria-label="recorded selection" value={appId} onChange={(e) => setAppId(e.target.value)}>{(apps.data?.applications ?? []).map((a) => <option key={a.id} value={a.id}>{a.policyId} at {a.node} · {a.selected}/{a.universe} selected</option>)}</select></div>
          {app.data ? <><div className="small muted">Recorded as of {fmtTime(app.data.asOf)} · query “{app.data.query}” · embeddings {app.data.embeddings}</div><ApplicationView stages={app.data.stages} records={app.data.records} final={app.data.final} /></>
            : <div className="empty pad">Choose a run that used a memory selection block.</div>}
        </div>
      )}
    </div>
  );
}
