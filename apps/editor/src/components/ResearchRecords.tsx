import { useEffect, useRef, useState } from "react";
import { api, errorText } from "../api";

interface Annotation { revision: number; note: string; tags: string[]; author: string | null; recordedAt: number | null }
interface RunRecord { runId: string; graphHash: string; projectId: string | null; kind: string; status: string; createdAt: number; updatedAt: number; annotation: Annotation; policy?: string }
interface Page { runs: RunRecord[]; nextAfter: string | null; policy: string }
interface Revisions { runId: string; revisions: Annotation[]; nextAfter: number | null; policy: string }

export function ResearchRecords({ projectId, onOpenRun }: { projectId: string; onOpenRun: (run: RunRecord) => void }) {
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState("");
  const [allProjects, setAllProjects] = useState(false);
  const [kind, setKind] = useState("");
  const [status, setStatus] = useState("");
  const [page, setPage] = useState<Page | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cursors, setCursors] = useState<string[]>([""]);
  const invalidate = () => { setPage(null); setSelected(null); setCursors([""]); };
  const read = async (next: string[]) => {
    setBusy(true); setError(null); setSelected(null);
    const params = new URLSearchParams({ query });
    if (tag) params.set("tag", tag);
    if (!allProjects) params.set("projectId", projectId);
    if (kind) params.set("kind", kind);
    if (status) params.set("status", status);
    if (next.at(-1)) params.set("after", next.at(-1)!);
    try { setPage(await api.get<Page>(`/api/research/runs?${params}`)); setCursors(next); }
    catch (e) { setPage(null); setError(errorText(e)); }
    finally { setBusy(false); }
  };
  return <div className="fullws"><section className="research-records production-workspace" aria-label="research run records">
    <h2>Research run records</h2>
    <p>Search recorded run identifiers, projects, graph hashes and authored notes/tags. Metadata is separate from measured execution results. All graph families can be listed; the original run keeps its existing inspector.</p>
    <section className="prod-card">
      <label>Search text <input aria-label="research search text" maxLength={200} disabled={busy} value={query} onChange={e => { setQuery(e.target.value); invalidate(); }} /></label>
      <label>Exact tag <input aria-label="research exact tag" maxLength={60} disabled={busy} value={tag} onChange={e => { setTag(e.target.value); invalidate(); }} /></label>
      <label><input aria-label="research all projects" type="checkbox" disabled={busy} checked={allProjects} onChange={e => { setAllProjects(e.target.checked); invalidate(); }} /> All projects (current: {projectId})</label>
      <label>Graph family <select aria-label="research graph family" disabled={busy} value={kind} onChange={e => { setKind(e.target.value); invalidate(); }}><option value="">All families</option>{["model", "tabular", "agent", "rl", "procedure", "unsup", "domain"].map(k => <option key={k}>{k}</option>)}</select></label>
      <label>Run status <select aria-label="research run status" disabled={busy} value={status} onChange={e => { setStatus(e.target.value); invalidate(); }}><option value="">All statuses</option>{["queued", "preparing", "running", "paused", "cancelling", "completed", "failed", "cancelled"].map(s => <option key={s}>{s}</option>)}</select></label>
      <button disabled={busy} onClick={() => void read([""])}>Search recorded runs</button>
      <button disabled={busy || cursors.length < 2} onClick={() => void read(cursors.slice(0, -1))}>Previous recorded runs</button>
      <button disabled={busy || !page?.nextAfter} onClick={() => void read([...cursors, page!.nextAfter!])}>Next recorded runs</button>
      {error && <p role="alert">{error}</p>}
      {page && <><p className="provenance">{page.policy} Lexicographic run-ID pages, at most 25 rows.</p>
        {page.runs.length === 0 ? <p>No recorded runs match these filters.</p> : <table className="conversation-discovery-table"><caption>Recorded runs and current authored metadata</caption><thead><tr><th>Run / project</th><th>Family / status</th><th>Notes / tags</th><th>Inspect</th></tr></thead><tbody>{page.runs.map(run => <tr key={run.runId}>
          <th scope="row">{run.runId}<br />{run.projectId ?? "no recorded project"}</th><td>{run.kind} · {run.status}</td>
          <td>{run.annotation.note.slice(0, 180) || "No current note"}<br />{run.annotation.tags.join(", ") || "No current tags"} · metadata revision {run.annotation.revision}</td>
          <td><button disabled={busy} aria-label={`inspect research ${run.runId}`} onClick={() => setSelected(run.runId)}>Inspect record</button><button disabled={busy || run.projectId !== projectId} title={run.projectId !== projectId ? "Open this recorded project before inspecting its original run" : "Open the existing native run inspector"} onClick={() => onOpenRun(run)}>Open original run</button></td>
        </tr>)}</tbody></table>}
      </>}
    </section>
    {selected && <AnnotationEditor key={selected} runId={selected} onChanged={() => setPage(null)} />}
  </section></div>;
}

function AnnotationEditor({ runId, onChanged }: { runId: string; onChanged: () => void }) {
  const [record, setRecord] = useState<RunRecord | null>(null);
  const [note, setNote] = useState("");
  const [tags, setTags] = useState("");
  const [author, setAuthor] = useState("local researcher");
  const [history, setHistory] = useState<Revisions | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [receipt, setReceipt] = useState<string | null>(null);
  const attempts = useRef(new Map<string, string>());
  const accept = (value: RunRecord) => { setRecord(value); setNote(value.annotation.note); setTags(value.annotation.tags.join("\n")); setAuthor(value.annotation.author ?? "local researcher"); };
  useEffect(() => {
    let disposed = false;
    api.get<RunRecord>(`/api/research/runs/${encodeURIComponent(runId)}`).then(value => { if (!disposed) accept(value); }).catch(e => { if (!disposed) setError(errorText(e)); });
    return () => { disposed = true; };
  }, [runId]);
  const act = async (fn: () => Promise<void>) => { setBusy(true); setError(null); try { await fn(); } catch (e) { setError(errorText(e)); } finally { setBusy(false); } };
  const save = () => act(async () => {
    const values = { expectedRevision: record!.annotation.revision, expectedGraphHash: record!.graphHash,
      author, note, tags: tags.split("\n").map(s => s.trim()).filter(Boolean) };
    const key = JSON.stringify(values);
    const id = attempts.current.get(key) ?? crypto.randomUUID(); attempts.current.set(key, id);
    const saved = await api.put<RunRecord & { mutationId: string }>(`/api/research/runs/${encodeURIComponent(runId)}/annotation`, { mutationId: id, ...values });
    accept(saved); setReceipt(saved.mutationId); setHistory(null); onChanged();
  });
  return <section className="prod-card" aria-label="research annotation editor">
    <h3>Authored metadata for {runId}</h3>
    {error && <p role="alert">{error}</p>}
    {record && <>
      <p className="provenance">Graph {record.graphHash} · project {record.projectId ?? "not recorded"} · {record.kind} · {record.status} · metadata revision {record.annotation.revision} · {record.annotation.recordedAt == null ? "not annotated" : `recorded ${new Date(record.annotation.recordedAt * 1000).toISOString()}`}</p>
      <p>{record.policy}</p>
      <label>Author label <input aria-label="research annotation author" maxLength={100} disabled={busy} value={author} onChange={e => setAuthor(e.target.value)} /></label>
      <label>Research note <textarea aria-label="research annotation note" maxLength={5000} disabled={busy} value={note} onChange={e => setNote(e.target.value)} /></label>
      <label>Tags (one per line, at most 20; each 1–60 characters) <textarea aria-label="research annotation tags" maxLength={1240} disabled={busy} value={tags} onChange={e => setTags(e.target.value)} /></label>
      <button disabled={busy || !author.trim()} onClick={() => void save()}>Save reviewed annotation</button>
      <button disabled={busy} onClick={() => void act(async () => { accept(await api.get<RunRecord>(`/api/research/runs/${encodeURIComponent(runId)}`)); setReceipt(null); setHistory(null); })}>Reload stored annotation</button>
      <button disabled={busy} onClick={() => void act(async () => setHistory(await api.get<Revisions>(`/api/research/runs/${encodeURIComponent(runId)}/annotation/history`)))}>Read annotation revisions</button>
      {receipt && <p className="provenance">Recorded mutation {receipt}</p>}
      {history && <><pre className="prod-recorded">{JSON.stringify(history, null, 2)}</pre><button disabled={busy || history.nextAfter == null} onClick={() => void act(async () => setHistory(await api.get<Revisions>(`/api/research/runs/${encodeURIComponent(runId)}/annotation/history?after=${history.nextAfter}`)))}>Next annotation revisions</button></>}
    </>}
  </section>;
}
