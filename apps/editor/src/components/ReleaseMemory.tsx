import { useEffect, useState } from "react";
import { api, errorText } from "../api";

interface MemoryRecord { id: string; namespace: string; kind: string; text: string; request: string; created: number; generated: boolean; evidence: unknown }
interface MemoryList { releaseId: string; user: string; records: MemoryRecord[]; count: number; maxRecords: number; policy: string }

/** Long-term memory a release keeps for one user (ADR 0075). Successful turns write it; the user (or an admin) can delete records. */
export function ReleaseMemory({ releaseId, user, refresh, disabled }: { releaseId: string; user: string; refresh: unknown; disabled: boolean }) {
  const [data, setData] = useState<MemoryList | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const url = `/api/production/releases/${releaseId}/memory?user=${encodeURIComponent(user)}`;
  const load = () => api.get<MemoryList>(url).then((d) => { setData(d); setErr(null); }, (e) => setErr(errorText(e)));
  useEffect(() => { void load(); }, [url, refresh]);
  async function remove(id: string) {
    try { await api.del(`/api/production/releases/${releaseId}/memory/${encodeURIComponent(id)}?user=${encodeURIComponent(user)}`); await load(); }
    catch (e) { setErr(errorText(e)); }
  }
  return (
    <section className="release-memory" aria-label="release memory">
      <h4>Long-term memory of user {user} in this release {data && <span className="badge">{data.count} / {data.maxRecords} records</span>}</h4>
      <p className="small muted">Each turn sees only this user's records in this release. Records are written when a turn succeeds; failed, cancelled and replayed requests write nothing. Deleting removes the record and its text.</p>
      {err && <div className="error small">{err}</div>}
      {data && data.records.length === 0 && <div className="empty small">No records yet.</div>}
      {data && data.records.length > 0 && <div className="tablewrap"><table className="dtable" aria-label="memory records"><thead><tr><th>text</th><th>kind</th><th>written</th><th></th></tr></thead>
        <tbody>{data.records.map((r) => <tr key={r.id}><td>{r.text}{r.generated ? <span className="badge" title="model-generated; storing it does not make it true">generated</span> : null}</td><td>{r.kind}</td>
          <td title={`request ${r.request}`}>{new Date(r.created * 1000).toLocaleTimeString()} · <code>{r.request.slice(0, 8)}</code></td>
          <td><button disabled={disabled} aria-label={`delete memory record ${r.id}`} onClick={() => void remove(r.id)}>Delete</button></td></tr>)}</tbody></table></div>}
    </section>
  );
}
