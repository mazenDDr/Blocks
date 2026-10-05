import { useState } from "react";
import { api, errorText } from "../api";

interface SessionRow { session: string; status: "checkpoint" | "reset"; head: { revision: number; checkpointSha256: string | null; lastRequestId: string | null } }
interface Page { releaseId: string; versionId: string; user: string; prefix: string; after: string | null; sessions: SessionRow[]; nextAfter: string | null; policy: string }

export function ConversationDiscovery({ releaseId, user, disabled, onInspect }: {
  releaseId: string; user: string; disabled: boolean; onInspect: (session: string) => Promise<void>;
}) {
  const [prefix, setPrefix] = useState("");
  const [page, setPage] = useState<Page | null>(null);
  const [back, setBack] = useState<(string | null)[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const valid = /^[A-Za-z0-9_-]{1,64}$/.test(user) && /^[A-Za-z0-9_-]{0,64}$/.test(prefix);
  const load = async (after: string | null, history: (string | null)[]) => {
    setBusy(true); setError(null);
    try {
      const query = new URLSearchParams({ user, prefix, limit: "25", ...(after ? { after } : {}) });
      const result = await api.get<Page>(`/api/production/releases/${releaseId}/conversations?${query}`);
      setPage(result); setBack(history);
    } catch (e) { setPage(null); setBack([]); setError(errorText(e)); }
    finally { setBusy(false); }
  };
  // Parent keys this component by release/user. Prefix edits invalidate old pages.
  return <section className="conversation-discovery" aria-label="conversation discovery">
    <h4>Find recorded conversations</h4>
    <p>Lists only session identifiers and committed head metadata for this release/user. Select a session to inspect its actual checkpoint. User namespaces are declared isolation keys.</p>
    <label>Session prefix <input aria-label="conversation session prefix" maxLength={64} disabled={disabled || busy} value={prefix} onChange={e => { setPrefix(e.target.value); setPage(null); setBack([]); }} /></label>
    <button disabled={disabled || busy || !valid} onClick={() => void load(null, [])}>{busy ? "Reading conversations…" : "Discover conversations"}</button>
    {error && <p role="alert">{error}</p>}
    {page && <>
      <p className="provenance">Release {page.releaseId} · version {page.versionId} · user {page.user}</p>
      {page.sessions.length === 0 ? <p>No recorded conversations match this scope and prefix.</p> : <table><caption>{page.sessions.length} recorded sessions on this page</caption><thead><tr><th>Session</th><th>Recorded head</th><th>Checkpoint / producing request</th><th>Inspect</th></tr></thead>
        <tbody>{page.sessions.map(row => <tr key={row.session}><th scope="row">{row.session}</th><td>revision {row.head.revision} · {row.status === "reset" ? "reset; no live checkpoint" : "checkpoint"}</td><td>{row.head.checkpointSha256 ?? "none"}<br />{row.head.lastRequestId ?? "no producing request"}</td><td><button disabled={disabled || busy} onClick={() => void onInspect(row.session)}>Inspect {row.session}</button></td></tr>)}</tbody></table>}
      <button disabled={disabled || busy || back.length === 0} onClick={() => void load(back.at(-1)!, back.slice(0, -1))}>Previous conversations</button>
      <button disabled={disabled || busy || !page.nextAfter} onClick={() => void load(page.nextAfter, [...back, page.after])}>Next conversations</button>
      <p>{page.policy}</p>
    </>}
  </section>;
}
