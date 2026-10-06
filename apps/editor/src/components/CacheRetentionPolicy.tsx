import { useState } from "react";
import { api, errorText } from "../api";
import { usePolling } from "../hooks";

interface Config { enabled: boolean; keepLatestPerNode: number; olderThanHours: number; intervalSeconds: number }
interface Status { projectId: string; policy: { config: Config; revision: number; nextDue: number; updatedAt: number } | null; running: boolean; lastError: string | null; receipts: { seq: number; revision: number; started: number; finished: number; result: unknown; error: string | null }[] }
const DEFAULT: Config = { enabled: false, keepLatestPerNode: 1, olderThanHours: 24, intervalSeconds: 3600 };
const FIELDS = [
  { key: "keepLatestPerNode", label: "Newest results to keep per node", min: 1, max: 1000, step: 1 },
  { key: "olderThanHours", label: "Minimum age to remove (hours)", min: 0, max: 87600, step: "any" },
  { key: "intervalSeconds", label: "Check interval (seconds)", min: 5, max: 604800, step: 1 },
] as const;

/** Only an explicit reviewed save enables deletion of obsolete cache entries. */
export function CacheRetentionPolicy({ projectId }: { projectId: string }) {
  const path = `/api/cache/nodes/policies/${encodeURIComponent(projectId)}`;
  const validProject = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(projectId);
  const state = usePolling<Status>(validProject ? path : null, 2000);
  const [draft, setDraft] = useState<{ config: Config; revision: number } | null>(null);
  const [preview, setPreview] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const update = (change: Partial<Config>) => { setDraft(d => d ? { ...d, config: { ...d.config, ...change } } : null); setPreview(null); };
  const act = async (save: boolean) => {
    if (!draft) return;
    setBusy(true); setError(null);
    try {
      if (save) {
        const result = await api.put<Status>(path, { expectedRevision: draft.revision, policy: draft.config });
        setDraft(result.policy ? { config: result.policy.config, revision: result.policy.revision } : null);
        setPreview(null); state.reload();
      } else setPreview(await api.post(path + "/preview", draft.config));
    } catch (e) { setError(errorText(e)); } finally { setBusy(false); }
  };
  return <details className="cache-retention-policy"><summary>Automatic cache retention for this saved project</summary>
    <p>Disabled until explicitly configured. Keep the newest results per node and remove only additional entries older than the declared age. Recorded run artifacts are protected. Checks defer while native research runs are active or paused. Removed cache results are recomputed when needed. One owning local control process; stop it before offline backup.</p>
    {!validProject && <p>Use a valid project ID (1–64 characters, starting with a letter or digit) and save the project before configuring retention.</p>}
    {state.error && <p role="alert">{state.error}</p>}
    {state.data && <><p className="provenance">Project {projectId} · {state.data.running ? "local scheduler running" : "local scheduler stopped"} · {state.data.policy ? `stored policy revision ${state.data.policy.revision}` : "no policy recorded"}</p>
      {state.data.lastError && <p role="alert">Scheduler error: {state.data.lastError}</p>}
      <pre>{JSON.stringify(state.data.policy, null, 2)}</pre>
      <button disabled={busy} onClick={() => { setDraft({ config: { ...(state.data?.policy?.config ?? DEFAULT) }, revision: state.data?.policy?.revision ?? 0 }); setPreview(null); setError(null); }}>Edit retention policy</button>
      {draft && <><p>Draft based on recorded revision {draft.revision}; preview changes nothing. Save updates this project’s future scheduled checks.</p>
        <div style={{ display: "grid", gap: 8, marginBottom: 8 }}>
          <label><input aria-label="automatic cache retention enabled" type="checkbox" checked={draft.config.enabled} disabled={busy} onChange={e => update({ enabled: e.target.checked })} /> Enable automatic retention</label>
          {FIELDS.map(({ key, label, min, max, step }) => <label key={key} style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>{label}<input aria-label={`cache retention ${key}`} type="number" min={min} max={max} step={step} disabled={busy} value={draft.config[key]} onChange={e => update({ [key]: Number(e.target.value) })} /></label>)}
        </div>
        <button disabled={busy} onClick={() => act(false)}>Preview retention candidates</button>{" "}<button disabled={busy} onClick={() => act(true)}>Save retention policy</button>
      </>}
      {preview != null && <><h4>Native dry-run candidates; nothing removed</h4><pre aria-label="cache retention preview">{JSON.stringify(preview, null, 2)}</pre></>}
      <h4>Actual scheduled check receipts (newest100 for this project)</h4>
      {state.data.receipts.length ? <pre aria-label="cache retention receipts">{JSON.stringify(state.data.receipts, null, 2)}</pre> : <p>No scheduled check recorded.</p>}
    </>}
    {error && <p role="alert">{error}</p>}
  </details>;
}
