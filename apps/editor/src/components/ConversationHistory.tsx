import { useRef, useState } from "react";
import type { ConversationSnapshot } from "./ConversationActions";

export interface HistoricalSnapshot {
  releaseId: string; user: string; session: string;
  sourceRequestId: string; sourceTraceSha256: string; sourceCheckpointSha256: string;
  sourceHead: { revision: number; checkpointSha256: string; lastRequestId: string };
  sourceThreadId: string; state: unknown; readOnly: true; policy: string;
}

/** Both source and destination must be inspected in the selected scope before a restore. */
export function ConversationHistory({ snapshot, sourceRequestId, disabled, onPreview, onRestore }: {
  snapshot: ConversationSnapshot; sourceRequestId: string | null; disabled: boolean;
  onPreview: (requestId: string) => Promise<HistoricalSnapshot | null>;
  onRestore: (body: Record<string, unknown>) => Promise<void>;
}) {
  const [source, setSource] = useState<HistoricalSnapshot | null>(null);
  const [reason, setReason] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const attempts = useRef(new Map<string, string>());
  const selected = source?.sourceRequestId === sourceRequestId ? source : null;
  const preview = async () => {
    setSource(null); setConfirmed(false);
    const result = await onPreview(sourceRequestId!);
    setSource(result);
  };
  const restore = async () => {
    const values = { user: snapshot.user, session: snapshot.session,
      expectedRevision: snapshot.head!.revision, expectedCheckpointSha256: snapshot.head!.checkpointSha256,
      sourceRequestId: selected!.sourceRequestId, sourceTraceSha256: selected!.sourceTraceSha256,
      sourceCheckpointSha256: selected!.sourceCheckpointSha256, reason };
    const key = JSON.stringify({ releaseId: snapshot.releaseId, ...values });
    const id = attempts.current.get(key) ?? crypto.randomUUID();
    attempts.current.set(key, id);
    await onRestore({ actionId: id, ...values });
  };
  return <section aria-label="historical conversation restore">
    <h4>Restore an earlier successful turn</h4>
    <p>Select a recorded successful request from this release, user and session, then preview its saved state. Restoration copies that state into a fresh native thread and advances the current revision. Earlier evidence remains recorded. Preview and restoration make no model call.</p>
    <p className="provenance">Current reviewed revision {snapshot.head?.revision ?? "no recorded head"} · checkpoint {snapshot.head?.checkpointSha256 ?? "empty after reset"} · selected request {sourceRequestId ?? "select a matching recorded request"}</p>
    <button disabled={disabled || !sourceRequestId || !snapshot.head} onClick={() => void preview()}>Preview historical checkpoint</button>
    {selected && <>
      <details open><summary>Verified historical checkpoint and provenance</summary><pre className="prod-recorded">{JSON.stringify(selected, null, 2)}</pre></details>
      <label>Restoration reason <input aria-label="conversation restoration reason" disabled={disabled} maxLength={500} value={reason} onChange={e => setReason(e.target.value)} /></label>
      <label><input aria-label="confirm historical restoration" type="checkbox" disabled={disabled} checked={confirmed} onChange={e => setConfirmed(e.target.checked)} /> Replace the reviewed live head with this earlier state, keeping all prior evidence</label>
      <button disabled={disabled || !confirmed || !reason.trim()} onClick={() => void restore()}>Restore reviewed historical checkpoint</button>
    </>}
  </section>;
}
