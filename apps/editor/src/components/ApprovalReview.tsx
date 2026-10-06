import { useState } from "react";
import { api } from "../api";

type Snapshot = { releaseId: string; user: string; session: string; head: { revision: number; checkpointSha256: string } | null; pending: { id: string; value: { prompt: string; proposed: unknown; actions: string[]; editField: string | null } } | null; records: Record<string, string>[] | null; state: unknown; budget: unknown };

export function ApprovalReview({ releaseId, target, namespace, user, session, disabled, onAction, onResult }: {
  releaseId: string; target: string; namespace: string; user: string; session: string; disabled: boolean;
  onAction: (title: string, action: () => Promise<void>) => Promise<void>; onResult: (trace: any) => void;
}) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [edit, setEdit] = useState("");
  const inspect = async () => {
    const next = await api.get<Snapshot>(`/api/production/releases/${releaseId}/approval?user=${encodeURIComponent(user)}&session=${encodeURIComponent(session)}`);
    setSnapshot(next); setEdit(typeof next.pending?.value.proposed === "string" ? next.pending.value.proposed : "");
  };
  const resume = (action: string) => onAction("Resuming reviewed native checkpoint", async () => {
    if (!snapshot?.head || !snapshot.pending || !snapshot.records) return;
    const trace = await api.post<any>(`/api/serve/${target}/${namespace}/resume`, {
      requestId: crypto.randomUUID(), records: snapshot.records, user, session, expectedRelease: releaseId,
      approval: { expectedRevision: snapshot.head.revision, expectedCheckpointSha256: snapshot.head.checkpointSha256,
        interruptId: snapshot.pending.id, action, ...(action === "edit" ? { value: edit } : {}) },
    });
    onResult(trace); await inspect();
  });
  return <section aria-label="production approval review">
    <h4>Native approval checkpoint</h4>
    <button disabled={disabled} onClick={() => onAction("Reading committed approval checkpoint", inspect)}>Inspect pending approval</button>
    <p>Review the committed session checkpoint before choosing a decision. Resuming preserves the paused turn inputs. Native state and the review persist even when trace capture is off.</p>
    {snapshot && <>
      <p>Revision {snapshot.head?.revision ?? 0} · {snapshot.pending ? "pending review" : "no pending review"}</p>
      {snapshot.pending && <>
        <p>{snapshot.pending.value.prompt}</p><pre>{JSON.stringify(snapshot.pending.value.proposed, null, 2)}</pre>
        {snapshot.pending.value.actions.includes("edit") && <label>Edited proposal <textarea aria-label="approval edit value" maxLength={2000} value={edit} onChange={e => setEdit(e.target.value)} /></label>}
        {snapshot.pending.value.actions.map(action => <button key={action} disabled={disabled} onClick={() => resume(action)}>{action === "approve" ? "Approve and resume" : action === "reject" ? "Reject and resume" : "Edit and resume"}</button>)}
      </>}
      <details><summary>Reviewed checkpoint, state and active-turn budget</summary><pre>{JSON.stringify(snapshot, null, 2)}</pre></details>
    </>}
  </section>;
}
