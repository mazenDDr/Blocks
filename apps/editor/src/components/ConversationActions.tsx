import { useRef, useState } from "react";

export interface ConversationSnapshot {
  releaseId: string;
  user: string;
  session: string;
  head: { revision: number; checkpointSha256: string | null; lastRequestId: string | null } | null;
  state: unknown;
}

type Operation = "reset" | "fork";

/** Actions target exactly the inspected scope/checkpoint, never an inferred latest head. */
export function ConversationActions({ snapshot, disabled, onApply }: {
  snapshot: ConversationSnapshot;
  disabled: boolean;
  onApply: (operation: Operation, body: Record<string, unknown>) => Promise<void>;
}) {
  const [destination, setDestination] = useState("alternative");
  const [reason, setReason] = useState("Explore a new continuation");
  const [reviewedReset, setReviewedReset] = useState(false);
  const attempts = useRef(new Map<string, string>());
  const head = snapshot.head;
  const apply = (operation: Operation) => {
    const values = { user: snapshot.user, session: snapshot.session, expectedRevision: head!.revision,
      expectedCheckpointSha256: head!.checkpointSha256, reason, ...(operation === "fork" ? { destinationSession: destination } : {}) };
    const key = JSON.stringify({ releaseId: snapshot.releaseId, operation, ...values });
    const id = attempts.current.get(key) ?? crypto.randomUUID();
    attempts.current.set(key, id); // Retrying identical reviewed values reuses the durable action receipt.
    return onApply(operation, { actionId: id, ...values });
  };
  if (!head?.checkpointSha256) return <p>This session has no live checkpoint. Its next successful turn starts from the graph’s declared defaults. Earlier traces and snapshots remain recorded.</p>;
  return <section aria-label="reviewed conversation actions">
    <h4>Manage the inspected conversation</h4>
    <p className="provenance">Release {snapshot.releaseId} · user {snapshot.user} · session {snapshot.session} · revision {head.revision} · checkpoint {head.checkpointSha256}</p>
    <p>Actions wait for active turns and refuse a changed checkpoint. Fork preserves this state in a new independent thread. Reset clears the live head; earlier snapshots, traces and captured replay remain available. Neither action invokes a model.</p>
    <label>Conversation action reason <input aria-label="conversation action reason" disabled={disabled} maxLength={500} value={reason} onChange={(e) => setReason(e.target.value)} /></label>
    <label>Fork destination session <input aria-label="fork destination session" disabled={disabled} maxLength={64} value={destination} onChange={(e) => setDestination(e.target.value)} /></label>
    <button disabled={disabled || !reason.trim() || !destination || destination === snapshot.session} onClick={() => apply("fork")}>Fork inspected checkpoint</button>
    <label><input aria-label="confirm conversation reset" type="checkbox" disabled={disabled} checked={reviewedReset} onChange={(e) => setReviewedReset(e.target.checked)} /> Start this session fresh on its next turn, retaining all earlier evidence</label>
    <button disabled={disabled || !reason.trim() || !reviewedReset} onClick={() => apply("reset")}>Reset inspected conversation</button>
  </section>;
}
