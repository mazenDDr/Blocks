import { useState } from "react";
import type { GNode } from "../types";
import { readAnnotation, type NodeAnnotation } from "../nodeAnnotations";

export function NodeComment({ node, value, identity, pending, onSave, onRemove }: {
  node: GNode; value: unknown; identity: { kind: "graph" | "module"; hash: string } | null; pending: boolean;
  onSave: (note: NodeAnnotation) => string | null; onRemove: () => string | null;
}) {
  let note: NodeAnnotation | null = null, malformed: string | null = null;
  try { note = readAnnotation(value); } catch (error) { malformed = String(error); }
  const [text, setText] = useState(note?.text ?? "");
  const [author, setAuthor] = useState(note?.author ?? "");
  const [error, setError] = useState<string | null>(null);
  const current = note && identity && note.source.hash === identity.hash && note.source.kind === identity.kind && note.source.nodeId === node.id && note.source.nodeType === node.type && note.source.nodeVersion === node.version;
  const changed = text.trim() !== (note?.text ?? "") || author.trim() !== (note?.author ?? "");
  const needsReview = !!note && !!identity && !current;
  return <details className="node-comment"><summary>Node comment{note ? " · saved" : ""}</summary>
    <p>User-authored observations for this node. Author labels and browser time are declarations; comments are separate from measured values.</p>
    {note && <div className="provenance"><p>Authored by {note.author || "unspecified author"} at <time>{note.authoredAt}</time>.</p>
      <p>Captured {note.source.kind} identity <code>{note.source.hash}</code>, node {note.source.nodeId} · {note.source.nodeType}@{note.source.nodeVersion}.</p>
      <p>{pending || !identity ? "Current native identity unavailable while validation is pending." : current ? "Comment refers to the current native identity." : "Comment refers to an earlier node or native identity. Review it before updating."}</p></div>}
    {malformed ? <p role="alert">{malformed}</p> : <>
      <label>Author label<input aria-label="node comment author" maxLength={120} value={author} onChange={e => setAuthor(e.target.value)} /></label>
      <label>Comment<textarea aria-label="node comment text" maxLength={4000} value={text} onChange={e => setText(e.target.value)} /></label>
      <p className="hint">{text.length}/4000 characters. Apply captures the reviewed native identity in the draft; Save project persists it. Layout changes do not change its native identity.</p>
      <button disabled={!text.trim() || (!changed && !needsReview) || pending || !identity} onClick={() => {
        if (!identity) return;
        setError(onSave({ text: text.trim(), author: author.trim(), authoredAt: new Date().toISOString(), source: { nodeId: node.id, nodeType: node.type, nodeVersion: node.version, ...identity } }));
      }}>Apply node comment</button>
    </>}
    {value !== undefined && <button onClick={() => setError(onRemove())}>Remove node comment</button>}
    {error && <p role="alert">{error}</p>}
  </details>;
}
