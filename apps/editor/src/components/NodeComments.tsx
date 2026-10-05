import { useState } from "react";
import type { GNode } from "../types";
import { annotationsOf, readAnnotation, type NodeAnnotation } from "../nodeAnnotations";
import { NodeComment } from "./NodeComment";

export function NodeComments({ nodes, selected, value, prefix, identity, pending, onSelect, onSave, onRemove, onRemoveMalformed }: {
  nodes: GNode[]; selected: string[]; value: unknown; prefix: string;
  identity: { kind: "graph" | "module"; hash: string } | null; pending: boolean;
  onSelect: (id: string) => void; onSave: (key: string, note: NodeAnnotation) => string | null;
  onRemove: (key: string) => string | null; onRemoveMalformed: () => void;
}) {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  let notes: Record<string, unknown> = {}, malformed: string | null = null;
  try { notes = annotationsOf(value); } catch (e) { malformed = String(e); }
  const rows = Object.entries(notes).filter(([key, raw]) => {
    const local = prefix ? key.startsWith(prefix) : !key.includes("/");
    return local && (!search || `${key} ${JSON.stringify(raw)}`.toLowerCase().includes(search.toLowerCase()));
  });
  const currentPage = Math.min(page, Math.max(0, Math.ceil(rows.length / 25) - 1));
  const node = selected.length === 1 ? nodes.find(n => n.id === selected[0]) : undefined;
  const nodeValue = node && Object.hasOwn(notes, prefix + node.id) ? notes[prefix + node.id] : undefined;
  return <details className="node-comments"><summary>Node comments ({rows.length} in this scope)</summary>
    <p>Saved user-authored notes, separate from execution results. Select a node to add or review its comment. Module notes belong to that definition; root notes belong to their node. Comments stay with their source nodes when copying a graph selection.</p>
    {malformed ? <><p role="alert">{malformed}</p><button onClick={onRemoveMalformed}>Remove malformed comments collection</button></> : <>
      <label>Find comments<input aria-label="find node comments" value={search} onChange={e => { setSearch(e.target.value); setPage(0); }} /></label>
      <table><caption>Comments in {prefix || "root graph"}; page {currentPage + 1}</caption><thead><tr><th>Node</th><th>Authored comment</th><th>Actions</th></tr></thead>
        <tbody>{rows.slice(currentPage * 25, currentPage * 25 + 25).map(([key, raw]) => {
          const id = key.slice(prefix.length), present = nodes.some(n => n.id === id);
          let note: NodeAnnotation | null = null, problem: string | null = null;
          try { note = readAnnotation(raw); } catch (e) { problem = String(e); }
          return <tr key={key}><th>{id}{!present && <small> · node no longer present</small>}</th><td>{problem ? <span role="alert">{problem}</span> : <><pre>{note?.text}</pre><small>Author: {note?.author || "unspecified"}; authored {note?.authoredAt}</small></>}</td>
            <td><button aria-label={`inspect comment ${id}`} disabled={!present} onClick={() => onSelect(id)}>Inspect comment</button>{" "}<button aria-label={`remove comment ${id}`} onClick={() => setError(onRemove(key))}>Remove comment</button></td></tr>;
        })}</tbody></table>
      <button disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>Previous comments</button>{" "}
      <button disabled={(currentPage + 1) * 25 >= rows.length} onClick={() => setPage(currentPage + 1)}>Next comments</button>
      {node && <NodeComment key={`${prefix}${node.id}:${JSON.stringify(nodeValue)}`} node={node} value={nodeValue} identity={identity} pending={pending}
        onSave={note => onSave(prefix + node.id, note)} onRemove={() => onRemove(prefix + node.id)} />}
      {!node && <p>Select exactly one node in the current graph to edit its comment.</p>}
    </>}
    {error && <p role="alert">{error}</p>}
  </details>;
}
