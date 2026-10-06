import { NodeChecklist } from "./NodeChecklist";
import { useState } from "react";
import type { Graph } from "../types";
import type { GraphClipboard } from "../graphClipboard";

export function GraphClipboardTools({ graph, selected, clipboard, inModule, blocked, onSelect, onCopy, onPaste, onClear }: {
  graph: Graph; selected: string[]; clipboard: GraphClipboard | null; inModule: boolean; blocked?: string | null;
  onSelect: (ids: string[]) => void; onCopy: () => Promise<void>; onPaste: () => void; onClear: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const compatible = clipboard && graph.graphKind === clipboard.graphKind && graph.backend === clipboard.backend;
  return <details className="clipboard-tools"><summary>Copy and paste graph nodes</summary>
    <p className="hint">Copies draft configuration and internal wires into page memory. Boundary wires stay disconnected. Parameter sharing within the selection and model/&lt;this-node-id&gt; draft references are rebound to new nodes. Native weights, files and opaque state are not copied. Reloading clears this clipboard.</p>
    {inModule && <p>Copies stored module-internal nodes. Inputs/output interfaces stay unchanged; pasting edits the shared definition used by every instance. Boundary and generated nodes are excluded.</p>}
    {blocked && <p role="status">{blocked}</p>}
      <fieldset disabled={busy || !!blocked}><legend>Select {inModule ? "module internal" : "root"} nodes to copy ({selected.filter(id => graph.nodes.some(n => n.id === id)).length})</legend>
        <button disabled={!graph.nodes.length || graph.nodes.length > 100} onClick={() => onSelect(graph.nodes.map(n => n.id))}>Select all for copy</button>{" "}
        <button onClick={() => onSelect([])}>Clear copy selection</button>
        <NodeChecklist nodes={graph.nodes} selected={selected} label="copy node" onToggle={(id, checked) => onSelect(checked ? [...selected, id] : selected.filter(x => x !== id))} />
        <button disabled={!selected.length || selected.length > 100 || selected.some(id => graph.nodes.filter(n => n.id === id).length !== 1)} onClick={async () => { setBusy(true); try { await onCopy(); } finally { setBusy(false); } }}>Copy selected nodes</button>
      </fieldset>
    {clipboard && <div className="clipboard-provenance">
      <p>Copied from <b>{clipboard.sourceProject}</b>{clipboard.sourceScope && <> · module {clipboard.sourceScope.id}@{clipboard.sourceScope.version}</>} · {clipboard.graphKind}/{clipboard.backend} · native {clipboard.sourceScope ? "module" : "graph"} identity <code>{clipboard.sourceGraphHash}</code>.</p>
      <p>{clipboard.nodes.length} nodes, {clipboard.edges.length} internal wires, {clipboard.modules.length} modules, {clipboard.codeBlocks.length} code definitions. {clipboard.omittedBoundaryEdges} boundary wires omitted. Same-version definitions must match exactly. At most 100 nodes/100 definitions and 512 KiB estimated JSON.</p>
      <button disabled={busy || !!blocked || !compatible} onClick={onPaste}>Paste copied nodes</button>{" "}<button disabled={busy} onClick={onClear}>Clear graph clipboard</button>
      {!compatible && <p>Paste requires the copied graph kind and backend. Open a compatible graph to paste this snapshot.</p>}
    </div>}
  </details>;
}
