import { useState } from "react";
import { api, errorText } from "../../api";
import { copyAgentNodes, pasteAgentNodes, type AgentClipboard, type AgentNodeAccess } from "../../agentClipboard";
import type { Graph, UiDoc } from "../../types";
import { NodeChecklist } from "../NodeChecklist";

export function AgentClipboardTools({ projectId, graph, ui, clipboard, setClipboard, setGraph, setUi, onPasted, setMessage }: {
  projectId: string; graph: Graph; ui: UiDoc; clipboard: AgentClipboard | null; setClipboard: (c: AgentClipboard | null) => void;
  setGraph: (f: (g: Graph) => Graph) => void; setUi: (f: (u: UiDoc) => UiDoc) => void; onPasted: (ids: string[]) => void; setMessage: (m: string) => void;
}) {
  const [selected, setSelected] = useState<string[]>([]); const [busy, setBusy] = useState(false); const [pastes, setPastes] = useState(0);
  const ids = selected.filter(id => graph.nodes.some(n => n.id === id));
  const copy = async () => {
    setBusy(true);
    try {
      const snapshot = structuredClone(graph), layout = structuredClone(ui), selection = [...ids];
      // State access comes from native validation of this exact snapshot, never from the editor's guess.
      const result = await api.validate(snapshot);
      setClipboard(copyAgentNodes(snapshot, layout, selection, (result.nodes ?? {}) as AgentNodeAccess, projectId, result.graphHash)); setPastes(0);
      setMessage(`Copied ${selection.length} agent nodes from '${projectId}'. Native validation remains required after paste.`);
    } catch (error) { setMessage(errorText(error)); } finally { setBusy(false); }
  };
  const paste = () => {
    if (!clipboard) return;
    try {
      const result = pasteAgentNodes(graph, ui, clipboard, 80 * (pastes + 1));
      setGraph(() => result.graph); setUi(() => result.ui); setSelected(result.selected); setPastes(n => n + 1); onPasted(result.selected);
      const o = clipboard.omitted;
      setMessage(`Pasted ${result.selected.length} agent nodes with fresh IDs. ${o.transitions} transitions, ${o.routes} routes and ${o.joins} joins crossing the selection were not copied; inspect validation before running.`);
    } catch (error) { setMessage(errorText(error)); }
  };
  return <details className="clipboard-tools agent-clipboard"><summary>Copy and paste agent nodes</summary>
    <p className="hint">Copies node configuration, transitions and conditional routes inside the selection (END stays END), joins whose branches are all selected, and the exact state fields, indexes and memory policies the nodes use according to native validation. Transitions, routes and joins crossing the selection are left out, never retargeted. Pasting into another agent graph adds missing definitions and refuses differing ones. Runs, memory records and index contents are not copied. Reloading clears this clipboard.</p>
    <fieldset disabled={busy}><legend>Select agent nodes to copy ({ids.length})</legend>
      <button disabled={!graph.nodes.length || graph.nodes.length > 100} onClick={() => setSelected(graph.nodes.map(n => n.id))}>Select all agent nodes for copy</button>{" "}
      <button onClick={() => setSelected([])}>Clear agent copy selection</button>
      <NodeChecklist nodes={graph.nodes} selected={ids} label="copy agent node" onToggle={(id, checked) => setSelected(checked ? [...ids, id] : ids.filter(x => x !== id))} />
      <button disabled={!ids.length || ids.length > 100} onClick={() => void copy()}>Copy selected agent nodes</button>
    </fieldset>
    {clipboard && <div className="clipboard-provenance">
      <p>Copied from <b>{clipboard.sourceProject}</b> · native graph identity <code>{clipboard.sourceGraphHash}</code>.</p>
      <p>{clipboard.nodes.length} nodes, {clipboard.edges.length} transitions, {clipboard.routes.length} routes, {clipboard.joins.length} joins, {clipboard.state.length} state fields, {clipboard.indexes.length} indexes, {clipboard.policies.length} memory policies. Left out: {clipboard.omitted.transitions} transitions, {clipboard.omitted.routes} routes, {clipboard.omitted.joins} joins.</p>
      <button disabled={busy} onClick={paste}>Paste copied agent nodes</button>{" "}<button disabled={busy} onClick={() => { setClipboard(null); setPastes(0); }}>Clear agent clipboard</button>
    </div>}
  </details>;
}
