import { useMemo, useState } from "react";
import type { Graph, OpInfo } from "../types";

export function KeyboardGraphTools({ graph, ops, selectedNode, selectedWire, onNode, onWire, onConnect }: {
  graph: Graph; ops: Record<string, OpInfo>; onNode: (id: string) => void; onWire: (id: string) => void;
  selectedNode: string; selectedWire: string;
  onConnect: (target: string, port: string, source: { node: string; port: string }) => void;
}) {
  const [source, setSource] = useState(""); const [target, setTarget] = useState("");
  // Option lists depend only on the graph, so selecting a node reuses them instead of rebuilding every entry.
  const { outputs, inputs, nodeOptions, wireOptions, outputOptions, inputOptions } = useMemo(() => {
    const outputs = graph.nodes.flatMap((n) => (ops[n.type]?.outputs ?? []).map((p) => ({ key: `${n.id}:${p}`, node: n.id, port: p })));
    const inputs = graph.nodes.flatMap((n) => (ops[n.type]?.inputs ?? []).map((p) => ({ key: `${n.id}:${p}`, node: n.id, port: p })));
    return { outputs, inputs,
      nodeOptions: graph.nodes.map((n) => <option key={n.id} value={n.id}>{n.id} · {n.type}</option>),
      wireOptions: graph.edges.map((e) => <option key={e.id} value={e.id}>{e.from.node}.{e.from.port} → {e.to.node}.{e.to.port}</option>),
      outputOptions: outputs.map((p) => <option key={p.key} value={p.key}>{p.key}</option>),
      inputOptions: inputs.map((p) => <option key={p.key} value={p.key}>{p.key}</option>) };
  }, [graph.nodes, graph.edges, ops]);
  const s = outputs.find((p) => p.key === source); const t = inputs.find((p) => p.key === target);
  return <details className="keyboard-tools"><summary>Keyboard graph tools</summary><div className="row">
    <label>Inspect node <select aria-label="inspect graph node" value={selectedNode} onChange={(e) => onNode(e.target.value)}><option value="">choose…</option>{nodeOptions}</select></label>
    <label>Inspect wire <select aria-label="inspect graph wire" value={selectedWire} onChange={(e) => onWire(e.target.value)}><option value="">choose…</option>{wireOptions}</select></label>
    <label>Source port <select aria-label="connection source" value={source} onChange={(e) => setSource(e.target.value)}><option value="">choose…</option>{outputOptions}</select></label>
    <label>Target port <select aria-label="connection target" value={target} onChange={(e) => setTarget(e.target.value)}><option value="">choose…</option>{inputOptions}</select></label>
    <button disabled={!s || !t || s.node === t.node} onClick={() => { if (s && t) onConnect(t.node, t.port, { node: s.node, port: s.port }); }}>Connect selected ports</button>
  </div><p className="hint">Use Tab and Enter to create, select, configure, connect and inspect. Connecting replaces the selected input's prior wire; validation reports incompatible types.</p></details>;
}
