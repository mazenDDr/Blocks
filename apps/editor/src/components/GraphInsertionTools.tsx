import { useState } from "react";
import { insertionOperations } from "../graphInsertion";
import type { Graph, OpInfo } from "../types";

export function GraphInsertionTools({ graph, ops, selectedWire, blocked, onWire, onInsert }: {
  graph: Graph; ops: OpInfo[]; selectedWire: string; blocked: string | null;
  onWire: (id: string) => void; onInsert: (edge: string, op: OpInfo, input: string, output: string) => void;
}) {
  const [operation, setOperation] = useState("");
  const [input, setInput] = useState(""); const [output, setOutput] = useState("");
  const choices = insertionOperations(graph, ops), op = choices.find(o => o.type === operation);
  const edge = graph.edges.find(e => e.id === selectedWire);
  const mismatch = !!op && !!edge && !!input && !!output && (op.inputKinds[input] !== edge.kind || op.outputKinds[output] !== edge.kind);
  return <details className="insertion-tools"><summary>Insert block into a wire</summary>
    <p>Choose an existing wire, operation and its two ports. One Undo restores the original graph and layout. Configure the inserted node and inspect native validation before running; additional inputs stay disconnected.</p>
    <div className="row">
      <label>Existing wire <select aria-label="insertion wire" value={edge?.id ?? ""} onChange={e => onWire(e.target.value)}><option value="">choose…</option>{graph.edges.map(e => <option key={e.id} value={e.id}>{e.from.node}.{e.from.port} → {e.to.node}.{e.to.port} · {e.kind}</option>)}</select></label>
      <label>Installed operation <select aria-label="insertion operation" value={op?.type ?? ""} onChange={e => { setOperation(e.target.value); setInput(""); setOutput(""); }}><option value="">choose…</option>{choices.map(o => <option key={o.type} value={o.type}>{o.displayName} · {o.type}@{o.version} · {o.backend}</option>)}</select></label>
      <label>Input <select aria-label="insertion input" value={input} onChange={e => setInput(e.target.value)}><option value="">choose…</option>{op?.inputs.map(p => <option key={p} value={p}>{p} · {op.inputKinds[p]}</option>)}</select></label>
      <label>Output <select aria-label="insertion output" value={output} onChange={e => setOutput(e.target.value)}><option value="">choose…</option>{op?.outputs.map(p => <option key={p} value={p}>{p} · {op.outputKinds[p]}</option>)}</select></label>
      <button disabled={!!blocked || !edge || !op || !op.inputs.includes(input) || !op.outputs.includes(output) || mismatch} onClick={() => { if (edge && op) onInsert(edge.id, op, input, output); }}>Insert selected block</button>
    </div>
    {op && <p>{op.purpose} Registered defaults: <code>{JSON.stringify(op.defaults)}</code>.</p>}
    {mismatch && <p role="status">E_INSERT_KIND: Both ports must declare {edge?.kind}; choose another operation or ports.</p>}
    {blocked && <p role="status">{blocked}</p>}
  </details>;
}
