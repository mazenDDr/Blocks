import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
import { graphOutline } from "../graphOutline";
import type { OutlineRow } from "../graphOutline";
import type { Graph, OpInfo, Validation } from "../types";

type OutlineActions = { onInspect: (id: string) => void; onCenter: (id: string) => void; onOpenModule: (id: string) => void; onDiagnostic: (id: string) => void };

// Rows receive stable actions and primitive selection/inspectability, so selecting a node re-renders only the rows that changed.
const OutlineRowView = memo(function OutlineRowView({ row, selected, inspectable, actions }: { row: OutlineRow; selected: boolean; inspectable: string; actions: OutlineActions }) {
  return <tr className={selected ? "outline-selected" : ""}>
      <th scope="row">{row.id}<br />{row.title}<br /><code>{row.type}</code></th>
      <td>{row.module && <div>Module {row.module}</div>}{row.sharing && <div>Shares with {row.sharing}</div>}
        {row.incoming.map((wire, i) => <div key={`in${i}`}>In: {wire}</div>)}{row.outgoing.map((wire, i) => <div key={`out${i}`}>Out: {wire}</div>)}
        {!row.incoming.length && !row.outgoing.length && <div>No declared wires.</div>}</td>
      <td>{row.diagnostics.map((diagnostic, i) => <div key={i} className={diagnostic.severity === "error" ? "errbadge" : "warn"}>{diagnostic.code} · {diagnostic.nodeId ?? "graph"}{diagnostic.port ? `.${diagnostic.port}` : ""}: {diagnostic.message}
        {diagnostic.nodeId && inspectable[i] === "1" && <button aria-label={`inspect diagnostic ${diagnostic.nodeId} ${diagnostic.code}`} onClick={() => actions.onDiagnostic(diagnostic.nodeId!)}>Open diagnostic node</button>}</div>)}
        {row.nativeView ? <><div>{row.nativeView.typed ? "typed" : "unresolved"}{row.nativeView.params != null ? ` · ${row.nativeView.params} parameters` : ""}</div>
          <details><summary>Native port contracts</summary><pre>{JSON.stringify({inputs: row.nativeView.inputShapes ?? null, outputs: row.nativeView.outputShapes ?? null}, null, 2)}</pre></details></> : <div>Native node contracts unavailable.</div>}</td>
      <td><button aria-label={`inspect outline ${row.id}`} aria-pressed={selected} onClick={() => actions.onInspect(row.id)}>Inspect node</button>{" "}
        <button aria-label={`center outline ${row.id}`} onClick={() => actions.onCenter(row.id)}>Center node</button>{" "}
        {row.module && <button aria-label={`module outline ${row.id}`} onClick={() => actions.onOpenModule(row.id)}>Open module</button>}</td>
    </tr>;
});


export function GraphOutline({ graph, ops, validation, pending, error, selected, scope, onInspect, onCenter, onOpenModule, canInspectDiagnostic, onDiagnostic }: {
  graph: Graph; ops: Record<string, OpInfo>; validation: Validation | null; pending: boolean; error: string | null;
  selected: string[]; scope: string; onInspect: (id: string) => void; onCenter: (id: string) => void; onOpenModule: (id: string) => void;
  canInspectDiagnostic: (id: string) => boolean; onDiagnostic: (id: string) => void;
}) {
  const latest = useRef({ onInspect, onCenter, onOpenModule, onDiagnostic });
  useLayoutEffect(() => { latest.current = { onInspect, onCenter, onOpenModule, onDiagnostic }; });
  const actions = useMemo<OutlineActions>(() => ({ onInspect: id => latest.current.onInspect(id), onCenter: id => latest.current.onCenter(id),
    onOpenModule: id => latest.current.onOpenModule(id), onDiagnostic: id => latest.current.onDiagnostic(id) }), []);
  const [query, setQuery] = useState(""); const [errorsOnly, setErrorsOnly] = useState(false); const [page, setPage] = useState(0);
  const result = useMemo(() => graphOutline(graph, ops, validation, pending || !!error, query, errorsOnly), [graph, ops, validation, pending, error, query, errorsOnly]);
  const last = Math.max(0, Math.ceil(result.rows.length / 50) - 1); const current = Math.min(page, last);
  const rows = result.rows.slice(current * 50, (current + 1) * 50);
  return <details className="graph-outline"><summary>Structured graph outline</summary>
    <p className="provenance">{scope} · {pending ? "Native validation pending; prior shapes and diagnostics withheld." : error ? `Native validation unavailable: ${error}` : result.graphHash ? `Native validation identity ${result.graphHash}` : "Native validation has not been returned."} Values below describe graph structure and native validation, not recorded learned activations.</p>
    <label>Find node, operation, wire, module or diagnostic <input aria-label="outline search" maxLength={200} value={query} onChange={e => { setQuery(e.target.value); setPage(0); }} /></label>{" "}
    <label><input aria-label="outline native errors only" type="checkbox" checked={errorsOnly} onChange={e => { setErrorsOnly(e.target.checked); setPage(0); }} /> Native errors only</label>
    <p>{result.rows.length} matching nodes · page {current + 1} of {last + 1} · at most 50 rows per page. Node order follows this graph document; no automatic layout changes.</p>
    <button disabled={current === 0} onClick={() => setPage(current - 1)}>Previous outline nodes</button>{" "}<button disabled={current === last} onClick={() => setPage(current + 1)}>Next outline nodes</button>
    {rows.length ? <table className="outline-table"><caption>Graph nodes, declared wires and native validation</caption><thead><tr><th>Node / operation</th><th>Connections / references</th><th>Native validation</th><th>Navigate</th></tr></thead><tbody>{rows.map(row => <OutlineRowView key={row.id} row={row} selected={selected.includes(row.id)} actions={actions}
      inspectable={row.diagnostics.map(d => d.nodeId && canInspectDiagnostic(d.nodeId) ? "1" : "0").join("")} />)}</tbody></table> : <p>No nodes match these filters.{pending && " Pending validation supplies no current native errors."}</p>}
    {result.globalDiagnostics.length > 0 && <details><summary>Graph diagnostics without a visible node ({result.globalDiagnostics.length})</summary>{result.globalDiagnostics.map((diagnostic, i) => <p key={i}>{diagnostic.code} · {diagnostic.nodeId ?? "graph"}: {diagnostic.message}
      {diagnostic.nodeId && canInspectDiagnostic(diagnostic.nodeId) && <button aria-label={`inspect diagnostic ${diagnostic.nodeId} ${diagnostic.code}`} onClick={() => onDiagnostic(diagnostic.nodeId!)}>Open diagnostic node</button>}</p>)}</details>}
  </details>;
}
