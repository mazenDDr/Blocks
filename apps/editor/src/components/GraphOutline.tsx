import { useMemo, useState } from "react";
import { graphOutline } from "../graphOutline";
import type { Graph, OpInfo, Validation } from "../types";

export function GraphOutline({ graph, ops, validation, pending, error, selected, scope, onInspect, onCenter, onOpenModule }: {
  graph: Graph; ops: Record<string, OpInfo>; validation: Validation | null; pending: boolean; error: string | null;
  selected: string[]; scope: string; onInspect: (id: string) => void; onCenter: (id: string) => void; onOpenModule: (id: string) => void;
}) {
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
    {rows.length ? <table className="outline-table"><caption>Graph nodes, declared wires and native validation</caption><thead><tr><th>Node / operation</th><th>Connections / references</th><th>Native validation</th><th>Navigate</th></tr></thead><tbody>{rows.map(row => <tr key={row.id} className={selected.includes(row.id) ? "outline-selected" : ""}>
      <th scope="row">{row.id}<br />{row.title}<br /><code>{row.type}</code></th>
      <td>{row.module && <div>Module {row.module}</div>}{row.sharing && <div>Shares with {row.sharing}</div>}
        {row.incoming.map((wire, i) => <div key={`in${i}`}>In: {wire}</div>)}{row.outgoing.map((wire, i) => <div key={`out${i}`}>Out: {wire}</div>)}
        {!row.incoming.length && !row.outgoing.length && <div>No declared wires.</div>}</td>
      <td>{row.diagnostics.map((diagnostic, i) => <div key={i} className={diagnostic.severity === "error" ? "errbadge" : "warn"}>{diagnostic.code} · {diagnostic.nodeId ?? "graph"}{diagnostic.port ? `.${diagnostic.port}` : ""}: {diagnostic.message}</div>)}
        {row.nativeView ? <><div>{row.nativeView.typed ? "typed" : "unresolved"}{row.nativeView.params != null ? ` · ${row.nativeView.params} parameters` : ""}</div>
          <details><summary>Native port contracts</summary><pre>{JSON.stringify({inputs: row.nativeView.inputShapes ?? null, outputs: row.nativeView.outputShapes ?? null}, null, 2)}</pre></details></> : <div>Native node contracts unavailable.</div>}</td>
      <td><button aria-label={`inspect outline ${row.id}`} aria-pressed={selected.includes(row.id)} onClick={() => onInspect(row.id)}>Inspect node</button>{" "}
        <button aria-label={`center outline ${row.id}`} onClick={() => onCenter(row.id)}>Center node</button>{" "}
        {row.module && <button aria-label={`module outline ${row.id}`} onClick={() => onOpenModule(row.id)}>Open module</button>}</td>
    </tr>)}</tbody></table> : <p>No nodes match these filters.{pending && " Pending validation supplies no current native errors."}</p>}
    {result.globalDiagnostics.length > 0 && <details><summary>Graph diagnostics without a visible node ({result.globalDiagnostics.length})</summary>{result.globalDiagnostics.map((diagnostic, i) => <p key={i}>{diagnostic.code} · {diagnostic.nodeId ?? "graph"}: {diagnostic.message}</p>)}</details>}
  </details>;
}
