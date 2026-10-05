import { useMemo, useState } from "react";
import { agentOutline } from "../../agentOutline";
import type { Graph, OpInfo, Validation } from "../../types";
export function AgentOutline({ graph, ops, validation, pending, error, selected, onInspect }: {
  graph: Graph; ops: Record<string, OpInfo>; validation: Validation | null; pending: boolean; error: string | null;
  selected: string | null; onInspect: (id: string) => void;
}) {
  const [query, setQuery] = useState(""), [errorsOnly, setErrorsOnly] = useState(false), [page, setPage] = useState(0);
  const result = useMemo(() => agentOutline(graph, ops, validation, pending || !!error, query, errorsOnly), [graph, ops, validation, pending, error, query, errorsOnly]);
  const last = Math.max(0, Math.ceil(result.rows.length / 50) - 1), current = Math.min(page, last), rows = result.rows.slice(current * 50, (current + 1) * 50);
  return <details className="graph-outline agent-outline"><summary>Structured agent outline</summary>
    <p className="provenance">{pending ? "Native agent validation pending; prior analysis and diagnostics withheld." : error ? `Native agent validation unavailable: ${error}` : result.graphHash ? `Native agent validation identity ${result.graphHash}` : "Native agent validation has not been returned."} Control wires, routes and joins below are declared graph structure. Reads/writes/effects are computed native contracts, with no recorded execution values or taken route inferred.</p>
    <label>Find node, operation, state field, effect, route or diagnostic <input aria-label="agent outline search" maxLength={200} value={query} onChange={e => { setQuery(e.target.value);setPage(0); }} /></label>{" "}
    <label><input type="checkbox" aria-label="agent outline native errors only" checked={errorsOnly} onChange={e => { setErrorsOnly(e.target.checked);setPage(0); }} /> Native errors only</label>
    <p>{result.rows.length} matching nodes · page {current + 1} of {last + 1} · at most 50 rows/page. START/END remain declared terminal references, not editable nodes.</p>
    <button disabled={current === 0} onClick={() => setPage(current - 1)}>Previous agent outline nodes</button>{" "}<button disabled={current === last} onClick={() => setPage(current + 1)}>Next agent outline nodes</button>
    {rows.length ? <table className="outline-table"><caption>Agent nodes, declared control/state references and native analysis</caption><thead><tr><th>Node / operation</th><th>Declared control references</th><th>Native contracts / diagnostics</th><th>Navigate</th></tr></thead><tbody>{rows.map(row => <tr key={row.id} className={selected === row.id ? "outline-selected" : ""}>
      <th scope="row">{row.id}<br />{row.title}<br /><code>{row.type}</code></th>
      <td>{row.incoming.map((wire,i)=><div key={`in${i}`}>In: {wire}</div>)}{row.outgoing.map((wire,i)=><div key={`out${i}`}>Out: {wire}</div>)}
        {!!row.routes.length && <details><summary>Declared conditional routes</summary><pre>{JSON.stringify(row.routes,null,2)}</pre></details>}{!!row.joins.length && <details><summary>Declared joins</summary><pre>{JSON.stringify(row.joins,null,2)}</pre></details>}</td>
      <td>{row.native ? <><div>Reads: {row.native.reads ? row.native.reads.join(", ") || "none" : "not returned"}</div><div>Writes: {row.native.writes ? row.native.writes.join(", ") || "none" : "not returned"}</div><div>Effects: {row.native.effects ? row.native.effects.join(", ") || "none" : "not returned"}</div>{row.native.fixtureModel && <div>Scripted FIXTURE model; no real model-quality claim.</div>}</> : <div>Native node analysis unavailable.</div>}
        {row.diagnostics.map((d,i)=><div key={i} className={d.severity === "error" ? "errbadge" : "warn"}>{d.code} · {d.nodeId ?? "graph"}: {d.message}</div>)}</td>
      <td><button aria-label={`inspect agent outline ${row.id}`} aria-pressed={selected === row.id} disabled={graph.nodes.filter(n=>n.id===row.id).length!==1} onClick={() => onInspect(row.id)}>Inspect agent node</button></td>
    </tr>)}</tbody></table> : <p>No nodes match these filters.</p>}
    {!!result.globalDiagnostics.length && <details><summary>Agent diagnostics without a visible node ({result.globalDiagnostics.length})</summary>{result.globalDiagnostics.map((d,i)=><p key={i}>{d.code}: {d.message}</p>)}</details>}
  </details>;
}
