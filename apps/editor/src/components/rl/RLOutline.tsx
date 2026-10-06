import { useMemo, useRef, useState } from "react";
import { rlOutline } from "../../rlOutline";
import type { Graph, OpInfo, Validation } from "../../types";

export function RLOutline({ graph, ops, validation, pending, error, onOpenPanel }: {
  graph: Graph; ops: Record<string, OpInfo>; validation: Validation | null; pending: boolean; error: string | null;
  onOpenPanel: (tab: "env" | "learner") => void;
}) {
  const [query, setQuery] = useState(""), [errorsOnly, setErrorsOnly] = useState(false), [page, setPage] = useState(0), [selected, setSelected] = useState<string | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const inspect = (id: string) => { setSelected(id); requestAnimationFrame(() => { if (heading.current?.dataset.node === id) heading.current.focus(); }); };
  const result = useMemo(() => rlOutline(graph, ops, validation, pending || !!error, query, errorsOnly), [graph, ops, validation, pending, error, query, errorsOnly]);
  const last = Math.max(0, Math.ceil(result.rows.length / 50) - 1), current = Math.min(page, last), rows = result.rows.slice(current * 50, (current + 1) * 50);
  const matches = result.allRows.filter(row => row.id === selected), inspected = matches.length === 1 && matches[0].node ? matches[0] : null;
  return <div className="scroll pad rl-outline">
    <h3>Structured RL outline</h3>
    <p className="provenance">{pending ? "Native RL validation pending; prior contracts and diagnostics withheld." : error ? `Native RL validation unavailable: ${error}` : result.graphHash ? `Native RL validation identity ${result.graphHash}` : "Native RL validation has not been returned."} Wires and configuration are declared structure. Native contracts describe observation/action spaces, rewards and the learner; no trajectory, trained Q-values or update measurements are inferred.</p>
    <label>Find node, operation, wire, configuration, native contract or diagnostic <input aria-label="rl outline search" maxLength={200} value={query} onChange={e => { setQuery(e.target.value); setPage(0); }} /></label>{" "}
    <label><input type="checkbox" aria-label="rl outline native errors only" checked={errorsOnly} onChange={e => { setErrorsOnly(e.target.checked); setPage(0); }} /> Native errors only</label>
    <p>{result.rows.length} matching nodes · page {current + 1} of {last + 1} · at most 50 rows/page.</p>
    <button disabled={current === 0} onClick={() => setPage(current - 1)}>Previous RL outline nodes</button>{" "}<button disabled={current === last} onClick={() => setPage(current + 1)}>Next RL outline nodes</button>
    {rows.length ? <table className="outline-table"><caption>RL nodes, declared wires and current native contracts</caption><thead><tr><th>Node / operation</th><th>Declared wires</th><th>Native status / diagnostics</th><th>Inspect</th></tr></thead><tbody>{rows.map((row, i) => <tr key={`${row.id}:${i}`} className={selected === row.id ? "outline-selected" : ""}>
      <th scope="row">{row.id}<br />{row.title}<br /><code>{row.type}</code></th>
      <td>{row.incoming.map((wire,i)=><div key={`in${i}`}>In: {wire}</div>)}{row.outgoing.map((wire,i)=><div key={`out${i}`}>Out: {wire}</div>)}</td>
      <td>{row.nativeView ? <div>Native typing: {row.nativeView.typed ? "typed" : "not typed"}{row.nativeView.params != null && <> · {row.nativeView.params} parameters</>}</div> : <div>Native node contracts unavailable.</div>}{row.diagnostics.map((d,i)=><div key={i} className={d.severity === "error" ? "errbadge" : "warn"}>{d.code} · {d.nodeId ?? "graph"}: {d.message}</div>)}</td>
      <td><button aria-label={`inspect rl outline ${row.id}`} aria-pressed={selected === row.id} disabled={!row.node} onClick={() => inspect(row.id)}>Inspect RL node</button></td>
    </tr>)}</tbody></table> : <p>No nodes match these filters.</p>}
    {!!result.globalDiagnostics.length && <details><summary>RL diagnostics without a visible node ({result.globalDiagnostics.length})</summary>{result.globalDiagnostics.map((d,i)=><p key={i}>{d.code}: {d.message}</p>)}</details>}
    {inspected && <section aria-label="RL node inspector">
      <h4 ref={heading} tabIndex={-1} data-node={inspected.id}>RL node inspector · {inspected.id}</h4><code>{inspected.type}</code>
      <p>Exact stored node configuration and current native per-node validation. Navigation does not change the graph, layout or draft history.</p>
      {inspected.panel && <button onClick={() => onOpenPanel(inspected.panel!)}>Open {inspected.panel === "env" ? "Environment" : "Learner"} controls for {inspected.id}</button>}
      <details><summary>Declared configuration</summary><pre aria-label="rl declared configuration">{JSON.stringify(inspected.node!.config,null,2)}</pre></details>
      {inspected.nativeView ? <><h4>Native input contracts</h4><pre aria-label="rl native input contracts">{inspected.nativeView.inputShapes ? JSON.stringify(inspected.nativeView.inputShapes,null,2) : "not returned"}</pre>
        <h4>Native output contracts</h4><pre aria-label="rl native output contracts">{inspected.nativeView.outputShapes ? JSON.stringify(inspected.nativeView.outputShapes,null,2) : "not returned"}</pre>
        <details><summary>Native resolved configuration and explanation</summary><pre>{JSON.stringify({resolvedConfig:inspected.nativeView.resolvedConfig ?? "not returned",explain:inspected.nativeView.explain ?? "not returned"},null,2)}</pre></details></> : <p>Native node contracts unavailable; no previous shapes, rewards or learner equation displayed.</p>}
    </section>}
  </div>;
}
