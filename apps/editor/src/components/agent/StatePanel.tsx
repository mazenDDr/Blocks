import type { Graph } from "../../types";
import { Check, JsonBox, Num, Row, Sel, useCatalog } from "./common";
import { PredicateBuilder, renderPredicate, statePaths } from "./PredicateBuilder";
import { specOf, type AgentAnalysis, type AgentSpec, type Case, type FieldType, type ReducerKind, type Route, type StateField } from "./types";

export type SetSpec = (fn: (s: AgentSpec) => AgentSpec) => void;

const PROP_TYPES = ["text", "integer", "number", "boolean", "list", "object"];

function FieldRow({ f, onChange, onRemove, writers }: { f: StateField; onChange: (f: StateField) => void; onRemove: () => void; writers: string[] }) {
  const cat = useCatalog();
  const allowed = (cat?.reducers ?? []).filter((r) => r.appliesTo.includes(f.type)).map((r) => r.kind);
  const doc = cat?.reducers.find((r) => r.kind === f.reducer.kind)?.doc;
  return (
    <tr>
      <td><input aria-label={`field name ${f.name}`} value={f.name} size={14} onChange={(e) => onChange({ ...f, name: e.target.value })} /></td>
      <td><Sel label={`type of ${f.name}`} value={f.type} options={(cat?.fieldTypes ?? [f.type]) as FieldType[]}
        onChange={(t) => onChange({ ...f, type: t, reducer: { kind: (cat?.reducers.find((r) => r.kind === f.reducer.kind)?.appliesTo.includes(t) ? f.reducer.kind : "replace") as ReducerKind }, properties: t === "object" ? f.properties : undefined })} /></td>
      <td title={doc}>
        <Sel label={`reducer of ${f.name}`} value={f.reducer.kind} options={(allowed.length ? allowed : [f.reducer.kind]) as ReducerKind[]} onChange={(k) => onChange({ ...f, reducer: { kind: k, n: k === "keep_last_n" ? f.reducer.n ?? 6 : null } })} />
        {f.reducer.kind === "keep_last_n" && <> N <Num label={`keep last N of ${f.name}`} integer min={1} value={f.reducer.n ?? 6} onChange={(n) => onChange({ ...f, reducer: { kind: "keep_last_n", n: n ?? 1 } })} width={56} /></>}
        <div className="small muted">{doc}</div>
      </td>
      <td><Sel label={`scope of ${f.name}`} value={f.scope ?? "turn"} options={[{ value: "turn", label: "turn (reset each run)" }, { value: "thread", label: "thread (persists)" }]} onChange={(s) => onChange({ ...f, scope: s })} /></td>
      <td>
        {f.type === "text" ? <input aria-label={`default of ${f.name}`} value={String(f.default ?? "")} onChange={(e) => onChange({ ...f, default: e.target.value === "" ? undefined : e.target.value })} />
          : f.type === "integer" || f.type === "number" ? <Num label={`default of ${f.name}`} nullable value={typeof f.default === "number" ? f.default : null} onChange={(n) => onChange({ ...f, default: n ?? undefined })} />
            : f.type === "boolean" ? <Check label={`default of ${f.name}`} value={f.default === true} onChange={(b) => onChange({ ...f, default: b })} />
              : <JsonBox label={`default of ${f.name}`} rows={1} value={f.default ?? null} onChange={(v) => onChange({ ...f, default: v === null ? undefined : v })} />}
        {f.type === "object" && (
          <div className="props small">properties:{" "}
            {Object.entries(f.properties ?? {}).map(([k, t]) => (
              <span key={k} className="prop"><input aria-label={`property name ${k}`} size={8} defaultValue={k} onBlur={(e) => { if (e.target.value && e.target.value !== k) { const { [k]: v, ...rest } = f.properties ?? {}; onChange({ ...f, properties: { ...rest, [e.target.value]: v } }); } }} />
                <Sel label={`property type ${k}`} value={t as string} options={PROP_TYPES} onChange={(nt) => onChange({ ...f, properties: { ...f.properties, [k]: nt } })} />
                <button className="danger small" aria-label={`remove property ${k}`} onClick={() => { const { [k]: _x, ...rest } = f.properties ?? {}; onChange({ ...f, properties: rest }); }}>×</button></span>
            ))}
            <button className="small" onClick={() => onChange({ ...f, properties: { ...f.properties, [`prop${Object.keys(f.properties ?? {}).length + 1}`]: "text" } })}>+ property</button></div>
        )}
      </td>
      <td className="small muted">{writers.length ? writers.join(", ") : <i>input (not written by a node)</i>}</td>
      <td><button className="danger" aria-label={`remove field ${f.name}`} onClick={onRemove}>×</button></td>
    </tr>
  );
}

export function StateSchemaEditor({ graph, setSpec, writes }: { graph: Graph; setSpec: SetSpec; writes: Record<string, string[]> }) {
  const spec = specOf(graph);
  return (
    <section aria-label="state schema">
      <h3>State schema <small className="muted">typed fields; the reducer decides how concurrent or repeated writes combine (native LangGraph channels)</small></h3>
      <table className="astate">
        <thead><tr><th>field</th><th>type</th><th>reducer</th><th>scope</th><th>default / properties</th><th>written by</th><th /></tr></thead>
        <tbody>
          {spec.state.map((f, i) => <FieldRow key={i} f={f} writers={writes[f.name] ?? []} onRemove={() => setSpec((s) => ({ ...s, state: s.state.filter((_, j) => j !== i) }))}
            onChange={(nf) => setSpec((s) => ({ ...s, state: s.state.map((x, j) => (j === i ? nf : x)) }))} />)}
        </tbody>
      </table>
      <button onClick={() => setSpec((s) => ({ ...s, state: [...s.state, { name: `field_${s.state.length + 1}`, type: "text", reducer: { kind: "replace" }, scope: "turn" }] }))}>Add field</button>
      <p className="hint">Turn-scoped fields are reset at the start of every run on a thread; thread-scoped fields (a conversation history) keep the value in the thread's checkpoint.</p>
    </section>
  );
}

export function LimitsEditor({ graph, setSpec }: { graph: Graph; setSpec: SetSpec }) {
  const l = specOf(graph).limits;
  const set = (k: string, v: number | null) => setSpec((s) => ({ ...s, limits: { ...s.limits, [k]: v } }));
  return (
    <section aria-label="loop limits">
      <h3>Loop limits and budgets <small className="muted">a loop always ends: by its route predicate, or by one of these</small></h3>
      <Row label="Max steps" hint="LangGraph recursion limit: supersteps per run"><Num label="max steps" integer min={1} value={l.maxSteps} onChange={(n) => set("maxSteps", n ?? 25)} /></Row>
      <Row label="Max model calls"><Num label="max model calls" integer min={1} nullable value={l.maxModelCalls ?? null} onChange={(n) => set("maxModelCalls", n)} /></Row>
      <Row label="Max tokens" hint="provider-reported input+output; estimated when a provider reports none"><Num label="max tokens" integer min={1} nullable value={l.maxTokens ?? null} onChange={(n) => set("maxTokens", n)} /></Row>
      <Row label="Max seconds"><Num label="max seconds" min={0.001} nullable value={l.maxSeconds ?? null} onChange={(n) => set("maxSeconds", n)} /></Row>
      <Row label="Max tool calls"><Num label="max tool calls" integer min={1} nullable value={l.maxToolCalls ?? null} onChange={(n) => set("maxToolCalls", n)} /></Row>
    </section>
  );
}

export function RouteEditor({ route, graph, setSpec }: { route: Route; graph: Graph; setSpec: SetSpec }) {
  const cat = useCatalog();
  const spec = specOf(graph);
  const paths = statePaths(spec);
  const targets = [...graph.nodes.map((n) => n.id), "END"];
  const upd = (fn: (r: Route) => Route) => setSpec((s) => ({ ...s, routes: s.routes.map((r) => (r.id === route.id ? fn(r) : r)) }));
  return (
    <div className="route" aria-label={`route from ${route.from}`}>
      <div className="small muted">First matching case wins; if none matches, the default branch is taken.</div>
      {route.cases.map((c, i) => (
        <div key={c.id} className="rcase">
          <div className="rcasehead">
            <b>{i + 1}.</b> <input aria-label={`label of case ${i + 1}`} value={c.label} placeholder="label" onChange={(e) => upd((r) => ({ ...r, cases: r.cases.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)) }))} />
            {" → "}<Sel label={`target of case ${i + 1}`} value={c.to} options={targets} onChange={(to) => upd((r) => ({ ...r, cases: r.cases.map((x, j) => (j === i ? { ...x, to } : x)) }))} />
            <button className="small" disabled={i === 0} aria-label={`move case ${i + 1} up`} onClick={() => upd((r) => { const cs = [...r.cases]; [cs[i - 1], cs[i]] = [cs[i], cs[i - 1]]; return { ...r, cases: cs }; })}>↑</button>
            <button className="danger small" aria-label={`remove case ${i + 1}`} onClick={() => upd((r) => ({ ...r, cases: r.cases.filter((_, j) => j !== i) }))}>×</button>
          </div>
          <PredicateBuilder value={c.when} paths={paths} catalog={cat} onChange={(w) => upd((r) => ({ ...r, cases: r.cases.map((x, j) => (j === i ? { ...x, when: w } : x)) }))} />
          <div className="small muted">when {renderPredicate(c.when)}</div>
        </div>
      ))}
      <button onClick={() => upd((r) => ({ ...r, cases: [...r.cases, { id: `case${r.cases.length + 1}_${Math.random().toString(36).slice(2, 5)}`, label: "", when: { always: true }, to: "END" } as Case] }))}>Add case</button>
      <div className="rdefault">otherwise <input aria-label="default branch label" value={route.defaultLabel ?? ""} placeholder="label" onChange={(e) => upd((r) => ({ ...r, defaultLabel: e.target.value }))} />
        {" → "}<Sel label="default target" value={route.default} options={targets} onChange={(to) => upd((r) => ({ ...r, default: to }))} /></div>
    </div>
  );
}

export function RoutesPanel({ graph, setSpec, setGraph }: { graph: Graph; setSpec: SetSpec; setGraph: (f: (g: Graph) => Graph) => void }) {
  const spec = specOf(graph);
  const without = spec.routes.map((r) => r.from);
  const free = graph.nodes.map((n) => n.id).filter((id) => !without.includes(id));
  return (
    <section aria-label="routes">
      <h3>Conditional routes <small className="muted">typed predicates built from field / operator / value pickers</small></h3>
      {spec.routes.length === 0 && <div className="empty">No routes. A node that needs to branch gets a route instead of fixed transitions.</div>}
      {spec.routes.map((r) => (
        <div key={r.id} className="routebox">
          <h4>after {r.from} <button className="danger small" onClick={() => setSpec((s) => ({ ...s, routes: s.routes.filter((x) => x.id !== r.id) }))}>remove route</button></h4>
          <RouteEditor route={r} graph={graph} setSpec={setSpec} />
        </div>
      ))}
      {free.length > 0 && <AddRoute free={free} setGraph={setGraph} />}
    </section>
  );
}

import { useState } from "react";
function AddRoute({ free, setGraph }: { free: string[]; setGraph: (f: (g: Graph) => Graph) => void }) {
  const [id, setId] = useState("");
  return (
    <div>
      <select aria-label="add route after" value={id} onChange={(e) => setId(e.target.value)}>
        <option value="">add a route after node…</option>{free.map((x) => <option key={x} value={x}>{x}</option>)}
      </select>{" "}
      <button disabled={!id} onClick={() => {
        setGraph((g) => ({ ...g, edges: g.edges.filter((e) => e.from.node !== id), agent: { ...(g.agent ?? {}), routes: [...specOf(g).routes, { id: `route_${id}`, from: id, cases: [{ id: "case1", label: "", when: { always: true }, to: "END" }], default: "END", defaultLabel: "otherwise" }] } }));
        setId("");
      }}>Add route</button>
      <span className="hint"> Adding a route replaces the node's fixed outgoing transitions.</span>
    </div>
  );
}

export function GraphAnalysis({ analysis }: { analysis: AgentAnalysis | undefined }) {
  if (!analysis) return null;
  return (
    <section aria-label="graph analysis">
      <h3>Loops, branches and joins <small className="muted">found by validation</small></h3>
      {analysis.loops.length === 0 && <div className="muted">No cycles.</div>}
      {analysis.loops.map((l, i) => <div key={i} className={l.hasConditionalExit ? "" : "warn"}>Cycle {l.nodes.join(" → ")} — bounded by {l.boundedBy}.</div>)}
      {analysis.parallel.map((p, i) => <div key={i}>Parallel branches after <b>{p.from}</b>: {p.branches.join(", ")} (they run in the same step)</div>)}
      {analysis.joins.map((j, i) => <div key={i}>Join at <b>{j.node}</b> waits for {j.waitFor.join(", ")}</div>)}
    </section>
  );
}

export function JoinsEditor({ graph, setSpec }: { graph: Graph; setSpec: SetSpec }) {
  const spec = specOf(graph);
  const ids = graph.nodes.map((n) => n.id);
  return (
    <section aria-label="joins">
      <h4>Joins</h4>
      {spec.joins.map((j, i) => (
        <div key={i} className="rcasehead">wait at <Sel label="join node" value={j.node} options={ids} onChange={(node) => setSpec((s) => ({ ...s, joins: s.joins.map((x, k) => (k === i ? { ...x, node } : x)) }))} /> for
          <input aria-label="join sources" value={j.waitFor.join(", ")} placeholder="node ids, comma separated" onChange={(e) => setSpec((s) => ({ ...s, joins: s.joins.map((x, k) => (k === i ? { ...x, waitFor: e.target.value.split(",").map((t) => t.trim()).filter(Boolean) } : x)) }))} />
          <button className="danger small" aria-label="remove join" onClick={() => setSpec((s) => ({ ...s, joins: s.joins.filter((_, k) => k !== i) }))}>×</button></div>
      ))}
      <button onClick={() => setSpec((s) => ({ ...s, joins: [...s.joins, { node: ids[0] ?? "", waitFor: [] }] }))} disabled={!ids.length}>Add join</button>
    </section>
  );
}

export function StateTab({ graph, setGraph, analysis, writes }: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; analysis?: AgentAnalysis; writes: Record<string, string[]> }) {
  const setSpec: SetSpec = (fn) => setGraph((g) => ({ ...g, agent: fn(specOf(g)) as unknown as Record<string, unknown> }));
  return (
    <div className="statetab">
      <StateSchemaEditor graph={graph} setSpec={setSpec} writes={writes} />
      <div className="twocol">
        <div><RoutesPanel graph={graph} setSpec={setSpec} setGraph={setGraph} /></div>
        <div><LimitsEditor graph={graph} setSpec={setSpec} /><JoinsEditor graph={graph} setSpec={setSpec} /><GraphAnalysis analysis={analysis} /></div>
      </div>
    </div>
  );
}
