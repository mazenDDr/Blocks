import { useCallback, useMemo } from "react";
import {
  Background, Controls, Handle, MarkerType, Position, ReactFlow, applyNodeChanges, type Connection, type Edge, type Node, type NodeChange, type NodeProps,
} from "@xyflow/react";
import type { Diagnostic, GNode, Graph, OpInfo, UiDoc, Validation } from "../../types";
import { nextId } from "../../util";
import { Library } from "../Library";
import { FixtureBadge, j, fmtMs } from "./common";
import { NodeForm } from "./NodeForms";
import { RouteEditor, type SetSpec } from "./StatePanel";
import { renderPredicate } from "./PredicateBuilder";
import { specOf, type AgentNodeView, type AgentSpec, type Trace } from "./types";

interface CardData extends Record<string, unknown> {
  gnode: GNode; op?: OpInfo; view?: AgentNodeView; ran?: { count: number; last?: number; replay: boolean }; paused?: boolean; routed: boolean; terminal?: "START" | "END";
}
type AgentFlowNode = Node<CardData, "acard">;

function cardSummary(n: GNode): string[] {
  const c = n.config as Record<string, any>;
  switch (n.type) {
    case "agent.chat_model": case "agent.structured_output": return [`${c.model?.provider}: ${c.model?.model}`, c.model?.provider === "fixture" ? "scripted FIXTURE" : `T=${c.model?.temperature ?? "default"} · max ${c.model?.max_tokens ?? "?"} tokens${c.model?.seed != null && c.model?.provider === "ollama" ? ` · seed ${c.model.seed}` : ""}`];
    case "agent.prompt": return [`${(c.items ?? []).length} message sources → ${c.output_field}`, (c.items ?? []).map((i: any) => i.kind === "template" ? i.role : i.kind).join(" · ")];
    case "agent.retrieve": return [`index ${c.index || "?"} · k=${c.k}${c.score_threshold != null ? ` · score ≥ ${c.score_threshold}` : ""}`, `query: ${j(c.query, 40)}`];
    case "agent.tool_call": return [`${c.tool}(${Object.keys(c.args ?? {}).join(", ")})`];
    case "agent.set_state": return (c.assignments ?? []).slice(0, 3).map((a: any) => `${a.field} ${a.kind === "increment" ? "+=" : a.kind === "append_item" ? "+" : "="} ${a.kind === "copy" ? a.source : a.kind === "template" ? j(a.template, 24) : a.kind === "literal" ? j(a.value, 24) : a.kind === "increment" ? a.by : a.kind === "append_item" && !a.source ? j(a.value, 24) : a.source}`);
    case "agent.memory_select": return [`policy ${c.policy || "?"} → ${c.output_field}`];
    case "agent.memory_write": return [`${c.target === "short_term" ? "thread messages" : `long-term ${c.namespace}/${c.scope}`}${c.mode === "approve" ? " · needs approval" : ""}`];
    case "agent.human_interrupt": return [`pauses · actions ${(c.actions ?? []).join("/")}`];
    case "agent.citations": return [`${c.text_field} vs ${c.docs_field}`];
    case "agent.embed_text": return [`${c.provider}${c.provider === "ollama" ? ` ${c.model}` : ` d${c.dimension}`}`];
    default: return [];
  }
}

function AgentCard({ data, selected }: NodeProps<AgentFlowNode>) {
  const { gnode, op, view, ran, paused, terminal } = data;
  if (terminal) {
    return (
      <div className={`acard terminal ${terminal.toLowerCase()} ${selected ? "sel" : ""}`}>
        {terminal === "END" && <Handle id="in" type="target" position={Position.Left} />}
        <b>{terminal}</b>
        {terminal === "START" && <Handle id="out" type="source" position={Position.Right} />}
      </div>
    );
  }
  const errors = view?.diagnostics.filter((d) => d.severity === "error") ?? [];
  const warns = view?.diagnostics.filter((d) => d.severity === "warning") ?? [];
  const effects = view?.effects ?? [];
  const external = effects.some((e) => e === "file_write" || e === "network");
  return (
    <div className={`acard ${selected ? "sel" : ""} ${errors.length ? "err" : ""} ${paused ? "paused" : ""} ${ran ? "ran" : ""}`}>
      <Handle id="in" type="target" position={Position.Left} title="control input" />
      <Handle id="out" type="source" position={Position.Right} title="control output" />
      <div className="card-head"><b>{op?.displayName ?? gnode.type}</b><span>{view?.fixtureModel && <FixtureBadge fixture />}{gnode.type === "agent.human_interrupt" && <span className="badge warnb">interrupt</span>}{external && <span className="badge warnb" title="External effect: needs an approval interrupt">external effect</span>}</span></div>
      <div className="card-id">{gnode.id}{ran && <span className="runmark finished">ran ×{ran.count}{ran.last != null ? ` · ${fmtMs(ran.last)}` : ""}{ran.replay ? " · replayed" : ""}</span>}{paused && <span className="runmark paused">PAUSED here</span>}</div>
      {cardSummary(gnode).map((s, i) => <div key={i} className="card-sum">{s}</div>)}
      {(view?.reads?.length || view?.writes?.length) ? <div className="rw small">{view?.reads?.length ? <>reads <b>{Array.from(new Set(view.reads.map((r) => r.split(".")[0]))).join(", ")}</b></> : null}{view?.writes?.length ? <> · writes <b>{view.writes.join(", ")}</b></> : null}</div> : null}
      {errors.length > 0 && <div className="errbadge">{errors[0].message}{errors.length > 1 ? ` (+${errors.length - 1})` : ""}</div>}
      {errors.length === 0 && warns.length > 0 && <div className="warn small">{warns[0].message}</div>}
    </div>
  );
}
const nodeTypes = { acard: AgentCard };

const AUTO_FIELDS: Record<string, [string, string, string?][]> = {
  "agent.prompt": [["output_field", "list"]], "agent.chat_model": [["messages_field", "list"], ["output_field", "text"]],
  "agent.structured_output": [["messages_field", "list"], ["output_field", "object"], ["error_field", "text"]], "agent.retrieve": [["output_field", "documents"]],
  "agent.citations": [["text_field", "text"], ["docs_field", "documents"], ["output_field", "object"]], "agent.tool_call": [["output_field", "object"]],
  "agent.human_interrupt": [["decision_field", "text"]], "agent.memory_select": [["output_field", "records"]], "agent.embed_text": [["from_field", "text"], ["output_field", "object"]],
};

/** Declare the state fields a freshly added block names, so a new block does not start with undeclared-field errors. */
function withFields(spec: AgentSpec, type: string, cfg: Record<string, any>): AgentSpec {
  const have = new Set(spec.state.map((f) => f.name));
  const add = [...spec.state];
  for (const [key, ftype] of AUTO_FIELDS[type] ?? []) {
    const name = cfg[key];
    if (typeof name === "string" && name && !have.has(name)) { have.add(name); add.push({ name, type: ftype as any, reducer: { kind: "replace" }, scope: "turn", ...(ftype === "object" ? { properties: {} } : {}) }); }
  }
  return { ...spec, state: add };
}

export function AgentCanvas({ graph, setGraph, ui, setUi, validation, ops, selNode, setSelNode, trace, setMessage, runValues }: {
  graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; ui: UiDoc; setUi: (f: (u: UiDoc) => UiDoc, dragging?: boolean) => void; validation: Validation | null; ops: OpInfo[];
  selNode: string | null; setSelNode: (id: string | null) => void; trace: Trace | null; setMessage: (m: string) => void; runValues?: (nodeId: string) => Record<string, any> | undefined;
}) {
  const spec = specOf(graph);
  const opsByType = useMemo(() => Object.fromEntries(ops.map((o) => [o.type, o])), [ops]);
  const views = (validation?.nodes ?? {}) as unknown as Record<string, AgentNodeView>;
  const setSpec: SetSpec = (fn) => setGraph((g) => ({ ...g, agent: fn(specOf(g)) as unknown as Record<string, unknown> }));

  const ranInfo = useMemo(() => {
    const m: Record<string, { count: number; last?: number; replay: boolean }> = {};
    for (const s of trace?.steps ?? []) { const r = m[s.node] ?? { count: 0, replay: false }; r.count++; r.last = s.durationMs; r.replay = r.replay || s.replay; m[s.node] = r; }
    return m;
  }, [trace]);
  const taken = useMemo(() => new Set((trace?.steps ?? []).filter((s) => s.route).map((s) => `route:${s.route!.route}:${s.route!.taken}`)), [trace]);

  const positions = ui.positions;
  const pos = (id: string, i: number) => positions[id] ?? { x: 40 + (i % 4) * 280, y: 60 + Math.floor(i / 4) * 170 };
  const nodes: AgentFlowNode[] = useMemo(() => {
    const routed = new Set(spec.routes.map((r) => r.from));
    const out: AgentFlowNode[] = [
      { id: "START", type: "acard", position: positions["START"] ?? { x: 0, y: 60 }, data: { gnode: { id: "START", type: "START", version: "", config: {} }, routed: false, terminal: "START" }, deletable: false },
      ...graph.nodes.map((n, i) => ({
        id: n.id, type: "acard" as const, position: pos(n.id, i + 1), selected: selNode === n.id,
        data: { gnode: n, op: opsByType[n.type], view: views[n.id], ran: ranInfo[n.id], paused: trace?.status === "paused" && trace.pendingInterrupt?.node === n.id, routed: routed.has(n.id) },
      })),
      { id: "END", type: "acard", position: positions["END"] ?? { x: 40 + 4 * 280, y: 60 }, data: { gnode: { id: "END", type: "END", version: "", config: {} }, routed: false, terminal: "END" }, deletable: false },
    ];
    return out;
  }, [graph.nodes, positions, selNode, opsByType, views, ranInfo, trace, spec.routes]); // eslint-disable-line react-hooks/exhaustive-deps

  const edges: Edge[] = useMemo(() => {
    const arrow = { type: MarkerType.ArrowClosed };
    const out: Edge[] = graph.edges.map((e) => ({ id: e.id, source: e.from.node, target: e.to.node, sourceHandle: "out", targetHandle: "in", markerEnd: arrow, interactionWidth: 24 }));
    spec.routes.forEach((r) => {
      r.cases.forEach((c) => {
        const id = `route:${r.id}:${c.id}`;
        const hit = taken.has(id);
        out.push({ id, source: r.from, target: c.to, sourceHandle: "out", targetHandle: "in", markerEnd: arrow, deletable: false, animated: hit,
          label: `${c.label || c.id}: ${renderPredicate(c.when)}`, labelStyle: { fontSize: 10 }, labelBgStyle: { fill: "#fff", fillOpacity: 0.9 }, labelBgPadding: [3, 2],
          style: { stroke: hit ? "#2a9d4b" : "#7c3aed", strokeDasharray: "6 3", strokeWidth: hit ? 3 : 1.5 } });
      });
      const id = `route:${r.id}:__default__`;
      out.push({ id, source: r.from, target: r.default, sourceHandle: "out", targetHandle: "in", markerEnd: arrow, deletable: false, animated: taken.has(id),
        label: r.defaultLabel || "otherwise", labelStyle: { fontSize: 10 }, labelBgStyle: { fill: "#fff", fillOpacity: 0.9 }, labelBgPadding: [3, 2],
        style: { stroke: taken.has(id) ? "#2a9d4b" : "#9b7fd1", strokeDasharray: "2 3", strokeWidth: taken.has(id) ? 3 : 1.5 } });
    });
    return out;
  }, [graph.edges, spec.routes, taken]);

  const onNodesChange = useCallback((changes: NodeChange<AgentFlowNode>[]) => {
    const moved = applyNodeChanges(changes, nodes);
    for (const c of changes) if (c.type === "select" && c.selected) setSelNode(c.id === "START" || c.id === "END" ? null : c.id);
    if (changes.some((c) => c.type === "position")) setUi((u) => ({ ...u, positions: { ...u.positions, ...Object.fromEntries(moved.map((n) => [n.id, { x: n.position.x, y: n.position.y }])) } }), changes.some(c => c.type === "position" && c.dragging === true));
  }, [nodes, setUi, setSelNode]);

  const connect = (c: Connection) => {
    if (!c.source || !c.target) return;
    if (c.source === "END" || c.target === "START") { setMessage("END has no outgoing and START no incoming transitions."); return; }
    if (spec.routes.some((r) => r.from === c.source)) { setMessage(`'${c.source}' leaves through a conditional route; edit the route's targets instead of adding a fixed transition.`); return; }
    setGraph((g) => (g.edges.some((e) => e.from.node === c.source && e.to.node === c.target) ? g : { ...g, edges: [...g.edges, { id: `${c.source}__${c.target}`, kind: "control", from: { node: c.source!, port: "out" }, to: { node: c.target!, port: "in" } }] }));
  };

  const addBlock = (op: OpInfo) => {
    const taken = new Set(graph.nodes.map((n) => n.id));
    const id = nextId(op.type.split(".")[1], taken);
    const cfg = JSON.parse(JSON.stringify(op.defaults));
    const prev = selNode && graph.nodes.some((n) => n.id === selNode) && !spec.routes.some((r) => r.from === selNode) && !graph.edges.some((e) => e.from.node === selNode) ? selNode : null;
    setGraph((g) => ({
      ...g, nodes: [...g.nodes, { id, type: op.type, version: op.version, config: cfg }],
      edges: prev ? [...g.edges, { id: `${prev}__${id}`, kind: "control", from: { node: prev, port: "out" }, to: { node: id, port: "in" } }] : g.edges,
      agent: withFields(specOf(g), op.type, cfg) as unknown as Record<string, unknown>,
    }));
    setUi((u) => ({ ...u, positions: { ...u.positions, [id]: { x: 80 + (graph.nodes.length % 4) * 280, y: 260 + Math.floor(graph.nodes.length / 4) * 170 } } }));
    setSelNode(id);
  };

  const removeNode = (id: string) => {
    setGraph((g) => {
      const s = specOf(g);
      return {
        ...g, nodes: g.nodes.filter((n) => n.id !== id), edges: g.edges.filter((e) => e.from.node !== id && e.to.node !== id),
        agent: { ...(g.agent ?? {}), routes: s.routes.filter((r) => r.from !== id).map((r) => ({ ...r, cases: r.cases.filter((c) => c.to !== id), default: r.default === id ? "END" : r.default })),
          joins: s.joins.filter((jn) => jn.node !== id).map((jn) => ({ ...jn, waitFor: jn.waitFor.filter((w) => w !== id) })) },
      };
    });
    setSelNode(null);
  };

  const renameNode = (id: string, to: string) => {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(to) || to === "START" || to === "END" || graph.nodes.some((n) => n.id === to)) { setMessage(`'${to}' is not a free node id (letters, digits, underscores; not START/END).`); return; }
    const sw = (x: string) => (x === id ? to : x);
    setGraph((g) => {
      const s = specOf(g);
      return {
        ...g, nodes: g.nodes.map((n) => (n.id === id ? { ...n, id: to } : n)),
        edges: g.edges.map((e) => ({ ...e, id: `${sw(e.from.node)}__${sw(e.to.node)}`, from: { ...e.from, node: sw(e.from.node) }, to: { ...e.to, node: sw(e.to.node) } })),
        agent: { ...(g.agent ?? {}), routes: s.routes.map((r) => ({ ...r, from: sw(r.from), default: sw(r.default), cases: r.cases.map((c) => ({ ...c, to: sw(c.to) })) })),
          joins: s.joins.map((jn) => ({ ...jn, node: sw(jn.node), waitFor: jn.waitFor.map(sw) })) },
      };
    });
    setUi((u) => { const { [id]: p, ...rest } = u.positions; return { ...u, positions: p ? { ...rest, [to]: p } : rest }; });
    setSelNode(to);
  };

  const sel = selNode ? graph.nodes.find((n) => n.id === selNode) : undefined;
  const selView = sel ? views[sel.id] : undefined;
  const route = sel ? spec.routes.find((r) => r.from === sel.id) : undefined;
  const lastStep = sel && trace ? [...trace.steps].reverse().find((s) => s.node === sel.id) : undefined;
  const diags: Diagnostic[] = validation?.diagnostics.filter((d) => !d.nodeId) ?? [];
  const setConfig = (patch: Record<string, unknown>) => {
    if (!sel) return;
    setGraph((g) => {
      const n = g.nodes.find((x) => x.id === sel.id)!;
      const cfg = { ...n.config, ...patch };
      return { ...g, nodes: g.nodes.map((x) => (x.id === sel.id ? { ...x, config: cfg } : x)), agent: withFields(specOf(g), n.type, cfg) as unknown as Record<string, unknown> };
    });
  };

  return (
    <div className="acanvas">
      <aside className="left"><Library ops={ops} onAdd={addBlock} /></aside>
      <main className="center">
        <ReactFlow<AgentFlowNode, Edge> nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onConnect={connect} fitView minZoom={0.2} proOptions={{ hideAttribution: true }}
          onPaneClick={() => setSelNode(null)} deleteKeyCode={["Backspace", "Delete"]}
          onNodesDelete={(ns) => ns.forEach((n) => removeNode(n.id))}
          onEdgesDelete={(es) => setGraph((g) => ({ ...g, edges: g.edges.filter((e) => !es.some((x) => x.id === e.id)) }))}>
          <Background gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
        {graph.nodes.length === 0 && <div className="canvas-empty">Empty agent graph. Add blocks from the library, connect START to the first block, or open an agent example.</div>}
        <div className="canvas-hint small">Solid arrows: fixed transitions. Dashed purple: conditional routes (first matching case wins). Green: taken in the selected run.</div>
      </main>
      <aside className="right ainspector" aria-label="inspector">
        {sel ? (
          <div className="pad">
            <h3>{opsByType[sel.type]?.displayName ?? sel.type} <small className="muted">{sel.type}</small></h3>
            <div className="row"><label>Node id <input aria-label="node id" defaultValue={sel.id} key={sel.id} size={14} onBlur={(e) => { if (e.target.value !== sel.id) renameNode(sel.id, e.target.value); }} /></label>{" "}
              <button className="danger" onClick={() => removeNode(sel.id)}>Delete node</button></div>
            <p className="small">{selView?.explain?.summary ?? opsByType[sel.type]?.purpose}</p>
            {(selView?.effects?.length ?? 0) > 0 && <div className="small">Declared effects: <b>{selView!.effects!.join(", ")}</b></div>}
            {selView?.diagnostics.map((d, i) => <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}>{d.code}: {d.message}</div>)}
            <NodeForm key={sel.id} type={sel.type} cfg={sel.config as Record<string, any>} spec={spec} onConfig={setConfig} runValues={runValues?.(sel.id)} />
            <h4>Transitions out of {sel.id}</h4>
            {route ? (
              <>
                <RouteEditor route={route} graph={graph} setSpec={setSpec} />
                <button className="danger small" onClick={() => setSpec((s) => ({ ...s, routes: s.routes.filter((r) => r.id !== route.id) }))}>Remove route</button>
              </>
            ) : (
              <>
                {graph.edges.filter((e) => e.from.node === sel.id).map((e) => <div key={e.id} className="small">→ {e.to.node} <button className="danger small" aria-label={`remove transition to ${e.to.node}`} onClick={() => setGraph((g) => ({ ...g, edges: g.edges.filter((x) => x.id !== e.id) }))}>×</button></div>)}
                {graph.edges.every((e) => e.from.node !== sel.id) && <div className="muted small">No outgoing transition: connect the right handle to another block or END.</div>}
                <button className="small" onClick={() => setGraph((g) => ({ ...g, edges: g.edges.filter((e) => e.from.node !== sel.id), agent: { ...(g.agent ?? {}), routes: [...specOf(g).routes, { id: `route_${sel.id}`, from: sel.id, cases: [{ id: "case1", label: "", when: { always: true }, to: "END" }], default: "END", defaultLabel: "otherwise" }] } }))}>Branch with a conditional route…</button>
              </>
            )}
            {lastStep && (
              <div className="recorded">
                <h4>Recorded in run {trace!.runId.slice(0, 8)}</h4>
                <div className="small muted">step {lastStep.step} · {lastStep.durationMs != null ? fmtMs(lastStep.durationMs) : "…"}{lastStep.replay ? " · replay after resume" : ""}</div>
                {lastStep.changes.map((c) => <div key={c.field} className="small"><b>{c.field}</b> <span className="muted">({c.reducer})</span>: {j(c.before, 60)} → {j(c.after, 80)}</div>)}
                {lastStep.modelCalls.map((m) => <div key={m.callId} className="small">model call {m.callId.slice(-6)} <FixtureBadge fixture={m.fixture} /> {fmtMs(m.latencyMs)}</div>)}
              </div>
            )}
          </div>
        ) : (
          <div className="pad">
            <div className="empty">Select a block to edit it. Fixed transitions connect handles; use a conditional route when the next step depends on state.</div>
            {diags.map((d, i) => <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}>{d.code}: {d.message}</div>)}
            <h4>What a block card tells you</h4>
            <p className="small muted">Each card lists what the block reads and writes, its declared effects and, after a run, how many times it ran. The Context tab shows exactly what each model call received.</p>
          </div>
        )}
      </aside>
    </div>
  );
}
