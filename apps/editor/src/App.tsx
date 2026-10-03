import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background, Controls, MarkerType, ReactFlow, ReactFlowProvider, applyEdgeChanges, applyNodeChanges,
  type Connection, type Edge, type EdgeChange, type NodeChange,
} from "@xyflow/react";
import { api, errorText } from "./api";
import { ExportView } from "./components/ExportView";
import { NodeInspector, InspectionBar, WireInspector, useInspectionData } from "./components/Inspector";
import type { Ctx } from "./components/InspectorTabs";
import { Library } from "./components/Library";
import { OpNodeCard, type CardNode } from "./components/OpNode";
import { RunPanel } from "./components/RunPanel";
import { useRuns, useValidation } from "./hooks";
import type { GNode, Graph, OpInfo, UiDoc } from "./types";
import { fmtInt, fmtShape, nextId, shortName } from "./util";

const EMPTY: Graph = { schemaVersion: "1.0.0", graphKind: "model", backend: "pytorch", nodes: [], edges: [] };
const EMPTY_UI: UiDoc = { schemaVersion: "1.0.0", positions: {} };
const LAST_KEY = "void.lastProject";
const nodeTypes = { card: OpNodeCard };

const lsGet = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable: remembering the last project is optional */ } };

export function App() {
  return <ReactFlowProvider><Workbench /></ReactFlowProvider>;
}

function Workbench() {
  const [ops, setOps] = useState<OpInfo[]>([]);
  const [graph, setGraph] = useState<Graph>(EMPTY);
  const [ui, setUi] = useState<UiDoc>(EMPTY_UI);
  const [projectId, setProjectId] = useState("reference_cnn");
  const [saved, setSaved] = useState<string>("");
  const [projects, setProjects] = useState<string[]>([]);
  const [examples, setExamples] = useState<string[]>([]);
  const [selNodes, setSelNodes] = useState<string[]>([]);
  const [selEdges, setSelEdges] = useState<string[]>([]);
  const [ctx, setCtx] = useState<Ctx>({ runId: null, step: null, sample: null });
  const [message, setMessage] = useState<string | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [booted, setBooted] = useState(false);

  const opsByType = useMemo(() => Object.fromEntries(ops.map((o) => [o.type, o])), [ops]);
  const validation = useValidation(graph);
  const v = validation.data;
  const { runs, reload: reloadRuns } = useRuns(projectId);
  const insData = useInspectionData(ctx.runId);

  const sig = useMemo(() => JSON.stringify([graph, ui]), [graph, ui]);
  const dirty = booted && sig !== saved;

  const refreshLists = useCallback(() => {
    api.get<{ projects: string[] }>("/api/projects").then((r) => setProjects(r.projects)).catch(() => {});
    api.get<{ examples: string[] }>("/api/examples").then((r) => setExamples(r.examples)).catch(() => {});
  }, []);

  const adopt = useCallback((id: string, g: Graph, u: UiDoc | null, savedState: boolean) => {
    const uu = u ?? EMPTY_UI;
    setProjectId(id); setGraph(g); setUi(uu); setSelNodes([]); setSelEdges([]);
    setSaved(savedState ? JSON.stringify([g, uu]) : "");
    setCtx({ runId: null, step: null, sample: null });
  }, []);

  // boot: registry, then the last saved project, else the reference example as an unsaved draft
  useEffect(() => {
    (async () => {
      try {
        const reg = await api.get<{ ops: OpInfo[] }>("/api/registry");
        setOps(reg.ops);
        refreshLists();
        const last = lsGet(LAST_KEY);
        if (last) {
          try { const p = await api.get<any>(`/api/projects/${encodeURIComponent(last)}`); adopt(p.id, p.graph, p.ui, true); setBooted(true); return; } catch { /* fall through to the example */ }
        }
        const ex = await api.get<any>("/api/examples/reference_cnn");
        adopt("reference_cnn", ex.graph, ex.ui, false);
      } catch (e) { setMessage(`Cannot reach the control service: ${errorText(e)}`); }
      setBooted(true);
    })();
  }, [adopt, refreshLists]);

  // default inspection context: newest run, first validation sample
  useEffect(() => { if (!ctx.runId && runs.length) setCtx((c) => ({ ...c, runId: runs[runs.length - 1].id })); }, [runs, ctx.runId]);
  useEffect(() => { if (ctx.runId && ctx.sample == null && insData.samples.length) setCtx((c) => ({ ...c, sample: 0 })); }, [ctx.runId, ctx.sample, insData.samples.length]);

  // ---------------------------------------------------------------- graph edits (the spec is the single source)
  const edit = (fn: (g: Graph) => Graph) => setGraph((g) => fn(g));
  const setConfig = (id: string, patch: Record<string, unknown>) =>
    edit((g) => ({ ...g, nodes: g.nodes.map((n) => (n.id === id ? { ...n, config: Object.fromEntries(Object.entries({ ...n.config, ...patch }).filter(([, x]) => x !== undefined)) } : n)) }));

  const connect = (toNode: string, toPort: string, from: { node: string; port: string } | null) =>
    edit((g) => {
      const edges = g.edges.filter((e) => !(e.to.node === toNode && e.to.port === toPort));
      if (from) edges.push({ id: `${from.node}_${from.port}__${toNode}_${toPort}`, kind: "tensor", from, to: { node: toNode, port: toPort } });
      return { ...g, edges };
    });

  const removeNodes = (ids: string[]) => {
    const s = new Set(ids);
    edit((g) => ({ ...g, nodes: g.nodes.filter((n) => !s.has(n.id)), edges: g.edges.filter((e) => !s.has(e.from.node) && !s.has(e.to.node)) }));
    setUi((u) => ({ ...u, positions: Object.fromEntries(Object.entries(u.positions).filter(([k]) => !s.has(k))) }));
    setSelNodes((a) => a.filter((x) => !s.has(x)));
  };
  const removeEdges = (ids: string[]) => { const s = new Set(ids); edit((g) => ({ ...g, edges: g.edges.filter((e) => !s.has(e.id)) })); setSelEdges((a) => a.filter((x) => !s.has(x))); };

  const rename = (oldId: string, newId: string): string | null => {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(newId)) return "Use letters, digits and underscores, starting with a letter.";
    if (graph.nodes.some((n) => n.id === newId)) return `Node id '${newId}' is already used.`;
    edit((g) => ({
      ...g,
      nodes: g.nodes.map((n) => (n.id === oldId ? { ...n, id: newId, stateRef: n.stateRef ? n.stateRef.replace(new RegExp(`${oldId}$`), newId) : n.stateRef } : n)),
      edges: g.edges.map((e) => ({ ...e, id: e.id.split(oldId).join(newId), from: e.from.node === oldId ? { ...e.from, node: newId } : e.from, to: e.to.node === oldId ? { ...e.to, node: newId } : e.to })),
    }));
    setUi((u) => { const { [oldId]: p, ...rest } = u.positions; return { ...u, positions: p ? { ...rest, [newId]: p } : rest }; });
    setSelNodes([newId]);
    return null;
  };

  const addBlock = (op: OpInfo) => {
    const id = nextId(shortName(op), new Set(graph.nodes.map((n) => n.id)));
    const node: GNode = { id, type: op.type, version: op.version, config: JSON.parse(JSON.stringify(op.defaults)), stateRef: null };
    const anchor = selNodes.length === 1 ? graph.nodes.find((n) => n.id === selNodes[0]) : undefined;
    const ap = anchor ? ui.positions[anchor.id] : undefined;
    const maxX = Math.max(-200, ...Object.values(ui.positions).map((p) => p.x));
    const pos = ap ? { x: ap.x + 260, y: ap.y } : { x: maxX + 260, y: 120 };
    edit((g) => {
      const edges = [...g.edges];
      const aop = anchor ? opsByType[anchor.type] : undefined;
      if (anchor && aop && aop.outputs.length && op.inputs.length) {
        const from = { node: anchor.id, port: aop.outputs[0] }, to = { node: id, port: op.inputs[0] };
        edges.push({ id: `${from.node}_${from.port}__${to.node}_${to.port}`, kind: "tensor", from, to });
      }
      return { ...g, nodes: [...g.nodes, node], edges };
    });
    setUi((u) => ({ ...u, positions: { ...u.positions, [id]: pos } }));
    setSelNodes([id]); setSelEdges([]);
  };

  // ---------------------------------------------------------------- project IO
  const save = useCallback(async () => {
    try {
      const r = await api.saveProject(projectId, graph, ui);
      setSaved(sig); lsSet(LAST_KEY, projectId); refreshLists();
      setMessage(`Saved '${projectId}' (graph ${r.graphHash.slice(0, 8)}).`);
    } catch (e) { setMessage(errorText(e)); throw e; }
  }, [projectId, graph, ui, sig, refreshLists]);
  const ensureSaved = useCallback(async () => { await save(); }, [save]);

  const load = async (kind: "project" | "example", id: string) => {
    if (dirty && !window.confirm("Discard unsaved changes?")) return;
    try {
      const p = await api.get<any>(kind === "project" ? `/api/projects/${encodeURIComponent(id)}` : `/api/examples/${encodeURIComponent(id)}`);
      adopt(p.id, p.graph, p.ui, kind === "project");
      if (kind === "project") lsSet(LAST_KEY, id);
      setMessage(`Loaded ${kind} '${id}'.`);
    } catch (e) { setMessage(errorText(e)); }
  };

  // ---------------------------------------------------------------- React Flow wiring
  const posOf = (id: string, i: number) => ui.positions[id] ?? { x: 60 + i * 260, y: 120 };
  const rfNodes: CardNode[] = useMemo(() => graph.nodes.map((n, i) => ({
    id: n.id, type: "card" as const, position: posOf(n.id, i), selected: selNodes.includes(n.id),
    data: { gnode: n, op: opsByType[n.type], view: v?.nodes[n.id], pending: validation.pending },
  })), [graph.nodes, ui.positions, selNodes, opsByType, v, validation.pending]); // eslint-disable-line react-hooks/exhaustive-deps

  const rfEdges: Edge[] = useMemo(() => graph.edges.map((e) => {
    const t = v?.nodes[e.from.node]?.outputShapes?.[e.from.port];
    const bad = (v?.nodes[e.to.node]?.diagnostics ?? []).some((d) => d.severity === "error" && d.port === e.to.port);
    return {
      id: e.id, source: e.from.node, sourceHandle: e.from.port, target: e.to.node, targetHandle: e.to.port, selected: selEdges.includes(e.id),
      label: bad ? `${fmtShape(t)} ✖` : fmtShape(t), interactionWidth: 28, markerEnd: { type: MarkerType.ArrowClosed },
      style: bad ? { stroke: "#c62828", strokeWidth: 2 } : undefined, labelStyle: { fontSize: 10, fill: bad ? "#c62828" : "#444" },
      labelBgStyle: { fill: "#fff", fillOpacity: 0.85 }, labelBgPadding: [3, 2] as [number, number],
    };
  }), [graph.edges, selEdges, v]);

  const onNodesChange = useCallback((changes: NodeChange<CardNode>[]) => {
    const moved = applyNodeChanges(changes, rfNodes);
    const sel = changes.some((c) => c.type === "select");
    if (sel) setSelNodes(moved.filter((n) => n.selected).map((n) => n.id));
    if (changes.some((c) => c.type === "position")) setUi((u) => ({ ...u, positions: { ...u.positions, ...Object.fromEntries(moved.map((n) => [n.id, { x: n.position.x, y: n.position.y }])) } }));
  }, [rfNodes]);
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    if (changes.some((c) => c.type === "select")) setSelEdges(applyEdgeChanges(changes, rfEdges).filter((e) => e.selected).map((e) => e.id));
  }, [rfEdges]);
  const onConnect = (c: Connection) => { if (c.source && c.target && c.sourceHandle && c.targetHandle) connect(c.target, c.targetHandle, { node: c.source, port: c.sourceHandle }); };

  const selNode = selNodes.length === 1 && selEdges.length === 0 ? graph.nodes.find((n) => n.id === selNodes[0]) : undefined;
  const selEdge = selEdges.length === 1 && selNodes.length === 0 ? graph.edges.find((e) => e.id === selEdges[0]) : undefined;
  const errCount = v ? v.diagnostics.filter((d) => d.severity === "error").length : 0;

  return (
    <div className="app">
      <header className="topbar">
        <b className="brand">Project Void</b>
        <label>Project <input value={projectId} onChange={(e) => setProjectId(e.target.value)} aria-label="project id" size={16} /></label>
        <button onClick={() => save().catch(() => {})}>Save{dirty ? " *" : ""}</button>
        <label>Load <select value="" onChange={(e) => { const [k, ...r] = e.target.value.split(":"); if (k) load(k as "project" | "example", r.join(":")); }} aria-label="load project">
          <option value="">choose…</option>
          {projects.length > 0 && <optgroup label="Saved projects">{projects.map((p) => <option key={p} value={`project:${p}`}>{p}</option>)}</optgroup>}
          <optgroup label="Examples">{examples.map((p) => <option key={p} value={`example:${p}`}>{p}</option>)}</optgroup>
        </select></label>
        <button onClick={() => { if (!dirty || window.confirm("Discard unsaved changes?")) adopt("untitled", EMPTY, null, false); }}>New</button>
        <button onClick={() => setShowCode(true)} disabled={!v?.ok} title={v?.ok ? "Show generated PyTorch" : "Fix the graph errors to export"}>Export PyTorch</button>
        <span className="spacer" />
        <span className={`vsum ${errCount ? "bad" : "good"}`} aria-live="polite">
          {validation.error ? `validation unavailable: ${validation.error}` : validation.pending ? "validating…" : v ? (errCount ? `${errCount} error${errCount > 1 ? "s" : ""}` : `valid · ${fmtInt(v.totalParams)} parameters`) : ""}
          {v && <small> · graph {v.graphHash.slice(0, 8)}</small>}
        </span>
      </header>
      {message && <div className="toast" role="status" onClick={() => setMessage(null)}>{message} <small>(click to dismiss)</small></div>}

      <aside className="left"><Library ops={ops} onAdd={addBlock} /></aside>

      <main className="center">
        <ReactFlow<CardNode, Edge> nodes={rfNodes} edges={rfEdges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
          onConnect={onConnect} onNodesDelete={(ns) => removeNodes(ns.map((n) => n.id))} onEdgesDelete={(es) => removeEdges(es.map((e) => e.id))}
          deleteKeyCode={["Backspace", "Delete"]} fitView minZoom={0.2} onPaneClick={() => { setSelNodes([]); setSelEdges([]); }} proOptions={{ hideAttribution: true }}>
          <Background gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
        {graph.nodes.length === 0 && <div className="canvas-empty">Empty graph. Add blocks from the library, or Load the reference_cnn example.</div>}
      </main>

      <aside className="right">
        <InspectionBar runs={runs} ctx={ctx} setCtx={setCtx} currentHash={v?.graphHash} checkpoints={insData.checkpoints} samples={insData.samples} />
        {selNode ? (
          <NodeInspector key={selNode.id} node={selNode} op={opsByType[selNode.type]} ops={opsByType} view={v?.nodes[selNode.id]} graph={graph} ctx={ctx}
            onConfig={(patch) => setConfig(selNode.id, patch)} onConnect={connect} onDelete={() => removeNodes([selNode.id])} onRename={(nid) => rename(selNode.id, nid)} />
        ) : selEdge ? (
          <WireInspector edge={selEdge} graph={graph} validation={v} ctx={ctx} ops={opsByType} />
        ) : (
          <div className="empty pad">Select a node to edit it, or click a wire to inspect the value crossing it.</div>
        )}
      </aside>

      <section className="bottom">
        <RunPanel projectId={projectId} graph={graph} validation={v} runs={runs} reloadRuns={reloadRuns} ctx={ctx} setCtx={setCtx}
          baseline={ui.pinnedBaseline ?? null} setBaseline={(id) => setUi((u) => ({ ...u, pinnedBaseline: id }))} ensureSaved={ensureSaved} />
      </section>

      {showCode && <ExportView projectId={projectId} ensureSaved={ensureSaved} onClose={() => setShowCode(false)} />}
    </div>
  );
}
