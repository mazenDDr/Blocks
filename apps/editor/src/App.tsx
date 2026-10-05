import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background, Controls, MarkerType, ReactFlow, ReactFlowProvider, applyEdgeChanges, applyNodeChanges, useReactFlow,
  type Connection, type Edge, type EdgeChange, type NodeChange,
} from "@xyflow/react";
import { api, errorText } from "./api";
import { AgentWorkspace } from "./components/agent/AgentWorkspace";
import { DomainWorkspace } from "./components/DomainWorkspace";
import { ProductionWorkspace } from "./components/ProductionWorkspace";
import { ScaleWorkspace } from "./components/ScaleWorkspace";
import { KeyboardGraphTools } from "./components/KeyboardGraphTools";
import { RLWorkspace } from "./components/rl/RLWorkspace";
import { GraphContext } from "./components/UnsupViews";
import { AttentionWorkspace } from "./components/Attention";
import { CodeBlockEditor } from "./components/CodeBlockEditor";
import { RepoImport } from "./components/RepoImport";
import { DataWorkspace } from "./components/DataWorkspace";
import { DebuggerWorkspace } from "./components/Debugger";
import { Experiments } from "./components/Experiments";
import { BackendWorkspace, useBackends } from "./components/BackendWorkspace";
import { CoverageView } from "./components/Coverage";
import { ExportView } from "./components/ExportView";
import { GroupCard, type GroupNode } from "./components/GroupCard";
import { NodeInspector, InspectionBar, WireInspector, useInspectionData, type InnerActions } from "./components/Inspector";
import type { Ctx } from "./components/InspectorTabs";
import { Library } from "./components/Library";
import { ModuleLibrary } from "./components/ModuleLibrary";
import { ModulePanel } from "./components/ModulePanel";
import { OpNodeCard, type CardNode } from "./components/OpNode";
import { RunPanel } from "./components/RunPanel";
import { TabularRunBar, TabularRunPanel, TabularWireInspector, tabularTabs } from "./components/TabularPanels";
import { TrainingWorkspace } from "./components/Training";
import { useCompat, useModuleValidation, useRuns, useValidation } from "./hooks";
import {
  PSEUDO_OPS, autoLayout, defToView, findModule, isPseudo, modKey, newModule, replaceModule, resolveDefTarget, viewToDef,
} from "./modules";
import {
  type CompatNode, isModelRun, isProcedureRun, isTabularRun, type CodeBlockDef, type GNode, type Graph, type ModuleDef, type NodeView, type OpInfo, type ProcedureRunSummary, type RunSummary, type UiDoc,
} from "./types";
import { fmtInt, fmtShape, nextId, shortName } from "./util";
import { useDocumentHistory } from "./useDocumentHistory";

const EMPTY: Graph = { schemaVersion: "1.0.0", graphKind: "model", backend: "pytorch", nodes: [], edges: [] };
const EMPTY_AGENT: Graph = { schemaVersion: "1.0.0", graphKind: "agent", backend: "langgraph", nodes: [], edges: [], agent: { state: [{ name: "question", type: "text", reducer: { kind: "replace" }, scope: "turn" }], routes: [], joins: [], limits: { maxSteps: 25 }, indexes: [], policies: [] } };
const EMPTY_TABULAR: Graph = { schemaVersion: "1.0.0", graphKind: "tabular", backend: "python", nodes: [], edges: [] };
interface ProjectInfo { id: string; graphKind: string; description: string | null; synthetic: boolean }
const EMPTY_UI: UiDoc = { schemaVersion: "1.0.0", positions: {} };
const LAST_KEY = "void.lastProject";
const nodeTypes = { card: OpNodeCard, group: GroupCard };
type AnyNode = CardNode | GroupNode;
type View = "graph" | "data" | "experiments" | "training" | "debug" | "attention" | "backends" | "coverage" | "domain" | "production" | "scale";
interface Scope { module: string; version: string; via: string }

const lsGet = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable: remembering the last project is optional */ } };

export function App() {
  return <ReactFlowProvider><Workbench /></ReactFlowProvider>;
}

/** Every module@version that an instance (anywhere, including inside other modules) refers to. */
function usedModules(g: Graph): Set<string> {
  const s = new Set<string>();
  const scan = (nodes: GNode[]) => {
    for (const n of nodes) {
      const c = n.config as any;
      if (n.type === "core.composite" || n.type === "core.repeat") s.add(`${c.module}@${c.version ?? "1.0.0"}`);
      if (n.type === "core.select") { for (const b of [c.then, c.otherwise]) if (b) s.add(`${b.module}@${b.version ?? "1.0.0"}`); }
    }
  };
  scan(g.nodes);
  for (const m of g.modules ?? []) scan(m.nodes);
  if (g.training?.loss?.module) s.add(`${g.training.loss.module}@${g.training.loss.version ?? "1.0.0"}`);
  return s;
}

function Workbench() {
  const [projectId, setProjectId] = useState("untitled");
  const { graph, ui, setGraph, setUi, reset: resetDraft, move: moveDraft, canUndo, canRedo } = useDocumentHistory(EMPTY, EMPTY_UI);
  const [saved, setSaved] = useState<string>("");
  const [ops, setOps] = useState<OpInfo[]>([]);
  const [projects, setProjects] = useState<ProjectInfo[]>([]);
  const [examples, setExamples] = useState<ProjectInfo[]>([]);
  const [selNodes, setSelNodes] = useState<string[]>([]);
  const [selEdges, setSelEdges] = useState<string[]>([]);
  const [ctx, setCtx] = useState<Ctx>({ runId: null, step: null, sample: null });
  const [message, setMessage] = useState<string | null>(null);
  const [showCode, setShowCode] = useState(false);
  const [booted, setBooted] = useState(false);
  const [view, setView] = useState<View>("graph");
  const [loadToken, setLoadToken] = useState(0);
  const [scope, setScope] = useState<Scope[]>([]);
  const [expanded, setExpanded] = useState<string[]>([]);
  const [leftTab, setLeftTab] = useState<"Blocks" | "Modules">("Blocks");
  const [codeEdit, setCodeEdit] = useState<{ id: string; version: string } | null>(null);
  const [repoImport, setRepoImport] = useState(false);
  const [debugRun, setDebugRun] = useState<string | null>(null);
  const { fitView } = useReactFlow();

  const opsByType = useMemo(() => Object.fromEntries([...ops, ...PSEUDO_OPS].map((o) => [o.type, o])), [ops]);
  const domain = graph.graphKind === "domain";
  const tabular = graph.graphKind === "tabular" || domain;
  const agent = graph.graphKind === "agent";
  const rl = graph.graphKind === "rl";

  // ---- what the canvas shows: the project graph, or the definition of the module being edited
  const def: ModuleDef | undefined = scope.length ? findModule(graph, scope[scope.length - 1].module, scope[scope.length - 1].version) : undefined;
  useEffect(() => { if (scope.length && !def) setScope([]); }, [scope.length, def]);
  const cur: Graph = useMemo(() => (def ? defToView(def) : graph), [def, graph]);
  const rootValidation = useValidation(graph);
  const modValidation = useModuleValidation(graph, def?.id ?? null, def?.version ?? null);
  const validation = def ? modValidation : rootValidation;
  const v = validation.data;
  const rv = rootValidation.data;
  const backendList = useBackends();
  const compat = useCompat(graph, graph.backend, !tabular && !agent && !rl);
  /** Compatibility verdict for one canvas node: its own flat entry, or (for a module instance) the worst of its expanded children. Shown only when it carries information. */
  const compatFor = useCallback((id: string): CompatNode | undefined => {
    const nodes = compat.data?.nodes;
    if (!nodes) return undefined;
    let n: CompatNode | undefined = nodes[id];
    if (!n) {
      const kids = Object.entries(nodes).filter(([k]) => k.startsWith(`${id}/`));
      if (!kids.length) return undefined;
      const bad = kids.find(([, c]) => c.status === "unsupported");
      n = bad ? { type: "module", status: "unsupported", conversions: [], code: bad[1].code, reason: `${bad[0]}: ${bad[1].reason}` }
        : { type: "module", status: kids.some(([, c]) => c.status === "converted") ? "converted" : "supported", conversions: kids.flatMap(([k, c]) => c.conversions.map((x) => ({ kind: x.kind, detail: `${k}: ${x.detail}` }))), code: null, reason: null };
    }
    return graph.backend !== "pytorch" || n.status === "unsupported" ? n : undefined;
  }, [compat.data, graph.backend]);
  const { runs: allRuns, reload: reloadRuns } = useRuns(projectId);
  const runs = useMemo(() => allRuns.filter(isModelRun) as RunSummary[], [allRuns]);
  const tabRuns = useMemo(() => allRuns.filter(isTabularRun), [allRuns]);
  const procRuns = useMemo(() => allRuns.filter(isProcedureRun) as ProcedureRunSummary[], [allRuns]);
  const insData = useInspectionData(tabular || rl || agent ? null : ctx.runId);
  const tabRun = tabular ? tabRuns.find((r) => r.id === ctx.runId) : undefined;

  const sig = useMemo(() => JSON.stringify([graph, ui]), [graph, ui]);
  const dirty = booted && sig !== saved;

  const refreshLists = useCallback(() => {
    api.get<{ details: ProjectInfo[] }>("/api/projects").then((r) => setProjects(r.details)).catch(() => {});
    api.get<{ details: ProjectInfo[] }>("/api/examples").then((r) => setExamples(r.details)).catch(() => {});
  }, []);

  const adopt = useCallback((id: string, g: Graph, u: UiDoc | null, savedState: boolean) => {
    const uu = u ?? EMPTY_UI;
    setProjectId(id); resetDraft({ graph: g, ui: uu }); setSelNodes([]); setSelEdges([]); setScope([]); setExpanded([]); setView(g.graphKind === "domain" ? "domain" : "graph");
    setSaved(savedState ? JSON.stringify([g, uu]) : "");
    setLoadToken((n) => n + 1);
    setCtx({ runId: null, step: null, sample: null });
  }, []);

  const moveHistory = useCallback((direction: "undo" | "redo") => {
    moveDraft(direction); setSelNodes([]); setSelEdges([]); setScope([]); setExpanded([]); setCodeEdit(null);
    setCtx({ runId: null, step: null, sample: null });
    setMessage(direction === "undo" ? "Undid draft edit." : "Redid draft edit.");
  }, [moveDraft]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest('input, textarea, select, [contenteditable], .cm-editor') || event.altKey || !(event.metaKey || event.ctrlKey)
          || view === "production" || view === "scale") return;
      const direction = event.key.toLowerCase() === "z" ? (event.shiftKey ? "redo" : "undo") : event.key.toLowerCase() === "y" && event.ctrlKey ? "redo" : null;
      if (!direction) return;
      event.preventDefault();
      if (direction === "undo" ? canUndo : canRedo) moveHistory(direction);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [view, canUndo, canRedo, moveHistory]);

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

  // show the whole graph after a project is opened or the scope changes (React Flow only fits on its first render)
  useEffect(() => { const t = setTimeout(() => fitView({ padding: 0.12, maxZoom: 1 }), 150); return () => clearTimeout(t); }, [loadToken, scope.length, fitView]);

  // default inspection context: newest run, first validation sample
  useEffect(() => {
    const list = tabular ? tabRuns : runs;
    if (!ctx.runId && list.length) setCtx((c) => ({ ...c, runId: list[list.length - 1].id }));
  }, [runs, tabRuns, tabular, ctx.runId]);
  useEffect(() => { if (ctx.runId && ctx.sample == null && insData.samples.length) setCtx((c) => ({ ...c, sample: 0 })); }, [ctx.runId, ctx.sample, insData.samples.length]);
  useEffect(() => { if (!debugRun && procRuns.length) setDebugRun(procRuns[procRuns.length - 1].id); }, [procRuns, debugRun]);

  // ---------------------------------------------------------------- graph edits (the spec is the single source)
  /** Apply an edit to the graph on the canvas: the project graph, or (inside a module) the module definition. */
  const edit = (fn: (g: Graph) => Graph) => setGraph((g) => {
    if (!scope.length) return fn(g);
    const d = findModule(g, scope[scope.length - 1].module, scope[scope.length - 1].version);
    return d ? replaceModule(g, d, viewToDef(fn(defToView(d)), d)) : g;
  });
  const posKey = (id: string) => (def ? `${modKey(def)}/${id}` : id);

  const setConfig = (id: string, patch: Record<string, unknown>) => {
    const apply = (n: GNode): GNode => ({ ...n, config: Object.fromEntries(Object.entries({ ...n.config, ...patch }).filter(([, x]) => x !== undefined)) });
    if (id.includes("/") && !def) {
      // an inner node of an expanded instance: the edit goes to the module definition that contains it
      const t = resolveDefTarget(graph, id, rv?.instances);
      if (!t) { setMessage(`'${id}' is generated (not a node of a module definition) and cannot be edited directly.`); return; }
      setGraph((g) => { const d = findModule(g, t.module.id, t.module.version); return d ? replaceModule(g, d, { ...d, nodes: d.nodes.map((n) => (n.id === t.nodeId ? apply(n) : n)) }) : g; });
      return;
    }
    edit((g) => ({ ...g, nodes: g.nodes.map((n) => (n.id === id ? apply(n) : n)) }));
  };
  const setShare = (id: string, target: string | null) => edit((g) => ({ ...g, nodes: g.nodes.map((n) => (n.id === id ? { ...n, sharedWith: target } : n)) }));

  /** A wire carries the kind its producing port declares (tensor / table / fit_state / ...). */
  const wireKind = (g: Graph, node: string, port: string) => {
    const n = g.nodes.find((x) => x.id === node);
    return (n && opsByType[n.type]?.outputKinds?.[port]) || "tensor";
  };
  const connect = (toNode: string, toPort: string, from: { node: string; port: string } | null) =>
    edit((g) => {
      const edges = g.edges.filter((e) => !(e.to.node === toNode && e.to.port === toPort));
      if (from) edges.push({ id: `${from.node}_${from.port}__${toNode}_${toPort}`, kind: wireKind(g, from.node, from.port), from, to: { node: toNode, port: toPort } });
      return { ...g, edges };
    });

  const removeNodes = (ids: string[]) => {
    const s = new Set(ids.filter((i) => !isPseudo(i) && !i.includes("/")));
    if (!s.size) return;
    edit((g) => ({ ...g, nodes: g.nodes.filter((n) => !s.has(n.id)), edges: g.edges.filter((e) => !s.has(e.from.node) && !s.has(e.to.node)) }));
    setUi((u) => ({ ...u, positions: Object.fromEntries(Object.entries(u.positions).filter(([k]) => ![...s].some((x) => k === posKey(x)))) }));
    setSelNodes((a) => a.filter((x) => !s.has(x)));
    setExpanded((a) => a.filter((x) => !s.has(x)));
  };
  const removeEdges = (ids: string[]) => { const s = new Set(ids.filter((i) => !i.startsWith("inner:") && !i.startsWith("bnd:"))); edit((g) => ({ ...g, edges: g.edges.filter((e) => !s.has(e.id)) })); setSelEdges((a) => a.filter((x) => !s.has(x))); };

  const rename = (oldId: string, newId: string): string | null => {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(newId)) return "Use letters, digits and underscores, starting with a letter.";
    if (cur.nodes.some((n) => n.id === newId)) return `Node id '${newId}' is already used.`;
    edit((g) => ({
      ...g,
      nodes: g.nodes.map((n) => (n.id === oldId ? { ...n, id: newId, stateRef: n.stateRef ? n.stateRef.replace(new RegExp(`${oldId}$`), newId) : n.stateRef } : n.sharedWith === oldId ? { ...n, sharedWith: newId } : n)),
      edges: g.edges.map((e) => ({ ...e, id: e.id.split(oldId).join(newId), from: e.from.node === oldId ? { ...e.from, node: newId } : e.from, to: e.to.node === oldId ? { ...e.to, node: newId } : e.to })),
    }));
    setUi((u) => { const { [posKey(oldId)]: p, ...rest } = u.positions; return { ...u, positions: p ? { ...rest, [posKey(newId)]: p } : rest }; });
    setSelNodes([newId]);
    return null;
  };

  const viewsNow = v?.nodes ?? {};
  const portsOf = (nid: string, type: string, dir: "inputs" | "outputs") => {
    const vw = viewsNow[nid];
    return (dir === "inputs" ? vw?.inputPorts : vw?.outputPorts) ?? opsByType[type]?.[dir] ?? [];
  };

  /** Settings for a new structural / code node, picked from what the project defines. */
  const structuralDefaults = (type: string): Record<string, unknown> | string => {
    const mods = graph.modules ?? [];
    const m0 = mods[0];
    if (type === "core.composite" || type === "core.repeat" || type === "core.select") {
      if (!m0) return "This project has no modules yet. Create one under Modules, or import a starter module.";
      const args = (m: ModuleDef) => Object.fromEntries(m.params.map((p) => [p.name, p.default]));
      if (type === "core.composite") return { module: m0.id, version: m0.version, args: args(m0), share: "clone" };
      if (type === "core.repeat") return { module: m0.id, version: m0.version, args: args(m0), count: 2, carry: [], termination: { kind: "fixed_count" }, share: "tied" };
      const m1 = mods[1] ?? m0;
      return { then: { module: m0.id, version: m0.version, args: args(m0) }, otherwise: { module: m1.id, version: m1.version, args: args(m1) } };
    }
    if (type === "code.block") {
      const c0 = (graph.codeBlocks ?? [])[0];
      return c0 ? { block: c0.id, version: c0.version, params: {}, seed: 0 } : "This project has no code blocks yet. Create one under Modules ▸ New code block.";
    }
    return {};
  };

  const addNode = (type: string, config: Record<string, unknown>, version = "1.0.0") => {
    const id = nextId(type === "core.composite" ? String((config as any).module) : type === "code.block" ? String((config as any).block) : type.split(".").pop()!, new Set(cur.nodes.map((n) => n.id)));
    const anchor = selNodes.length === 1 ? cur.nodes.find((n) => n.id === selNodes[0]) : undefined;
    const ap = anchor ? ui.positions[posKey(anchor.id)] : undefined;
    const maxX = Math.max(-200, ...cur.nodes.map((n) => ui.positions[posKey(n.id)]?.x ?? 0));
    const pos = ap ? { x: ap.x + 260, y: ap.y } : { x: maxX + 260, y: 120 };
    edit((g) => ({ ...g, nodes: [...g.nodes, { id, type, version, config, stateRef: null }] }));
    setUi((u) => ({ ...u, positions: { ...u.positions, [posKey(id)]: pos } }));
    setSelNodes([id]); setSelEdges([]);
    return id;
  };

  const addBlock = (op: OpInfo) => {
    if (op.type === "code.block" && def) { setMessage("Code blocks are added from the project graph; leave the module first."); return; }
    const special = ["core.composite", "core.repeat", "core.select", "code.block"].includes(op.type) ? structuralDefaults(op.type) : null;
    if (typeof special === "string") { setMessage(special); setLeftTab("Modules"); return; }
    const id = nextId(shortName(op), new Set(cur.nodes.map((n) => n.id)));
    const node: GNode = { id, type: op.type, version: op.version, config: special ?? JSON.parse(JSON.stringify(op.defaults)), stateRef: null };
    const anchor = selNodes.length === 1 ? cur.nodes.find((n) => n.id === selNodes[0]) : undefined;
    const ap = anchor ? ui.positions[posKey(anchor.id)] : undefined;
    const maxX = Math.max(-200, ...cur.nodes.map((n) => ui.positions[posKey(n.id)]?.x ?? 0));
    const pos = ap ? { x: ap.x + 260, y: ap.y } : { x: maxX + 260, y: 120 };
    edit((g) => {
      const edges = [...g.edges];
      const aop = anchor ? opsByType[anchor.type] : undefined;
      if (anchor && aop && op.inputs.length) {
        const aouts = portsOf(anchor.id, anchor.type, "outputs");
        // first free input port that accepts one of the anchor's outputs (same wire kind)
        const pair = aouts.flatMap((po) => op.inputs.filter((pi) => (op.inputKinds?.[pi] ?? "tensor") === (aop.outputKinds?.[po] ?? "tensor")).map((pi) => [po, pi] as const))[0];
        if (pair) {
          const from = { node: anchor.id, port: pair[0] }, to = { node: id, port: pair[1] };
          edges.push({ id: `${from.node}_${from.port}__${to.node}_${to.port}`, kind: aop.outputKinds?.[pair[0]] ?? "tensor", from, to });
        }
      }
      return { ...g, nodes: [...g.nodes, node], edges };
    });
    setUi((u) => ({ ...u, positions: { ...u.positions, [posKey(id)]: pos } }));
    setSelNodes([id]); setSelEdges([]);
  };

  /** Called by the Data workspace: add a connector source node to the open (tabular) graph. Returns a message for the user. */
  const addSource = (type: string, config: Record<string, unknown>): string | null => {
    const op = opsByType[type];
    if (!op) return `The registry has no '${type}' block.`;
    if (graph.graphKind !== "tabular") return "The open graph is a model graph. Use 'New tabular graph' (or open a tabular project) first.";
    const id = nextId(shortName(op), new Set(graph.nodes.map((n) => n.id)));
    const node: GNode = { id, type, version: op.version, config: { ...JSON.parse(JSON.stringify(op.defaults)), ...config }, stateRef: null };
    const maxX = Math.max(-200, ...Object.values(ui.positions).map((p) => p.x));
    setGraph((g) => ({ ...g, nodes: [...g.nodes, node] }));
    setUi((u) => ({ ...u, positions: { ...u.positions, [id]: { x: maxX + 260, y: 120 } } }));
    setSelNodes([id]);
    return `Added node '${id}' to the graph '${projectId}'. Switch to the Graph view to connect and run it.`;
  };
  const openRun = (runId: string) => {
    if (allRuns.some((r) => r.id === runId)) { setCtx({ runId, step: null, sample: null }); setView("graph"); setMessage(`Inspecting run ${runId}.`); }
    else setMessage(`Run ${runId} belongs to another project; open that project to inspect it.`);
  };

  // ---------------------------------------------------------------- modules and code blocks
  const openModule = (m: { id: string; version: string }, via: string) => { setScope((s) => [...s, { module: m.id, version: m.version, via }]); setSelNodes([]); setSelEdges([]); setView("graph"); };
  const openInstance = (nid: string) => {
    const n = cur.nodes.find((x) => x.id === nid);
    if (!n) return;
    const c = n.config as any;
    const ref = n.type === "core.select" ? c.then : c;
    if (!ref?.module) { setMessage("This instance does not name a module yet."); return; }
    const m = findModule(graph, ref.module, ref.version ?? "1.0.0");
    if (!m) { setMessage(`Module ${ref.module} v${ref.version ?? "1.0.0"} is not defined in this project.`); return; }
    openModule(m, nid);
  };
  const toggleExpand = (path: string) => setExpanded((a) => (a.includes(path) ? a.filter((x) => x !== path) : [...a, path]));
  const addInstance = (m: ModuleDef) => {
    addNode("core.composite", { module: m.id, version: m.version, args: Object.fromEntries(m.params.map((p) => [p.name, p.default])), share: "clone" });
  };
  const importModule = (m: ModuleDef, silent?: boolean) => {
    if (findModule(graph, m.id, m.version)) { if (!silent) setMessage(`${m.id} v${m.version} is already in this project.`); return; }
    setGraph((g) => ({ ...g, modules: [...(g.modules ?? []), JSON.parse(JSON.stringify(m))] }));
    setMessage(`Imported ${m.id} v${m.version} into this project (a copy: the project's semantic hash now covers it).`);
  };
  const createModule = () => {
    const m = newModule("my_module", graph.modules ?? []);
    setGraph((g) => ({ ...g, modules: [...(g.modules ?? []), m] }));
    openModule(m, "");
    setMessage(`Created module ${m.id}. Add blocks, wire them to the input box and to the output box, then use Interface to name the ports.`);
  };
  const removeModule = (m: ModuleDef) => setGraph((g) => ({ ...g, modules: (g.modules ?? []).filter((x) => !(x.id === m.id && x.version === m.version)) }));
  const addCodeNode = (d: CodeBlockDef) => { addNode("code.block", { block: d.id, version: d.version, params: {}, seed: 0 }); };
  const createCode = async () => {
    let id = "my_block", i = 1;
    while ((graph.codeBlocks ?? []).some((c) => c.id === id)) id = `my_block_${i++}`;
    const base: CodeBlockDef = { id, version: "1.0.0", description: "", inputs: [{ name: "x", dtype: "float32", shape: ["N", 4] }], outputs: [{ name: "y", dtype: "float32", same_as: "x" }], config: [], state: [], effects: [], randomness: "none", differentiable: true, dependencies: [], source: "", fixtures: [{ name: "random batch", inputs: { x: { shape: [2, 4], seed: 0 } } }], limits: {} };
    try { base.source = (await api.post<{ source: string }>("/api/codeblocks/template", { block: base })).source; } catch (e) { setMessage(errorText(e)); }
    setGraph((g) => ({ ...g, codeBlocks: [...(g.codeBlocks ?? []), base] }));
    setCodeEdit({ id, version: base.version });
  };
  const importCode = (d: CodeBlockDef) => { setGraph((g) => ((g.codeBlocks ?? []).some((c) => c.id === d.id && c.version === d.version) ? g : { ...g, codeBlocks: [...(g.codeBlocks ?? []), d] })); setMessage(`Imported code block ${d.id} v${d.version}.`); };
  const codeDef = codeEdit ? (graph.codeBlocks ?? []).find((c) => c.id === codeEdit.id && c.version === codeEdit.version) : undefined;
  const codeUsedBy = codeEdit ? graph.nodes.filter((n) => n.type === "code.block" && (n.config as any).block === codeEdit.id).map((n) => n.id) : [];

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
  const auto = useMemo(() => (def ? autoLayout(cur) : {}), [def, cur]);
  /** Instances drawn expanded (project level only): their inner nodes and wires from the validation. */
  const expandedGroups = useMemo(() => {
    if (def || !rv?.instances) return [];
    return expanded.filter((p) => rv.instances![p] && graph.nodes.some((n) => n.id === p)).map((p) => {
      const inst = rv.instances![p];
      const members = inst.members.filter((m) => rv.flat?.[m]);
      const lay = autoLayout({ ...EMPTY, nodes: members.map((id) => ({ id, type: "", version: "", config: {} })), edges: (inst.innerEdges ?? []).map((e) => ({ id: e.id, kind: "tensor", from: e.from, to: e.to })) }, 262, 150, 28, 52);
      const w = Math.max(320, ...Object.values(lay).map((q) => q.x + 262)), h = Math.max(160, ...Object.values(lay).map((q) => q.y + 150));
      return { path: p, inst, members, lay, w, h };
    });
  }, [def, rv, expanded, graph.nodes]);
  const expandedPaths = new Set(expandedGroups.map((g) => g.path));
  const expandedGroupsRef = { current: expandedGroups };
  const basePos = (id: string, i: number) => ui.positions[posKey(id)] ?? (def ? auto[id] : undefined) ?? { x: 60 + i * 260, y: 120 };
  /** nodes to the right of an expanded instance move right by the extra width of its frame, so nothing is drawn on top of it */
  const posOf = (id: string, i: number) => {
    const p = basePos(id, i);
    let dx = 0;
    for (const g of expandedGroupsRef.current) { const gi = cur.nodes.findIndex((n) => n.id === g.path); if (g.path !== id && gi >= 0 && basePos(g.path, gi).x < p.x) dx += g.w - 250; }
    return dx ? { x: p.x + dx, y: p.y } : p;
  };


  const rfNodes: AnyNode[] = useMemo(() => {
    const out: AnyNode[] = [];
    cur.nodes.forEach((n, i) => {
      const nv: NodeView | undefined = v?.nodes[n.id];
      const structural = ["core.composite", "core.repeat", "core.select"].includes(n.type);
      const grp = expandedGroups.find((g) => g.path === n.id);
      if (grp) {
        out.push({ id: n.id, type: "group", position: posOf(n.id, i), selected: selNodes.includes(n.id), style: { width: grp.w, height: grp.h },
          data: { path: n.id, module: grp.inst.module, version: grp.inst.version, kind: grp.inst.kind, inputs: grp.inst.inputs, outputs: grp.inst.outputs, width: grp.w, height: grp.h, note: grp.inst.note,
            params: grp.inst.params, errors: grp.inst.diagnostics.filter((d) => d.severity === "error").length, sharedWith: grp.inst.sharedWith, onCollapse: () => toggleExpand(n.id), onOpen: () => openInstance(n.id) } });
        for (const m of grp.members) {
          const fv = rv!.flat![m];
          const op = opsByType[fv.type ?? ""];
          out.push({ id: m, type: "card", parentId: n.id, extent: "parent", draggable: false, position: grp.lay[m], selected: selNodes.includes(m),
            data: { gnode: { id: m, type: fv.type ?? "", version: fv.nodeVersion ?? "1.0.0", config: (fv.resolvedConfig ?? {}) as Record<string, unknown> }, op, view: fv, pending: false, inner: true, localName: m.slice(n.id.length + 1), compat: compatFor(m) } });
        }
        return;
      }
      out.push({
        id: n.id, type: "card" as const, position: posOf(n.id, i), selected: selNodes.includes(n.id),
        data: { gnode: n, op: opsByType[n.type], view: nv, pending: validation.pending, compat: def ? undefined : compatFor(n.id), runStatus: tabRun?.nodes.find((x) => x.node === n.id),
          onOpen: structural ? () => openInstance(n.id) : undefined, onToggle: structural && !def && n.type !== "core.select" ? () => toggleExpand(n.id) : undefined, expanded: expandedPaths.has(n.id) },
      });
    });
    return out;
  }, [cur.nodes, ui.positions, selNodes, opsByType, v, rv, validation.pending, tabRun, expandedGroups, def, compatFor]); // eslint-disable-line react-hooks/exhaustive-deps

  const rfEdges: Edge[] = useMemo(() => {
    const base = cur.edges.map((e) => {
      const t = v?.nodes[e.from.node]?.outputShapes?.[e.from.port];
      const bad = (v?.nodes[e.to.node]?.diagnostics ?? []).some((d) => d.severity === "error" && d.port === e.to.port);
      return {
        id: e.id, source: e.from.node, sourceHandle: e.from.port, target: e.to.node, targetHandle: e.to.port, selected: selEdges.includes(e.id),
        label: bad ? `${fmtShape(t)} ✖` : fmtShape(t), interactionWidth: 28, markerEnd: { type: MarkerType.ArrowClosed },
        style: bad ? { stroke: "#c62828", strokeWidth: 2 } : undefined, labelStyle: { fontSize: 10, fill: bad ? "#c62828" : "#444" },
        labelBgStyle: { fill: "#fff", fillOpacity: 0.85 }, labelBgPadding: [3, 2] as [number, number],
      } as Edge;
    });
    for (const g of expandedGroups) {
      for (const e of g.inst.innerEdges ?? []) base.push({ id: `inner:${e.id}`, source: e.from.node, sourceHandle: e.from.port, target: e.to.node, targetHandle: e.to.port, markerEnd: { type: MarkerType.ArrowClosed }, style: { stroke: "#7c8aa5" }, selectable: false } as Edge);
      for (const b of g.inst.boundary ?? []) {
        if (!b.node) continue;
        base.push(b.kind === "in"
          ? { id: `bnd:in:${g.path}:${b.port}:${b.node}:${b.nodePort}`, source: g.path, sourceHandle: `in-${b.port}`, target: b.node, targetHandle: b.nodePort, markerEnd: { type: MarkerType.ArrowClosed }, style: { stroke: "#7c8aa5", strokeDasharray: "4 3" }, selectable: false } as Edge
          : { id: `bnd:out:${g.path}:${b.port}`, source: b.node, sourceHandle: b.nodePort, target: g.path, targetHandle: `out-${b.port}`, markerEnd: { type: MarkerType.ArrowClosed }, style: { stroke: "#7c8aa5", strokeDasharray: "4 3" }, selectable: false } as Edge);
      }
    }
    return base;
  }, [cur.edges, selEdges, v, expandedGroups]);

  const onNodesChange = useCallback((changes: NodeChange<AnyNode>[]) => {
    const moved = applyNodeChanges(changes, rfNodes);
    const sel = changes.some((c) => c.type === "select");
    if (sel) setSelNodes(moved.filter((n) => n.selected).map((n) => n.id));
    if (changes.some((c) => c.type === "position")) setUi((u) => ({ ...u, positions: { ...u.positions, ...Object.fromEntries(moved.filter((n) => !n.parentId).map((n) => [posKey(n.id), { x: n.position.x, y: n.position.y }])) } }), changes.some(c => c.type === "position" && c.dragging === true));
  }, [rfNodes]); // eslint-disable-line react-hooks/exhaustive-deps
  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    if (changes.some((c) => c.type === "select")) setSelEdges(applyEdgeChanges(changes, rfEdges).filter((e) => e.selected).map((e) => e.id));
  }, [rfEdges]);
  const onConnect = (c: Connection) => { if (c.source && c.target && c.sourceHandle && c.targetHandle) connect(c.target, c.targetHandle, { node: c.source, port: c.sourceHandle }); };

  // ---- selection -> inspector
  const flatId = selNodes.length === 1 && selNodes[0].includes("/") ? selNodes[0] : null;
  const flatView: NodeView | undefined = flatId ? rv?.flat?.[flatId] : undefined;
  const innerTarget = flatId ? resolveDefTarget(graph, flatId, rv?.instances) : null;
  const selNode: GNode | undefined = flatId && flatView
    ? { id: flatId, type: flatView.type ?? "", version: flatView.nodeVersion ?? "1.0.0", config: (flatView.resolvedConfig ?? {}) as Record<string, unknown>, sharedWith: flatView.sharedWith }
    : selNodes.length === 1 && selEdges.length === 0 ? cur.nodes.find((n) => n.id === selNodes[0]) : undefined;
  const selView: NodeView | undefined = flatId ? flatView : selNode ? v?.nodes[selNode.id] : undefined;
  const selEdge = selEdges.length === 1 && selNodes.length === 0 ? cur.edges.find((e) => e.id === selEdges[0]) : undefined;
  const errCount = rv ? rv.diagnostics.filter((d) => d.severity === "error").length : 0;
  const extra: InnerActions | undefined = selNode ? {
    inner: flatId && innerTarget ? { module: innerTarget.module.id, version: innerTarget.module.version, instances: rv?.modules?.find((m) => m.id === innerTarget.module.id && m.version === innerTarget.module.version)?.usedBy.length ?? 0, local: innerTarget.nodeId } : null,
    allViews: viewsNow, onOpenModule: ["core.composite", "core.repeat", "core.select"].includes(selNode.type) ? () => openInstance(selNode.id) : undefined,
    onToggleExpand: !def && ["core.composite", "core.repeat"].includes(selNode.type) ? () => toggleExpand(selNode.id) : undefined, expanded: expandedPaths.has(selNode.id),
    onShare: (t) => setShare(selNode.id, t), onEditCode: (id, version) => setCodeEdit({ id, version }),
  } : undefined;
  const instUsed = def ? (rv?.modules?.find((m) => m.id === def.id && m.version === def.version)?.usedBy ?? []) : [];

  const unusedNote = !tabular && graph.modules && graph.modules.length > 0 && !graph.nodes.some((n) => ["core.composite", "core.repeat", "core.select"].includes(n.type));

  return (
    <GraphContext.Provider value={graph}>
    <div className={`app${view === "graph" && !agent && !rl ? " keyboard-layout" : ""}`}>
      <header className="topbar">
        <b className="brand">Project Void</b>
        <label>Project <input value={projectId} onChange={(e) => setProjectId(e.target.value)} aria-label="project id" size={16} /></label>
        <button onClick={() => save().catch(() => {})}>Save{dirty ? " *" : ""}</button>
        <button aria-label="undo draft edit" title="Undo graph settings/layout; recorded runs and external actions are retained" disabled={!canUndo} onClick={() => moveHistory("undo")}>Undo</button>
        <button aria-label="redo draft edit" title="Redo graph settings/layout" disabled={!canRedo} onClick={() => moveHistory("redo")}>Redo</button>
        <label>Open <select value="" onChange={(e) => { const [k, ...r] = e.target.value.split(":"); if (k) load(k as "project" | "example", r.join(":")); }} aria-label="open project">
          <option value="">choose…</option>
          {projects.length > 0 && <optgroup label="Saved projects">{projects.map((p) => <option key={p.id} value={`project:${p.id}`}>{p.id} [{p.graphKind}]</option>)}</optgroup>}
          <optgroup label="Examples — model graphs">{examples.filter((p) => p.graphKind === "model").map((p) => <option key={p.id} value={`example:${p.id}`}>{p.id}{p.synthetic ? " (synthetic data)" : ""}</option>)}</optgroup>
          <optgroup label="Examples — agent graphs (LangGraph)">{examples.filter((p) => p.graphKind === "agent").map((p) => <option key={p.id} value={`example:${p.id}`}>{p.id}{p.synthetic ? " (synthetic data)" : ""}</option>)}</optgroup>
          <optgroup label="Examples — reinforcement learning (Gymnasium)">{examples.filter((p) => p.graphKind === "rl").map((p) => <option key={p.id} value={`example:${p.id}`}>{p.id}{p.synthetic ? " (synthetic data)" : ""}</option>)}</optgroup>
          <optgroup label="Examples — vision / NLP / speech">{examples.filter((p) => p.graphKind === "domain").map((p) => <option key={p.id} value={`example:${p.id}`}>{p.id} (SYNTHETIC)</option>)}</optgroup>
          <optgroup label="Examples — tabular / statistics graphs">{examples.filter((p) => p.graphKind === "tabular").map((p) => <option key={p.id} value={`example:${p.id}`}>{p.id}{p.synthetic ? " (synthetic data)" : ""}</option>)}</optgroup>
        </select></label>
        {!tabular && !agent && !rl && (
          <label title="Backend that executes this model graph. Switching never rewrites settings; unsupported nodes are reported before execution.">Backend{" "}
            <select value={graph.backend} onChange={(e) => setGraph((g) => ({ ...g, backend: e.target.value }))} aria-label="backend">
              {(backendList.length ? backendList : [{ id: "pytorch", title: "PyTorch", available: true }, { id: "keras", title: "TensorFlow / Keras 3", available: true }, { id: "jax", title: "JAX", available: true }] as { id: string; title: string; available: boolean }[]).map((b) =>
                <option key={b.id} value={b.id}>{b.title}{b.available ? "" : " (unavailable)"}</option>)}
              {!backendList.some((b) => b.id === graph.backend) && backendList.length > 0 && <option value={graph.backend}>{graph.backend} (unsupported)</option>}
            </select>
            {compat.data && graph.backend !== "pytorch" && <span className={`badge ${compat.data.ok ? "compat-supported" : "compat-unsupported"}`} title="Backend compatibility of this graph">{compat.data.ok ? "compatible" : `${compat.data.counts.unsupported} unsupported`}</span>}
          </label>
        )}
        <span className="viewtabs" role="tablist" aria-label="workspace">
          <button role="tab" aria-selected={view === "scale"} className={view === "scale" ? "on" : ""} onClick={() => setView("scale")}>Integrations</button>
          <button role="tab" aria-selected={view === "production"} className={view === "production" ? "on" : ""} onClick={() => setView("production")}>Production</button>
          {(domain ? [["domain", "Domain workspace"], ["graph", "Graph"], ["coverage", "Coverage"]] as [View, string][] : rl ? [["graph", "RL lab"]] as [View, string][] : agent ? [["graph", "Agent"], ["data", "Data"]] as [View, string][] : [["graph", "Graph"], ["data", "Data"], ["experiments", "Experiments"], ...(tabular ? [] : [["training", "Training"], ["debug", "Debug"], ["attention", "Attention"], ["backends", "Backend"]]), ["coverage", "Coverage"]] as [View, string][]).map(([k, label]) => (
            <button key={k} role="tab" aria-selected={view === k} className={view === k ? "on" : ""} onClick={() => setView(k)}>{label}</button>))}
        </span>
        <span className="badge kind" title="Graph kind: wires of different kinds never mean the same thing">{graph.graphKind} graph</span>
        <button onClick={() => { if (!dirty || window.confirm("Discard unsaved changes?")) adopt("untitled", EMPTY, null, false); }}>New model graph</button>
        <button onClick={() => { if (!dirty || window.confirm("Discard unsaved changes?")) adopt("untitled_tabular", EMPTY_TABULAR, null, false); }}>New tabular graph</button>
        <button onClick={() => { if (!dirty || window.confirm("Discard unsaved changes?")) adopt("untitled_agent", EMPTY_AGENT, null, false); }}>New agent graph</button>
        <button onClick={() => { if (dirty && !window.confirm("Discard unsaved changes?")) return; api.get<any>("/api/examples/rl_cartpole_dqn").then((ex) => adopt("untitled_rl", ex.graph, ex.ui, false)).catch((e) => setMessage(errorText(e))); }} title="Starts from a complete, valid CartPole DQN graph (all six RL nodes wired); change the environment on the Environment tab">New RL graph</button>
        {!tabular && !agent && !rl && <button onClick={() => setShowCode(true)} title="Show generated native code for a backend (compatibility is checked first)">Export code</button>}
        <span className="spacer" />
        <span className={`vsum ${errCount ? "bad" : "good"}`} aria-live="polite">
          {rootValidation.error ? `validation unavailable: ${rootValidation.error}` : rootValidation.pending ? "validating…" : rv ? (errCount ? `${errCount} error${errCount > 1 ? "s" : ""}` : ((tabular || agent || rl) ? `valid · ${graph.nodes.length} nodes` : `valid · ${fmtInt(rv.totalParams)} parameters`)) : ""}
          {rv && <small> · graph {rv.graphHash.slice(0, 8)}</small>}
        </span>
      </header>
      {view === "graph" && !agent && !rl && ui.description && <div className={`notice ${ui.synthetic ? "synthetic" : ""}`}>{ui.synthetic && <b>Synthetic / teaching data. </b>}{ui.description}</div>}
      {message && <div className="toast" role="status" onClick={() => setMessage(null)}>{message} <small>(click to dismiss)</small></div>}

      {view === "production" && <ProductionWorkspace onOpenRun={openRun} />}
      {view === "scale" && <ScaleWorkspace graph={graph} ui={ui} projectId={projectId} onImport={(p) => { adopt(p.projectId, p.graph, p.ui, true); refreshLists(); }} onOpenRun={openRun} />}

      {view === "domain" && domain && <DomainWorkspace key={projectId} projectId={projectId} graph={graph} validation={rv ?? null} ops={opsByType} runs={tabRuns} runId={ctx.runId}
        setRunId={(id) => setCtx({ runId: id, step: null, sample: null })} reloadRuns={reloadRuns} ensureSaved={ensureSaved} onConfig={setConfig} onDataset={(d) => {
          const source = { vision: "domain.vision_source", nlp: "domain.nlp_source", speech: "domain.audio_source" }[d.family];
          const trainer = { vision: "domain.vision_segmenter", nlp: "domain.nlp_tagger", speech: "domain.speech_ctc" }[d.family];
          setGraph((g) => ({ ...g, nodes: g.nodes.map((n) => ({ ...n, config:
            n.type === source ? { ...n.config, path: d.path, n: null } :
            n.type === trainer ? { ...n.config, resume_model_id: null } :
            d.family === "nlp" && n.type === "domain.nlp_tokenizer" ? { ...n.config, fitted_model_id: null } : n.config })) }));
          setUi((u) => ({ ...u, synthetic: d.synthetic, description: `Imported ${d.kind}; synthetic=${d.synthetic} (user declared); ${d.license.declaration}` }));
        }} />}
      {view === "graph" && rl && (
        <RLWorkspace projectId={projectId} graph={graph} setGraph={setGraph} ui={ui} validation={rv ?? null} allRuns={allRuns} reloadRuns={reloadRuns} ensureSaved={ensureSaved} />
      )}
      {view === "backends" && !tabular && !agent && !rl && (
        <BackendWorkspace graph={graph} setBackend={(b) => setGraph((g) => ({ ...g, backend: b }))} report={compat.data} pending={compat.pending} error={compat.error} backends={backendList} onExport={() => setShowCode(true)} />
      )}
      {view === "coverage" && !rl && !agent && <CoverageView />}
      {view === "data" && !rl && <DataWorkspace onAddSource={addSource} tabular={tabular} />}
      {view === "graph" && agent && (
        <AgentWorkspace projectId={projectId} graph={graph} setGraph={setGraph} ui={ui} setUi={setUi} validation={rv ?? null} ops={ops} allRuns={allRuns} reloadRuns={reloadRuns} setMessage={setMessage} requestedRunId={ctx.runId} />
      )}
      {view === "experiments" && !rl && <Experiments graph={graph} validation={rv ?? null} projectId={projectId} ops={opsByType} ensureSaved={ensureSaved} onOpenRun={openRun} />}
      {view === "training" && !tabular && (
        <div className="fullws"><TrainingWorkspace projectId={projectId} graph={graph} setGraph={setGraph} runs={procRuns} reloadRuns={reloadRuns} ensureSaved={ensureSaved} valid={!!rv?.ok}
          onOpenDebug={(id) => { setDebugRun(id); setView("debug"); }} setMessage={setMessage} /></div>
      )}
      {view === "debug" && !tabular && (
        <div className="fullws"><DebuggerWorkspace graph={graph} validation={rv ?? null} runs={procRuns} runId={debugRun} setRunId={setDebugRun} setMessage={setMessage} onEditProcedure={() => setView("training")} /></div>
      )}
      {view === "attention" && !tabular && <div className="fullws"><AttentionWorkspace graph={graph} runs={procRuns} setMessage={setMessage} /></div>}
      {view === "graph" && !agent && !rl && <>
      <KeyboardGraphTools graph={cur} ops={opsByType} selectedNode={selNodes[0] ?? ""} selectedWire={selEdges[0] ?? ""} onNode={(id) => { setSelNodes(id ? [id] : []); setSelEdges([]); }} onWire={(id) => { setSelEdges(id ? [id] : []); setSelNodes([]); }} onConnect={connect} />
      <aside className="left">
        {!tabular && (
          <div className="tabs" role="tablist" style={{ marginTop: 0 }}>
            {(["Blocks", "Modules"] as const).map((t) => <button key={t} role="tab" aria-selected={leftTab === t} className={leftTab === t ? "on" : ""} onClick={() => setLeftTab(t)}>{t === "Modules" ? "Modules & code" : t}</button>)}
          </div>
        )}
        {(tabular || leftTab === "Blocks") ? <Library ops={ops.filter((o) => o.graphKind === graph.graphKind)} onAdd={addBlock} />
          : <ModuleLibrary graph={graph} inModule={!!def} onAddInstance={addInstance} onOpen={(m) => openModule(m, "")} onNew={createModule} onImport={importModule} onRemove={removeModule}
            onAddCodeNode={addCodeNode} onEditCode={(d) => setCodeEdit({ id: d.id, version: d.version })} onNewCode={createCode} onImportCode={importCode} onRepoImport={() => setRepoImport(true)} usedModules={usedModules(graph)} setMessage={setMessage} />}
      </aside>

      <main className="center">
        {(scope.length > 0 || expanded.length > 0) && (
          <nav className="crumbs" aria-label="breadcrumb">
            <button className="link" onClick={() => setScope([])}>{projectId}</button>
            {scope.map((s, i) => (
              <span key={i}> › <button className="link" onClick={() => setScope(scope.slice(0, i + 1))}>{s.via ? `${s.via} (${s.module} v${s.version})` : `${s.module} v${s.version}`}</button></span>
            ))}
            {def && <span className="muted small"> — editing the module definition: changes apply to every instance</span>}
          </nav>
        )}
        <ReactFlow<AnyNode, Edge> nodes={rfNodes} edges={rfEdges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
          onConnect={onConnect} onNodesDelete={(ns) => removeNodes(ns.map((n) => n.id))} onEdgesDelete={(es) => removeEdges(es.map((e) => e.id))}
          deleteKeyCode={["Backspace", "Delete"]} fitView minZoom={0.2} onPaneClick={() => { setSelNodes([]); setSelEdges([]); }} proOptions={{ hideAttribution: true }}>
          <Background gap={20} />
          <Controls showInteractive={false} />
        </ReactFlow>
        {cur.nodes.length === 0 && <div className="canvas-empty">{tabular ? "Empty tabular graph. Add a CSV table source from the library, or open an example." : "Empty graph. Add blocks from the library, or open the reference_cnn example."}</div>}
        {unusedNote && <div className="canvas-hint">This project defines modules; add instances from Modules &amp; code.</div>}
      </main>

      <aside className="right">
        {tabular ? <TabularRunBar runs={tabRuns} runId={ctx.runId} setRunId={(id) => setCtx({ runId: id, step: null, sample: null })} currentHash={rv?.graphHash} />
          : <InspectionBar runs={runs} ctx={ctx} setCtx={setCtx} currentHash={rv?.graphHash} checkpoints={insData.checkpoints} samples={insData.samples} />}
        {selNode ? (
          <NodeInspector key={selNode.id} node={selNode} op={opsByType[selNode.type]} ops={opsByType} view={selView} graph={cur} ctx={ctx}
            tabSet={tabular ? tabularTabs(opsByType[selNode.type], selNode, selView, ctx.runId, (patch) => setConfig(selNode.id, patch), tabRun?.maxSeq) : undefined}
            extra={{ ...extra, moduleEditing: !!def }}
            onConfig={(patch) => setConfig(selNode.id, patch)} onConnect={connect} onDelete={() => removeNodes([selNode.id])} onRename={(nid) => rename(selNode.id, nid)} />
        ) : selEdge ? (
          tabular ? <TabularWireInspector edge={selEdge} graph={cur} validation={v} runId={ctx.runId} ops={opsByType} />
            : <WireInspector edge={selEdge} graph={cur} validation={v} ctx={ctx} ops={opsByType} />
        ) : (
          <div className="empty pad">Select a node to edit it, or click a wire to inspect the value crossing it.{def ? " You are inside a module: nodes here are its definition." : ""}</div>
        )}
      </aside>

      <section className="bottom">
        {def ? (
          <ModulePanel graph={graph} setGraph={(f) => setGraph(f)} def={def} validation={modValidation.data} usedBy={instUsed} onExit={() => setScope([])} setMessage={setMessage} />
        ) : tabular ? (
          <TabularRunPanel projectId={projectId} graph={graph} validation={v} runs={tabRuns} reloadRuns={reloadRuns} runId={ctx.runId}
            setRunId={(id) => setCtx({ runId: id, step: null, sample: null })} ensureSaved={ensureSaved} onSelectNode={(id) => { setSelNodes([id]); setSelEdges([]); }} />
        ) : graph.backend !== "pytorch" ? (
          <div className="pad">
            <b>Training runs are PyTorch-only.</b> This graph targets <b>{graph.backend}</b>: the {graph.backend} backend executes forward, loss, gradients and one SGD step (tested against PyTorch, see the Backend tab),
            but there is no {graph.backend} training run, checkpoint or run history in this version. Switch the backend to PyTorch to train; the graph itself is unchanged.
            {compat.data && !compat.data.ok && <div className="error">The graph is also not compatible with {graph.backend} (see the badges on the canvas).</div>}
          </div>
        ) : (
          <RunPanel projectId={projectId} graph={graph} validation={v} runs={runs} reloadRuns={reloadRuns} ctx={ctx} setCtx={setCtx}
            baseline={ui.pinnedBaseline ?? null} setBaseline={(id) => setUi((u) => ({ ...u, pinnedBaseline: id }))} ensureSaved={ensureSaved} />
        )}
      </section>
      </>}

      {codeEdit && codeDef && (
        <CodeBlockEditor def={codeDef} usedBy={codeUsedBy} setMessage={setMessage} onClose={() => setCodeEdit(null)}
          onChange={(d) => {
            setGraph((g) => ({ ...g, codeBlocks: (g.codeBlocks ?? []).map((c) => (c.id === codeEdit.id && c.version === codeEdit.version ? d : c)), nodes: g.nodes.map((n) => (n.type === "code.block" && (n.config as any).block === codeEdit.id && (n.config as any).version === codeEdit.version && (d.id !== codeEdit.id || d.version !== codeEdit.version) ? { ...n, config: { ...n.config, block: d.id, version: d.version } } : n)) }));
            if (d.id !== codeEdit.id || d.version !== codeEdit.version) setCodeEdit({ id: d.id, version: d.version });
          }} />
      )}
      {repoImport && <RepoImport setMessage={setMessage} onClose={() => setRepoImport(false)} onImport={(d) => { importCode(d); setCodeEdit({ id: d.id, version: d.version }); }} />}
      {showCode && <ExportView projectId={projectId} graph={graph} initialBackend={graph.backend} onClose={() => setShowCode(false)} />}
    </div>
    </GraphContext.Provider>
  );
}
