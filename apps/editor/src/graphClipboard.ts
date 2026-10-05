import type { CodeBlockDef, GEdge, GNode, Graph, ModuleDef, UiDoc } from "./types";

export interface GraphClipboard {
  graphKind: string; backend: string; sourceProject: string; sourceGraphHash: string;
  nodes: GNode[]; edges: GEdge[]; modules: ModuleDef[]; codeBlocks: CodeBlockDef[];
  positions: UiDoc["positions"]; omittedBoundaryEdges: number;
  packageDependencies: Record<string, unknown>[];
}
export class ClipboardError extends Error {
  code: string;
  constructor(code: string, message: string) { super(`${code}: ${message}`); this.code = code; }
}
const fail = (code: string, message: string): never => { throw new ClipboardError(code, message); };
const key = (value: { id: string; version: string }) => `${value.id}@${value.version}`;
const clone = <T,>(value: T): T => structuredClone(value);
const stable = (value: unknown): string => JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(k => [k, item[k]])) : item);
function references(nodes: GNode[]) {
  const modules: { id: string; version: string }[] = []; const blocks: { id: string; version: string }[] = [];
  const module = (value: Record<string, unknown>) => {
    if (typeof value.module !== "string") fail("E_CLIPBOARD_DEFINITION", "Selected structural node has no declared module identity.");
    modules.push({ id: value.module as string, version: typeof value.version === "string" ? value.version : "1.0.0" });
  };
  for (const node of nodes) {
    if (node.type === "core.composite" || node.type === "core.repeat") module(node.config);
    if (node.type === "core.select") {
      for (const field of ["then", "otherwise"]) {
        const branch = node.config[field];
        if (!branch || typeof branch !== "object" || Array.isArray(branch)) fail("E_CLIPBOARD_DEFINITION", "Selected branch has no declared module identity.");
        module(branch as Record<string, unknown>);
      }
    }
    if (node.type === "code.block") {
      if (typeof node.config.block !== "string") fail("E_CLIPBOARD_DEFINITION", "Selected code block has no declared identity.");
      blocks.push({ id: node.config.block as string, version: typeof node.config.version === "string" ? node.config.version : "1.0.0" });
    }
  }
  return { modules, blocks };
}
function bounded(clipboard: GraphClipboard) {
  if (clipboard.nodes.length > 100 || clipboard.modules.length + clipboard.codeBlocks.length > 100 || JSON.stringify(clipboard).length * 2 > 512 * 1024)
    fail("E_CLIPBOARD_BOUNDS", "Copy at most 100 nodes/100 definitions within 512 KiB estimated JSON size.");
}
export function copyGraphNodes(graph: Graph, ui: UiDoc, selected: string[], sourceProject: string, sourceGraphHash: string): GraphClipboard {
  if (!["model", "tabular", "domain"].includes(graph.graphKind)) fail("E_CLIPBOARD_KIND", "Copy supports root model, tabular and domain graphs.");
  const ids = new Set(selected);
  if (!ids.size || ids.size > 100 || selected.some(id => !graph.nodes.some(n => n.id === id))) fail("E_CLIPBOARD_SELECTION", "Select 1–100 actual root graph nodes; generated/module boundary nodes cannot be copied.");
  const nodes = graph.nodes.filter(n => ids.has(n.id));
  for (const node of nodes) {
    if (node.stateRef && node.stateRef !== `model/${node.id}`) fail("E_CLIPBOARD_STATE_REF", "Opaque state references cannot be rebound safely; only the declared model/<this-node-id> draft convention is rebound.");
    if (node.sharedWith && !ids.has(node.sharedWith)) fail("E_CLIPBOARD_SHARE", `Include parameter source '${node.sharedWith}' when copying '${node.id}'.`);
    if (node.type === "core.composite" && typeof node.config.share === "string" && !["clone", ""].includes(node.config.share) && !ids.has(node.config.share))
      fail("E_CLIPBOARD_SHARE", `Include shared module instance '${node.config.share}' when copying '${node.id}'.`);
  }
  const edges = graph.edges.filter(e => ids.has(e.from.node) && ids.has(e.to.node));
  const modules: ModuleDef[] = []; const codeBlocks: CodeBlockDef[] = [];
  const visitedModules = new Set<string>(); const visitedBlocks = new Set<string>();
  const collect = (group: GNode[], nested = false) => {
    if (nested && group.some(node => node.stateRef)) fail("E_CLIPBOARD_STATE_REF", "State references inside a required module cannot be rebound safely.");
    const refs = references(group);
    for (const ref of refs.modules) {
      const id = key(ref); if (visitedModules.has(id)) continue;
      const def = graph.modules?.find(m => key(m) === id);
      if (!def) fail("E_CLIPBOARD_DEFINITION", `Required module '${id}' is missing.`);
      visitedModules.add(id); modules.push(def!);
      if (modules.length + codeBlocks.length > 100) fail("E_CLIPBOARD_BOUNDS", "Copy at most 100 required definitions.");
      collect(def!.nodes, true);
    }
    for (const ref of refs.blocks) {
      const id = key(ref); if (visitedBlocks.has(id)) continue;
      const def = graph.codeBlocks?.find(b => key(b) === id);
      if (!def) fail("E_CLIPBOARD_DEFINITION", `Required code definition '${id}' is missing.`);
      visitedBlocks.add(id); codeBlocks.push(def!);
      if (modules.length + codeBlocks.length > 100) fail("E_CLIPBOARD_BOUNDS", "Copy at most 100 required definitions.");
    }
  };
  collect(nodes);
  const clipboard = clone({ graphKind: graph.graphKind, backend: graph.backend, sourceProject, sourceGraphHash, nodes, edges, modules, codeBlocks, packageDependencies: graph.packageDependencies ?? [],
    positions: Object.fromEntries(nodes.map(node => [node.id, ui.positions[node.id] ?? { x: 60 + graph.nodes.indexOf(node) * 260, y: 120 }])),
    omittedBoundaryEdges: graph.edges.filter(e => ids.has(e.from.node) !== ids.has(e.to.node)).length });
  bounded(clipboard); return clipboard;
}
function fresh(id: string, taken: Set<string>) {
  let base = id.replace(/[^A-Za-z0-9_]/g, "_").slice(0, 48);
  if (!/^[A-Za-z]/.test(base)) base = "node_" + base;
  base += "_copy";
  let candidate = base; let suffix = 2;
  while (taken.has(candidate)) candidate = `${base}_${suffix++}`;
  taken.add(candidate); return candidate;
}
function merge<T extends { id: string; version: string }>(target: T[] | undefined, copied: T[]) {
  const result = [...(target ?? [])];
  for (const definition of copied) {
    const existing = result.find(d => key(d) === key(definition));
    if (existing && stable(existing) !== stable(definition)) fail("E_CLIPBOARD_DEFINITION_CONFLICT", `Definition '${key(definition)}' differs in this project; it is never silently replaced.`);
    if (!existing) result.push(clone(definition));
  }
  return result;
}
export function pasteGraphNodes(graph: Graph, ui: UiDoc, clipboard: GraphClipboard, offset = 80) {
  if (graph.graphKind !== clipboard.graphKind || graph.backend !== clipboard.backend) fail("E_CLIPBOARD_KIND", "Paste requires the same graph kind and backend as the copied snapshot.");
  bounded(clipboard);
  const taken = new Set(graph.nodes.map(n => n.id));
  const mapping = Object.fromEntries(clipboard.nodes.map(n => [n.id, fresh(n.id, taken)]));
  const nodes = clipboard.nodes.map(original => {
    const node = clone(original); node.id = mapping[original.id];
    if (node.stateRef) node.stateRef = `model/${node.id}`;
    if (node.sharedWith) node.sharedWith = mapping[node.sharedWith] ?? fail("E_CLIPBOARD_SHARE", "Copied parameter source is missing.");
    if (node.type === "core.composite" && typeof node.config.share === "string" && !["clone", ""].includes(node.config.share))
      node.config.share = mapping[node.config.share] ?? fail("E_CLIPBOARD_SHARE", "Copied module parameter source is missing.");
    return node;
  });
  const edgeIds = new Set(graph.edges.map(e => e.id));
  const edges = clipboard.edges.map(original => {
    const edge = clone(original); edge.id = fresh(edge.id, edgeIds);
    edge.from.node = mapping[edge.from.node] ?? fail("E_CLIPBOARD_EDGE", "Copied edge source is missing.");
    edge.to.node = mapping[edge.to.node] ?? fail("E_CLIPBOARD_EDGE", "Copied edge target is missing."); return edge;
  });
  const modules = merge(graph.modules, clipboard.modules); const codeBlocks = merge(graph.codeBlocks, clipboard.codeBlocks);
  const packageDependencies = [...(graph.packageDependencies ?? [])];
  for (const dependency of clipboard.packageDependencies) {
    const existing = packageDependencies.find(d => d.operation === dependency.operation);
    if (existing && stable(existing) !== stable(dependency)) fail("E_CLIPBOARD_DEFINITION_CONFLICT", "Required package operation identity differs; install/review the exact dependency explicitly.");
    if (!existing) packageDependencies.push(clone(dependency));
  }
  const positions = { ...ui.positions, ...Object.fromEntries(clipboard.nodes.map(n => {
    const point = clipboard.positions[n.id]; return [mapping[n.id], { x: point.x + offset, y: point.y + offset }];
  })) };
  return { graph: { ...graph, nodes: [...graph.nodes, ...nodes], edges: [...graph.edges, ...edges],
    ...(clipboard.modules.length ? { modules } : {}), ...(clipboard.codeBlocks.length ? { codeBlocks } : {}), ...(clipboard.packageDependencies.length ? { packageDependencies } : {}) },
    ui: { ...ui, positions }, mapping, selected: nodes.map(n => n.id) };
}
