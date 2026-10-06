import type { GEdge, GNode, Graph, UiDoc } from "./types";
import { specOf, type IndexSpec, type Join, type Policy, type Predicate, type Route, type StateField } from "./components/agent/types.ts";
import { ClipboardError } from "./graphClipboard.ts";

/** Copied agent nodes with the state fields, routes, joins, indexes and memory policies they need. */
export interface AgentClipboard {
  graphKind: "agent"; backend: string; sourceProject: string; sourceGraphHash: string;
  nodes: GNode[]; edges: GEdge[]; routes: Route[]; joins: Join[];
  state: StateField[]; indexes: IndexSpec[]; policies: Policy[];
  positions: UiDoc["positions"];
  omitted: { transitions: number; routes: number; joins: number };
}
/** Native validation's per-node state access for the exact copied snapshot. */
export type AgentNodeAccess = Record<string, { reads?: string[]; writes?: string[] } | undefined>;

const fail = (code: string, message: string): never => { throw new ClipboardError(code, message); };
const clone = <T,>(value: T): T => structuredClone(value);
const stable = (value: unknown): string => JSON.stringify(value, (_key, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(k => [k, item[k]])) : item);
const TERMINAL = new Set(["START", "END"]);
const root = (path: string) => path.split(".")[0];

function predicateFields(when: Predicate | undefined, out: Set<string>) {
  if (!when) return;
  if (typeof when.field === "string") out.add(root(when.field));
  if (typeof when.other === "string") out.add(root(when.other));
  for (const p of [...(when.all ?? []), ...(when.any ?? []), ...(when.not ? [when.not] : [])]) predicateFields(p, out);
}

export function checkAgentClipboardBounds(clipboard: AgentClipboard) {
  if (clipboard.nodes.length > 100 || JSON.stringify(clipboard).length * 2 > 512 * 1024)
    fail("E_CLIPBOARD_BOUNDS", "Copy at most 100 agent nodes within 512 KiB estimated JSON size.");
}

export function copyAgentNodes(graph: Graph, ui: UiDoc, selected: string[], access: AgentNodeAccess, sourceProject: string, sourceGraphHash: string): AgentClipboard {
  if (graph.graphKind !== "agent") fail("E_CLIPBOARD_KIND", "Agent copy requires an agent graph.");
  const ids = new Set(selected);
  if (!ids.size || ids.size > 100 || ids.size !== selected.length || selected.some(id => TERMINAL.has(id) || graph.nodes.filter(n => n.id === id).length !== 1))
    fail("E_CLIPBOARD_SELECTION", "Select 1–100 uniquely identified agent nodes; START and END cannot be copied.");
  const spec = specOf(graph);
  const inside = (id: string) => ids.has(id) || id === "END";
  const nodes = graph.nodes.filter(n => ids.has(n.id));
  // Transitions and routes leaving the selection are omitted rather than retargeted, so no copied path silently changes.
  const edges = graph.edges.filter(e => ids.has(e.from.node) && inside(e.to.node));
  const fromSelected = spec.routes.filter(r => ids.has(r.from));
  const routes = fromSelected.filter(r => inside(r.default) && r.cases.every(c => inside(c.to)));
  const joinsTouching = spec.joins.filter(j => ids.has(j.node));
  const joins = joinsTouching.filter(j => j.waitFor.every(w => ids.has(w)));

  const fields = new Set<string>();
  for (const node of nodes) {
    const info = access[node.id];
    if (!info) fail("E_CLIPBOARD_PENDING", `Native validation has no state access for '${node.id}'; wait for current validation and copy again.`);
    for (const f of [...(info!.reads ?? []), ...(info!.writes ?? [])]) fields.add(root(f));
  }
  for (const r of routes) for (const c of r.cases) predicateFields(c.when, fields);
  const state = [...fields].sort().map(name => spec.state.find(f => f.name === name) ?? fail("E_CLIPBOARD_STATE", `State field '${name}' is used by the selection but not declared in this graph.`));

  const indexIds = new Set(nodes.map(n => n.config.index).filter((v): v is string => typeof v === "string" && !!v));
  const policyIds = new Set(nodes.map(n => n.config.policy).filter((v): v is string => typeof v === "string" && !!v));
  const indexes = [...indexIds].map(id => spec.indexes.find(i => i.id === id) ?? fail("E_CLIPBOARD_DEFINITION", `Required index '${id}' is not declared in this graph.`));
  const policies = [...policyIds].map(id => spec.policies.find(p => p.id === id) ?? fail("E_CLIPBOARD_DEFINITION", `Required memory policy '${id}' is not declared in this graph.`));

  const clipboard: AgentClipboard = clone({ graphKind: "agent" as const, backend: graph.backend, sourceProject, sourceGraphHash, nodes, edges, routes, joins, state, indexes, policies,
    positions: Object.fromEntries(nodes.map(node => [node.id, (Object.hasOwn(ui.positions, node.id) ? ui.positions[node.id] : undefined) ?? { x: 40 + graph.nodes.indexOf(node) * 280, y: 60 }])),
    omitted: {
      transitions: graph.edges.filter(e => ids.has(e.from.node) !== ids.has(e.to.node) && !(ids.has(e.from.node) && e.to.node === "END")).length,
      routes: fromSelected.length - routes.length + spec.routes.filter(r => !ids.has(r.from) && (ids.has(r.default) || r.cases.some(c => ids.has(c.to)))).length,
      joins: joinsTouching.length - joins.length,
    } });
  checkAgentClipboardBounds(clipboard); return clipboard;
}

function fresh(id: string, taken: Set<string>) {
  let base = id.replace(/[^A-Za-z0-9_]/g, "_").slice(0, 48);
  if (!/^[A-Za-z]/.test(base)) base = "node_" + base;
  base += "_copy";
  let candidate = base; let suffix = 2;
  while (taken.has(candidate)) candidate = `${base}_${suffix++}`;
  taken.add(candidate); return candidate;
}
function merge<T>(target: T[], copied: T[], name: (item: T) => string, what: string) {
  const result = [...target];
  for (const item of copied) {
    const existing = result.find(x => name(x) === name(item));
    if (existing && stable(existing) !== stable(item)) fail("E_CLIPBOARD_DEFINITION_CONFLICT", `${what} '${name(item)}' differs in this graph; it is never silently replaced.`);
    if (!existing) result.push(clone(item));
  }
  return result;
}

export function pasteAgentNodes(graph: Graph, ui: UiDoc, clipboard: AgentClipboard, offset = 80) {
  if (graph.graphKind !== "agent" || graph.backend !== clipboard.backend) fail("E_CLIPBOARD_KIND", "Paste requires an agent graph with the copied backend.");
  checkAgentClipboardBounds(clipboard);
  const spec = specOf(graph);
  // Shared definitions merge by name; a differing definition refuses before any change.
  const state = merge(spec.state, clipboard.state, f => f.name, "State field");
  const indexes = merge(spec.indexes, clipboard.indexes, i => i.id, "Index");
  const policies = merge(spec.policies, clipboard.policies, p => p.id, "Memory policy");
  const taken = new Set([...TERMINAL, ...graph.nodes.map(n => n.id), ...Object.keys(ui.positions),
    ...(ui.nodeComments && typeof ui.nodeComments === "object" && !Array.isArray(ui.nodeComments) ? Object.keys(ui.nodeComments) : [])]);
  const mapping: Record<string, string> = Object.fromEntries(clipboard.nodes.map(n => [n.id, fresh(n.id, taken)]));
  const to = (id: string) => id === "END" ? "END" : mapping[id] ?? fail("E_CLIPBOARD_EDGE", `Copied transition target '${id}' is missing.`);
  const nodes = clipboard.nodes.map(original => ({ ...clone(original), id: mapping[original.id] }));
  const edgeIds = new Set(graph.edges.map(e => e.id));
  const edges = clipboard.edges.map(original => {
    const edge = clone(original); edge.from.node = to(edge.from.node); edge.to.node = to(edge.to.node);
    let id = `${edge.from.node}__${edge.to.node}`, suffix = 2;
    while (edgeIds.has(id)) id = `${edge.from.node}__${edge.to.node}_${suffix++}`;
    edgeIds.add(id); edge.id = id; return edge;
  });
  const routeIds = new Set(spec.routes.map(r => r.id));
  const routes = clipboard.routes.map(original => {
    const route = clone(original); route.id = fresh(route.id, routeIds); route.from = to(route.from); route.default = to(route.default);
    route.cases = route.cases.map(c => ({ ...c, to: to(c.to) })); return route;
  });
  const joins = clipboard.joins.map(j => ({ node: to(j.node), waitFor: j.waitFor.map(to) }));
  const positions = { ...ui.positions, ...Object.fromEntries(clipboard.nodes.map(n => {
    const point = clipboard.positions[n.id]; return [mapping[n.id], { x: point.x + offset, y: point.y + offset }];
  })) };
  const agent = { ...(graph.agent ?? {}), state, indexes, policies, routes: [...spec.routes, ...routes], joins: [...spec.joins, ...joins] };
  return { graph: { ...graph, nodes: [...graph.nodes, ...nodes], edges: [...graph.edges, ...edges], agent: agent as unknown as Record<string, unknown> },
    ui: { ...ui, positions }, mapping, selected: nodes.map(n => n.id) };
}
