import type { Graph, OpInfo, UiDoc } from "./types";

export class InsertionError extends Error {
  code: string;
  constructor(code: string, message: string) { super(`${code}: ${message}`); this.code = code; }
}
const fail = (code: string, message: string): never => { throw new InsertionError(code, message); };
const STATIC = new Set(["core.composite", "core.repeat", "core.select", "code.block", "__module_input", "__module_output"]);
export function insertionOperations(graph: Graph, ops: OpInfo[]) {
  return ops.filter(op => op.graphKind === graph.graphKind && (graph.graphKind === "model" ? op.backend === graph.backend : graph.backend === "python") && !STATIC.has(op.type) && op.inputs.length && op.outputs.length);
}
/** Split exactly one actual wire. Native validation still owns shapes/config/additional inputs. */
export function insertOnWire(graph: Graph, ui: UiDoc, edgeId: string, op: OpInfo, input: string, output: string, prefix = "") {
  if (!["model", "tabular", "domain"].includes(graph.graphKind)) fail("E_INSERT_SCOPE", "Insertion supports model, tabular and domain graphs.");
  if (!insertionOperations(graph, [op]).length) fail("E_INSERT_OPERATION", "Choose an installed static-port operation for this graph kind and backend.");
  const matches = graph.edges.filter(e => e.id === edgeId);
  if (matches.length !== 1) fail("E_INSERT_WIRE", "Choose one uniquely identified existing wire.");
  const edge = matches[0];
  if (!op.inputs.includes(input) || !op.outputs.includes(output)) fail("E_INSERT_PORT", "Choose declared input and output ports.");
  if (op.inputKinds[input] !== edge.kind || op.outputKinds[output] !== edge.kind) fail("E_INSERT_KIND", "Both selected ports must declare the existing wire kind; shape compatibility is validated natively.");
  if (!graph.nodes.some(n => n.id === edge.from.node) || !graph.nodes.some(n => n.id === edge.to.node)) fail("E_INSERT_WIRE", "Wire endpoints must be actual nodes in this view.");
  if (graph.edges.filter(e => e.to.node === edge.to.node && e.to.port === edge.to.port).length !== 1) fail("E_INSERT_WIRE", "Resolve duplicate producers for the selected target input before insertion.");
  const taken = new Set(graph.nodes.map(n => n.id));
  for (const values of [ui.positions, ui.nodeComments]) if (values && typeof values === "object" && !Array.isArray(values))
    for (const key of Object.keys(values)) if (key.startsWith(prefix) && !key.slice(prefix.length).includes("/")) taken.add(key.slice(prefix.length));
  const base = (op.type.split(".").pop() ?? "node").replace(/[^A-Za-z0-9_]/g, "_");
  let i = 1; while (taken.has(`${base}_${i}`)) i++;
  const id = `${base}_${i}`;
  const edgeIds = new Set(graph.edges.map(e => e.id));
  const freshEdge = (suffix: string) => { let key = `${id}_${suffix}`, n = 2; while (edgeIds.has(key)) key = `${id}_${suffix}_${n++}`; edgeIds.add(key); return key; };
  // Extension metadata stays on the downstream wire to its original consumer only.
  const incoming = { id: freshEdge("in"), kind: edge.kind, from: { ...edge.from }, to: { node: id, port: input } };
  const outgoing = { ...edge, id: freshEdge("out"), from: { node: id, port: output }, to: { ...edge.to } };
  const point = (node: string) => {
    const key = prefix + node, p = Object.hasOwn(ui.positions, key) ? ui.positions[key] : undefined;
    const index = graph.nodes.findIndex(n => n.id === node);
    const fallback = { x: 60 + index * 260, y: 120 };
    const value = p ?? fallback;
    if (!Number.isFinite(value.x) || !Number.isFinite(value.y)) fail("E_INSERT_POSITION", "Repair non-finite endpoint layout before insertion.");
    return value;
  };
  const a = point(edge.from.node), b = point(edge.to.node);
  const position = { x: a.x / 2 + b.x / 2, y: a.y / 2 + b.y / 2 };
  return {
    id, replacement: [incoming, outgoing],
    graph: { ...graph, nodes: [...graph.nodes, { id, type: op.type, version: op.version, config: structuredClone(op.defaults), stateRef: null }],
      edges: graph.edges.flatMap(e => e.id === edgeId ? [incoming, outgoing] : [e]) },
    ui: { ...ui, positions: { ...ui.positions, [prefix + id]: position } },
  };
}
