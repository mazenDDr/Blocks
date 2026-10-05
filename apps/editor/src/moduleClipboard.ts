import type { Graph, ModuleDef, UiDoc } from "./types";
import { ClipboardError, checkClipboardBounds, copyGraphNodes, pasteGraphNodes, type GraphClipboard } from "./graphClipboard.ts";
import { autoLayout, defToView, modKey, replaceModule } from "./modules.ts";

function checked(graph: Graph, definition: ModuleDef) {
  if (graph.graphKind !== "model" || graph.backend !== "pytorch" || graph.modules?.filter(d => modKey(d) === modKey(definition)).length !== 1)
    throw new ClipboardError("E_CLIPBOARD_SCOPE", "Module transfer requires one actual stored module in a PyTorch model graph.");
  return graph.modules.find(d => modKey(d) === modKey(definition))!;
}
function localUi(ui: UiDoc, definition: ModuleDef, includeFallback: boolean): UiDoc {
  const prefix = modKey(definition) + "/";
  const positions = Object.fromEntries(Object.entries(ui.positions).filter(([key]) => key.startsWith(prefix)).map(([key, p]) => [key.slice(prefix.length), p]));
  if (includeFallback) {
    const fallback = autoLayout(defToView(definition));
    for (const node of definition.nodes) if (!Object.hasOwn(positions, node.id)) positions[node.id] = fallback[node.id];
  }
  const comments = ui.nodeComments;
  return { ...ui, positions, ...(comments && typeof comments === "object" && !Array.isArray(comments)
    ? { nodeComments: Object.fromEntries(Object.entries(comments).filter(([key]) => key.startsWith(prefix)).map(([key, note]) => [key.slice(prefix.length), note])) } : {}) };
}
export function copyModuleNodes(graph: Graph, ui: UiDoc, definition: ModuleDef, selected: string[], project: string, hash: string): GraphClipboard {
  const actual = checked(graph, definition);
  if (actual.nodes.some(node => selected.includes(node.id) && node.stateRef))
    throw new ClipboardError("E_CLIPBOARD_STATE_REF", "Module-local state references are opaque and cannot be safely rebound.");
  const snapshot = copyGraphNodes({ ...graph, nodes: actual.nodes, edges: actual.edges }, localUi(ui, actual, true), selected, project, hash);
  snapshot.omittedBoundaryEdges += actual.outputs.filter(output => selected.includes(output.from.node)).length;
  snapshot.sourceScope = { kind: "module", id: actual.id, version: actual.version };
  checkClipboardBounds(snapshot);
  return snapshot;
}
export function pasteModuleNodes(graph: Graph, ui: UiDoc, definition: ModuleDef, clipboard: GraphClipboard, offset = 80) {
  const actual = checked(graph, definition);
  // Work directly on stored internal nodes/edges: interface lists and $in wires are never reconstructed.
  const result = pasteGraphNodes({ ...graph, nodes: actual.nodes, edges: actual.edges }, localUi(ui, actual, false), clipboard, offset);
  const next = { ...actual, nodes: result.graph.nodes, edges: result.graph.edges };
  const merged = replaceModule({ ...result.graph, nodes: graph.nodes, edges: graph.edges }, actual, next);
  const prefix = modKey(actual) + "/";
  const positions = Object.fromEntries(result.selected.map(id => [prefix + id, result.ui.positions[id]]));
  return { graph: merged, ui: { ...ui, positions: { ...ui.positions, ...positions } }, selected: result.selected, mapping: result.mapping };
}
