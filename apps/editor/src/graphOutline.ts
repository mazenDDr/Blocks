import type { Diagnostic, Graph, NodeView, OpInfo, Validation } from "./types";

export interface OutlineRow {
  id: string; type: string; title: string; incoming: string[]; outgoing: string[];
  diagnostics: Diagnostic[]; nativeView: NodeView | null; module: string | null; sharing: string | null;
}
/** Recorded validation only; pending/unavailable reports never supply current shapes/errors. */
export function graphOutline(graph: Graph, ops: Record<string, OpInfo>, validation: Validation | null, pending: boolean, query = "", errorsOnly = false) {
  const report = pending ? null : validation;
  const text = query.toLowerCase();
  const ids = new Set(graph.nodes.map(n => n.id));
  const incoming = new Map<string, string[]>(); const outgoing = new Map<string, string[]>();
  const append = <T,>(map: Map<string, T[]>, id: string, value: T) => {
    const items = map.get(id); if (items) items.push(value); else map.set(id, [value]);
  };
  for (const edge of graph.edges) {
    append(incoming, edge.to.node, `${edge.from.node}.${edge.from.port} → ${edge.to.port} (${edge.kind})`);
    append(outgoing, edge.from.node, `${edge.from.port} → ${edge.to.node}.${edge.to.port} (${edge.kind})`);
  }
  const byNode = new Map<string, Diagnostic[]>(); const globalDiagnostics: Diagnostic[] = [];
  for (const diagnostic of report?.diagnostics ?? []) {
    let owner = diagnostic.nodeId;
    while (owner && !ids.has(owner)) owner = owner.includes("/") ? owner.slice(0, owner.lastIndexOf("/")) : null;
    if (owner) append(byNode, owner, diagnostic); else globalDiagnostics.push(diagnostic);
  }
  const rows: OutlineRow[] = graph.nodes.map(node => {
    const diagnostics = byNode.get(node.id) ?? [];
    const structural = ["core.composite", "core.repeat"].includes(node.type);
    const module = structural && typeof node.config.module === "string" ? `${node.config.module}@${node.config.version ?? "1.0.0"}` : null;
    const sharing = node.sharedWith ?? (node.type === "core.composite" && typeof node.config.share === "string" && !["clone", ""].includes(node.config.share) ? node.config.share : null);
    return { id: node.id, type: node.type, title: ops[node.type]?.displayName ?? node.type,
      incoming: incoming.get(node.id) ?? [], outgoing: outgoing.get(node.id) ?? [], diagnostics, nativeView: report?.nodes[node.id] ?? null, module, sharing };
  }).filter(row => (!errorsOnly || row.diagnostics.some(d => d.severity === "error")) &&
    [row.id, row.type, row.title, row.module ?? "", row.sharing ?? "", ...row.incoming, ...row.outgoing, ...row.diagnostics.map(d => `${d.code} ${d.message}`)].some(value => value.toLowerCase().includes(text)));
  return { rows, globalDiagnostics, graphHash: report?.graphHash ?? null };
}
