import { graphOutline } from "./graphOutline.ts";
import { specOf, type AgentNodeView } from "./components/agent/types.ts";
import type { Graph, OpInfo, Validation } from "./types";
/** Actual declared control/state references and current native agent validation only. */
export function agentOutline(graph: Graph, ops: Record<string, OpInfo>, validation: Validation | null, pending: boolean, query = "", errorsOnly = false) {
  const base = graphOutline(graph, ops, validation, pending, "", errorsOnly), spec = specOf(graph), text = query.toLowerCase();
  const rows = base.rows.map(row => {
    const routes = spec.routes.filter(route => route.from === row.id);
    const joins = spec.joins.filter(join => join.node === row.id);
    const native = row.nativeView as unknown as AgentNodeView | null;
    const searchable = [row.id,row.type,row.title,...row.incoming,...row.outgoing,...row.diagnostics.map(d=>d.code+" "+d.message),
      ...(native?.reads ?? []),...(native?.writes ?? []),...(native?.effects ?? []),JSON.stringify(routes),JSON.stringify(joins)];
    return { ...row, native, routes, joins, searchable };
  }).filter(row => row.searchable.some(value => value.toLowerCase().includes(text)));
  return { ...base, rows };
}
