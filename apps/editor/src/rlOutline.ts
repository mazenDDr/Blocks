import { graphOutline } from "./graphOutline.ts";
import type { Graph, OpInfo, Validation } from "./types";

/** Declared RL structure and exact current native per-node contracts. No catalog fallback. */
export function rlOutline(graph: Graph, ops: Record<string, OpInfo>, validation: Validation | null, pending: boolean, query = "", errorsOnly = false) {
  const base = graphOutline(graph, ops, validation, pending);
  const allRows = base.rows.map(row => {
    const matches = graph.nodes.filter(n => n.id === row.id);
    const node = matches.length === 1 ? matches[0] : null;
    const panel = node && graph.nodes.filter(n => n.type === node.type).length === 1
      ? ["rl.environment", "rl.reward"].includes(node.type) ? "env" as const
        : ["rl.q_network", "rl.replay_buffer", "rl.dqn_learner", "rl.evaluation"].includes(node.type) ? "learner" as const : null
      : null;
    return { ...row, node, panel };
  });
  const text = query.toLowerCase();
  const rows = allRows.filter(row => (!errorsOnly || row.diagnostics.some(d => d.severity === "error")) &&
    [row.id, row.type, row.title, ...row.incoming, ...row.outgoing, JSON.stringify(row.node?.config ?? {}),
      JSON.stringify(row.nativeView?.inputShapes ?? {}), JSON.stringify(row.nativeView?.outputShapes ?? {}),
      ...row.diagnostics.map(d => d.code + " " + d.message)].some(value => value.toLowerCase().includes(text)));
  return { ...base, rows, allRows };
}
