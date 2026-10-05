// Helpers for reusable modules: a module definition is shown on the canvas as a graph with two kinds of boundary pseudo-nodes
// (the module's inputs and outputs). Editing the canvas edits the definition; converting back removes the pseudo-nodes.
import type { GEdge, GNode, Graph, InstanceInfo, ModuleDef, OpInfo, Validation } from "./types";

export const IN_PREFIX = "__in:";
export const OUT_PREFIX = "__out:";
export const MODULE_INPUT = "__module_input";
export const MODULE_OUTPUT = "__module_output";
export const STRUCTURAL = ["core.composite", "core.repeat", "core.select"];
export const isPseudo = (id: string) => id.startsWith(IN_PREFIX) || id.startsWith(OUT_PREFIX);

const base = (type: string, name: string, inputs: string[], outputs: string[]): OpInfo => ({
  type, version: "1.0.0", backend: "module", graphKind: "model", summaryKind: null, displayName: name, category: "Modules", purpose: "Boundary of the module being edited.",
  inputs, outputs, inputKinds: Object.fromEntries(inputs.map((p) => [p, "tensor"])), outputKinds: Object.fromEntries(outputs.map((p) => [p, "tensor"])),
  configSchema: { properties: {} }, defaults: {},
});
export const PSEUDO_OPS: OpInfo[] = [base(MODULE_INPUT, "Module input", [], ["value"]), base(MODULE_OUTPUT, "Module output", ["value"], [])];

export const modKey = (m: { id: string; version: string }) => `${m.id}@${m.version}`;
export const findModule = (g: Graph, id: string, version?: string) => (g.modules ?? []).find((m) => m.id === id && (version === undefined || m.version === version));

/** The module definition as a canvas graph: inner nodes and edges plus one pseudo-node per input and per output. */
export function defToView(d: ModuleDef): Graph {
  const nodes: GNode[] = [
    ...d.inputs.map((p) => ({ id: IN_PREFIX + p.name, type: MODULE_INPUT, version: "1.0.0", config: { name: p.name, dtype: p.dtype, shape: p.shape } })),
    ...d.nodes,
    ...d.outputs.map((o) => ({ id: OUT_PREFIX + o.name, type: MODULE_OUTPUT, version: "1.0.0", config: { name: o.name } })),
  ];
  const edges: GEdge[] = d.edges.map((e) => (e.from.node === "$in" ? { ...e, from: { node: IN_PREFIX + e.from.port, port: "value" } } : e));
  for (const o of d.outputs) {
    if (o.from.node) {
      const src = o.from.node === "$in" ? { node: IN_PREFIX + o.from.port, port: "value" } : o.from;
      edges.push({ id: `__outwire_${o.name}`, kind: "tensor", from: src, to: { node: OUT_PREFIX + o.name, port: "value" } });
    }
  }
  return { schemaVersion: "1.0.0", graphKind: "model", backend: "pytorch", nodes, edges };
}

/** Back from the canvas graph to the definition (interface lists are kept; wires to/from the pseudo-nodes become `$in` references and output sources). */
export function viewToDef(v: Graph, d: ModuleDef): ModuleDef {
  const edges: GEdge[] = [];
  const outFrom = new Map<string, { node: string; port: string }>();
  for (const e of v.edges) {
    if (e.to.node.startsWith(OUT_PREFIX)) {
      const src = e.from.node.startsWith(IN_PREFIX) ? { node: "$in", port: e.from.node.slice(IN_PREFIX.length) } : e.from;
      outFrom.set(e.to.node.slice(OUT_PREFIX.length), src);
    } else if (e.from.node.startsWith(IN_PREFIX)) {
      const port = e.from.node.slice(IN_PREFIX.length);
      edges.push({ ...e, from: { node: "$in", port } });
    } else edges.push(e);
  }
  return {
    ...d, nodes: v.nodes.filter((n) => !isPseudo(n.id)), edges,
    outputs: d.outputs.map((o) => ({ ...o, from: outFrom.get(o.name) ?? { node: "", port: "" } })),
  };
}

export const replaceModule = (g: Graph, old: ModuleDef, next: ModuleDef): Graph => ({ ...g, modules: (g.modules ?? []).map((m) => (m.id === old.id && m.version === old.version ? next : m)) });

/** A path like block/ra/conv_a: which module definition (and which node inside it) does that inner node come from? */
export function resolveDefTarget(g: Graph, path: string, inst: Record<string, InstanceInfo> | undefined): { module: ModuleDef; nodeId: string } | null {
  const parts = path.split("/");
  let scopeNodes: GNode[] = g.nodes;
  let mod: ModuleDef | undefined;
  for (let i = 0; i < parts.length; i++) {
    const n = scopeNodes.find((x) => x.id === parts[i]);
    if (!n) return null;
    if (i === parts.length - 1) return mod ? { module: mod, nodeId: n.id } : null;
    const prefix = parts.slice(0, i + 1).join("/");
    const info = inst?.[prefix];
    const cfg = n.config as { module?: string; version?: string; then?: { module: string; version?: string }; otherwise?: { module: string; version?: string } };
    let m: ModuleDef | undefined;
    if (n.type === "core.composite") m = findModule(g, cfg.module ?? "", cfg.version ?? "1.0.0");
    else if (n.type === "core.repeat") {
      // path component after a repeat is itN, then the module's own nodes
      m = findModule(g, cfg.module ?? "", cfg.version ?? "1.0.0");
      i += 1; // skip itN
    } else if (n.type === "core.select") {
      const br = parts[i + 1] === "then" ? cfg.then : cfg.otherwise;
      m = br ? findModule(g, br.module, br.version ?? "1.0.0") : undefined;
      i += 1;
    }
    if (!m) return null;
    void info;
    mod = m;
    scopeNodes = m.nodes;
  }
  return null;
}

/** Simple layered layout (left to right by dependency depth). Used for canvas views that have no stored positions. */
export function autoLayout(g: Graph, gx = 250, gy = 130, ox = 40, oy = 40): Record<string, { x: number; y: number }> {
  const ids = new Set(g.nodes.map((n) => n.id));
  const srcs: Record<string, string[]> = {};
  for (const e of g.edges) if (ids.has(e.from.node) && ids.has(e.to.node)) (srcs[e.to.node] ??= []).push(e.from.node);
  const depth: Record<string, number> = {};
  const visiting = new Set<string>();
  const d = (n: string): number => {
    if (depth[n] !== undefined) return depth[n];
    if (visiting.has(n)) return 0;
    visiting.add(n);
    depth[n] = 1 + Math.max(-1, ...(srcs[n] ?? []).map(d));
    visiting.delete(n);
    return depth[n];
  };
  const rows: Record<number, number> = {};
  const out: Record<string, { x: number; y: number }> = {};
  for (const n of g.nodes) {
    const c = d(n.id);
    const r = rows[c] ?? 0;
    rows[c] = r + 1;
    out[n.id] = { x: ox + c * gx, y: oy + r * gy };
  }
  return out;
}

export function newModule(id: string, existing: ModuleDef[]): ModuleDef {
  let name = id, i = 1;
  while (existing.some((m) => m.id === name)) name = `${id}_${i++}`;
  return { id: name, version: "1.0.0", description: "", inputs: [{ name: "x", dtype: "float32", shape: null }], outputs: [{ name: "y", from: { node: "", port: "" } }], params: [], nodes: [], edges: [] };
}

export const bumpVersion = (v: string, part: "major" | "minor" | "patch" = "minor") => {
  const [a, b, c] = v.split(".").map(Number);
  return part === "major" ? `${a + 1}.0.0` : part === "minor" ? `${a}.${b + 1}.0` : `${a}.${b}.${c + 1}`;
};

/** Validation of a module on its own, shaped like a project validation so the canvas can use it unchanged. */
export type ModuleValidation = Validation & { outputs?: Record<string, { shape: (number | string)[]; dtype: string }>; params?: number };
