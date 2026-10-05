import type { GNode, Graph, ModuleDef } from "./types";
export interface DiagnosticScope { module: string; version: string; via: string }
export interface DiagnosticTarget { node: string; scopes: DiagnosticScope[] }
/** Resolve declared native diagnostic paths to actual stored nodes; no runtime values inferred. */
export function diagnosticTarget(graph: Graph, nodeId: string | null | undefined, start: { id: string; version: string } | null = null): DiagnosticTarget | null {
  if (!nodeId || nodeId.length > 4096 || !["model", "tabular", "domain"].includes(graph.graphKind)) return null;
  const parts = nodeId.split("/");
  if (parts.length > 64 || parts.some(part => !part)) return null;
  const find = (id: unknown, version: unknown): ModuleDef | null => {
    if (typeof id !== "string" || typeof version !== "string") return null;
    const definitions = graph.modules?.filter(d => d.id === id && d.version === version) ?? [];
    return definitions.length === 1 ? definitions[0] : null;
  };
  const initial = start ? find(start.id, start.version) : null;
  if (start && !initial) return null;
  let nodes = initial ? initial.nodes : graph.nodes;
  const scopes: DiagnosticScope[] = [], seen = new Set(initial ? [initial.id + "@" + initial.version] : []);
  for (let i = 0; i < parts.length; i++) {
    const matches: GNode[] = nodes.filter(node => node.id === parts[i]);
    if (matches.length !== 1) return null;
    const node = matches[0];
    if (i === parts.length - 1) return { node: node.id, scopes };
    let reference: Record<string, unknown> = node.config, via = node.id;
    if (node.type === "core.repeat") {
      const count = node.config.count ?? 2, marker = parts[++i];
      if (typeof count !== "number" || !Number.isInteger(count) || count < 1 || count > 64 || !/^it(0|[1-9]\d*)$/.test(marker ?? "") || Number(marker.slice(2)) >= count) return null;
      via += "/" + marker;
    } else if (node.type === "core.select") {
      const branch = parts[++i];
      if (branch !== "then" && branch !== "otherwise") return null;
      const value = node.config[branch];
      if (!value || typeof value !== "object" || Array.isArray(value)) return null;
      reference = value as Record<string, unknown>;via += "/" + branch;
    } else if (node.type !== "core.composite") return null;
    const definition = find(reference.module, reference.version ?? "1.0.0");
    if (!definition || seen.has(definition.id + "@" + definition.version)) return null;
    seen.add(definition.id + "@" + definition.version);
    scopes.push({ module: definition.id, version: definition.version, via });nodes = definition.nodes;
  }
  return null;
}
