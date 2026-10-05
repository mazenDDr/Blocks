import type { Graph, OpInfo } from "./types";
export type WorkspaceView = "graph" | "data" | "experiments" | "training" | "debug" | "attention" | "backends" | "coverage" | "domain" | "production" | "scale" | "records";
export type GraphCommand = { id: string; label: string; detail: string; disabled?: string } & (
  { action: "view"; view: WorkspaceView } | { action: "inspect"; node: string } | { action: "create"; operation: string } |
  { action: "undo" | "redo" | "root" }
);
export function workspaceChoices(kind: string): [WorkspaceView, string][] {
  const shared: [WorkspaceView, string][] = [["scale", "Integrations"], ["production", "Production"], ["records", "Records"]];
  const work: [WorkspaceView, string][] = kind === "domain" ? [["domain", "Domain workspace"], ["graph", "Graph"], ["coverage", "Coverage"]]
    : kind === "rl" ? [["graph", "RL lab"]] : kind === "agent" ? [["graph", "Agent"], ["data", "Data"]]
    : [["graph", "Graph"], ["data", "Data"], ["experiments", "Experiments"], ...(kind === "tabular" ? [] : [["training", "Training"], ["debug", "Debug"], ["attention", "Attention"], ["backends", "Backend"]] as [WorkspaceView, string][]), ["coverage", "Coverage"]];
  return [...shared, ...work];
}
/** Catalogue of existing actions, not native values or graph edits. */
export function graphCommands(root: Graph, current: Graph, ops: OpInfo[], canUndo: boolean, canRedo: boolean, module: boolean): GraphCommand[] {
  const commands: GraphCommand[] = workspaceChoices(root.graphKind).map(([view, title]) => ({ id: `view:${view}`, label: `Open ${title}`, detail: "Existing workspace in this project", action: "view", view }));
  const editable = !["agent", "rl"].includes(root.graphKind);
  if (editable) {
    for (const node of current.nodes) commands.push({ id: `inspect:${node.id}`, label: `Inspect ${node.id}`, detail: `${module ? "Module definition" : "Root graph"} · ${node.type}@${node.version}`, action: "inspect", node: node.id });
    for (const op of ops) if (op.graphKind === current.graphKind && (current.graphKind === "model" ? op.backend === current.backend : current.backend === "python") && !["core.composite", "core.repeat", "core.select", "code.block"].includes(op.type))
      commands.push({ id: `create:${op.type}`, label: `Add ${op.displayName}`, detail: `${op.type}@${op.version} · ${op.backend} · ${op.purpose} · disconnected registered defaults; inspect validation`, action: "create", operation: op.type });
  }
  commands.push({ id: "undo", label: "Undo draft edit", detail: "Whole graph and layout; returns to the root", action: "undo", disabled: canUndo ? undefined : "No prior draft edit" },
    { id: "redo", label: "Redo draft edit", detail: "Whole graph and layout; returns to the root", action: "redo", disabled: canRedo ? undefined : "No draft edit to redo" });
  if (module) commands.push({ id: "root", label: "Back to project root", detail: "Leave the module definition without editing", action: "root" });
  return commands;
}
export function findGraphCommands(commands: GraphCommand[], query: string) {
  const text = query.toLowerCase();
  return commands.filter(c => `${c.label} ${c.detail}`.toLowerCase().includes(text));
}
