import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export interface GroupData extends Record<string, unknown> {
  path: string; module?: string | null; version?: string | null; kind?: string; inputs: string[]; outputs: string[]; width: number; height: number;
  onCollapse: () => void; onOpen: () => void; note: string; errors: number; params: number; sharedWith?: string | null;
}
export type GroupNode = Node<GroupData, "group">;
const top = (i: number, n: number) => `${((i + 1) * 100) / (n + 1)}%`;

/** An instance expanded in place: its inner nodes are drawn inside this frame; the instance's own ports stay on the border so outside wires do not move. */
export function GroupCard({ data, selected }: NodeProps<GroupNode>) {
  const d = data;
  return (
    <div className={`groupcard ${selected ? "sel" : ""} ${d.errors ? "err" : ""} ${d.sharedWith ? "shared" : ""}`} style={{ width: d.width, height: d.height }}>
      {d.inputs.map((p, i) => <Handle key={"i" + p} id={p} type="target" position={Position.Left} style={{ top: top(i, d.inputs.length) }} title={`input ${p}`} />)}
      {d.inputs.map((p, i) => <Handle key={"ii" + p} id={`in-${p}`} type="source" position={Position.Left} style={{ top: top(i, d.inputs.length), left: 14 }} title={`inside: ${p}`} />)}
      {d.outputs.map((p, i) => <Handle key={"o" + p} id={p} type="source" position={Position.Right} style={{ top: top(i, d.outputs.length) }} title={`output ${p}`} />)}
      {d.outputs.map((p, i) => <Handle key={"oo" + p} id={`out-${p}`} type="target" position={Position.Right} style={{ top: top(i, d.outputs.length), right: 14 }} title={`inside: ${p}`} />)}
      <div className="grouphead">
        <b>{d.path}</b> <span className="badge">{d.kind?.replace("core.", "")} {d.module} {d.version ? `v${d.version}` : ""}</span>
        {d.sharedWith && <span className="badge shared-badge">{"\u{1F517}"} shares {d.sharedWith}</span>}
        <span className="muted small"> {d.params.toLocaleString("en-US")} parameters · expanded: {d.note}</span>
        <button className="nodrag" onClick={(e) => { e.stopPropagation(); d.onOpen(); }}>Open</button>
        <button className="nodrag" onClick={(e) => { e.stopPropagation(); d.onCollapse(); }}>Collapse</button>
      </div>
    </div>
  );
}
