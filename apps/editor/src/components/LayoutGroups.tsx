import { useState } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import type { LayoutGroup } from "../layoutGroups";

export interface FrameData extends Record<string, unknown> { label: string; members: number; missing: string[]; width: number; height: number }
export type FrameNode = Node<FrameData, "frame">;

/** A persistent layout group drawn behind its cards. Dragging it moves every member; clicking selects them. */
export function GroupFrame({ data, selected }: NodeProps<FrameNode>) {
  return <div className={`layout-frame ${selected ? "sel" : ""}`} style={{ width: data.width, height: data.height }} aria-label={`layout group ${data.label}`}>
    <div className="layout-frame-head"><b>{data.label}</b> <span className="muted small">{data.members} cards{data.missing.length ? ` · ${data.missing.length} missing` : ""}</span></div>
  </div>;
}

export function LayoutGroupsPanel({ groups, nodes, selected, scope, blocked, onCreate, onUpdate, onDelete, onSelect }: {
  groups: LayoutGroup[]; nodes: Set<string>; selected: string[]; scope: string; blocked: string | null;
  onCreate: (label: string) => void; onUpdate: (id: string, change: { label?: string; members?: string[] }) => void;
  onDelete: (id: string) => void; onSelect: (members: string[]) => void;
}) {
  const [label, setLabel] = useState("");
  const chosen = selected.filter(id => nodes.has(id));
  return <details className="layout-groups"><summary>Layout groups ({groups.length})</summary>
    <p>{scope}. Named frames around cards, saved with the layout only: the graph, its native identity and execution are unchanged. Dragging a frame moves its cards; clicking a frame selects them for the move, arrange and copy tools. Each change is one layout Undo edit.</p>
    <div className="row"><label>New group name <input aria-label="new layout group name" maxLength={80} value={label} onChange={e => setLabel(e.target.value)} /></label>
      <button disabled={!!blocked || !label.trim() || !chosen.length} onClick={() => { onCreate(label); setLabel(""); }}>Group selected cards ({chosen.length})</button></div>
    {blocked && <p role="status">{blocked}</p>}
    {groups.length ? <ul className="layout-group-list">{groups.map(g => {
      const missing = g.members.filter(m => !nodes.has(m));
      return <li key={g.id}>
        <input aria-label={`layout group name ${g.id}`} maxLength={80} defaultValue={g.label} key={`${g.id}:${g.label}`} onBlur={e => { if (e.target.value.trim() && e.target.value.trim() !== g.label) onUpdate(g.id, { label: e.target.value }); }} />
        {" "}<span className="muted small">{g.members.length - missing.length} cards{missing.length ? ` · missing ${missing.join(", ")}` : ""}</span>{" "}
        <button onClick={() => onSelect(g.members.filter(m => nodes.has(m)))}>Select cards</button>{" "}
        <button disabled={!!blocked || !chosen.some(id => !g.members.includes(id))} onClick={() => onUpdate(g.id, { members: [...g.members.filter(m => nodes.has(m)), ...chosen.filter(id => !g.members.includes(id))] })}>Add selected</button>{" "}
        <button disabled={!!blocked || !chosen.some(id => g.members.includes(id)) || g.members.filter(m => nodes.has(m) && !chosen.includes(m)).length === 0}
          onClick={() => onUpdate(g.id, { members: g.members.filter(m => nodes.has(m) && !chosen.includes(m)) })}>Remove selected</button>{" "}
        <button aria-label={`delete layout group ${g.id}`} disabled={!!blocked} onClick={() => onDelete(g.id)}>Delete group</button>
      </li>;
    })}</ul> : <p className="muted">No groups in this layout.</p>}
  </details>;
}
