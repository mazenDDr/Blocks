import { memo, useCallback, useLayoutEffect, useRef, type ReactElement } from "react";
import type { GNode } from "../types";

// Rows keep a stable toggle handler, so a selection change re-renders only the rows whose checked state changed.
const NodeCheck = memo(function NodeCheck({ id, type, label, checked, onToggle }: {
  id: string; type: string; label: string; checked: boolean; onToggle: (id: string, checked: boolean) => void;
}) {
  return <label><input type="checkbox" aria-label={`${label} ${id}`} checked={checked} onChange={e => onToggle(id, e.target.checked)} />{id} · {type}</label>;
});

export function NodeChecklist({ nodes, selected, label, onToggle }: {
  nodes: GNode[]; selected: string[]; label: string; onToggle: (id: string, checked: boolean) => void;
}) {
  const latest = useRef(onToggle);
  useLayoutEffect(() => { latest.current = onToggle; });
  const toggle = useCallback((id: string, checked: boolean) => latest.current(id, checked), []);
  // Reusing an unchanged row's element lets React skip it without creating a new element per node.
  const cache = useRef(new Map<string, { node: GNode; checked: boolean; element: ReactElement }>());
  const chosen = new Set(selected), next = new Map<string, { node: GNode; checked: boolean; element: ReactElement }>();
  const rows = nodes.map(n => {
    const checked = chosen.has(n.id), old = cache.current.get(n.id);
    const row = old && old.node === n && old.checked === checked ? old
      : { node: n, checked, element: <NodeCheck key={n.id} id={n.id} type={n.type} label={label} checked={checked} onToggle={toggle} /> };
    next.set(n.id, row); return row.element;
  });
  useLayoutEffect(() => { cache.current = next; });
  return <div className="clipboard-selection">{rows}</div>;
}
