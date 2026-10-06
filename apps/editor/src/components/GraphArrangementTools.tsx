import { NodeChecklist } from "./NodeChecklist";
import type { GNode } from "../types";
import { ARRANGEMENTS, type Arrangement } from "../graphArrangement";

export function GraphArrangementTools({ nodes, selected, scope, blocked, onSelect, onArrange, onCollapse }: {
  nodes: GNode[]; selected: string[]; scope: string; blocked: string | null;
  onSelect: (ids: string[]) => void; onArrange: (operation: Arrangement) => void; onCollapse?: () => void;
}) {
  const ids = selected.filter(id => nodes.some(n => n.id === id));
  return <details className="arrangement-tools"><summary>Arrange graph nodes</summary>
    <p>{scope}. Alignment uses actual card dimensions and can overlap cards. Distribution keeps the two end cards fixed and requires enough space for equal gaps. Each action is one layout Undo edit.</p>
    <fieldset><legend>Select nodes to arrange ({ids.length})</legend>
      <button disabled={!nodes.length || nodes.length > 100} onClick={() => onSelect(nodes.map(n => n.id))}>Select all for arrangement</button>{" "}
      <button onClick={() => onSelect([])}>Clear arrangement selection</button>
      <NodeChecklist nodes={nodes} selected={ids} label="arrange node" onToggle={(id, checked) => onSelect(checked ? [...ids, id] : ids.filter(x => x !== id))} />
    </fieldset>
    <div className="arrangement-actions">{ARRANGEMENTS.map(o => <button key={o.value} disabled={!!blocked || ids.length < o.minimum || ids.length > 100} onClick={() => onArrange(o.value)}>{o.label}</button>)}</div>
    {blocked && <p role="status">{blocked} {onCollapse && <button onClick={onCollapse}>Collapse modules for arrangement</button>}</p>}
  </details>;
}
