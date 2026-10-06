import { NodeChecklist } from "./NodeChecklist";
import { useState } from "react";
import type { GNode } from "../types";
export function GraphMovementTools({ nodes, selected, scope, blocked, onSelect, onMove, onCollapse }: {
  nodes: GNode[]; selected: string[]; scope: string; blocked: string | null;
  onSelect: (ids: string[]) => void; onMove: (dx: number, dy: number) => void; onCollapse?: () => void;
}) {
  const [horizontal, setHorizontal] = useState("0"), [vertical, setVertical] = useState("0");
  const ids = selected.filter(id => nodes.some(n => n.id === id));
  const dx = Number(horizontal), dy = Number(vertical);
  const valid = !!horizontal.trim() && !!vertical.trim() && [dx, dy].every(n => Number.isFinite(n) && Math.abs(n) <= 1_000_000);
  return <details className="movement-tools"><summary>Move selected nodes together</summary>
    <p>{scope}. Enter offsets in canvas layout units; the same offsets apply to every selected origin. Each action is one layout Undo. A zero offset makes no edit. Placement can overlap and does not choose a graph sequence automatically.</p>
    <fieldset><legend>Select layout cards ({ids.length})</legend><button disabled={!nodes.length || nodes.length > 100} onClick={() => onSelect(nodes.map(n => n.id))}>Select all for movement</button>{" "}<button onClick={() => onSelect([])}>Clear movement selection</button>
      <NodeChecklist nodes={nodes} selected={ids} label="move node" onToggle={(id, checked) => onSelect(checked ? [...ids, id] : ids.filter(x => x !== id))} />
    </fieldset>
    <div className="movement-actions"><label>Horizontal offset <input type="number" step="any" aria-label="group horizontal offset" value={horizontal} onChange={e => setHorizontal(e.target.value)} /></label><label>Vertical offset <input type="number" step="any" aria-label="group vertical offset" value={vertical} onChange={e => setVertical(e.target.value)} /></label>
      <button disabled={!!blocked || !valid || !ids.length || ids.length > 100} onClick={() => onMove(dx, dy)}>Move selected layout cards</button></div>
    {!valid && <p role="status">E_MOVE_DELTA: Enter both finite offsets within ±1,000,000 layout units.</p>}
    {blocked && <p role="status">{blocked} {onCollapse && <button onClick={onCollapse}>Collapse modules for movement</button>}</p>}
  </details>;
}
