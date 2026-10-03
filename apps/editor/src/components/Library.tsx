import { useMemo, useState } from "react";
import type { OpInfo } from "../types";

/** Block library: search by name, type id or purpose. Enter on the search box or on a block adds it. */
export function Library({ ops, onAdd }: { ops: OpInfo[]; onAdd: (op: OpInfo) => void }) {
  const [q, setQ] = useState("");
  const hits = useMemo(() => {
    const s = q.trim().toLowerCase();
    return ops.filter((o) => !s || `${o.displayName} ${o.type} ${o.purpose} ${o.category}`.toLowerCase().includes(s));
  }, [ops, q]);
  const ORDER = ["Layers", "Loss", "Core", "Tensor ops"];
  const cats = [...new Set(hits.map((o) => o.category))].sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  return (
    <div className="library" aria-label="Block library">
      <h3>Library</h3>
      <input value={q} placeholder="Search blocks (e.g. Conv2d)" aria-label="Search blocks"
        onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && hits[0]) onAdd(hits[0]); }} />
      {cats.map((c) => (
        <div key={c}>
          <h4>{c}</h4>
          {hits.filter((o) => o.category === c).map((o) => (
            <button key={o.type} className="lib-item" onClick={() => onAdd(o)} title={o.purpose}>
              <b>{o.displayName}</b><span className="badge">{o.backend}</span>
              <small>{o.inputs.length ? o.inputs.join(", ") : "no inputs"} {"→"} {o.outputs.join(", ")}</small>
            </button>
          ))}
        </div>
      ))}
      {hits.length === 0 && <div className="empty">No block matches.</div>}
      <p className="hint">Click or press Enter to add. If a node is selected with a free output, the new block is connected to it.</p>
    </div>
  );
}
