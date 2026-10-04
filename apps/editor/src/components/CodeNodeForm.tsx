import type { GNode, Graph, NodeView } from "../types";

type Patch = Record<string, unknown>;

/** Config of a code.block node: which block (id + version), its declared config fields, the seed, and a summary of the declared behaviour. */
export function CodeNodeForm({ node, graph, view, onConfig, onEdit }: { node: GNode; graph: Graph; view?: NodeView; onConfig: (p: Patch) => void; onEdit?: (id: string, version: string) => void }) {
  const c = node.config as { block?: string; version?: string; params?: Record<string, unknown>; seed?: number };
  const defs = graph.codeBlocks ?? [];
  const def = defs.find((d) => d.id === c.block && d.version === (c.version ?? "1.0.0"));
  const iface = (view?.resolvedConfig as any)?.interface;
  const params = c.params ?? {};
  return (
    <div className="form">
      <div className="row"><label className="lbl">Code block</label>
        <select aria-label="code block" value={`${c.block ?? ""}@${c.version ?? "1.0.0"}`} onChange={(e) => { const [b, v] = e.target.value.split("@"); onConfig({ block: b, version: v, params: {} }); }}>
          {!def && <option value={`${c.block ?? ""}@${c.version ?? "1.0.0"}`}>{c.block ? `${c.block} v${c.version} (not in this project)` : "choose…"}</option>}
          {defs.map((d) => <option key={`${d.id}@${d.version}`} value={`${d.id}@${d.version}`}>{d.id} v{d.version}</option>)}
        </select>
        {def && onEdit && <button className="primary" onClick={() => onEdit(def.id, def.version)}>Edit code</button>}
      </div>
      {def && def.config.map((f) => {
        const v = f.name in params ? params[f.name] : f.default;
        const set = (x: unknown) => onConfig({ params: { ...params, [f.name]: x } });
        return (
          <div className="row" key={f.name}>
            <label className="lbl">{f.name}{!(f.name in params) && <small> (default)</small>}</label>
            {f.type === "bool" ? <input type="checkbox" aria-label={`code config ${f.name}`} checked={!!v} onChange={(e) => set(e.target.checked)} />
              : f.type === "str" ? <input aria-label={`code config ${f.name}`} value={String(v)} onChange={(e) => set(e.target.value)} />
                : <input type="number" aria-label={`code config ${f.name}`} step={f.type === "int" ? 1 : "any"} value={Number(v)} onChange={(e) => { const n = Number(e.target.value); if (e.target.value !== "" && Number.isFinite(n)) set(f.type === "int" ? Math.round(n) : n); }} />}
          </div>
        );
      })}
      {iface && (
        <table className="valtable"><tbody>
          <tr><td>source sha256</td><td><code>{String(iface.sourceSha256).slice(0, 16)}</code> (part of the semantic hash)</td></tr>
          <tr><td>differentiable</td><td>{iface.differentiable ? "yes: gradients computed inside the sandbox by torch.autograd" : "NO: an explicit gradient boundary"}</td></tr>
          <tr><td>randomness / effects</td><td>{iface.randomness} / {iface.effects.length ? iface.effects.join(", ") : "none declared"}</td></tr>
          <tr><td>pinned dependencies</td><td>{iface.dependencies.length ? iface.dependencies.join(", ") : "none beyond torch"}</td></tr>
        </tbody></table>
      )}
      <div className="muted small">The block runs in an isolated subprocess with CPU, file-size and wall-clock limits; the interface above is what the visual graph sees, and it stays the same when you edit the code.</div>
    </div>
  );
}
