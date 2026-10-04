import type { GNode, Graph, ModuleDef } from "../types";
import { findModule } from "../modules";

type Patch = Record<string, unknown>;
const ref = (m: ModuleDef) => `${m.id}@${m.version}`;
const sig = (m: ModuleDef) => `${m.inputs.map((p) => p.name).join(", ") || "no inputs"} → ${m.outputs.map((o) => o.name).join(", ") || "no outputs"}`;

/** Pick a module definition of this project (id and version are one choice: a version is never upgraded silently). */
export function ModulePicker({ graph, module, version, onPick, label }: { graph: Graph; module: string; version: string; onPick: (id: string, version: string) => void; label: string }) {
  const mods = graph.modules ?? [];
  const cur = `${module}@${version}`;
  const known = mods.some((m) => ref(m) === cur);
  return (
    <select aria-label={label} value={cur} onChange={(e) => { const [i, v] = e.target.value.split("@"); onPick(i, v); }}>
      {!known && <option value={cur}>{module ? `${cur} (not in this project)` : "choose a module…"}</option>}
      {mods.map((m) => <option key={ref(m)} value={ref(m)}>{m.id} v{m.version}: {sig(m)}</option>)}
    </select>
  );
}

function ArgsEditor({ mod, args, onChange }: { mod: ModuleDef | undefined; args: Record<string, unknown>; onChange: (a: Record<string, unknown>) => void }) {
  if (!mod) return null;
  if (!mod.params.length) return <div className="muted small">This module declares no arguments.</div>;
  return (
    <>
      {mod.params.map((p) => {
        const v = p.name in args ? args[p.name] : p.default;
        const set = (x: unknown) => onChange({ ...args, [p.name]: x });
        return (
          <div className="row" key={p.name}>
            <label className="lbl">{p.name}{!(p.name in args) && <small> (default)</small>}</label>
            {p.type === "bool" ? <input type="checkbox" aria-label={`argument ${p.name}`} checked={!!v} onChange={(e) => set(e.target.checked)} />
              : p.type === "int" || p.type === "float" ? <input type="number" aria-label={`argument ${p.name}`} step={p.type === "int" ? 1 : "any"} value={Number(v)} onChange={(e) => { const n = Number(e.target.value); if (e.target.value !== "" && Number.isFinite(n)) set(p.type === "int" ? Math.round(n) : n); }} />
                : <input aria-label={`argument ${p.name}`} value={typeof v === "string" ? v : JSON.stringify(v)} onChange={(e) => set(e.target.value)} />}
            {p.type && <small className="muted"> {p.type}</small>}
          </div>
        );
      })}
    </>
  );
}

export function CompositeForm({ node, graph, onConfig }: { node: GNode; graph: Graph; onConfig: (p: Patch) => void }) {
  const c = node.config as { module?: string; version?: string; args?: Record<string, unknown>; share?: string };
  const mod = findModule(graph, c.module ?? "", c.version ?? "1.0.0");
  const others = graph.nodes.filter((n) => n.id !== node.id && n.type === "core.composite" && (n.config as any).module === c.module && (n.config as any).version === c.version && ((n.config as any).share ?? "clone") === "clone");
  const share = c.share ?? "clone";
  return (
    <div className="form">
      <div className="row"><label className="lbl">Module</label>
        <ModulePicker graph={graph} module={c.module ?? ""} version={c.version ?? "1.0.0"} label="module" onPick={(id, v) => onConfig({ module: id, version: v, args: {}, share: "clone" })} /></div>
      {mod && <div className="muted small">{mod.description || "No description."} Signature: {sig(mod)}.</div>}
      <h4>Arguments</h4>
      <ArgsEditor mod={mod} args={c.args ?? {}} onChange={(a) => onConfig({ args: a })} />
      <h4>Parameters</h4>
      <div className="row"><label className="lbl">Parameter sharing</label>
        <select aria-label="parameter sharing" value={share} onChange={(e) => onConfig({ share: e.target.value })}>
          <option value="clone">clone: this instance has its OWN parameters</option>
          {others.map((o) => <option key={o.id} value={o.id}>share with {o.id}: the SAME tensors</option>)}
        </select></div>
      <div className="muted small">{share === "clone"
        ? "Default instantiation clones: every instance of a module is initialised independently and trained separately. To reuse one set of weights at several places, choose \"share with\" on the second instance."
        : `Shared: every inner node of this instance uses the parameter tensors of ${share}. Parameters are counted once, and gradients from both call sites accumulate on the same tensors.`}</div>
    </div>
  );
}

export function RepeatForm({ node, graph, onConfig }: { node: GNode; graph: Graph; onConfig: (p: Patch) => void }) {
  const c = node.config as { module?: string; version?: string; args?: Record<string, unknown>; count?: number; carry?: { input: string; output: string }[]; termination?: { kind: string }; share?: string };
  const mod = findModule(graph, c.module ?? "", c.version ?? "1.0.0");
  const carry = c.carry ?? [];
  return (
    <div className="form">
      <div className="row"><label className="lbl">Body module</label>
        <ModulePicker graph={graph} module={c.module ?? ""} version={c.version ?? "1.0.0"} label="body module" onPick={(id, v) => onConfig({ module: id, version: v, args: {}, carry: [] })} /></div>
      <div className="row"><label className="lbl">Iterations</label>
        <input type="number" aria-label="iterations" min={1} max={64} value={c.count ?? 2} onChange={(e) => { const n = Math.round(Number(e.target.value)); if (n >= 1 && n <= 64) onConfig({ count: n }); }} />
        <small className="muted"> static count, at most 64</small></div>
      <div className="row"><label className="lbl">Termination</label>
        <select aria-label="termination" value={c.termination?.kind ?? "fixed_count"} onChange={(e) => onConfig({ termination: { kind: e.target.value } })}>
          <option value="fixed_count">fixed_count: run exactly the declared count</option>
        </select></div>
      <div className="muted small">Data-dependent termination is not implemented on this backend; the loop is bounded by its count. The loop is unrolled: each iteration is a separate node (it0, it1, ...).</div>
      <h4>Loop-carried state</h4>
      {carry.map((cr, i) => (
        <div className="row" key={i}>
          <select aria-label={`carried input ${i + 1}`} value={cr.input} onChange={(e) => onConfig({ carry: carry.map((x, j) => (j === i ? { ...x, input: e.target.value } : x)) })}>{(mod?.inputs ?? []).map((p) => <option key={p.name}>{p.name}</option>)}</select>
          {" ← "}
          <select aria-label={`carried output ${i + 1}`} value={cr.output} onChange={(e) => onConfig({ carry: carry.map((x, j) => (j === i ? { ...x, output: e.target.value } : x)) })}>{(mod?.outputs ?? []).map((o) => <option key={o.name}>{o.name}</option>)}</select>
          <button className="danger" aria-label={`remove carry ${i + 1}`} onClick={() => onConfig({ carry: carry.filter((_, j) => j !== i) })}>×</button>
        </div>
      ))}
      <button disabled={!mod || !mod.inputs.length || !mod.outputs.length} onClick={() => onConfig({ carry: [...carry, { input: mod!.inputs[0].name, output: mod!.outputs[0].name }] })}>Carry an output into an input</button>
      <div className="muted small">A carried input takes the external value on iteration 0 and the previous iteration's output after that; the type must stay the same (checked).</div>
      <h4>Arguments</h4>
      <ArgsEditor mod={mod} args={c.args ?? {}} onChange={(a) => onConfig({ args: a })} />
      <h4>Parameters</h4>
      <div className="row"><label className="lbl">Across iterations</label>
        <select aria-label="iteration parameter sharing" value={c.share ?? "tied"} onChange={(e) => onConfig({ share: e.target.value })}>
          <option value="tied">tied: one set of tensors reused by every iteration</option>
          <option value="clone">clone: separate parameters per iteration</option>
        </select></div>
    </div>
  );
}

export function SelectForm({ node, graph, onConfig }: { node: GNode; graph: Graph; onConfig: (p: Patch) => void }) {
  const c = node.config as { then?: { module: string; version?: string; args?: Record<string, unknown> }; otherwise?: { module: string; version?: string; args?: Record<string, unknown> } };
  const branch = (key: "then" | "otherwise", title: string) => {
    const b = c[key] ?? { module: "", version: "1.0.0" };
    const mod = findModule(graph, b.module, b.version ?? "1.0.0");
    return (
      <>
        <h4>{title}</h4>
        <div className="row"><label className="lbl">Module</label>
          <ModulePicker graph={graph} module={b.module} version={b.version ?? "1.0.0"} label={`${key} module`} onPick={(id, v) => onConfig({ [key]: { module: id, version: v, args: {} } })} /></div>
        <ArgsEditor mod={mod} args={b.args ?? {}} onChange={(a) => onConfig({ [key]: { ...b, args: a } })} />
      </>
    );
  };
  return (
    <div className="form">
      <div className="muted small">Typed branches: both modules must have the same input and output port names and produce the same types. The predicate input <code>pred</code> must be a scalar bool (reduce a comparison with Any / all).
        Both branches are evaluated; the predicate selects which outputs are used, and gradients flow only through the selected branch.</div>
      {branch("then", "Then branch (pred is true)")}
      {branch("otherwise", "Otherwise branch (pred is false)")}
    </div>
  );
}

/** Node-level parameter sharing for a plain node: use the parameters of another node of the same type and configuration. */
export function SharingRow({ node, graph, onShare }: { node: GNode; graph: Graph; onShare: (target: string | null) => void }) {
  const cands = graph.nodes.filter((n) => n.id !== node.id && n.type === node.type && !n.sharedWith);
  if (!cands.length && !node.sharedWith) return null;
  return (
    <div className="row"><label className="lbl">Share parameters with</label>
      <select aria-label="share parameters with" value={node.sharedWith ?? ""} onChange={(e) => onShare(e.target.value || null)}>
        <option value="">(own parameters)</option>
        {cands.map((n) => <option key={n.id} value={n.id}>{n.id}</option>)}
      </select>
      <div className="muted small">Both nodes must have the same operation and settings. The very same tensors are used at both call sites.</div>
    </div>
  );
}
