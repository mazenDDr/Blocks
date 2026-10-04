import { useEffect, useState } from "react";
import { api, errorText } from "../api";
import { bumpVersion, replaceModule } from "../modules";
import type { Graph, ModuleDef, Validation } from "../types";
import { fmtShape } from "../util";

const DTYPES = ["float32", "float64", "int64", "bool"];
const shapeText = (s: (number | string | null)[] | null | undefined) => (s ? s.map((d) => (d === null ? "?" : String(d))).join(", ") : "");
const parseShape = (t: string): (number | string | null)[] | null => {
  if (!t.trim()) return null;
  return t.split(",").map((x) => x.trim()).map((x) => (x === "?" || x === "" ? null : /^\d+$/.test(x) ? Number(x) : x));
};

interface Props {
  graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; def: ModuleDef; validation: (Validation & { outputs?: Record<string, any>; params?: number }) | null;
  usedBy: string[]; onExit: () => void; setMessage: (m: string) => void;
}

function InterfaceTab({ graph, setGraph, def }: Props) {
  const upd = (patch: Partial<ModuleDef>) => setGraph((g) => replaceModule(g, def, { ...def, ...patch }));
  const rewire = (from: string, to: string) => def.edges.map((e) => (e.from.node === "$in" && e.from.port === from ? { ...e, from: { ...e.from, port: to } } : e));
  const used = (graph.nodes.filter((n) => n.type === "core.composite" && (n.config as any).module === def.id).length);
  return (
    <div className="tabbody">
      <div className="row"><label className="lbl">Description</label><textarea rows={2} aria-label="module description" value={def.description} onChange={(e) => upd({ description: e.target.value })} /></div>
      <h4>Inputs (typed ports)</h4>
      {def.inputs.map((p, i) => (
        <div className="row" key={i}>
          <input aria-label={`input ${i + 1} name`} value={p.name} size={8} onChange={(e) => { const nm = e.target.value; if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(nm)) upd({ inputs: def.inputs.map((x, j) => (j === i ? { ...x, name: nm } : x)), edges: rewire(p.name, nm) }); }} />
          <select aria-label={`input ${i + 1} dtype`} value={p.dtype} onChange={(e) => upd({ inputs: def.inputs.map((x, j) => (j === i ? { ...x, dtype: e.target.value } : x)) })}>{DTYPES.map((d) => <option key={d}>{d}</option>)}</select>
          <input aria-label={`input ${i + 1} shape`} placeholder="shape: ?, 16, T (blank = any)" defaultValue={shapeText(p.shape)} key={shapeText(p.shape)} size={18}
            onBlur={(e) => upd({ inputs: def.inputs.map((x, j) => (j === i ? { ...x, shape: parseShape(e.target.value) } : x)) })} />
          <button className="danger" aria-label={`remove input ${i + 1}`} onClick={() => upd({ inputs: def.inputs.filter((_, j) => j !== i), edges: def.edges.filter((e) => !(e.from.node === "$in" && e.from.port === p.name)) })}>×</button>
        </div>
      ))}
      <button onClick={() => upd({ inputs: [...def.inputs, { name: `in${def.inputs.length + 1}`, dtype: "float32", shape: null }] })}>Add input</button>
      <div className="muted small">Shape entries: a number, N (the batch axis), a name such as T (bound consistently across ports) or ? (any). Checked wherever the module is instantiated.</div>
      <h4>Outputs</h4>
      {def.outputs.map((o, i) => (
        <div className="row" key={i}>
          <input aria-label={`output ${i + 1} name`} value={o.name} size={8} onChange={(e) => { const nm = e.target.value; if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(nm)) upd({ outputs: def.outputs.map((x, j) => (j === i ? { ...x, name: nm } : x)) }); }} />
          <input aria-label={`output ${i + 1} shape`} placeholder="declared shape (optional)" defaultValue={shapeText(o.shape)} key={shapeText(o.shape)} size={18}
            onBlur={(e) => upd({ outputs: def.outputs.map((x, j) => (j === i ? { ...x, shape: parseShape(e.target.value) } : x)) })} />
          <span className="muted small">{o.from.node ? `from ${o.from.node}.${o.from.port}` : "not wired"}</span>
          <button className="danger" aria-label={`remove output ${i + 1}`} onClick={() => upd({ outputs: def.outputs.filter((_, j) => j !== i) })}>×</button>
        </div>
      ))}
      <button onClick={() => upd({ outputs: [...def.outputs, { name: `out${def.outputs.length + 1}`, from: { node: "", port: "" } }] })}>Add output</button>
      <div className="muted small">Wire a node into the output box on the canvas to decide what it returns.</div>
      <h4>Arguments (config parameters)</h4>
      {def.params.map((p, i) => (
        <div className="row" key={i}>
          <input aria-label={`parameter ${i + 1} name`} value={p.name} size={8} onChange={(e) => { const nm = e.target.value; if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(nm)) upd({ params: def.params.map((x, j) => (j === i ? { ...x, name: nm } : x)) }); }} />
          <select aria-label={`parameter ${i + 1} type`} value={p.type ?? ""} onChange={(e) => upd({ params: def.params.map((x, j) => (j === i ? { ...x, type: e.target.value || null } : x)) })}>
            <option value="">any</option>{["int", "float", "str", "bool"].map((t) => <option key={t}>{t}</option>)}</select>
          <input aria-label={`parameter ${i + 1} default`} defaultValue={JSON.stringify(p.default)} key={JSON.stringify(p.default)} size={8}
            onBlur={(e) => { try { upd({ params: def.params.map((x, j) => (j === i ? { ...x, default: JSON.parse(e.target.value) } : x)) }); } catch { /* keep the previous default */ } }} />
          <button className="danger" aria-label={`remove parameter ${i + 1}`} onClick={() => upd({ params: def.params.filter((_, j) => j !== i) })}>×</button>
        </div>
      ))}
      <button onClick={() => upd({ params: [...def.params, { name: `p${def.params.length + 1}`, default: 8, type: "int" }] })}>Add argument</button>
      <div className="muted small">Inside the module, set any node setting to <code>{`{"$param": "name"}`}</code> (JSON) to use an argument; each instance supplies its own value.</div>
      {def.reduction && (
        <>
          <h4>Declared reduction semantics</h4>
          <table><tbody>{Object.entries(def.reduction).map(([k, v]) => <tr key={k}><td>{k}</td><td>{String(v)}</td></tr>)}</tbody></table>
        </>
      )}
      <div className="muted small">{used} instance(s) of this module are in this project. Changing the interface may leave their wires dangling; they are reported as errors, never repaired silently.</div>
    </div>
  );
}

function TestTab({ graph, def }: Props) {
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [args, setArgs] = useState("{}");
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setTexts(Object.fromEntries(def.inputs.map((p) => [p.name, texts[p.name] ?? JSON.stringify({ shape: (p.shape ?? ["N", 4]).map((d) => (typeof d === "number" ? d : 3)), seed: 0, dtype: p.dtype })]))); }, [def.inputs.map((p) => p.name).join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const run = async () => {
    setErr(null);
    try {
      const inputs = Object.fromEntries(Object.entries(texts).map(([k, t]) => [k, JSON.parse(t)]));
      setRes(await api.post<any>("/api/modules/test", { graph, moduleId: def.id, version: def.version, inputs, args: JSON.parse(args || "{}"), grads: true }));
    } catch (e) { setRes(null); setErr(errorText(e)); }
  };
  return (
    <div className="tabbody">
      <div className="muted small">Runs this module on the tensors below through the normal lowering path (a tiny numerical example). Give a nested list of numbers, or {`{"shape": [..], "seed": 0}`} for a seeded random tensor. The result is computed now; it is not a recorded run.</div>
      {def.inputs.map((p) => (
        <div className="row" key={p.name}><label className="lbl">{p.name}</label><textarea rows={2} aria-label={`test input ${p.name}`} value={texts[p.name] ?? ""} onChange={(e) => setTexts({ ...texts, [p.name]: e.target.value })} /></div>
      ))}
      {def.params.length > 0 && <div className="row"><label className="lbl">Arguments</label><input aria-label="test arguments" value={args} onChange={(e) => setArgs(e.target.value)} /></div>}
      <button className="primary" onClick={run}>Run the module</button>
      {err && <div className="error pre">{err}</div>}
      {res && (
        <>
          <h4>Outputs</h4>
          {Object.entries(res.outputs).map(([k, v]: [string, any]) => <div key={k}><b>{k}</b> {fmtShape({ shape: v.shape, dtype: v.dtype })} <code>{JSON.stringify(v.values.map((x: number) => Number(x.toPrecision(6))))}</code></div>)}
          {Object.keys(res.grads).length > 0 && <><h4>Gradients of the scalar output</h4>{Object.entries(res.grads).map(([k, v]: [string, any]) => <div key={k}><b>d/d {k}</b> <code>{JSON.stringify(v.values.map((x: number) => Number(x.toPrecision(6))))}</code></div>)}</>}
          {res.reduction && <div className="muted small">Declared reduction: {res.reduction.divisor ?? ""} {res.reduction.formula ? `· ${res.reduction.formula}` : ""}</div>}
          <div className="muted small">{res.provenance.kind}</div>
        </>
      )}
    </div>
  );
}

function VersionsTab({ graph, setGraph, def, usedBy, setMessage }: Props) {
  const [info, setInfo] = useState<any>(null);
  const [note, setNote] = useState<string>("");
  const load = () => api.get<any>(`/api/modules/${encodeURIComponent(def.id)}`).then(setInfo).catch(() => setInfo(null));
  useEffect(() => { load(); }, [def.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const publish = async () => {
    try {
      const r = await api.post<any>("/api/modules/publish", { module: def, note });
      setMessage(`${r.alreadyPublished ? "Already published" : "Published"} ${def.id} v${def.version} (content ${r.contentHash.slice(0, 8)}).`);
      load();
    } catch (e: any) {
      setMessage(errorText(e) + (e?.detail?.nextVersion ? `\nUse "New version in this project" to create v${e.detail.nextVersion}.` : ""));
    }
  };
  const newVersion = () => {
    const v = bumpVersion(def.version);
    setGraph((g) => ({ ...g, modules: [...(g.modules ?? []), { ...JSON.parse(JSON.stringify(def)), version: v }] }));
    setMessage(`Created ${def.id} v${v} as a copy in this project. Instances still use v${def.version} until you move them (no silent upgrade).`);
  };
  const siblings = (graph.modules ?? []).filter((m) => m.id === def.id && m.version !== def.version);
  const moveInstances = (to: string) => {
    setGraph((g) => ({ ...g, nodes: g.nodes.map((n) => (n.type === "core.composite" && (n.config as any).module === def.id && (n.config as any).version === def.version ? { ...n, config: { ...n.config, version: to } } : n)) }));
    setMessage(`Moved ${usedBy.length} instance(s) from v${def.version} to v${to}.`);
  };
  return (
    <div className="tabbody">
      <div>Editing <b>{def.id}</b> v{def.version}. {usedBy.length ? `Used by ${usedBy.join(", ")}.` : "Not used by any instance yet."}</div>
      <div className="muted small">Edits here change this definition inside the project (and so its semantic hash). Published versions in My modules are immutable.</div>
      <div className="row"><input aria-label="publish note" placeholder="note for this version (optional)" value={note} onChange={(e) => setNote(e.target.value)} /><button className="primary" onClick={publish}>Publish v{def.version} to My modules</button></div>
      <div className="row"><button onClick={newVersion}>New version in this project (v{bumpVersion(def.version)})</button></div>
      {siblings.length > 0 && (
        <>
          <h4>Other versions in this project</h4>
          {siblings.map((m) => <div key={m.version}>v{m.version} <button disabled={!usedBy.length} onClick={() => moveInstances(m.version)}>Move this version's {usedBy.length} instance(s) to v{m.version}</button></div>)}
        </>
      )}
      <h4>Published versions (My modules)</h4>
      {info ? <div>{info.versions.map((v: string) => <span key={v} className="badge ok">v{v}</span>)}</div> : <div className="muted">None published yet.</div>}
    </div>
  );
}

export function ModulePanel(p: Props) {
  const [tab, setTab] = useState<"Interface" | "Test" | "Versions">("Interface");
  const v = p.validation;
  return (
    <div className="runpanel">
      <div className="tabs" role="tablist">
        <b style={{ padding: "4px 8px" }}>Module {p.def.id} v{p.def.version}</b>
        {(["Interface", "Test", "Versions"] as const).map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}
        <span className="spacer" style={{ flex: 1 }} />
        {v && <span className={`vsum ${v.ok ? "good" : "bad"}`} style={{ padding: "4px 8px" }}>{v.ok ? `valid · ${v.params ?? 0} parameters` : `${v.diagnostics.filter((d) => d.severity === "error").length} error(s)`}</span>}
        <button onClick={p.onExit}>Back to the project</button>
      </div>
      <div style={{ overflow: "auto", flex: 1 }}>
        {tab === "Interface" && <InterfaceTab {...p} />}
        {tab === "Test" && <TestTab {...p} />}
        {tab === "Versions" && <VersionsTab {...p} />}
      </div>
    </div>
  );
}
