import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../api";
import { bumpVersion } from "../modules";
import type { CodeBlockDef, CodeIO } from "../types";
import { fmtNum } from "../util";
import { CodeEditor, type SourceDiagnostic } from "./CodeEditor";

const DTYPES = ["float32", "float64", "int64", "bool"];
const EFFECTS = ["file_read", "file_write", "network"];
const shapeText = (s?: (number | string | null)[] | null) => (s ? s.map((d) => (d === null ? "?" : String(d))).join(", ") : "");
const parseShape = (t: string) => (t.trim() ? t.split(",").map((x) => x.trim()).map((x) => (x === "?" || x === "" ? null : /^\d+$/.test(x) ? Number(x) : x)) : null);

function IORows({ rows, set, kind }: { rows: CodeIO[]; set: (r: CodeIO[]) => void; kind: "input" | "output" }) {
  const upd = (i: number, p: Partial<CodeIO>) => set(rows.map((r, j) => (j === i ? { ...r, ...p } : r)));
  return (
    <>
      {rows.map((r, i) => (
        <div className="row" key={i} style={{ gridTemplateColumns: "1fr" }}>
          <span>
            <input aria-label={`${kind} ${i + 1} name`} value={r.name} size={7} onChange={(e) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(e.target.value) && upd(i, { name: e.target.value })} />{" "}
            {kind === "output" && r.same_as !== undefined && r.same_as !== null ? (
              <span>same as input <input aria-label={`output ${i + 1} same as`} value={r.same_as} size={6} onChange={(e) => upd(i, { same_as: e.target.value })} /> <button className="link" onClick={() => upd(i, { same_as: null, shape: [] })}>declare type</button></span>
            ) : (
              <>
                <select aria-label={`${kind} ${i + 1} dtype`} value={r.dtype} onChange={(e) => upd(i, { dtype: e.target.value })}>{DTYPES.map((d) => <option key={d}>{d}</option>)}</select>{" "}
                <input aria-label={`${kind} ${i + 1} shape`} placeholder="shape: N, 3, D" size={12} defaultValue={shapeText(r.shape)} key={shapeText(r.shape)} onBlur={(e) => upd(i, { shape: parseShape(e.target.value) })} />
                {kind === "output" && <button className="link" onClick={() => upd(i, { same_as: rows.length ? "" : "" })}>same as input</button>}
              </>
            )}
            <button className="danger" aria-label={`remove ${kind} ${i + 1}`} onClick={() => set(rows.filter((_, j) => j !== i))}>×</button>
          </span>
        </div>
      ))}
      <button onClick={() => set([...rows, { name: `${kind === "input" ? "in" : "out"}${rows.length + 1}`, dtype: "float32", shape: kind === "input" ? ["N", 4] : null }])}>Add {kind}</button>
    </>
  );
}

/** Where an imported block came from, and whether its source still equals the pinned import (checked by the backend). */
function OriginBanner({ def }: { def: CodeBlockDef }) {
  const [st, setSt] = useState<{ locallyModified: boolean } | null>(null);
  useEffect(() => { api.post<{ origin: { locallyModified: boolean } | null }>("/api/repos/origin-status", { block: def }).then((r) => setSt(r.origin)).catch(() => setSt(null)); }, [def.source, def.origin]);
  const o = def.origin!;
  return (
    <div className="small repoorigin">
      <b>Imported</b> {o.function} from <code>{o.path}</code> at commit <code>{o.commit.slice(0, 12)}</code> of <code>{o.url}</code> · import <code>{o.importId.slice(0, 12)}</code>
      {st && (st.locallyModified ? <span className="badge old">locally modified since import</span> : <span className="badge ok">unmodified pinned import</span>)}
      <span className="muted"> · runs only in the isolated sandbox. To update deliberately, re-import from another revision (Import from repository… ▸ Compare).</span>
    </div>
  );
}

export function CodeBlockEditor({ def, onChange, onClose, setMessage, usedBy }: { def: CodeBlockDef; onChange: (d: CodeBlockDef) => void; onClose: () => void; setMessage: (m: string) => void; usedBy: string[] }) {
  const [res, setRes] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [jump, setJump] = useState<{ line: number; nonce: number } | null>(null);
  const [deps, setDeps] = useState<any[] | null>(null);
  const [fxText, setFxText] = useState<Record<number, string>>({});
  const upd = (p: Partial<CodeBlockDef>) => onChange({ ...def, ...p });

  const diagnostics: SourceDiagnostic[] = useMemo(() => {
    const out: SourceDiagnostic[] = [];
    for (const f of res?.fixtures ?? []) for (const fr of f.error?.frames ?? []) out.push({ line: fr.line, severity: "error", message: `${f.error.code}: ${f.error.message}` });
    return out;
  }, [res, def.source]);

  const template = async () => {
    if (def.source.trim() && !window.confirm("Replace your source with a fresh template? Your code is never overwritten unless you confirm.")) return;
    try { const r = await api.post<{ source: string }>("/api/codeblocks/template", { block: { ...def, source: "" } }); upd({ source: r.source }); } catch (e) { setMessage(errorText(e)); }
  };
  const test = async () => {
    setBusy(true);
    try { setRes(await api.post<any>("/api/codeblocks/test", { block: def })); } catch (e) { setRes(null); setMessage(errorText(e)); } finally { setBusy(false); }
  };
  const publish = async () => {
    try { const r = await api.post<any>("/api/codeblocks/publish", { block: def }); setMessage(`Saved ${def.id} v${def.version} to My code blocks (${r.alreadyPublished ? "already there, identical" : "new"}; identity ${r.identity.slice(0, 10)}).`); }
    catch (e: any) { setMessage(errorText(e) + (e?.detail?.nextVersion ? `\nChange the version to ${e.detail.nextVersion} and save again.` : "")); }
  };
  const checkDeps = async () => { try { setDeps((await api.post<{ pins: any[] }>("/api/codeblocks/dependencies", { pins: def.dependencies })).pins); } catch (e) { setMessage(errorText(e)); } };
  const fx = def.fixtures;
  const setFx = (i: number, p: Record<string, any>) => upd({ fixtures: fx.map((f, j) => (j === i ? { ...f, ...p } : f)) });

  return (
    <div className="modal" role="dialog" aria-label="code block editor">
      <div className="modalbody codeedit">
        <div className="actions">
          <h3 style={{ margin: 0 }}>Python code block <code>{def.id}</code> v{def.version}</h3>
          <span className="muted small">{usedBy.length ? `used by ${usedBy.join(", ")}` : "not used by a node yet"} · {def.source.split("\n").length} lines · runs in an isolated subprocess</span>
          <span style={{ flex: 1 }} />
          <button onClick={template}>Generate template from the interface</button>
          <button className="primary" disabled={busy} onClick={test}>Test with fixtures</button>
          <button onClick={publish}>Save as version…</button>
          <button onClick={onClose}>Close</button>
        </div>
        {def.origin && <OriginBanner def={def} />}
        <div className="codegrid">
          <div className="cbside">
            <h4>Interface (kept when you edit the code)</h4>
            <div className="row"><label className="lbl">Id / version</label><span><input aria-label="block id" value={def.id} size={10} onChange={(e) => /^[A-Za-z][A-Za-z0-9_]*$/.test(e.target.value) && upd({ id: e.target.value })} /> <input aria-label="block version" value={def.version} size={6} onChange={(e) => upd({ version: e.target.value })} />
              <button className="link" onClick={() => upd({ version: bumpVersion(def.version) })}>bump</button></span></div>
            <div className="row"><label className="lbl">Description</label><input aria-label="block description" value={def.description} onChange={(e) => upd({ description: e.target.value })} /></div>
            <h4>Inputs</h4><IORows rows={def.inputs} set={(r) => upd({ inputs: r })} kind="input" />
            <h4>Outputs</h4><IORows rows={def.outputs} set={(r) => upd({ outputs: r })} kind="output" />
            <h4>Config (editable on the node)</h4>
            {def.config.map((c, i) => (
              <div className="row" key={i} style={{ gridTemplateColumns: "1fr" }}><span>
                <input aria-label={`config ${i + 1} name`} value={c.name} size={7} onChange={(e) => upd({ config: def.config.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)) })} />{" "}
                <select aria-label={`config ${i + 1} type`} value={c.type} onChange={(e) => upd({ config: def.config.map((x, j) => (j === i ? { ...x, type: e.target.value, default: e.target.value === "str" ? "" : e.target.value === "bool" ? false : 0 } : x)) })}>{["float", "int", "str", "bool"].map((t) => <option key={t}>{t}</option>)}</select>{" "}
                <input aria-label={`config ${i + 1} default`} size={6} value={String(c.default)} onChange={(e) => { const v = e.target.value; upd({ config: def.config.map((x, j) => (j === i ? { ...x, default: c.type === "float" || c.type === "int" ? (Number.isFinite(Number(v)) ? Number(v) : x.default) : c.type === "bool" ? v === "true" : v } : x)) }); }} />
                <button className="danger" aria-label={`remove config ${i + 1}`} onClick={() => upd({ config: def.config.filter((_, j) => j !== i) })}>×</button></span></div>
            ))}
            <button onClick={() => upd({ config: [...def.config, { name: `k${def.config.length + 1}`, type: "float", default: 1.0 }] })}>Add config field</button>
            <h4>State (persists between calls)</h4>
            {def.state.map((s, i) => (
              <div className="row" key={i} style={{ gridTemplateColumns: "1fr" }}><span>
                <input aria-label={`state ${i + 1} name`} value={s.name} size={7} onChange={(e) => upd({ state: def.state.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)) })} /> shape <input aria-label={`state ${i + 1} shape`} size={5} defaultValue={s.shape.join(",")} onBlur={(e) => upd({ state: def.state.map((x, j) => (j === i ? { ...x, shape: e.target.value.split(",").map(Number).filter((n) => n > 0) } : x)) })} /> init <input aria-label={`state ${i + 1} init`} type="number" value={s.init} onChange={(e) => upd({ state: def.state.map((x, j) => (j === i ? { ...x, init: Number(e.target.value) } : x)) })} />
                <button className="danger" aria-label={`remove state ${i + 1}`} onClick={() => upd({ state: def.state.filter((_, j) => j !== i) })}>×</button></span></div>
            ))}
            <button onClick={() => upd({ state: [...def.state, { name: `s${def.state.length + 1}`, shape: [1], dtype: "float32", init: 0 }] })}>Add state field</button>
            <h4>Declared behaviour (tested, never inferred)</h4>
            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
              <span>effects: {EFFECTS.map((f) => <label key={f}><input type="checkbox" aria-label={`effect ${f}`} checked={def.effects.includes(f)} onChange={(e) => upd({ effects: e.target.checked ? [...def.effects, f] : def.effects.filter((x) => x !== f) })} /> {f} </label>)}{def.effects.length === 0 && <small className="muted">none declared: undeclared network/file access fails</small>}</span>
              <span>randomness <select aria-label="randomness" value={def.randomness} onChange={(e) => upd({ randomness: e.target.value })}><option value="none">none (checked: no random numbers consumed)</option><option value="seeded">seeded (given a seed from the run's random stream)</option></select></span>
              <span><label><input type="checkbox" aria-label="differentiable" checked={def.differentiable} onChange={(e) => upd({ differentiable: e.target.checked })} /> differentiable</label> <small className="muted">{def.differentiable ? "only if the code uses torch operations on the inputs; checked with a finite difference" : "an explicit gradient boundary: outputs carry no gradient"}</small></span>
            </div>
            <h4>Pinned dependencies (part of the block's identity)</h4>
            <input aria-label="dependencies" placeholder="name==version, comma separated" value={def.dependencies.join(", ")} onChange={(e) => upd({ dependencies: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
            <button onClick={checkDeps}>Check the environment</button>
            {deps && <ul className="small">{deps.map((d) => <li key={d.pin} className={d.status === "ok" ? "" : "exec"}>{d.pin}: {d.status}{d.installed ? ` (installed ${d.installed})` : ""}</li>)}</ul>}
            <div className="muted small">Dependencies are checked, not installed, in this version.</div>
            <h4>Limits</h4>
            <div className="row" style={{ gridTemplateColumns: "1fr" }}><span>CPU s <input type="number" aria-label="cpu seconds" size={4} value={def.limits.cpu_seconds ?? 30} onChange={(e) => upd({ limits: { ...def.limits, cpu_seconds: Number(e.target.value) } })} /> wall s <input type="number" aria-label="wall seconds" value={def.limits.wall_seconds ?? 30} onChange={(e) => upd({ limits: { ...def.limits, wall_seconds: Number(e.target.value) } })} /></span></div>
          </div>
          <div className="cbmain">
            <CodeEditor value={def.source} onChange={(v) => upd({ source: v })} diagnostics={diagnostics} jump={jump} />
            <h4>Fixtures (test inputs and optional expected outputs)</h4>
            {fx.map((f, i) => (
              <div className="fixture" key={i}>
                <input aria-label={`fixture ${i + 1} name`} value={f.name ?? ""} size={12} onChange={(e) => setFx(i, { name: e.target.value })} />
                <textarea aria-label={`fixture ${i + 1} json`} rows={3} spellCheck={false} value={fxText[i] ?? JSON.stringify({ inputs: f.inputs, config: f.config, expect: f.expect }, null, 1)}
                  onChange={(e) => { setFxText({ ...fxText, [i]: e.target.value }); try { const j = JSON.parse(e.target.value); setFx(i, { inputs: j.inputs, config: j.config, expect: j.expect }); } catch { /* keep typing */ } }} />
                <button className="danger" aria-label={`remove fixture ${i + 1}`} onClick={() => upd({ fixtures: fx.filter((_, j) => j !== i) })}>×</button>
              </div>
            ))}
            <button onClick={() => upd({ fixtures: [...fx, { name: `fixture ${fx.length + 1}`, inputs: Object.fromEntries(def.inputs.map((p) => [p.name, { shape: (p.shape ?? ["N", 4]).map((d) => (typeof d === "number" ? d : 2)), seed: 0 }])) }] })}>Add fixture</button>
            <div className="muted small">Input: {"{ \"shape\": [2, 3], \"seed\": 0 }"} (seeded random) or {"{ \"values\": [[...]] }"}. Expected: {"{ \"y\": { \"values\": [[...]], \"atol\": 1e-6 } }"}.</div>
            {res && (
              <div className="testres">
                <h4>Test result: {res.ok ? <span className="badge ok">all fixtures pass</span> : <span className="badge old">failures</span>} <small className="muted">source sha256 {res.sourceSha256.slice(0, 12)} · identity {res.identity.slice(0, 12)}</small></h4>
                {res.fixtures.map((f: any) => (
                  <div key={f.name + f.cacheKey} className="fxres">
                    <b>{f.ok ? "✓" : "✗"} {f.name}</b> <small className="muted">{f.cached ? "cached result for this exact source, pins and fixture" : `ran in ${f.seconds.toFixed(2)} s`}</small>
                    <ul className="small">{f.checks.map((c: any, i: number) => <li key={i} className={c.ok ? "" : "exec"}>{c.ok ? "✓" : "✗"} {c.name}{c.detail ? ` — ${c.detail}` : ""}</li>)}</ul>
                    {f.outputs && Object.entries(f.outputs).map(([k, o]: [string, any]) => <div key={k} className="small">output <b>{k}</b> {o.dtype}[{o.shape.join(", ")}]: <code>{o.values.slice(0, 8).map((v: number) => fmtNum(v, 4)).join(" ")}{o.values.length > 8 ? " …" : ""}</code></div>)}
                    {f.error && (
                      <div className="errbadge">
                        <b>{f.error.code}</b> {f.error.message}
                        {f.error.frames.map((fr: any, i: number) => <div key={i}><button className="link" onClick={() => setJump({ line: fr.line, nonce: Date.now() })}>line {fr.line} in {fr.function}</button> <code>{fr.text}</code></div>)}
                        {f.error.traceback && <details><summary>original traceback</summary><pre>{f.error.traceback}</pre></details>}
                      </div>
                    )}
                    {f.stdout && <pre>{f.stdout}</pre>}
                  </div>
                ))}
              </div>
            )}
            <div className="muted small">Source debugging: exceptions are shown at their source lines. Breakpoints and variable inspection inside the code are not available in this version.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
