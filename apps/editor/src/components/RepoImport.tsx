import { useMemo, useState } from "react";
import { api, errorText } from "../api";
import type { CodeBlockDef } from "../types";
import { shortHash } from "../util";

interface Resolved { repoId: string; url: string; commit: string; tree: string; subject: string; committedAt: string; refs: Record<string, string>; note: string }
interface Entry { path: string; mode: string; type: string; blob: string; size: number | null; kind: string }
interface Dep { spec: string; name: string; constraint: string | null; pinned: boolean; pin: string | null; source: string }
interface Tree { entries: Entry[]; counts: Record<string, number>; dependencies: { declared: Dep[]; notes: string[] }; license: { path: string; spdxGuess: string | null; sha256: string; note: string } | null }
interface Fn { name: string; line: number; params: { name: string; default: unknown; hasDefault: boolean }[]; varargs: boolean; doc: string | null }
interface PyInfo { path: string; blob: string; sha256: string; functions: Fn[]; imports: string[]; localImports: string[]; wrappable: boolean; problems: string[]; note: string }

const KIND_LABEL: Record<string, string> = {
  python_source: "Python", installation_script: "installation script (never run)", dependency_manifest: "dependencies", license: "license", dataset: "dataset",
  model_weights: "model weights", configuration: "configuration", documentation: "docs", notebook: "notebook", lfs_pointer: "large-file pointer (not downloaded)",
  submodule: "submodule (not fetched)", symlink: "symlink (not followed)", other: "other",
};

/** Browse a Git repository at a pinned commit (nothing is checked out, installed or executed) and wrap one function as a code block. */
export function RepoImport({ onImport, onClose, setMessage }: { onImport: (d: CodeBlockDef) => void; onClose: () => void; setMessage: (m: string) => void }) {
  const [url, setUrl] = useState("");
  const [rev, setRev] = useState("HEAD");
  const [res, setRes] = useState<Resolved | null>(null);
  const [tree, setTree] = useState<Tree | null>(null);
  const [filter, setFilter] = useState("");
  const [file, setFile] = useState<{ path: string; text: string | null; sha256: string; blob: string; note?: string } | null>(null);
  const [py, setPy] = useState<PyInfo | null>(null);
  const [fn, setFn] = useState<string>("");
  const [roles, setRoles] = useState<Record<string, "input" | "config" | "default">>({});
  const [outputs, setOutputs] = useState("y");
  const [sameAs, setSameAs] = useState(true);
  const [pins, setPins] = useState<string[]>([]);
  const [cmpRev, setCmpRev] = useState("");
  const [cmp, setCmp] = useState<{ head: string; changed: { status: string; path: string }[]; patch: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const go = async <T,>(f: () => Promise<T>) => { setBusy(true); setErr(null); try { return await f(); } catch (e) { setErr(errorText(e)); return undefined; } finally { setBusy(false); } };
  const resolve = () => go(async () => {
    const r = await api.post<Resolved>("/api/repos/resolve", { url, rev });
    setRes(r); setFile(null); setPy(null); setCmp(null);
    setTree(await api.get<Tree>(`/api/repos/${r.repoId}/tree?commit=${r.commit}`));
  });
  const open = (e: Entry) => go(async () => {
    if (!res || e.type !== "blob" || e.kind === "symlink") return;
    const q = `commit=${res.commit}&path=${encodeURIComponent(e.path)}`;
    setFile(await api.get<any>(`/api/repos/${res.repoId}/file?${q}`));
    setPy(null); setFn(""); setCmp(null);
    if (e.kind === "python_source" || e.kind === "installation_script" && e.path.endsWith(".py")) setPy(await api.get<PyInfo>(`/api/repos/${res.repoId}/python?${q}`));
  });
  const choose = (name: string) => {
    setFn(name);
    const f = py?.functions.find((x) => x.name === name);
    setRoles(Object.fromEntries((f?.params ?? []).map((p, i) => [p.name, i === 0 || !p.hasDefault ? "input" : "default"])));
  };
  const compare = () => go(async () => {
    if (!res || !file) return;
    const other = await api.post<Resolved>("/api/repos/resolve", { url: res.url, rev: cmpRev });
    setCmp(await api.post<any>(`/api/repos/${res.repoId}/compare`, { base: res.commit, head: other.commit, paths: [file.path] }));
  });
  const f = py?.functions.find((x) => x.name === fn);
  const declaredPins = useMemo(() => (tree?.dependencies.declared ?? []).filter((d) => d.pinned && d.pin).map((d) => d.pin!) , [tree]);
  const doImport = () => go(async () => {
    if (!res || !py || !f) return;
    const ins = f.params.filter((p) => roles[p.name] === "input").map((p) => ({ name: p.name, dtype: "float32", shape: ["N", 4] }));
    const cfg = f.params.filter((p) => roles[p.name] === "config").map((p) => ({ name: p.name, type: typeof p.default === "number" ? (Number.isInteger(p.default) ? "int" : "float") : typeof p.default === "boolean" ? "bool" : "str", default: p.default }));
    const outs = outputs.split(",").map((s) => s.trim()).filter(Boolean).map((name, i) => (sameAs && i === 0 && ins.length ? { name, dtype: "float32", same_as: ins[0].name } : { name, dtype: "float32", shape: null }));
    const id = `${f.name}_${res.commit.slice(0, 7)}`.replace(/[^A-Za-z0-9_]/g, "_");
    const iface = { id, version: "1.0.0", description: (f.doc ?? `${f.name} from ${py.path}`).split("\n")[0], inputs: ins, outputs: outs, config: cfg, state: [], effects: [], randomness: "none", differentiable: false,
      fixtures: [{ name: "random batch", inputs: Object.fromEntries(ins.map((i) => [i.name, { shape: [3, 4], seed: 0 }])) }], limits: {} };
    const r = await api.post<{ importId: string; block: CodeBlockDef }>(`/api/repos/${res.repoId}/import`, { commit: res.commit, path: py.path, function: f.name, interface: iface, pins });
    onImport(r.block);
    setMessage(`Imported ${f.name} from ${py.path} at ${shortHash(res.commit)} as code block ${r.block.id} (import ${shortHash(r.importId)}). Adjust shapes and test it in the code editor; it runs only in the sandbox.`);
    onClose();
  });
  const shown = (tree?.entries ?? []).filter((e) => !filter || e.path.toLowerCase().includes(filter.toLowerCase()) || (KIND_LABEL[e.kind] ?? e.kind).includes(filter.toLowerCase()));

  return (
    <div className="modal" role="dialog" aria-label="import from repository">
      <div className="modalbody codeedit repoimport">
        <div className="actions">
          <h3 style={{ margin: 0 }}>Import code from a Git repository</h3>
          <span className="muted small">browsed at a pinned commit · nothing is checked out, installed or executed</span>
          <span style={{ flex: 1 }} />
          <button onClick={onClose}>Close</button>
        </div>
        <div className="actions">
          <label>Repository <input aria-label="repository url" size={48} placeholder="/abs/path, file://, https:// or ssh:// (no credentials in the URL)" value={url} onChange={(e) => setUrl(e.target.value)} /></label>
          <label>Revision <input aria-label="revision" size={14} value={rev} onChange={(e) => setRev(e.target.value)} /></label>
          <button className="primary" disabled={busy || !url.trim()} onClick={resolve}>Resolve and browse</button>
        </div>
        {err && <div className="error pre">{err}</div>}
        {res && (
          <div className="small"><b>commit</b> <code>{res.commit}</code> · tree <code>{shortHash(res.tree)}</code> · {res.subject} · {res.committedAt} <span className="muted">— {res.note}</span></div>
        )}
        {tree && (
          <div className="codegrid">
            <div className="cbside">
              <h4>Contents ({tree.entries.length} entries)</h4>
              <div className="small muted">{Object.entries(tree.counts).map(([k, n]) => `${KIND_LABEL[k] ?? k}: ${n}`).join(" · ")}</div>
              <input aria-label="filter entries" placeholder="filter by path or kind" value={filter} onChange={(e) => setFilter(e.target.value)} />
              <div className="repotree">
                <table><tbody>{shown.slice(0, 400).map((e) => (
                  <tr key={e.path} className={file?.path === e.path ? "sel" : ""}>
                    <td>{e.type === "blob" && e.kind !== "symlink" ? <button className="link" onClick={() => open(e)}>{e.path}</button> : e.path}</td>
                    <td><span className={`badge kind-${e.kind}`}>{KIND_LABEL[e.kind] ?? e.kind}</span></td><td className="muted">{e.size ?? ""}</td>
                  </tr>))}</tbody></table>
                {shown.length > 400 && <div className="muted small">{shown.length - 400} more; filter to narrow.</div>}
              </div>
              <h4>License</h4>
              {tree.license ? <div className="small">{tree.license.path}: {tree.license.spdxGuess ?? "not recognised"} <span className="muted">({tree.license.note})</span></div> : <div className="small muted">No license file at the repository root.</div>}
              <h4>Declared dependencies (parsed, not installed)</h4>
              <ul className="small">{tree.dependencies.declared.map((d, i) => (
                <li key={i}>{d.pinned
                  ? <label><input type="checkbox" aria-label={`pin ${d.pin}`} checked={pins.includes(d.pin!)} onChange={(e) => setPins(e.target.checked ? [...pins, d.pin!] : pins.filter((x) => x !== d.pin))} /> <code>{d.pin}</code></label>
                  : <><code>{d.spec}</code> <span className="muted">unpinned: cannot become a block pin</span></>} <span className="muted">({d.source})</span></li>))}</ul>
              {tree.dependencies.notes.map((n, i) => <div key={i} className="warn small">{n}</div>)}
              {declaredPins.length > 0 && <div className="muted small">Checked pins become the block's dependencies and are verified against the sandbox environment when it runs.</div>}
            </div>
            <div className="cbmain">
              {!file ? <div className="muted">Select a file to read it at this commit.</div> : (
                <>
                  <h4>{file.path} <small className="muted">blob {shortHash(file.blob)} · sha256 {shortHash(file.sha256)}</small></h4>
                  {file.text === null ? <div className="muted">{file.note}</div> : <pre className="repofile">{file.text}</pre>}
                  <div className="actions">
                    <label>Compare with revision <input aria-label="compare revision" size={14} value={cmpRev} onChange={(e) => setCmpRev(e.target.value)} /></label>
                    <button disabled={busy || !cmpRev.trim()} onClick={compare}>Compare</button>
                  </div>
                  {cmp && <div className="small">{cmp.changed.length === 0 ? `Unchanged at ${shortHash(cmp.head)}.` : <><b>changed at {shortHash(cmp.head)}</b><pre className="repofile">{cmp.patch}</pre></>}</div>}
                  {py && (
                    <div className="repopy">
                      <h4>Wrap a function as a code block <small className="muted">{py.note}</small></h4>
                      {!py.wrappable && py.problems.map((p, i) => <div key={i} className="errbadge">{p}</div>)}
                      {py.wrappable && (
                        <>
                          <div className="small">imports: {py.imports.join(", ") || "none"}</div>
                          <label>Function <select aria-label="entry function" value={fn} onChange={(e) => choose(e.target.value)}>
                            <option value="">(choose)</option>
                            {py.functions.filter((x) => !x.varargs).map((x) => <option key={x.name} value={x.name}>{x.name}({x.params.map((p) => p.name).join(", ")}) · line {x.line}</option>)}
                          </select></label>
                          {f && (
                            <>
                              <table className="small"><thead><tr><th>Parameter</th><th>Role</th><th>Default</th></tr></thead><tbody>
                                {f.params.map((p) => (
                                  <tr key={p.name}><td>{p.name}</td><td>
                                    <select aria-label={`role ${p.name}`} value={roles[p.name]} onChange={(e) => setRoles({ ...roles, [p.name]: e.target.value as any })}>
                                      <option value="input">tensor input</option>
                                      {p.hasDefault && <option value="config">node setting</option>}
                                      {p.hasDefault && <option value="default">keep the function's default</option>}
                                    </select></td><td>{p.hasDefault ? JSON.stringify(p.default) : "required"}</td></tr>))}
                              </tbody></table>
                              <label>Outputs <input aria-label="output names" size={16} value={outputs} onChange={(e) => setOutputs(e.target.value)} /></label>{" "}
                              <label><input type="checkbox" aria-label="first output same as first input" checked={sameAs} onChange={(e) => setSameAs(e.target.checked)} /> first output has the first input's shape</label>
                              <div className="actions"><button className="primary" disabled={busy || !Object.values(roles).includes("input")} onClick={doImport}>Import pinned function</button></div>
                              <div className="muted small">Records the URL, commit, path, blob, sha256, license, declared dependencies and chosen pins as an immutable import; the block keeps them as its origin (part of the graph's identity). Several outputs mean the function returns a tuple in that order.</div>
                            </>
                          )}
                        </>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
