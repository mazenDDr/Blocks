import { useEffect, useState } from "react";
import { api, errorText } from "../api";
import type { ConnType, Connection, DiscColumn, DiscTable, PreviewResult, S3Object, SecretRef } from "../types";
import { fmtInt } from "../util";
import { ErrorBox, PreviewTable, useConnections } from "./Connectors";

type AddSource = (type: string, config: Record<string, unknown>) => string | null;

// ---------------------------------------------------------------------------------------- secret reference input
type RefState = { kind: "none" | "env" | "file"; name: string; path: string; key: string };
const EMPTY_REF: RefState = { kind: "none", name: "", path: "", key: "" };
const toRef = (r: RefState): SecretRef | undefined => (r.kind === "env" ? { kind: "env", name: r.name } : r.kind === "file" ? { kind: "file", path: r.path, key: r.key } : undefined);

function SecretRefInput({ label, value, onChange }: { label: string; value: RefState; onChange: (v: RefState) => void }) {
  return (
    <div className="row secretref">
      <label className="lbl">{label}</label>
      <select aria-label={`${label} reference kind`} value={value.kind} onChange={(e) => onChange({ ...value, kind: e.target.value as RefState["kind"] })}>
        <option value="none">none</option><option value="env">environment variable</option><option value="file">secrets file (outside the project)</option>
      </select>
      {value.kind === "env" && <input aria-label={`${label} environment variable name`} placeholder="NAME_OF_ENV_VAR" value={value.name} onChange={(e) => onChange({ ...value, name: e.target.value })} />}
      {value.kind === "file" && <>
        <input aria-label={`${label} secrets file path`} placeholder="/absolute/path/secrets.json" value={value.path} onChange={(e) => onChange({ ...value, path: e.target.value })} />
        <input aria-label={`${label} secrets file key`} placeholder="key" size={10} value={value.key} onChange={(e) => onChange({ ...value, key: e.target.value })} /></>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- add connection
function AddConnection({ onDone }: { onDone: (id: string) => void }) {
  const [type, setType] = useState<ConnType>("postgres");
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [s, setS] = useState<Record<string, any>>({ host: "", port: 5432, dbname: "", user: "", sslmode: "prefer", connect_timeout_s: 5, bucket: "", region: "us-east-1", endpoint_url: "", verify_tls: true, repo: "", default_rev: "", remote: "" });
  const [refs, setRefs] = useState<Record<string, RefState>>({});
  const [err, setErr] = useState<unknown>(null);
  const ref = (k: string) => refs[k] ?? EMPTY_REF;
  const set = (k: string, v: unknown) => setS((x) => ({ ...x, [k]: v }));
  const secretFields = type === "postgres" ? [["password", "Password"]] : type === "s3" ? [["access_key_id", "Access key id"], ["secret_access_key", "Secret access key"], ["session_token", "Session token (optional)"]] : [];
  async function submit() {
    setErr(null);
    const settings = type === "postgres" ? { host: s.host, port: Number(s.port), dbname: s.dbname, user: s.user, sslmode: s.sslmode, connect_timeout_s: Number(s.connect_timeout_s) }
      : type === "s3" ? { bucket: s.bucket, region: s.region, endpoint_url: s.endpoint_url || null, verify_tls: s.verify_tls }
        : { repo: s.repo, default_rev: s.default_rev || null, remote: s.remote || null };
    const secrets: Record<string, SecretRef> = {};
    for (const [k] of secretFields) { const r = toRef(ref(k)); if (r) secrets[k] = r; }
    try { await api.post("/api/connections", { id, name: name || id, type, settings, secrets }); onDone(id); } catch (e) { setErr(e); }
  }
  const text = (k: string, label: string, ph = "") => <div className="row"><label className="lbl">{label}</label><input aria-label={label} placeholder={ph} value={String(s[k] ?? "")} onChange={(e) => set(k, e.target.value)} /></div>;
  return (
    <div className="addconn">
      <h3>Add a connection</h3>
      <div className="row"><label className="lbl">Type</label>
        <select aria-label="connection type" value={type} onChange={(e) => setType(e.target.value as ConnType)}>
          <option value="postgres">PostgreSQL</option><option value="s3">S3 / S3-compatible object storage</option><option value="dvc">DVC repository (Git + DVC)</option></select></div>
      <div className="row"><label className="lbl">Id</label><input aria-label="connection id" placeholder="lab_db" value={id} onChange={(e) => setId(e.target.value)} /></div>
      <div className="row"><label className="lbl">Name</label><input aria-label="connection name" value={name} onChange={(e) => setName(e.target.value)} /></div>
      {type === "postgres" && <>
        {text("host", "Host", "hostname, or a socket directory")}
        <div className="row"><label className="lbl">Port</label><input type="number" aria-label="Port" value={s.port} onChange={(e) => set("port", e.target.value)} /></div>
        {text("dbname", "Database")}{text("user", "User")}
        <div className="row"><label className="lbl">SSL mode</label><select aria-label="SSL mode" value={s.sslmode} onChange={(e) => set("sslmode", e.target.value)}>{["disable", "prefer", "require", "verify-ca", "verify-full"].map((m) => <option key={m}>{m}</option>)}</select></div></>}
      {type === "s3" && <>
        {text("bucket", "Bucket")}{text("region", "Region")}{text("endpoint_url", "Endpoint URL", "empty = AWS; set for S3-compatible stores")}
        <div className="row"><label className="lbl">Verify TLS</label><input type="checkbox" aria-label="Verify TLS" checked={!!s.verify_tls} onChange={(e) => set("verify_tls", e.target.checked)} /></div></>}
      {type === "dvc" && <>{text("repo", "Repository", "local path or Git URL")}{text("default_rev", "Default revision", "branch, tag or commit")}{text("remote", "DVC remote", "empty = repository default")}</>}
      {secretFields.map(([k, label]) => <SecretRefInput key={k} label={label} value={ref(k)} onChange={(v) => setRefs((r) => ({ ...r, [k]: v }))} />)}
      <div className="muted small">{type === "dvc" ? "DVC connections need no stored secret here; remote credentials come from the DVC remote configuration of the repository." : "Only a reference is stored (an environment variable name, or a key in a JSON secrets file outside the project). The secret value is resolved when it is used and is never saved, returned or exported."}</div>
      <ErrorBox error={err} />
      <button className="primary" disabled={!id} onClick={submit}>Add connection</button>
    </div>
  );
}

// ---------------------------------------------------------------------------------------- explorers
function PgExplorer({ conn, onAddSource }: { conn: Connection; onAddSource: AddSource }) {
  const [tables, setTables] = useState<DiscTable[] | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [sel, setSel] = useState<DiscTable | null>(null);
  const [cols, setCols] = useState<DiscColumn[]>([]);
  const [prev, setPrev] = useState<PreviewResult | null>(null);
  const [rows, setRows] = useState(20);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { setTables(null); setSel(null); setErr(null); api.get<{ tables: DiscTable[] }>(`/api/connections/${conn.id}/discover`).then((r) => setTables(r.tables)).catch(setErr); }, [conn.id]);
  useEffect(() => {
    setCols([]); setPrev(null);
    if (sel) api.get<{ columns: DiscColumn[] }>(`/api/connections/${conn.id}/discover?schema=${encodeURIComponent(sel.schema)}&table=${encodeURIComponent(sel.name)}`).then((r) => setCols(r.columns)).catch(setErr);
  }, [conn.id, sel]);
  const usable = cols.filter((c) => c.selectable);
  const query = sel ? { base: { schema: sel.schema, name: sel.name }, base_alias: "t", columns: usable.map((c) => ({ table: "t", column: c.name })), limit: 1000 } : null;
  async function preview() {
    if (!query) return;
    setErr(null);
    try { setPrev(await api.post<PreviewResult>(`/api/connections/${conn.id}/preview`, { mode: "visual", query: { ...query, limit: rows }, rows, timeoutMs: 5000 })); } catch (e) { setPrev(null); setErr(e); }
  }
  return (
    <div className="explorer">
      <h4>Tables visible to these credentials</h4>
      <ErrorBox error={err} />
      {tables && <div className="tablewrap short"><table className="dtable"><thead><tr><th>table</th><th>kind</th><th>est. rows</th><th>access</th></tr></thead><tbody>
        {tables.map((t) => (
          <tr key={`${t.schema}.${t.name}`} className={sel && sel.schema === t.schema && sel.name === t.name ? "sel" : ""} onClick={() => setSel(t)} style={{ cursor: "pointer" }}>
            <td>{t.schema}.{t.name}</td><td>{t.kind}</td><td className="num">{t.estimatedRows == null ? "?" : fmtInt(t.estimatedRows)}</td><td>{t.selectable ? "SELECT" : <span className="badge old">no SELECT permission</span>}</td></tr>))}
      </tbody></table></div>}
      <div className="muted small">Row counts are planner estimates. Only objects the credentials can see are listed.</div>
      {sel && (
        <>
          <h4>{sel.schema}.{sel.name}</h4>
          <table className="dtable"><thead><tr><th>column</th><th>type</th><th>null</th><th>key</th><th>access</th></tr></thead><tbody>
            {cols.map((c) => <tr key={c.name}><td>{c.name}</td><td>{c.type}</td><td>{c.nullable ? "yes" : "no"}</td><td>{c.primaryKey ? "primary" : ""}</td><td>{c.selectable ? "" : "denied"}</td></tr>)}</tbody></table>
          <div className="row"><label className="lbl">Preview rows</label><input type="number" aria-label="preview rows" min={1} max={200} value={rows} onChange={(e) => setRows(Math.max(1, Math.min(200, Number(e.target.value) || 1)))} />
            <button onClick={preview} disabled={!usable.length}>Preview (bounded)</button>
            <button className="primary" disabled={!usable.length} onClick={() => setMsg(onAddSource("postgres.query", { connection: conn.id, mode: "visual", query }))}>Add as a query node</button></div>
          {msg && <div className="small muted">{msg}</div>}
          {prev && <PreviewTable p={prev} />}
        </>
      )}
    </div>
  );
}

function S3Explorer({ conn, onAddSource }: { conn: Connection; onAddSource: AddSource }) {
  const [prefix, setPrefix] = useState("");
  const [token, setToken] = useState<string | null>(null);
  const [page, setPage] = useState<{ prefixes: string[]; objects: S3Object[]; nextToken: string | null; versioning: string; note: string | null } | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [sel, setSel] = useState<S3Object | null>(null);
  const [prev, setPrev] = useState<PreviewResult | null>(null);
  const [versions, setVersions] = useState<{ versionId: string; isLatest: boolean; size: number; etag: string; lastModified: string }[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { setPrefix(""); setToken(null); setSel(null); }, [conn.id]);
  useEffect(() => { setPage(null); setErr(null); api.get<any>(`/api/connections/${conn.id}/objects?prefix=${encodeURIComponent(prefix)}&maxKeys=50${token ? `&token=${encodeURIComponent(token)}` : ""}`).then(setPage).catch(setErr); }, [conn.id, prefix, token]);
  useEffect(() => {
    setPrev(null); setVersions([]);
    if (!sel) return;
    api.get<{ versions: typeof versions }>(`/api/connections/${conn.id}/object-versions?key=${encodeURIComponent(sel.key)}`).then((r) => setVersions(r.versions)).catch(() => {});
    api.post<PreviewResult>(`/api/connections/${conn.id}/preview`, { key: sel.key, rows: 20 }).then(setPrev).catch(setErr);
  }, [conn.id, sel]);
  const up = () => { const parts = prefix.split("/").filter(Boolean); parts.pop(); setPrefix(parts.length ? parts.join("/") + "/" : ""); setToken(null); };
  return (
    <div className="explorer">
      <h4>Bucket {conn.settings.bucket} <small className="muted">prefix /{prefix}</small></h4>
      <ErrorBox error={err} />
      {page && <div className="small">Versioning: <b>{page.versioning}</b>{page.note ? ` - ${page.note}` : ""}</div>}
      <div className="row">{prefix && <button onClick={up}>up</button>}
        <button onClick={() => setMsg(onAddSource("s3.object_listing", { connection: conn.id, prefix, max_objects: 1000 }))}>Add listing of this prefix as a node</button></div>
      {msg && <div className="small muted">{msg}</div>}
      {page && <div className="tablewrap short"><table className="dtable"><thead><tr><th>key</th><th>size</th><th>format</th><th>version</th></tr></thead><tbody>
        {page.prefixes.map((p) => <tr key={p} onClick={() => { setPrefix(p); setToken(null); }} style={{ cursor: "pointer" }}><td colSpan={4}><b>{p}</b></td></tr>)}
        {page.objects.map((o) => <tr key={o.key} className={sel?.key === o.key ? "sel" : ""} onClick={() => setSel(o)} style={{ cursor: "pointer" }}>
          <td>{o.key}</td><td className="num">{fmtInt(o.size ?? 0)}</td><td>{o.format}</td><td>{o.versionId ? <code>{o.versionId.slice(0, 8)}…{o.versionCount && o.versionCount > 1 ? ` (${o.versionCount} versions)` : ""}</code> : "-"}</td></tr>)}
      </tbody></table></div>}
      <div className="pager">{token && <button onClick={() => setToken(null)}>first page</button>}{page?.nextToken && <button onClick={() => setToken(page.nextToken)}>next page</button>}</div>
      {sel && (
        <>
          <h4>{sel.key}</h4>
          {versions.length > 0 && <div className="small">Versions: {versions.map((v) => <code key={v.versionId} title={v.lastModified}>{v.versionId.slice(0, 8)}{v.isLatest ? "*" : ""} </code>)}</div>}
          {versions.length === 0 && <div className="warn small">No version ids: reads of this object cannot be pinned to a version (reproducibility limited).</div>}
          {prev && <PreviewTable p={prev} />}
          <button className="primary" disabled={sel.format !== "csv" && sel.format !== "tsv"} onClick={() => setMsg(onAddSource("s3.csv_source", { connection: conn.id, key: sel.key, delimiter: sel.format === "tsv" ? "\t" : "," }))}>Add as a CSV source node</button>
        </>
      )}
    </div>
  );
}

function DvcExplorer({ conn, onAddSource }: { conn: Connection; onAddSource: AddSource }) {
  const [revs, setRevs] = useState<{ commit: string; author: string; date: string; subject: string; refs: string[] }[]>([]);
  const [rev, setRev] = useState<string>("");
  const [path, setPath] = useState("");
  const [tree, setTree] = useState<{ commit: string; entries: { path: string; type: string; size: number | null; dvcTracked: boolean; md5: string | null }[] } | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [file, setFile] = useState<string | null>(null);
  const [prev, setPrev] = useState<PreviewResult | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => { setRevs([]); setRev(""); setPath(""); setFile(null); setErr(null); api.get<{ revisions: typeof revs }>(`/api/connections/${conn.id}/revisions`).then((r) => { setRevs(r.revisions); }).catch(setErr); }, [conn.id]);
  useEffect(() => { setTree(null); api.get<any>(`/api/connections/${conn.id}/tree?path=${encodeURIComponent(path)}${rev ? `&rev=${encodeURIComponent(rev)}` : ""}`).then(setTree).catch(setErr); }, [conn.id, rev, path]);
  useEffect(() => { setPrev(null); if (file) api.post<PreviewResult>(`/api/connections/${conn.id}/preview`, { path: file, rev: rev || null, rows: 20 }).then(setPrev).catch(setErr); }, [conn.id, file, rev]);
  return (
    <div className="explorer">
      <h4>Revisions</h4>
      <ErrorBox error={err} />
      <div className="tablewrap short"><table className="dtable"><thead><tr><th>commit</th><th>refs</th><th>message</th></tr></thead><tbody>
        {revs.map((r) => <tr key={r.commit} className={rev === r.commit ? "sel" : ""} onClick={() => { setRev(r.commit); setPath(""); setFile(null); }} style={{ cursor: "pointer" }}><td><code>{r.commit.slice(0, 10)}</code></td><td>{r.refs.join(", ")}</td><td>{r.subject}</td></tr>)}</tbody></table></div>
      <div className="small">Browsing revision <b>{rev ? rev.slice(0, 10) : conn.settings.default_rev || "HEAD"}</b>{tree ? <> (commit <code>{tree.commit.slice(0, 10)}</code>)</> : null}, path /{path}</div>
      {path && <button onClick={() => setPath(path.split("/").slice(0, -1).join("/"))}>up</button>}
      {tree && <div className="tablewrap short"><table className="dtable"><thead><tr><th>path</th><th>type</th><th>size</th><th>DVC</th><th>md5</th></tr></thead><tbody>
        {tree.entries.map((e) => <tr key={e.path} onClick={() => (e.type === "directory" ? (setPath(e.path), setFile(null)) : setFile(e.path))} style={{ cursor: "pointer" }} className={file === e.path ? "sel" : ""}>
          <td>{e.path}</td><td>{e.type}</td><td className="num">{e.size == null ? "" : fmtInt(e.size)}</td><td>{e.dvcTracked ? "tracked" : "git"}</td><td><code>{e.md5?.slice(0, 10)}</code></td></tr>)}</tbody></table></div>}
      {file && (
        <>
          <h4>{file}</h4>
          {prev && <PreviewTable p={prev} />}
          {prev?.dvcMd5 && <div className="small">DVC md5 <code>{prev.dvcMd5}</code> at commit <code>{prev.commit?.slice(0, 12)}</code></div>}
          <button className="primary" onClick={() => setMsg(onAddSource("dvc.csv_source", { connection: conn.id, path: file, rev: rev || null }))}>Add as a DVC dataset node</button>
          {msg && <div className="small muted">{msg}</div>}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- workspace
export function DataWorkspace({ onAddSource, tabular }: { onAddSource: AddSource; tabular: boolean }) {
  const { connections, error, reload } = useConnections();
  const [sel, setSel] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testErr, setTestErr] = useState<string | null>(null);
  const conn = connections.find((c) => c.id === sel) ?? null;
  async function test() {
    if (!conn) return;
    setTesting(true); setTestErr(null);
    try { await api.post(`/api/connections/${conn.id}/test`, {}); reload(); } catch (e) { setTestErr(errorText(e)); } finally { setTesting(false); }
  }
  async function remove() {
    if (!conn || !window.confirm(`Remove connection '${conn.id}'? Projects keep referencing the id and will report E_SRC_CONNECTION_NOT_FOUND until it is re-added.`)) return;
    try { await api.del(`/api/connections/${conn.id}`); setSel(null); reload(); } catch (e) { setTestErr(errorText(e)); }
  }
  const lt = conn?.lastTest;
  return (
    <div className="workspace dataws">
      <aside className="wsleft">
        <h3>Connections</h3>
        <div className="muted small">Sources you connected and are authorized to browse. Secrets are references, never stored values.</div>
        {error && <div className="error">{error}</div>}
        {connections.length === 0 && <div className="muted">No connections yet.</div>}
        {connections.map((c) => (
          <button key={c.id} className={`connitem ${sel === c.id && !adding ? "on" : ""}`} onClick={() => { setSel(c.id); setAdding(false); }}>
            <b>{c.name}</b> <span className="badge">{c.type}</span>
            <small>{c.id} · {c.lastTest ? (c.lastTest.ok ? "healthy at last test" : `last test failed: ${c.lastTest.error?.code}`) : "not tested"}</small>
          </button>
        ))}
        <button className="primary" onClick={() => { setAdding(true); setSel(null); }}>Add connection</button>
      </aside>
      <main className="wsmain">
        {adding && <AddConnection onDone={(id) => { setAdding(false); setSel(id); reload(); }} />}
        {!adding && !conn && <div className="empty pad">Select a connection to test it and browse what it exposes, or add one. “All databases” means the sources you connect here and are authorized to read.</div>}
        {!adding && conn && (
          <div>
            <div className="ni-head"><div><h3>{conn.name} <span className="badge">{conn.type}</span></h3><code>{conn.id}</code></div>
              <div><button onClick={test} disabled={testing}>{testing ? "Testing…" : "Test connection"}</button> <button className="danger" onClick={remove}>Remove</button></div></div>
            <ErrorBox error={testErr ? new Error(testErr) : null} />
            <table className="kv"><tbody>
              {Object.entries(conn.settings).map(([k, v]) => <tr key={k}><td>{k}</td><td>{String(v ?? "")}</td></tr>)}
              {Object.entries(conn.secrets).map(([k, s]) => <tr key={k}><td>{k} <small>(secret)</small></td><td>{s.kind === "env" ? `env ${s.name}` : `file key ${s.key}`} · {s.resolvable ? "resolves here" : <b className="bad">does not resolve in this process</b>}{s.warning ? ` · ${s.warning}` : ""}</td></tr>)}
            </tbody></table>
            {lt && (
              <div className={lt.ok ? "okbox" : "errbadge"}>
                {lt.ok ? <><b>Connection OK</b> · {lt.latencyMs} ms{lt.serverVersion ? ` · ${lt.serverVersion}` : ""}{lt.versioning ? ` · bucket versioning: ${lt.versioning}` : ""}{lt.head ? ` · head ${lt.head.slice(0, 10)}` : ""}{lt.note ? <div className="small">{lt.note}</div> : null}</>
                  : <ErrorBox error={lt.error} />}
                {(lt as any).capabilities && <div className="small muted">Capabilities: {Object.entries((lt as any).capabilities).map(([k, v]) => `${k}: ${v === true ? "yes" : v === false ? "no" : String(v)}`).join(" · ")}</div>}
              </div>
            )}
            {!tabular && <div className="warn small">The open graph is a model graph. Open or create a tabular graph to add connected sources as nodes.</div>}
            {conn.type === "postgres" && <PgExplorer conn={conn} onAddSource={onAddSource} />}
            {conn.type === "s3" && <S3Explorer conn={conn} onAddSource={onAddSource} />}
            {conn.type === "dvc" && <DvcExplorer conn={conn} onAddSource={onAddSource} />}
          </div>
        )}
      </main>
    </div>
  );
}
