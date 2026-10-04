import { useEffect, useState } from "react";
import { api, ApiError, errorText } from "../api";
import { useInspect } from "../hooks";
import type { Compiled, Connection, DiscColumn, DiscTable, GNode, PreviewResult, S3Object, SourceErrorInfo, SummaryResult, TabUnavailable } from "../types";
import { fmtInt, fmtNum, shortHash } from "../util";
import { NotRecorded } from "./Provenance";
import { TabProv } from "./Tabular";

// ---------------------------------------------------------------------------------------- shared
/** Source failures carry a stable code and a hint; show them as such, not as a generic error string. */
export function sourceError(e: unknown): SourceErrorInfo | null {
  if (e instanceof ApiError && e.detail && typeof e.detail.code === "string" && e.detail.code.startsWith("E_SRC_")) return e.detail as SourceErrorInfo;
  return null;
}
export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const s = sourceError(error) ?? (typeof error === "object" && error && "code" in (error as object) ? (error as SourceErrorInfo) : null);
  if (!s) return <div className="error pre">{errorText(error)}</div>;
  return (
    <div className="errbadge srcerr" role="alert">
      <b>{s.code}</b> {s.message}
      {s.hint && <div className="small">{s.hint}</div>}
      <div className="small muted">{s.recoverable ? "Recoverable: the project and connection are unchanged. " : ""}{s.resource ? `Resource: ${s.resource}. ` : ""}{s.connectionId ? `Connection: ${s.connectionId}.` : ""}</div>
    </div>
  );
}

export function useConnections() {
  const [list, setList] = useState<Connection[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => { api.get<{ connections: Connection[] }>("/api/connections").then((r) => { setList(r.connections); setError(null); }).catch((e) => setError(errorText(e))); }, [tick]);
  return { connections: list, error, reload: () => setTick((t) => t + 1) };
}

const cell = (v: unknown) => (v === null || v === undefined ? <span className="missing">∅</span> : typeof v === "number" ? (Number.isInteger(v) ? String(v) : fmtNum(v, 6)) : String(v));

export function PreviewTable({ p }: { p: PreviewResult }) {
  if (!p.columns || !p.rows) return <div className="muted small">{p.note ?? "No tabular preview for this resource."}</div>;
  return (
    <div className="tablepreview">
      <div className="small">
        {p.rows.length} row{p.rows.length === 1 ? "" : "s"} shown{p.truncated ? " (more exist; the preview is bounded)" : ""}
        {p.elapsedMs != null ? ` · ${p.elapsedMs} ms` : ""}{p.bounds ? ` · bounds ${JSON.stringify(p.bounds)}` : ""}
      </div>
      <div className="tablewrap"><table className="dtable">
        <thead><tr>{p.columns.map((c) => <th key={c.name}>{c.name}<small>{c.dtype}</small></th>)}</tr></thead>
        <tbody>{p.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className="num">{cell(v)}</td>)}</tr>)}</tbody>
      </table></div>
      {p.note && <div className="muted small">{p.note}</div>}
      <div className="prov">Provenance: {p.provenance?.source ?? "bounded preview"} · not a recorded snapshot; a run records its own identity</div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------- query builder (PostgreSQL)
interface ColRef { table: string; column: string }
interface Typed { type: string; value: unknown }
interface Filter { column: ColRef; op: string; value?: Typed | null; values?: Typed[] | null }
interface Join { table: { schema: string; name: string }; alias: string; type: "inner" | "left"; on: { left: ColRef; right: ColRef }[] }
interface Agg { fn: string; column?: ColRef | null; alias: string }
export interface Query {
  base: { schema: string; name: string }; base_alias: string; columns: { table: string; column: string; alias?: string | null }[]; filters: Filter[]; joins: Join[];
  group_by: ColRef[]; aggregates: Agg[]; order_by: { by: string; direction: "asc" | "desc" }[]; limit: number;
}
const OPS = ["=", "!=", "<", "<=", ">", ">=", "in", "not_in", "is_null", "is_not_null", "like"];
const AGGS = ["count", "count_distinct", "sum", "avg", "min", "max"];
const typeForDtype = (d: string) => (d === "int" ? "int" : d === "float" ? "float" : d === "bool" ? "bool" : d === "datetime" ? "timestamp" : "string");
const refKey = (r: ColRef) => `${r.table}.${r.column}`;
const parseRef = (s: string): ColRef => { const i = s.indexOf("."); return { table: s.slice(0, i), column: s.slice(i + 1) }; };

function coerce(type: string, text: string): unknown {
  if (type === "int" || type === "float") return text.trim() === "" || !Number.isFinite(Number(text)) ? text : Number(text);
  if (type === "bool") return text === "true" ? true : text === "false" ? false : text;
  return text;
}

function useDiscovery(connection: string) {
  const [tables, setTables] = useState<DiscTable[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [cols, setCols] = useState<Record<string, DiscColumn[]>>({});
  useEffect(() => {
    setTables([]); setCols({}); setError(null);
    if (!connection) return;
    api.get<{ tables: DiscTable[] }>(`/api/connections/${encodeURIComponent(connection)}/discover`).then((r) => setTables(r.tables)).catch(setError);
  }, [connection]);
  const need = (schema: string, name: string) => {
    const k = `${schema}.${name}`;
    if (cols[k]) return;
    setCols((c) => ({ ...c, [k]: [] }));
    api.get<{ columns: DiscColumn[] }>(`/api/connections/${encodeURIComponent(connection)}/discover?schema=${encodeURIComponent(schema)}&table=${encodeURIComponent(name)}`)
      .then((r) => setCols((c) => ({ ...c, [k]: r.columns }))).catch(setError);
  };
  return { tables, cols, need, error };
}

/** A saved query may omit empty parts (the server fills defaults); the builder always works on the full shape. */
const normalize = (q: Partial<Query> | null): Query | null => (q && q.base ? { base_alias: "t", columns: [], filters: [], joins: [], group_by: [], aggregates: [], order_by: [], limit: 1000, ...q } as Query : null);

export function QueryBuilder({ connection, query, onChange }: { connection: string; query: Query | null; onChange: (q: Query) => void }) {
  const disc = useDiscovery(connection);
  const q = normalize(query);
  const tableOf = (alias: string) => (q ? (alias === q.base_alias ? q.base : q.joins.find((j) => j.alias === alias)?.table) : undefined);
  useEffect(() => { if (q) { disc.need(q.base.schema, q.base.name); q.joins.forEach((j) => disc.need(j.table.schema, j.table.name)); } }, [q?.base.schema, q?.base.name, JSON.stringify(q?.joins.map((j) => j.table))]); // eslint-disable-line react-hooks/exhaustive-deps
  const colsOf = (alias: string): DiscColumn[] => { const t = tableOf(alias); return t ? disc.cols[`${t.schema}.${t.name}`] ?? [] : []; };
  const aliases = q ? [q.base_alias, ...q.joins.map((j) => j.alias)] : [];
  const allRefs = aliases.flatMap((a) => colsOf(a).map((c) => ({ ref: `${a}.${c.name}`, dtype: c.dtype, selectable: c.selectable })));
  const dtypeOfRef = (r: ColRef) => allRefs.find((x) => x.ref === refKey(r))?.dtype ?? "string";
  const outNames = !q ? [] : q.aggregates.length || q.group_by.length ? [...q.group_by.map((g) => g.column), ...q.aggregates.map((a) => a.alias)] : q.columns.map((c) => c.alias || c.column);

  if (!connection) return <div className="muted small">Choose a PostgreSQL connection first.</div>;
  const set = (patch: Partial<Query>) => q && onChange({ ...q, ...patch });
  const baseKey = q ? `${q.base.schema}.${q.base.name}` : "";
  const tkey = (t: DiscTable) => `${t.schema}.${t.name}`;
  const tableSelect = (value: string, onPick: (t: DiscTable) => void, label: string) => (
    <select aria-label={label} value={value} onChange={(e) => { const t = disc.tables.find((x) => tkey(x) === e.target.value); if (t) onPick(t); }}>
      <option value="">choose a table…</option>
      {disc.tables.map((t) => <option key={tkey(t)} value={tkey(t)} disabled={!t.selectable}>{tkey(t)}{t.selectable ? "" : " (no SELECT permission)"}</option>)}
    </select>
  );
  const colSelect = (value: string, onPick: (s: string) => void, label: string, only?: string[]) => (
    <select aria-label={label} value={value} onChange={(e) => onPick(e.target.value)}>
      <option value="">column…</option>
      {allRefs.filter((r) => !only || only.some((a) => r.ref.startsWith(a + "."))).map((r) => <option key={r.ref} value={r.ref} disabled={!r.selectable}>{r.ref} ({r.dtype})</option>)}
    </select>
  );

  if (!q) {
    return (
      <div className="qb">
        <ErrorBox error={disc.error} />
        <div className="row"><label className="lbl">Source table</label>
          {tableSelect("", (t) => onChange({ base: { schema: t.schema, name: t.name }, base_alias: "t", columns: [], filters: [], joins: [], group_by: [], aggregates: [], order_by: [], limit: 1000 }), "source table")}</div>
        <div className="muted small">Pick a table to start; columns, filters, joins, grouping and ordering follow. No SQL needs to be written.</div>
      </div>
    );
  }
  const grouped = q.aggregates.length > 0 || q.group_by.length > 0;
  return (
    <div className="qb">
      <ErrorBox error={disc.error} />
      <div className="row"><label className="lbl">Source table</label>{tableSelect(baseKey, (t) => onChange({ ...q, base: { schema: t.schema, name: t.name }, columns: [], filters: [], joins: [], group_by: [], aggregates: [], order_by: [] }), "source table")}
        <span className="muted small"> as <code>{q.base_alias}</code></span></div>

      <h4>Joins</h4>
      {q.joins.map((j, i) => (
        <div className="qbbox" key={i}>
          <div className="row">
            {tableSelect(`${j.table.schema}.${j.table.name}`, (t) => set({ joins: q.joins.map((x, k) => (k === i ? { ...x, table: { schema: t.schema, name: t.name }, on: [] } : x)) }), `join ${i + 1} table`)}
            <select aria-label={`join ${i + 1} type`} value={j.type} onChange={(e) => set({ joins: q.joins.map((x, k) => (k === i ? { ...x, type: e.target.value as "inner" | "left" } : x)) })}>
              <option value="inner">inner (matching rows only)</option><option value="left">left (keep all left rows)</option></select>
            <button className="danger" aria-label={`remove join ${i + 1}`} onClick={() => set({ joins: q.joins.filter((_, k) => k !== i), columns: q.columns.filter((c) => c.table !== j.alias), filters: q.filters.filter((f) => f.column.table !== j.alias) })}>×</button>
          </div>
          {j.on.map((k, ki) => (
            <div className="row" key={ki}>
              {colSelect(refKey(k.left), (s) => set({ joins: q.joins.map((x, m) => (m === i ? { ...x, on: x.on.map((o, n) => (n === ki ? { ...o, left: parseRef(s) } : o)) } : x)) }), `join ${i + 1} left key`, [q.base_alias, ...q.joins.slice(0, i).map((x) => x.alias)])}
              <span>=</span>
              {colSelect(refKey(k.right), (s) => set({ joins: q.joins.map((x, m) => (m === i ? { ...x, on: x.on.map((o, n) => (n === ki ? { ...o, right: parseRef(s) } : o)) } : x)) }), `join ${i + 1} right key`, [j.alias])}
              <button className="danger" aria-label="remove key pair" onClick={() => set({ joins: q.joins.map((x, m) => (m === i ? { ...x, on: x.on.filter((_, n) => n !== ki) } : x)) })}>×</button>
            </div>
          ))}
          <button onClick={() => set({ joins: q.joins.map((x, m) => (m === i ? { ...x, on: [...x.on, { left: { table: q.base_alias, column: "" }, right: { table: j.alias, column: "" } }] } : x)) })}>Add key pair</button>
        </div>
      ))}
      <button onClick={() => { const t = disc.tables.find((x) => x.selectable); if (t) set({ joins: [...q.joins, { table: { schema: t.schema, name: t.name }, alias: `j${q.joins.length + 1}`, type: "inner", on: [] }] }); }}
        disabled={!disc.tables.some((t) => t.selectable)}>Add join</button>

      <h4>{grouped ? "Group by" : "Columns"}</h4>
      <div className="qbcols">
        {allRefs.map((r) => {
          const ref = parseRef(r.ref);
          const on = grouped ? q.group_by.some((g) => refKey(g) === r.ref) : q.columns.some((c) => `${c.table}.${c.column}` === r.ref);
          return (
            <label key={r.ref} className={r.selectable ? "" : "muted"} title={r.selectable ? r.dtype : "no SELECT permission on this column"}>
              <input type="checkbox" checked={on} disabled={!r.selectable} onChange={(e) => {
                if (grouped) set({ group_by: e.target.checked ? [...q.group_by, ref] : q.group_by.filter((g) => refKey(g) !== r.ref), columns: [] });
                else {
                  const dup = q.columns.some((c) => c.column === ref.column);
                  set({ columns: e.target.checked ? [...q.columns, { table: ref.table, column: ref.column, alias: dup || (aliases.length > 1 && ref.table !== q.base_alias && allRefs.some((x) => x.ref !== r.ref && x.ref.endsWith("." + ref.column) && q.columns.some((c) => `${c.table}.${c.column}` === x.ref))) ? `${ref.table}_${ref.column}` : null }] : q.columns.filter((c) => `${c.table}.${c.column}` !== r.ref) });
                }
              }} /> {r.ref} <small>{r.dtype}</small>
            </label>
          );
        })}
      </div>

      <h4>Aggregates</h4>
      {q.aggregates.map((a, i) => (
        <div className="row" key={i}>
          <select aria-label={`aggregate ${i + 1} function`} value={a.fn} onChange={(e) => set({ aggregates: q.aggregates.map((x, k) => (k === i ? { ...x, fn: e.target.value } : x)) })}>{AGGS.map((f) => <option key={f}>{f}</option>)}</select>
          {colSelect(a.column ? refKey(a.column) : "", (s) => set({ aggregates: q.aggregates.map((x, k) => (k === i ? { ...x, column: s ? parseRef(s) : null } : x)) }), `aggregate ${i + 1} column`)}
          <span>as</span><input aria-label={`aggregate ${i + 1} alias`} size={10} value={a.alias} onChange={(e) => set({ aggregates: q.aggregates.map((x, k) => (k === i ? { ...x, alias: e.target.value } : x)) })} />
          <button className="danger" aria-label={`remove aggregate ${i + 1}`} onClick={() => set({ aggregates: q.aggregates.filter((_, k) => k !== i) })}>×</button>
        </div>
      ))}
      <button onClick={() => set({ aggregates: [...q.aggregates, { fn: "count", column: null, alias: `n${q.aggregates.length + 1}` }], columns: [] })}>Add aggregate</button>
      {grouped && <div className="muted small">With grouping, the output is the group-by columns plus the aggregates (filters apply before grouping).</div>}

      <h4>Filters (typed comparisons)</h4>
      {q.filters.map((f, i) => {
        const dt = typeForDtype(dtypeOfRef(f.column));
        const upd = (patch: Partial<Filter>) => set({ filters: q.filters.map((x, k) => (k === i ? { ...x, ...patch } : x)) });
        const nullary = f.op === "is_null" || f.op === "is_not_null";
        const multi = f.op === "in" || f.op === "not_in";
        return (
          <div className="row" key={i}>
            {colSelect(refKey(f.column), (s) => upd({ column: parseRef(s), value: f.value ? { type: typeForDtype(allRefs.find((x) => x.ref === s)?.dtype ?? "string"), value: "" } : null }), `filter ${i + 1} column`)}
            <select aria-label={`filter ${i + 1} operator`} value={f.op} onChange={(e) => upd({ op: e.target.value, value: ["is_null", "is_not_null", "in", "not_in"].includes(e.target.value) ? null : f.value ?? { type: dt, value: "" }, values: ["in", "not_in"].includes(e.target.value) ? f.values ?? [] : null })}>{OPS.map((o) => <option key={o}>{o}</option>)}</select>
            {!nullary && !multi && <input aria-label={`filter ${i + 1} value`} size={12} value={String(f.value?.value ?? "")} placeholder={dt} onChange={(e) => upd({ value: { type: dt, value: coerce(dt, e.target.value) } })} />}
            {multi && <input aria-label={`filter ${i + 1} values`} size={16} placeholder={`${dt}, comma separated`} value={(f.values ?? []).map((v) => String(v.value)).join(", ")}
              onChange={(e) => upd({ values: e.target.value.split(",").map((s) => s.trim()).filter(Boolean).map((s) => ({ type: dt, value: coerce(dt, s) })) })} />}
            <small className="muted">{dt}</small>
            <button className="danger" aria-label={`remove filter ${i + 1}`} onClick={() => set({ filters: q.filters.filter((_, k) => k !== i) })}>×</button>
          </div>
        );
      })}
      <button disabled={!allRefs.length} onClick={() => { const r = allRefs.find((x) => x.selectable)!; set({ filters: [...q.filters, { column: parseRef(r.ref), op: "=", value: { type: typeForDtype(r.dtype), value: "" } }] }); }}>Add filter</button>

      <h4>Order and limit</h4>
      {q.order_by.map((o, i) => (
        <div className="row" key={i}>
          <select aria-label={`order ${i + 1} column`} value={o.by} onChange={(e) => set({ order_by: q.order_by.map((x, k) => (k === i ? { ...x, by: e.target.value } : x)) })}>{outNames.map((n) => <option key={n}>{n}</option>)}</select>
          <select aria-label={`order ${i + 1} direction`} value={o.direction} onChange={(e) => set({ order_by: q.order_by.map((x, k) => (k === i ? { ...x, direction: e.target.value as "asc" | "desc" } : x)) })}><option>asc</option><option>desc</option></select>
          <button className="danger" aria-label={`remove order ${i + 1}`} onClick={() => set({ order_by: q.order_by.filter((_, k) => k !== i) })}>×</button>
        </div>
      ))}
      <button disabled={!outNames.length} onClick={() => set({ order_by: [...q.order_by, { by: outNames[0], direction: "asc" }] })}>Add ordering</button>
      <div className="row"><label className="lbl">Row limit</label><input type="number" aria-label="row limit" min={1} max={1000000} value={q.limit} onChange={(e) => set({ limit: Math.max(1, Math.min(1_000_000, Number(e.target.value) || 1)) })} /></div>
    </div>
  );
}

/** Read-only compiled SQL (exactly the executed text) with its bound parameters. */
export function CompiledSql({ query: raw }: { query: Query | null }) {
  const query = normalize(raw);
  const [c, setC] = useState<Compiled | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const key = JSON.stringify(query);
  useEffect(() => {
    setErr(null);
    if (!query || (!query.columns.length && !query.aggregates.length && !query.group_by.length) || query.joins.some((j) => j.on.some((k) => !k.left.column || !k.right.column) || !j.on.length)) { setC(null); return; }
    const t = setTimeout(() => api.post<Compiled>("/api/queries/compile", { query }).then((r) => { setC(r); setErr(null); }).catch((e) => { setC(null); setErr(e); }), 250);
    return () => clearTimeout(t);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!query) return null;
  return (
    <div className="compiled">
      <h4>Compiled SQL <small className="muted">(read-only)</small></h4>
      {err ? <ErrorBox error={err} /> : !c ? <div className="muted small">Select columns (and complete every join key) to compile.</div> : (
        <>
          <pre aria-label="compiled sql">{c.sql}</pre>
          <table className="kv"><tbody>{c.params.map((p, i) => <tr key={i}><td>%s #{i + 1}</td><td><code>{JSON.stringify(p)}</code> <small className="muted">{c.paramTypes[i]}</small></td></tr>)}</tbody></table>
          <div className="muted small">{c.placeholderStyle}.</div>
          {c.warnings.map((w, i) => <div key={i} className="warn">{w}</div>)}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- node config editors
const TYPE_OF: Record<string, "postgres" | "s3" | "dvc"> = { "postgres.query": "postgres", "s3.csv_source": "s3", "s3.object_listing": "s3", "dvc.csv_source": "dvc" };
export const isConnectorOp = (t: string) => t in TYPE_OF;

function ConnectionSelect({ type, value, onChange }: { type: string; value: string; onChange: (v: string) => void }) {
  const { connections } = useConnections();
  const opts = connections.filter((c) => c.type === type);
  return (
    <div className="row"><label className="lbl">Connection</label>
      <select aria-label="connection" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">choose a {type} connection…</option>
        {opts.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.id}){c.lastTest && !c.lastTest.ok ? " - last test failed" : ""}</option>)}
        {value && !opts.some((c) => c.id === value) && <option value={value}>{value} (not defined here)</option>}
      </select>
      {opts.length === 0 && <span className="muted small"> add one in the Data workspace</span>}
    </div>
  );
}

function PinRow({ cfg, onChange }: { cfg: Record<string, any>; onChange: (p: Record<string, unknown>) => void }) {
  return (
    <div className="row"><label className="lbl">Pinned snapshot</label>
      {cfg.pin ? <><code title={cfg.pin}>{shortHash(cfg.pin)}</code> <button onClick={() => onChange({ pin: null })}>Unpin (read the live source)</button></>
        : <span className="muted small">none: each run reads the live source and records a new snapshot. After a run, pin one from the Source tab.</span>}
    </div>
  );
}

function SqlEditor({ cfg, onChange }: { cfg: Record<string, any>; onChange: (p: Record<string, unknown>) => void }) {
  const params: { name: string; type: string; value: unknown }[] = cfg.params ?? [];
  const setP = (i: number, patch: object) => onChange({ params: params.map((p, k) => (k === i ? { ...p, ...patch } : p)) });
  return (
    <div>
      <textarea aria-label="sql" rows={7} className="sqlarea" value={cfg.sql ?? ""} spellCheck={false} placeholder={"SELECT specimen_id, ph FROM specimens WHERE site = %(site)s ORDER BY specimen_id"}
        onChange={(e) => onChange({ sql: e.target.value })} />
      <div className="muted small">One read-only SELECT. It runs inside <code>SELECT * FROM (…) LIMIT n</code> in a READ ONLY transaction with a statement timeout; write values as <code>%(name)s</code> parameters, never inline.</div>
      <h4>Parameters</h4>
      {params.map((p, i) => (
        <div className="row" key={i}>
          <input aria-label={`parameter ${i + 1} name`} size={8} value={p.name} onChange={(e) => setP(i, { name: e.target.value })} />
          <select aria-label={`parameter ${i + 1} type`} value={p.type} onChange={(e) => setP(i, { type: e.target.value })}>{["int", "float", "string", "bool", "date", "timestamp"].map((t) => <option key={t}>{t}</option>)}</select>
          <input aria-label={`parameter ${i + 1} value`} size={10} value={String(p.value ?? "")} onChange={(e) => setP(i, { value: coerce(p.type, e.target.value) })} />
          <button className="danger" aria-label={`remove parameter ${i + 1}`} onClick={() => onChange({ params: params.filter((_, k) => k !== i) })}>×</button>
        </div>
      ))}
      <button onClick={() => onChange({ params: [...params, { name: `p${params.length + 1}`, type: "string", value: "" }] })}>Add parameter</button>
      <div className="row"><label className="lbl">Row limit</label><input type="number" aria-label="sql row limit" min={1} max={1000000} value={cfg.limit ?? 10000} onChange={(e) => onChange({ limit: Math.max(1, Math.min(1_000_000, Number(e.target.value) || 1)) })} /></div>
      <div className="row"><label className="lbl">Timeout (ms)</label><input type="number" aria-label="timeout ms" min={100} max={30000} value={cfg.timeout_ms ?? 10000} onChange={(e) => onChange({ timeout_ms: Math.max(100, Math.min(30000, Number(e.target.value) || 100)) })} /></div>
    </div>
  );
}

function PreviewBar({ connection, body, label }: { connection: string; body: Record<string, unknown> | null; label?: string }) {
  const [rows, setRows] = useState(20);
  const [res, setRes] = useState<PreviewResult | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  async function go() {
    if (!body) return;
    setBusy(true); setErr(null);
    try { setRes(await api.post<PreviewResult>(`/api/connections/${encodeURIComponent(connection)}/preview`, { ...body, rows, timeoutMs: 5000 })); }
    catch (e) { setRes(null); setErr(e); } finally { setBusy(false); }
  }
  return (
    <div className="previewbar">
      <h4>Preview <small className="muted">(bounded: row limit and 5 s timeout, enforced at the source)</small></h4>
      <div className="row"><label className="lbl">Rows</label><input type="number" aria-label="preview rows" min={1} max={200} value={rows} onChange={(e) => setRows(Math.max(1, Math.min(200, Number(e.target.value) || 1)))} />
        <button disabled={!connection || !body || busy} onClick={go}>{busy ? "Running…" : label ?? "Preview"}</button></div>
      <ErrorBox error={err} />
      {res && <PreviewTable p={res} />}
    </div>
  );
}

export function ConnectorConfigEditor({ node, onChange }: { node: GNode; onChange: (patch: Record<string, unknown>) => void }) {
  const cfg = node.config as Record<string, any>;
  const type = TYPE_OF[node.type];
  const conn: string = cfg.connection ?? "";
  const [objs, setObjs] = useState<{ prefixes: string[]; objects: S3Object[] } | null>(null);
  const [objErr, setObjErr] = useState<unknown>(null);
  const [prefix, setPrefix] = useState<string>(node.type === "s3.object_listing" ? cfg.prefix ?? "" : (cfg.key ?? "").split("/").slice(0, -1).join("/") + ((cfg.key ?? "").includes("/") ? "/" : ""));
  useEffect(() => {
    setObjs(null); setObjErr(null);
    if (type !== "s3" || !conn) return;
    api.get<{ prefixes: string[]; objects: S3Object[] }>(`/api/connections/${encodeURIComponent(conn)}/objects?prefix=${encodeURIComponent(prefix)}&maxKeys=50`).then(setObjs).catch(setObjErr);
  }, [type, conn, prefix]);
  const [versions, setVersions] = useState<{ versionId: string; isLatest: boolean; size: number; lastModified: string }[]>([]);
  useEffect(() => {
    setVersions([]);
    if (node.type !== "s3.csv_source" || !conn || !cfg.key) return;
    api.get<{ versions: typeof versions }>(`/api/connections/${encodeURIComponent(conn)}/object-versions?key=${encodeURIComponent(cfg.key)}`).then((r) => setVersions(r.versions)).catch(() => setVersions([]));
  }, [node.type, conn, cfg.key]);
  const [revs, setRevs] = useState<{ commit: string; subject: string; refs: string[] }[]>([]);
  useEffect(() => {
    setRevs([]);
    if (type !== "dvc" || !conn) return;
    api.get<{ revisions: typeof revs }>(`/api/connections/${encodeURIComponent(conn)}/revisions`).then((r) => setRevs(r.revisions)).catch(() => setRevs([]));
  }, [type, conn]);

  return (
    <div className="connector-config">
      <ConnectionSelect type={type} value={conn} onChange={(v) => onChange({ connection: v })} />
      {node.type === "postgres.query" && (
        <>
          <div className="row"><label className="lbl">Query mode</label>
            <label><input type="radio" checked={cfg.mode !== "sql"} onChange={() => onChange({ mode: "visual" })} /> visual builder</label>
            <label><input type="radio" checked={cfg.mode === "sql"} onChange={() => onChange({ mode: "sql" })} /> raw SQL (read-only)</label></div>
          {cfg.mode === "sql" ? <SqlEditor cfg={cfg} onChange={onChange} /> : <><QueryBuilder connection={conn} query={cfg.query ?? null} onChange={(q) => onChange({ query: q })} /><CompiledSql query={cfg.query ?? null} /></>}
          <PinRow cfg={cfg} onChange={onChange} />
          <PreviewBar connection={conn} body={cfg.mode === "sql" ? (cfg.sql ? { mode: "sql", sql: cfg.sql, params: cfg.params ?? [] } : null) : (cfg.query ? { mode: "visual", query: cfg.query } : null)} />
        </>
      )}
      {node.type === "s3.csv_source" && (
        <>
          <ErrorBox error={objErr} />
          <div className="row"><label className="lbl">Object key</label><input aria-label="object key" value={cfg.key ?? ""} onChange={(e) => onChange({ key: e.target.value })} /></div>
          {objs && (
            <div className="objlist small">
              <div><b>s3://…/{prefix}</b>{prefix && <button onClick={() => setPrefix(prefix.split("/").slice(0, -2).join("/") + (prefix.split("/").length > 2 ? "/" : ""))}>up</button>}</div>
              {objs.prefixes.map((p) => <div key={p}><button onClick={() => setPrefix(p)}>{p}</button></div>)}
              {objs.objects.map((o) => <div key={o.key}><button className={o.key === cfg.key ? "on" : ""} disabled={o.format !== "csv" && o.format !== "tsv"} onClick={() => onChange({ key: o.key, version_id: null, delimiter: o.format === "tsv" ? "\t" : "," })}>{o.key}</button> <span className="muted">{fmtInt(o.size ?? 0)} B · {o.format}</span></div>)}
            </div>
          )}
          <div className="row"><label className="lbl">Object version</label>
            <select aria-label="object version" value={cfg.version_id ?? ""} onChange={(e) => onChange({ version_id: e.target.value || null })}>
              <option value="">latest at run time (recorded in the snapshot)</option>
              {versions.map((v) => <option key={v.versionId} value={v.versionId}>{v.versionId.slice(0, 12)}… {v.isLatest ? "(latest) " : ""}{v.lastModified}</option>)}
            </select></div>
          {conn && cfg.key && versions.length === 0 && <div className="warn small">No object versions reported: if the bucket is not versioned, runs materialize a copy and flag limited reproducibility.</div>}
          <PinRow cfg={cfg} onChange={onChange} />
          <PreviewBar connection={conn} body={cfg.key ? { key: cfg.key, versionId: cfg.version_id } : null} />
        </>
      )}
      {node.type === "s3.object_listing" && (
        <>
          <div className="row"><label className="lbl">Prefix</label><input aria-label="prefix" value={cfg.prefix ?? ""} onChange={(e) => { onChange({ prefix: e.target.value }); setPrefix(e.target.value); }} /></div>
          <div className="row"><label className="lbl">Max objects</label><input type="number" aria-label="max objects" min={1} max={10000} value={cfg.max_objects ?? 1000} onChange={(e) => onChange({ max_objects: Math.max(1, Math.min(10000, Number(e.target.value) || 1)) })} /></div>
          <div className="row"><label className="lbl">Key → id regex</label><input aria-label="key regex" value={cfg.key_regex ?? ""} placeholder="images/(?P<id>SP\d+)\.png" onChange={(e) => onChange({ key_regex: e.target.value })} /></div>
          <div className="muted small">Optional. The named group <code>(?P&lt;id&gt;…)</code> becomes column <code>key_id</code>, usable as a join key. Object bodies are never read.</div>
          <ErrorBox error={objErr} />
          {objs && <div className="small">{objs.objects.length} object(s) and {objs.prefixes.length} sub-prefix(es) in this page of the prefix.</div>}
          <PinRow cfg={cfg} onChange={onChange} />
        </>
      )}
      {node.type === "dvc.csv_source" && (
        <>
          <div className="row"><label className="lbl">Dataset path</label><input aria-label="dataset path" value={cfg.path ?? ""} onChange={(e) => onChange({ path: e.target.value })} /></div>
          <div className="row"><label className="lbl">Revision</label>
            <input aria-label="revision" list="dvc-revs" value={cfg.rev ?? ""} placeholder="branch, tag or commit (default: the connection's)" onChange={(e) => onChange({ rev: e.target.value || null })} />
            <datalist id="dvc-revs">{revs.flatMap((r) => [...r.refs, r.commit]).map((x) => <option key={x} value={x} />)}</datalist></div>
          <div className="muted small">A branch or tag is resolved to a commit when the run starts; the commit and the DVC md5 are recorded in the snapshot.</div>
          <PinRow cfg={cfg} onChange={onChange} />
          <PreviewBar connection={conn} body={cfg.path ? { path: cfg.path, rev: cfg.rev } : null} />
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- result views
type SumRes = SummaryResult<any> | TabUnavailable;

function useSummary(runId: string | null, node: string) {
  return useInspect<SumRes>(runId ? `/api/runs/${runId}/inspect` : null, runId ? { kind: "summary", node } : null);
}

function Repro({ r }: { r: { level: string; limited: boolean; statement: string } }) {
  return <div className={r.limited ? "warn" : "okbox"}><b>{r.limited ? "Reproducibility limited" : "Reproducible"}</b> <span className="muted small">({r.level})</span><div className="small">{r.statement}</div></div>;
}

export function ConnectorSourceView({ runId, node, onPin, pinned }: { runId: string | null; node: string; onPin?: (id: string) => void; pinned?: string | null }) {
  const st = useSummary(runId, node);
  const [check, setCheck] = useState<any>(null);
  const [checkErr, setCheckErr] = useState<unknown>(null);
  if (!runId) return <NotRecorded message="No run selected. Snapshot identities are recorded by runs; nothing is guessed." />;
  if (st.error) return <div className="error pre">{st.error}</div>;
  if (!st.data) return <div className="muted">loading…</div>;
  if (!st.data.available) return <><NotRecorded message={st.data.message} /><TabProv p={st.data.provenance} /></>;
  const d = st.data.data;
  const m = d.snapshot;
  const sid: string = d.snapshotId;
  async function doCheck() {
    setCheck(null); setCheckErr(null);
    try { setCheck(await api.post<any>(`/api/snapshots/${sid}/check`, {})); } catch (e) { setCheckErr(e); }
  }
  return (
    <div className="srcview">
      <div><span className={`badge ${d.mode === "pinned" ? "part-train" : ""}`}>{d.mode === "pinned" ? "read from pinned snapshot" : "read from the live source"}</span>{" "}
        <b>{d.connector}</b> · {fmtInt(d.rows)} rows · {fmtInt(d.bytes ?? 0)} bytes</div>
      <Repro r={d.reproducibility} />
      <h4>Snapshot identity</h4>
      <table className="kv"><tbody>
        <tr><td>snapshot id</td><td><code title={sid}>{sid.slice(0, 16)}…</code></td></tr>
        <tr><td>taken at</td><td>{m.takenAt}</td></tr>
        {m.kind === "postgres" && <>
          <tr><td>server</td><td>{m.server.version}</td></tr>
          <tr><td>transaction snapshot</td><td><code>{m.snapshotToken}</code> <small className="muted">{m.transaction}</small></td></tr>
          <tr><td>result SHA-256</td><td><code>{m.result.sha256}</code></td></tr>
          <tr><td>ordered</td><td>{m.result.ordered ? "yes" : "no ORDER BY: row order not guaranteed"}{m.result.reachedLimit ? " · row limit reached" : ""}</td></tr></>}
        {m.kind === "s3_object" && <>
          <tr><td>object</td><td>s3://{m.object.bucket}/{m.object.key}</td></tr>
          <tr><td>version id</td><td>{m.object.versionId ? <code>{m.object.versionId}</code> : <b>none (bucket not versioned)</b>}</td></tr>
          <tr><td>ETag</td><td><code>{m.object.etag}</code></td></tr><tr><td>content SHA-256</td><td><code>{m.object.sha256}</code></td></tr></>}
        {m.kind === "s3_listing" && <><tr><td>prefix</td><td>s3://{m.listing.bucket}/{m.listing.prefix}{m.listing.truncated ? " (truncated at max_objects)" : ""}</td></tr><tr><td>listing SHA-256</td><td><code>{m.result.sha256}</code></td></tr></>}
        {m.kind === "dvc_file" && <>
          <tr><td>repository</td><td>{m.file.repo}</td></tr><tr><td>requested → resolved</td><td>{m.file.requestedRev} → <code>{m.file.commit}</code></td></tr>
          <tr><td>DVC md5</td><td><code>{m.file.dvcMd5}</code></td></tr><tr><td>content SHA-256</td><td><code>{m.file.sha256}</code></td></tr></>}
      </tbody></table>
      {d.sql && <><h4>Executed SQL</h4><pre aria-label="executed sql">{d.sql}</pre>
        {d.params?.length > 0 && <table className="kv"><tbody>{d.params.map((p: any, i: number) => <tr key={i}><td>{p.name ?? `%s #${p.position}`}</td><td><code>{JSON.stringify(p.value)}</code> <small className="muted">{p.type}</small></td></tr>)}</tbody></table>}</>}
      {(d.joinReports ?? []).length > 0 && (
        <><h4>Join report (in the database)</h4>
          {d.joinReports.map((j: any) => j.available ? (
            <div key={j.alias} className="joinrep small"><b>{j.left} {j.joinType} join {j.right}</b> on {j.leftKeys.join(", ")} = {j.rightKeys.join(", ")}: <b>{j.cardinality.replace(/_/g, "-")}</b>
              {j.rowMultiplicationPossible && <span className="badge old">rows can multiply</span>}
              <div>left {fmtInt(j.leftRows)} rows ({j.leftDistinctKeys} distinct keys, {j.leftNullKeys} NULL, {j.leftDuplicateKeyRows} duplicates) · right {fmtInt(j.rightRows)} rows ({j.rightDistinctKeys} distinct, {j.rightNullKeys} NULL, {j.rightDuplicateKeyRows} duplicates)</div>
              <div>unmatched: {j.unmatchedLeftRows} left rows have no right row · {j.unmatchedRightRows} right rows have no left row <span className="muted">({j.executedIn})</span></div></div>
          ) : <div key={j.alias} className="muted small">join {j.alias}: {j.reason}</div>)}</>
      )}
      <h4>Transfer</h4>
      <div className="small">{d.transfer.from} → {d.transfer.to} · {fmtInt(d.transfer.rows)} rows · {fmtInt(d.transfer.bytes ?? 0)} bytes <span className="muted">(cost estimates are not available for these connectors)</span></div>
      {(d.warnings ?? []).map((w: string, i: number) => <div key={i} className="warn small">{w}</div>)}
      <div className="actions">
        {onPin && <button disabled={pinned === sid} onClick={() => onPin(sid)} title="Set this node's `pin`: later runs read this recorded snapshot instead of the live source">{pinned === sid ? "Pinned" : "Pin this snapshot to the node"}</button>}
        <button onClick={doCheck}>Compare with the source now</button>
      </div>
      <ErrorBox error={checkErr} />
      {check && <div className={check.identical ? "okbox small" : "warn small"}><b>{check.identical ? "The source still matches this snapshot." : "The source differs from this snapshot."}</b> {check.note}
        <pre className="small">{JSON.stringify(check, null, 1)}</pre></div>}
      <TabProv p={st.data.provenance} label="snapshot manifest stored as a run artifact" />
    </div>
  );
}

export function JoinView({ runId, node }: { runId: string | null; node: string }) {
  const st = useSummary(runId, node);
  if (!runId) return <NotRecorded message="No run selected. The join report is computed by a run." />;
  if (st.error) return <div className="error pre">{st.error}</div>;
  if (!st.data) return <div className="muted">loading…</div>;
  if (!st.data.available) return <><NotRecorded message={st.data.message} /><TabProv p={st.data.provenance} /></>;
  const d = st.data.data;
  const side = (name: string, s: any) => (
    <tr><td>{name}</td><td className="num">{fmtInt(s.rows)}</td><td className="num">{fmtInt(s.distinctKeys)}</td><td className="num">{fmtInt(s.duplicateKeyRows)}</td><td className="num">{fmtInt(s.nullKeyRows)}</td><td className="num"><b>{fmtInt(s.unmatchedRows)}</b></td></tr>
  );
  return (
    <div className="joinview">
      <div><b>{d.how} join</b> on {d.leftKeys.join(", ")} = {d.rightKeys.join(", ")} · key cardinality <b>{String(d.cardinality).replace(/_/g, "-")}</b>{d.expected !== "any" && <> (expected {String(d.expected).replace(/_/g, "-")})</>}</div>
      <table className="dtable"><thead><tr><th>side</th><th>rows</th><th>distinct keys</th><th>duplicate-key rows</th><th>NULL keys</th><th>unmatched rows</th></tr></thead><tbody>{side("left", d.left)}{side("right", d.right)}</tbody></table>
      <div className="small">Result: <b>{fmtInt(d.resultRows)}</b> rows ({d.rowMultiplication == null ? "n/a" : `${fmtNum(d.rowMultiplication, 4)}× the left rows`}{d.rowsMultiplied ? " - rows were multiplied by repeated keys" : ""}).</div>
      {d.left.unmatchedSample.length > 0 && <div className="small"><b>Unmatched left keys (sample):</b> {d.left.unmatchedSample.map((u: any) => `${u.key.join("/")} (row ${u.rowId})`).join(", ")}</div>}
      {d.right.unmatchedSample.length > 0 && <div className="small"><b>Unmatched right keys (sample):</b> {d.right.unmatchedSample.map((u: any) => `${u.key.join("/")} (row ${u.rowId})`).join(", ")}</div>}
      <h4>Where it ran</h4>
      <div className="small">{d.execution.where}. {d.execution.note}</div>
      <h4>Lineage</h4>
      {(["left", "right"] as const).map((k) => (d.lineage[k] as any[]).map((s, i) => <div className="small" key={k + i}><b>{k}</b>: {s.kind} · {s.path} · snapshot <code>{shortHash(s.snapshotId)}</code> ({s.mode})</div>))}
      <div className="muted small">{d.semantics}</div>
      <TabProv p={st.data.provenance} />
    </div>
  );
}
