import type { AgentSpec, Catalog, FieldType, Predicate } from "./types";
import { Num, Sel } from "./common";

export interface PathInfo { path: string; type: FieldType }
const PROP_TYPE: Record<string, FieldType> = { text: "text", integer: "integer", number: "number", boolean: "boolean", list: "list", object: "object" };

/** State paths a predicate may read: each field, and each declared property of an object field. */
export function statePaths(spec: AgentSpec): PathInfo[] {
  const out: PathInfo[] = [];
  for (const f of spec.state) {
    out.push({ path: f.name, type: f.type });
    if (f.type === "object") for (const [k, t] of Object.entries(f.properties ?? {})) out.push({ path: `${f.name}.${k}`, type: PROP_TYPE[t] ?? "json" });
  }
  return out;
}

const LISTY: FieldType[] = ["list", "messages", "documents", "records"];
const NUMERIC: FieldType[] = ["integer", "number"];
function opsFor(t: FieldType, fn: string | undefined): string[] {
  if (fn === "len" || NUMERIC.includes(t)) return ["==", "!=", "<", "<=", ">", ">="];
  if (t === "boolean") return ["is_true", "is_false", "==", "!="];
  if (t === "text") return ["==", "!=", "contains", "not_contains", "startswith", "in", "is_empty", "not_empty"];
  if (LISTY.includes(t)) return ["contains", "not_contains", "is_empty", "not_empty"];
  return ["==", "!=", "is_empty", "not_empty"];
}
export const defaultCondition = (paths: PathInfo[]): Predicate => {
  const p = paths[0];
  if (!p) return { field: "", op: "==", value: "" };
  return NUMERIC.includes(p.type) ? { field: p.path, op: ">=", value: 1 } : p.type === "boolean" ? { field: p.path, op: "is_true" } : { field: p.path, op: p.type === "text" ? "==" : "not_empty", ...(p.type === "text" ? { value: "" } : {}) };
};

function Condition({ p, onChange, paths, catalog }: { p: Predicate; onChange: (p: Predicate) => void; paths: PathInfo[]; catalog: Catalog | null }) {
  const info = paths.find((x) => x.path === p.field);
  const t: FieldType = info?.type ?? "text";
  const effective: FieldType = p.fn === "len" ? "integer" : t;
  const ops = opsFor(t, p.fn);
  const op = p.op ?? "==";
  const unary = catalog?.predicate.operators.find((o) => o.op === op)?.unary ?? ["is_empty", "not_empty", "is_true", "is_false"].includes(op);
  const label = (o: string) => catalog?.predicate.operators.find((x) => x.op === o)?.label ?? o;
  const setField = (path: string) => {
    const ni = paths.find((x) => x.path === path);
    const nt = ni?.type ?? "text";
    const next: Predicate = { field: path, op: opsFor(nt, undefined).includes(op) ? op : opsFor(nt, undefined)[0] };
    if (NUMERIC.includes(nt)) next.value = typeof p.value === "number" ? p.value : 0;
    else if (nt === "text") next.value = typeof p.value === "string" ? p.value : "";
    onChange(next);
  };
  const setOp = (o: string) => {
    const n: Predicate = { ...p, op: o };
    if (["is_empty", "not_empty", "is_true", "is_false"].includes(o)) { delete n.value; delete n.other; }
    else if (n.value === undefined && n.other === undefined) n.value = NUMERIC.includes(effective) ? 0 : effective === "boolean" ? true : "";
    onChange(n);
  };
  const useOther = typeof p.other === "string";
  const others = paths.filter((x) => x.path !== p.field && (NUMERIC.includes(x.type) === NUMERIC.includes(effective)));
  return (
    <span className="cond">
      <select aria-label="state field" value={p.field ?? ""} onChange={(e) => setField(e.target.value)}>
        {!info && <option value={p.field ?? ""}>{p.field || "choose a field"}</option>}
        {paths.map((x) => <option key={x.path} value={x.path}>{x.path} ({x.type})</option>)}
      </select>
      {(LISTY.includes(t) || t === "text" || t === "object") && (
        <label className="small"><input type="checkbox" aria-label="use length" checked={p.fn === "len"} onChange={(e) => onChange({ ...p, fn: e.target.checked ? "len" : undefined, op: e.target.checked ? ">" : opsFor(t, undefined)[0], value: e.target.checked ? 0 : "" })} /> length</label>
      )}
      <select aria-label="operator" value={op} onChange={(e) => setOp(e.target.value)}>{ops.map((o) => <option key={o} value={o}>{label(o)}</option>)}</select>
      {!unary && (
        <>
          {useOther ? (
            <select aria-label="compare with field" value={p.other} onChange={(e) => onChange({ ...p, other: e.target.value })}>{others.map((x) => <option key={x.path} value={x.path}>{x.path}</option>)}</select>
          ) : NUMERIC.includes(effective) ? (
            <Num label="value" value={typeof p.value === "number" ? p.value : 0} onChange={(n) => onChange({ ...p, value: n ?? 0 })} />
          ) : effective === "boolean" ? (
            <Sel label="value" value={String(p.value === true) as "true" | "false"} options={["true", "false"]} onChange={(v) => onChange({ ...p, value: v === "true" })} />
          ) : op === "in" ? (
            <input aria-label="values" placeholder="a, b, c" value={Array.isArray(p.value) ? (p.value as string[]).join(", ") : ""} onChange={(e) => onChange({ ...p, value: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
          ) : (
            <input aria-label="value" value={String(p.value ?? "")} onChange={(e) => onChange({ ...p, value: e.target.value })} />
          )}
          {others.length > 0 && <label className="small" title="Compare with another state field instead of a constant"><input type="checkbox" aria-label="compare with another field" checked={useOther} onChange={(e) => { const n = { ...p }; if (e.target.checked) { delete n.value; n.other = others[0].path; } else { delete n.other; n.value = NUMERIC.includes(effective) ? 0 : ""; } onChange(n); }} /> field</label>}
        </>
      )}
    </span>
  );
}

/** Visual predicate builder: field / operator / value pickers combined with ALL / ANY / NOT. No code is typed anywhere. */
export function PredicateBuilder({ value, onChange, paths, catalog, root = true, onRemove }: {
  value: Predicate; onChange: (p: Predicate) => void; paths: PathInfo[]; catalog: Catalog | null; root?: boolean; onRemove?: () => void;
}) {
  const group = value.all ? "all" : value.any ? "any" : null;
  const wrap = (kids: Predicate[], kind: "all" | "any"): Predicate => (kind === "all" ? { all: kids } : { any: kids });
  const remove = onRemove && <button className="danger small" aria-label="remove condition" onClick={onRemove}>×</button>;
  if (group) {
    const kids = (value[group] ?? []) as Predicate[];
    return (
      <div className={`pgroup ${group}`}>
        <div className="phead">
          <select aria-label="group type" value={group} onChange={(e) => onChange(wrap(kids, e.target.value as "all" | "any"))}><option value="all">ALL of</option><option value="any">ANY of</option></select>
          <button className="small" onClick={() => onChange(wrap([...kids, defaultCondition(paths)], group))}>+ condition</button>
          <button className="small" onClick={() => onChange(wrap([...kids, { not: defaultCondition(paths) }], group))}>+ NOT</button>
          <button className="small" onClick={() => onChange(wrap([...kids, { all: [defaultCondition(paths)] }], group))}>+ group</button>
          {remove}
        </div>
        {kids.map((k, i) => <PredicateBuilder key={i} value={k} root={false} paths={paths} catalog={catalog} onChange={(n) => onChange(wrap(kids.map((x, j) => (j === i ? n : x)), group))}
          onRemove={kids.length > 1 || !root ? () => onChange(wrap(kids.filter((_, j) => j !== i), group)) : undefined} />)}
      </div>
    );
  }
  if (value.not) {
    return (
      <div className="pnot"><b className="small">NOT</b> <PredicateBuilder value={value.not} root={false} paths={paths} catalog={catalog} onChange={(n) => onChange({ not: n })} />{remove}</div>
    );
  }
  if (value.always) return <div className="pcond"><i className="muted">always true</i> <button className="small" onClick={() => onChange(defaultCondition(paths))}>set a condition</button>{remove}</div>;
  return (
    <div className="pcond">
      <Condition p={value} paths={paths} catalog={catalog} onChange={onChange} />
      {root && <button className="small" title="Combine several conditions" onClick={() => onChange({ all: [value, defaultCondition(paths)] })}>+ and/or</button>}
      {remove}
    </div>
  );
}

export function renderPredicate(p: Predicate): string {
  if (!p || p.always) return "always";
  if (p.all) return "(" + p.all.map(renderPredicate).join(" AND ") + ")";
  if (p.any) return "(" + p.any.map(renderPredicate).join(" OR ") + ")";
  if (p.not) return "NOT " + renderPredicate(p.not);
  const left = p.fn === "len" ? `len(${p.field})` : p.field;
  const sym: Record<string, string> = { not_contains: "does not contain", is_empty: "is empty", not_empty: "is not empty", is_true: "is true", is_false: "is false", startswith: "starts with" };
  const op = p.op ?? "==";
  if (["is_empty", "not_empty", "is_true", "is_false"].includes(op)) return `${left} ${sym[op]}`;
  return `${left} ${sym[op] ?? op} ${typeof p.other === "string" ? p.other : JSON.stringify(p.value)}`;
}
