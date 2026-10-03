import { useEffect, useState } from "react";
import type { GNode, JSchema, OpInfo } from "../types";

type Patch = Record<string, unknown>;

function NumInput({ value, onChange, integer, min, ariaLabel }: { value: number; onChange: (n: number) => void; integer: boolean; min?: number; ariaLabel: string }) {
  const [text, setText] = useState(String(value));
  useEffect(() => { setText((t) => (Number(t) === value ? t : String(value))); }, [value]);
  return (
    <input type="number" aria-label={ariaLabel} value={text} step={integer ? 1 : "any"} min={min}
      onChange={(e) => {
        setText(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value !== "" && Number.isFinite(n) && (!integer || Number.isInteger(n))) onChange(n);
      }} />
  );
}

function parseList(text: string): (number | string)[] {
  return text.split(",").map((s) => s.trim()).filter(Boolean).map((s) => (Number.isFinite(Number(s)) ? Number(s) : s));
}

function ListInput({ value, onChange, label }: { value: unknown[]; onChange: (v: unknown[]) => void; label: string }) {
  const [text, setText] = useState(value.join(", "));
  useEffect(() => { setText((t) => (JSON.stringify(parseList(t)) === JSON.stringify(value) ? t : value.join(", "))); }, [value]);
  return <input aria-label={label} value={text} placeholder="comma separated" onChange={(e) => { setText(e.target.value); onChange(parseList(e.target.value)); }} />;
}

function StringListInput({ value, onChange, label, multiline }: { value: string[]; onChange: (v: string[]) => void; label: string; multiline?: boolean }) {
  const join = (v: string[]) => v.join(multiline ? "\n" : ", ");
  const split = (t: string) => t.split(multiline ? "\n" : ",").map((x) => x.trim()).filter(Boolean);
  const [text, setText] = useState(join(value));
  useEffect(() => { setText((t) => (JSON.stringify(split(t)) === JSON.stringify(value) ? t : join(value))); }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  return multiline
    ? <textarea aria-label={label} rows={3} value={text} placeholder="one per line" onChange={(e) => { setText(e.target.value); onChange(split(e.target.value)); }} />
    : <input aria-label={label} value={text} placeholder="comma separated (empty = default)" onChange={(e) => { setText(e.target.value); onChange(split(e.target.value)); }} />;
}

interface ColSpec { name: string; dtype: string }
/** Typed column selection: one row per selected column with its declared type. */
function ColumnSpecEditor({ value, onChange, dtypes }: { value: ColSpec[]; onChange: (v: ColSpec[]) => void; dtypes: string[] }) {
  return (
    <div className="colspec">
      {value.map((c, i) => (
        <div key={i} className="colrow">
          <input aria-label={`column ${i + 1} name`} value={c.name} onChange={(e) => onChange(value.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))} />
          <select aria-label={`column ${i + 1} type`} value={c.dtype} onChange={(e) => onChange(value.map((x, j) => (j === i ? { ...x, dtype: e.target.value } : x)))}>{dtypes.map((d) => <option key={d}>{d}</option>)}</select>
          <button className="danger" aria-label={`remove column ${i + 1}`} onClick={() => onChange(value.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <button onClick={() => onChange([...value, { name: "", dtype: dtypes[0] }])}>Add column</button>
    </div>
  );
}

const isInferInt = (s: JSchema) => !!s.anyOf && s.anyOf.some((a) => a.const === "infer") && s.anyOf.some((a) => a.type === "integer");
const literals = (a: JSchema): string[] => (a.const !== undefined ? [String(a.const)] : a.enum ? a.enum.map(String) : []);

function Field({ name, schema, value, resolved, onChange, defs }: { name: string; schema: JSchema; value: unknown; resolved: unknown; onChange: (v: unknown) => void; defs?: Record<string, JSchema> }) {
  if (schema.type === "array" && schema.items?.$ref) {
    const def = defs?.[schema.items.$ref.split("/").pop()!];
    const dt = def?.properties?.dtype?.enum?.map(String);
    if (def?.properties?.name && dt) return <ColumnSpecEditor value={(Array.isArray(value) ? value : []) as ColSpec[]} dtypes={dt} onChange={onChange} />;
  }
  if (schema.type === "array" && schema.items?.type === "string") return <StringListInput label={name} multiline={name === "assumptions"} value={(Array.isArray(value) ? value : []) as string[]} onChange={onChange} />;
  if (isInferInt(schema)) {
    const locked = value !== "infer";
    return (
      <div className="infer">
        <label><input type="radio" checked={!locked} onChange={() => onChange("infer")} /> infer</label>
        <label><input type="radio" checked={locked} onChange={() => onChange(typeof resolved === "number" ? resolved : 1)} /> locked</label>
        {locked ? <NumInput ariaLabel={name} value={Number(value)} integer min={1} onChange={onChange} />
          : <span className="muted">{typeof resolved === "number" ? `= ${resolved} (from the connected tensor)` : "no connected input yet"}</span>}
      </div>
    );
  }
  if (schema.anyOf) {
    const alts = schema.anyOf;
    const lits = alts.flatMap(literals);
    const hasNull = alts.some((a) => a.type === "null");
    const typed = alts.find((a) => a.type !== "null" && literals(a).length === 0);
    const sel = value === null || value === undefined ? "__none" : typeof value === "string" && lits.includes(value) ? value : "__value";
    return (
      <div className="union">
        <select aria-label={name} value={sel}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "__none") onChange(null);
            else if (v === "__value") onChange(typed?.default ?? (typed?.prefixItems ? [1, 1] : 1));
            else onChange(v);
          }}>
          {typed && <option value="__value">custom value</option>}
          {lits.map((l) => <option key={l} value={l}>{l}</option>)}
          {hasNull && <option value="__none">none (default behaviour)</option>}
        </select>
        {sel === "__value" && typed && <Field name={name} schema={typed} value={value} resolved={resolved} onChange={onChange} />}
      </div>
    );
  }
  if (schema.enum) {
    return <select aria-label={name} value={String(value)} onChange={(e) => onChange(e.target.value)}>{schema.enum.map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}</select>;
  }
  switch (schema.type) {
    case "boolean": return <input type="checkbox" aria-label={name} checked={!!value} onChange={(e) => onChange(e.target.checked)} />;
    case "integer": case "number": {
      const min = schema.minimum ?? (schema.exclusiveMinimum !== undefined ? schema.exclusiveMinimum + (schema.type === "integer" ? 1 : 0) : undefined);
      return <NumInput ariaLabel={name} value={Number(value)} integer={schema.type === "integer"} min={min} onChange={onChange} />;
    }
    case "string": return <input aria-label={name} value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />;
    case "array": {
      const arr = Array.isArray(value) ? value : [value, value];
      if (schema.prefixItems && schema.prefixItems.length === 2) {
        const [a, b] = schema.prefixItems;
        const mn = (s: JSchema) => s.minimum ?? (s.exclusiveMinimum !== undefined ? s.exclusiveMinimum + 1 : undefined);
        return (
          <span className="pair">
            <NumInput ariaLabel={`${name} height`} value={Number(arr[0])} integer min={mn(a)} onChange={(n) => onChange([n, arr[1]])} />
            {"×"}
            <NumInput ariaLabel={`${name} width`} value={Number(arr[1])} integer min={mn(b)} onChange={(n) => onChange([arr[0], n])} />
          </span>
        );
      }
      return <ListInput label={name} value={arr as unknown[]} onChange={onChange} />;
    }
    default: return <code>{JSON.stringify(value)}</code>;
  }
}

/** Config form generated from the registry's JSON Schema; no per-op UI code. */
export function ConfigForm({ op, node, resolved, onChange }: { op: OpInfo; node: GNode; resolved?: Record<string, unknown>; onChange: (patch: Patch) => void }) {
  const props = op.configSchema.properties ?? {};
  const keys = Object.keys(props);
  if (!keys.length) return <div className="muted">This block has no settings.</div>;
  return (
    <div className="form">
      {keys.map((k) => {
        const s = props[k];
        const v = k in node.config ? node.config[k] : op.defaults[k];
        return (
          <div className="row" key={k}>
            <label className="lbl">{s.title ?? k}{!(k in node.config) && <small> (default)</small>}</label>
            <Field name={k} schema={s} value={v} resolved={resolved?.[k]} defs={op.configSchema.$defs} onChange={(nv) => onChange({ [k]: nv })} />
          </div>
        );
      })}
    </div>
  );
}
