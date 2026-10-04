import { useEffect, useState, type ReactNode } from "react";
import { api, errorText } from "../../api";
import type { Catalog } from "./types";

/** A labelled control row used by every agent form. */
export function Row({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return <label className="arow"><span className="alabel">{label}</span><span className="actl">{children}{hint && <small className="muted ahint">{hint}</small>}</span></label>;
}

export function Num({ value, onChange, integer, min, max, nullable, label, disabled, width }: {
  value: number | null | undefined; onChange: (n: number | null) => void; integer?: boolean; min?: number; max?: number; nullable?: boolean; label: string; disabled?: boolean; width?: number;
}) {
  const [text, setText] = useState(value == null ? "" : String(value));
  useEffect(() => { setText((t) => (value == null ? (t === "" ? t : "") : Number(t) === value ? t : String(value))); }, [value]);
  return (
    <input type="number" aria-label={label} disabled={disabled} value={text} step={integer ? 1 : "any"} min={min} max={max} style={width ? { width } : undefined}
      placeholder={nullable ? "none" : undefined}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value === "") { if (nullable) onChange(null); return; }
        const n = Number(e.target.value);
        if (Number.isFinite(n) && (!integer || Number.isInteger(n))) onChange(n);
      }} />
  );
}

export function Txt({ value, onChange, label, rows = 2, placeholder, disabled }: { value: string; onChange: (s: string) => void; label: string; rows?: number; placeholder?: string; disabled?: boolean }) {
  return <textarea aria-label={label} className="atext" rows={rows} value={value} placeholder={placeholder} disabled={disabled} onChange={(e) => onChange(e.target.value)} />;
}

/** Edits any JSON value as text; the value only changes when the text parses. */
export function JsonBox({ value, onChange, label, rows = 3 }: { value: unknown; onChange: (v: any) => void; label: string; rows?: number }) {
  const fmt = (v: unknown) => JSON.stringify(v ?? null, null, 1);
  const [text, setText] = useState(fmt(value));
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setText((t) => { try { return JSON.stringify(JSON.parse(t)) === JSON.stringify(value ?? null) ? t : fmt(value); } catch { return fmt(value); } }); }, [value]);
  return (
    <span className="jsonbox">
      <textarea aria-label={label} className="atext mono" rows={rows} value={text}
        onChange={(e) => { setText(e.target.value); try { onChange(JSON.parse(e.target.value)); setErr(null); } catch (x) { setErr(String((x as Error).message)); } }} />
      {err && <small className="error">not valid JSON: {err}</small>}
    </span>
  );
}

export function Sel<T extends string>({ value, options, onChange, label, disabled }: { value: T; options: (T | { value: T; label: string })[]; onChange: (v: T) => void; label: string; disabled?: boolean }) {
  return (
    <select aria-label={label} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
      {options.map((o) => typeof o === "string" ? <option key={o} value={o}>{o}</option> : <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function Check({ value, onChange, label, disabled }: { value: boolean; onChange: (b: boolean) => void; label: string; disabled?: boolean }) {
  return <input type="checkbox" aria-label={label} checked={value} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />;
}

let catalogPromise: Promise<Catalog> | null = null;
export function useCatalog(): Catalog | null {
  const [c, setC] = useState<Catalog | null>(null);
  useEffect(() => {
    let alive = true;
    if (!catalogPromise) catalogPromise = api.get<Catalog>("/api/agent/catalog");
    catalogPromise.then((x) => { if (alive) setC(x); }).catch(() => { catalogPromise = null; });
    return () => { alive = false; };
  }, []);
  return c;
}

export const j = (v: unknown, max = 160): string => {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s == null ? "∅" : s.length > max ? s.slice(0, max) + "…" : s;
};
export const uid = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`);
export const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`);
export const fmtTime = (t: number) => new Date(t * 1000).toLocaleString();

export const SOURCE_LABEL: Record<string, string> = {
  prompt_template: "prompt template", memory_record: "memory record", retrieved_chunk: "retrieved chunk", tool_result: "tool result", conversation_message: "conversation message",
  structured_output_instruction: "schema instruction", structured_output_retry_feedback: "retry feedback", model_reply: "earlier model reply", unlinked: "unlinked",
};
export function sourceText(s: Record<string, any>): string {
  switch (s.kind) {
    case "prompt_template": return `template · ${s.node}#${s.item}${s.variables?.length ? ` · {${s.variables.join(", ")}}` : ""}`;
    case "memory_record": return `memory record ${s.recordId} (${s.store ?? "?"}) · selection ${s.applicationId ?? "?"}`;
    case "retrieved_chunk": return `chunk ${s.chunkId} · score ${s.score} · ${String(s.sourcePath ?? "").split("/").pop()}`;
    case "tool_result": return `tool ${s.tool} · ${s.callId ?? ""}`;
    case "conversation_message": return `conversation ${s.recordId ?? ""}`;
    default: return SOURCE_LABEL[s.kind] ?? s.kind;
  }
}

export function ErrorLine({ text }: { text: string | null }) {
  return text ? <div className="error pre">{text}</div> : null;
}

/** Run an async action with a busy flag and an error line. */
export function useAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true); setError(null);
    try { return await fn(); } catch (e) { setError(errorText(e)); return undefined; } finally { setBusy(false); }
  };
  return { busy, error, run, setError };
}

export function FixtureBadge({ fixture }: { fixture: boolean }) {
  return fixture ? <span className="badge fixture" title="Scripted test model: its reply is NOT a real model response">FIXTURE</span> : null;
}
