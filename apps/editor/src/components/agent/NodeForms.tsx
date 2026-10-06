import { Check, FixtureBadge, JsonBox, Num, Row, Sel, Txt, useCatalog } from "./common";
import type { AgentSpec, Catalog, ModelSpec } from "./types";

type Cfg = Record<string, any>;
type Patch = (p: Cfg) => void;

/** Select one declared state field (optionally limited to some types). */
export function FieldSel({ spec, value, onChange, label, types, allowNone }: { spec: AgentSpec; value: string; onChange: (v: string) => void; label: string; types?: string[]; allowNone?: boolean }) {
  const names = spec.state.filter((f) => !types || types.includes(f.type)).map((f) => f.name);
  const known = names.includes(value) || (allowNone && value === "");
  return (
    <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value)} className={known ? "" : "bad"}>
      {allowNone && <option value="">(none)</option>}
      {!known && <option value={value}>{value} (not declared)</option>}
      {names.map((n) => <option key={n} value={n}>{n}</option>)}
    </select>
  );
}

function varsOf(t: string): string[] { return Array.from(new Set(Array.from(t.matchAll(/\{([A-Za-z_][A-Za-z0-9_.]*)\}/g)).map((m) => m[1]))); }
const getPath = (ns: Record<string, any>, p: string): any => p.split(".").reduce((c, k) => (c && typeof c === "object" ? c[k] : undefined), ns);
export function renderTemplate(t: string, ns: Record<string, any>): string {
  return t.replace(/\{([A-Za-z_][A-Za-z0-9_.]*)\}/g, (_m, p) => { const v = getPath(ns, p); return v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v); });
}

function VarChips({ text, spec }: { text: string; spec: AgentSpec }) {
  const vs = varsOf(text);
  if (!vs.length) return null;
  return <div className="small muted">variables: {vs.map((v) => { const f = spec.state.find((x) => x.name === v.split(".")[0]); return <span key={v} className={`vchip ${f ? "" : "bad"}`}>{v}: {f ? f.type : "undeclared"}</span>; })}</div>;
}

// ------------------------------------------------------------------------------------------------ model
export function ModelForm({ model, onChange }: { model: ModelSpec; onChange: (m: ModelSpec) => void }) {
  const cat = useCatalog();
  const prov = cat?.providers[model.provider];
  const supported = (k: string) => (prov ? prov.settings.includes(k) : true);
  const why = (k: string) => prov?.unsupported?.[k] ?? "not supported by this provider";
  const set = (p: Partial<ModelSpec>) => onChange({ ...model, ...p });
  const oll = cat?.providers.ollama?.ollama;
  return (
    <fieldset className="afield" aria-label="model invocation">
      <legend>Model invocation <small className="muted">(inference, not training)</small> <FixtureBadge fixture={model.provider === "fixture"} /></legend>
      <Row label="Provider"><Sel label="provider" value={model.provider} options={[{ value: "ollama", label: "Ollama (local)" }, { value: "anthropic", label: "Anthropic API" }, { value: "openai_compatible", label: "OpenAI-compatible endpoint" }, { value: "fixture", label: "FIXTURE (scripted test model)" }]}
        onChange={(p) => set({ provider: p, model: p === "anthropic" ? cat?.defaults.anthropicModel ?? "claude-sonnet-5-5" : p === "ollama" || p === "openai_compatible" ? cat?.defaults.ollamaModel ?? "qwen3.5:2b" : "fixture",
          base_url: p === "openai_compatible" ? model.base_url ?? "http://127.0.0.1:11434/v1" : p === "ollama" ? model.base_url : null, api_key: p === "ollama" || p === "fixture" ? null : model.api_key })} /></Row>
      {model.provider === "ollama" && oll && !oll.reachable && <div className="warn">Ollama is not reachable at http://localhost:11434: runs with this block will fail with model_unavailable.</div>}
      <Row label="Model">
        {model.provider === "ollama" && oll?.models.length ? <Sel label="model" value={model.model} options={oll.models.includes(model.model) ? oll.models : [model.model, ...oll.models]} onChange={(m) => set({ model: m })} />
          : <input aria-label="model" value={model.model} onChange={(e) => set({ model: e.target.value })} />}
      </Row>
      <Row label="Temperature" hint={supported("temperature") ? undefined : why("temperature")}><Num label="temperature" nullable min={0} max={2} value={model.temperature} disabled={!supported("temperature")} onChange={(n) => set({ temperature: n })} /></Row>
      <Row label="Max tokens" hint={supported("max_tokens") ? "output limit" : why("max_tokens")}><Num label="max tokens" nullable integer min={1} value={model.max_tokens} disabled={!supported("max_tokens")} onChange={(n) => set({ max_tokens: n })} /></Row>
      <Row label="Seed" hint={supported("seed") ? undefined : why("seed")}><Num label="seed" nullable integer value={model.seed} disabled={!supported("seed")} onChange={(n) => set({ seed: n })} /></Row>
      <Row label="Timeout (s)"><Num label="timeout seconds" min={1} value={model.timeout_s} onChange={(n) => set({ timeout_s: n ?? 180 })} /></Row>
      {model.provider === "ollama" && <Row label="Thinking" hint="request the reasoning trace (reasoning models)"><Check label="think" value={model.think} onChange={(b) => set({ think: b })} /></Row>}
      {model.provider === "ollama" && <Row label="Base URL"><input aria-label="base url" value={model.base_url ?? ""} placeholder="http://localhost:11434" onChange={(e) => set({ base_url: e.target.value || null })} /></Row>}
      {model.provider === "openai_compatible" && <Row label="Base URL" hint="the server's /v1 root; a key is only sent over https or to this machine"><input aria-label="base url" value={model.base_url ?? ""} placeholder="http://127.0.0.1:11434/v1" onChange={(e) => set({ base_url: e.target.value || null })} /></Row>}
      {model.provider === "openai_compatible" && <Row label="Reasoning effort" hint="sent only when set; reasoning models may otherwise spend max tokens on hidden reasoning, and some servers reject it">
        <Sel label="reasoning effort" value={model.reasoning_effort ?? ""} options={[{ value: "", label: "not sent (server default)" }, { value: "none", label: "none" }, { value: "low", label: "low" }, { value: "medium", label: "medium" }, { value: "high", label: "high" }]}
          onChange={(v) => set({ reasoning_effort: (v || null) as ModelSpec["reasoning_effort"] })} /></Row>}
      {model.provider === "openai_compatible" && <Row label="API key" hint="optional (local servers need none); a reference to a secret">
        <Check label="use api key" value={!!model.api_key} onChange={(b) => set({ api_key: b ? { kind: "env", name: "OPENAI_API_KEY" } : null })} />
        {model.api_key && <input aria-label="secret variable name" value={model.api_key?.name ?? ""} placeholder="OPENAI_API_KEY" onChange={(e) => set({ api_key: { kind: "env", name: e.target.value } })} />}
      </Row>}
      {model.provider === "anthropic" && (
        <Row label="API key" hint="a reference to a secret; the value is never stored">
          <Sel label="secret kind" value={model.api_key?.kind ?? "env"} options={[{ value: "env", label: "environment variable" }, { value: "file", label: "secrets file" }]} onChange={(k) => set({ api_key: k === "env" ? { kind: "env", name: "ANTHROPIC_API_KEY" } : { kind: "file", path: "", key: "anthropic" } })} />
          {(model.api_key?.kind ?? "env") === "env"
            ? <input aria-label="secret variable name" value={model.api_key?.name ?? ""} placeholder="ANTHROPIC_API_KEY" onChange={(e) => set({ api_key: { kind: "env", name: e.target.value } })} />
            : <><input aria-label="secrets file path" value={model.api_key?.path ?? ""} placeholder="/abs/path/secrets.json" onChange={(e) => set({ api_key: { ...(model.api_key as any), kind: "file", path: e.target.value } })} />
              <input aria-label="secrets file key" value={model.api_key?.key ?? ""} placeholder="key" onChange={(e) => set({ api_key: { ...(model.api_key as any), kind: "file", key: e.target.value } })} /></>}
        </Row>
      )}
      {model.provider === "fixture" && (
        <Row label="Script" hint="rules (when_contains → reply), responses cycled per call, fail_first, default. Test control flow only.">
          <JsonBox label="fixture script" rows={5} value={model.fixture} onChange={(v) => set({ fixture: v ?? {} })} />
        </Row>
      )}
      {prov && <div className="small muted">Structured output: {prov.structured}. Usage: {prov.usage}.{Object.keys(prov.unsupported ?? {}).length > 0 && ` Unsupported here: ${Object.keys(prov.unsupported!).join(", ")}.`}</div>}
    </fieldset>
  );
}

// ------------------------------------------------------------------------------------------------ individual blocks
function SetStateForm({ cfg, set, spec }: { cfg: Cfg; set: Patch; spec: AgentSpec }) {
  const as: Cfg[] = cfg.assignments ?? [];
  const upd = (i: number, p: Cfg) => set({ assignments: as.map((a, j) => (j === i ? { ...a, ...p } : a)) });
  const paths = spec.state.map((f) => f.name);
  return (
    <div>
      {as.map((a, i) => (
        <div key={i} className="arow2">
          <FieldSel spec={spec} label={`assign field ${i + 1}`} value={a.field} onChange={(v) => upd(i, { field: v })} />
          <Sel label={`assign kind ${i + 1}`} value={a.kind} options={[{ value: "literal", label: "= value" }, { value: "copy", label: "= copy of" }, { value: "template", label: "= text template" }, { value: "increment", label: "+= number" }, { value: "length", label: "= length of" }, { value: "append_item", label: "append item" }]} onChange={(k) => upd(i, { kind: k })} />
          {a.kind === "literal" && <JsonBox label={`literal ${i + 1}`} rows={1} value={a.value ?? null} onChange={(v) => upd(i, { value: v })} />}
          {(a.kind === "copy" || a.kind === "length") && <input aria-label={`source ${i + 1}`} list="statepaths" value={a.source} onChange={(e) => upd(i, { source: e.target.value })} />}
          {a.kind === "append_item" && <><input aria-label={`source ${i + 1}`} list="statepaths" value={a.source} placeholder="state path (or leave empty for a value)" onChange={(e) => upd(i, { source: e.target.value })} />{!a.source && <JsonBox label={`item value ${i + 1}`} rows={1} value={a.value ?? null} onChange={(v) => upd(i, { value: v })} />}</>}
          {a.kind === "template" && <Txt label={`template ${i + 1}`} value={a.template} onChange={(t) => upd(i, { template: t })} />}
          {a.kind === "increment" && <Num label={`increment ${i + 1}`} value={a.by} onChange={(n) => upd(i, { by: n ?? 1 })} />}
          <button className="danger small" aria-label={`remove assignment ${i + 1}`} onClick={() => set({ assignments: as.filter((_, j) => j !== i) })}>×</button>
          {a.kind === "template" && <VarChips text={a.template} spec={spec} />}
        </div>
      ))}
      <datalist id="statepaths">{paths.map((p) => <option key={p} value={p} />)}</datalist>
      <button onClick={() => set({ assignments: [...as, { field: spec.state[0]?.name ?? "", kind: "literal", value: null, source: "", template: "", by: 1 }] })}>Add assignment</button>
    </div>
  );
}

function PromptForm({ cfg, set, spec, values }: { cfg: Cfg; set: Patch; spec: AgentSpec; values?: Record<string, any> }) {
  const items: Cfg[] = cfg.items ?? [];
  const upd = (i: number, p: Cfg) => set({ items: items.map((a, j) => (j === i ? { ...a, ...p } : a)) });
  const FIELD_TYPES: Record<string, string[]> = { memory: ["records"], conversation: ["records", "messages"], documents: ["documents"], tool_results: ["list"] };
  return (
    <div>
      <Row label="Writes messages to"><FieldSel spec={spec} label="prompt output field" value={cfg.output_field} types={["list"]} onChange={(v) => set({ output_field: v })} /></Row>
      {items.map((it, i) => (
        <div key={i} className="pitem">
          <div className="rcasehead">
            <b>{i + 1}.</b>
            <Sel label={`item kind ${i + 1}`} value={it.kind} options={[{ value: "template", label: "message template" }, { value: "memory", label: "memory selection" }, { value: "conversation", label: "conversation history" }, { value: "documents", label: "retrieved chunks" }, { value: "tool_results", label: "tool results" }]} onChange={(k) => upd(i, { kind: k, field: "" })} />
            <Sel label={`item role ${i + 1}`} value={it.role} options={["system", "user", "assistant"]} onChange={(r) => upd(i, { role: r })} />
            <button className="small" disabled={i === 0} aria-label={`move item ${i + 1} up`} onClick={() => { const a = [...items]; [a[i - 1], a[i]] = [a[i], a[i - 1]]; set({ items: a }); }}>↑</button>
            <button className="small" disabled={i === items.length - 1} aria-label={`move item ${i + 1} down`} onClick={() => { const a = [...items]; [a[i + 1], a[i]] = [a[i], a[i + 1]]; set({ items: a }); }}>↓</button>
            <button className="danger small" aria-label={`remove item ${i + 1}`} onClick={() => set({ items: items.filter((_, j) => j !== i) })}>×</button>
          </div>
          {it.kind === "template" ? (
            <>
              <Txt label={`template ${i + 1}`} rows={3} value={it.template} onChange={(t) => upd(i, { template: t })} placeholder="Text with {state_field} placeholders" />
              <VarChips text={it.template} spec={spec} />
              {values && <div className="rendered" aria-label={`rendered preview ${i + 1}`}><span className="small muted">rendered with the values the selected run read: </span>{renderTemplate(it.template, values) || <i className="muted">(empty)</i>}</div>}
            </>
          ) : (
            <>
              <Row label="From field"><FieldSel spec={spec} label={`item field ${i + 1}`} value={it.field} types={FIELD_TYPES[it.kind]} onChange={(v) => upd(i, { field: v })} /></Row>
              {it.kind !== "conversation" && <Row label="Header line"><input aria-label={`header ${i + 1}`} value={it.header} onChange={(e) => upd(i, { header: e.target.value })} /></Row>}
              {it.kind !== "conversation" && <Row label="Per-item line" hint={it.kind === "memory" ? "fields: id, text, kind, store, scores" : it.kind === "documents" ? "fields: chunk_id, text, score, doc_id" : "fields: tool, args, result"}>
                <input aria-label={`item template ${i + 1}`} value={it.template} placeholder={it.kind === "memory" ? "- ({id}) {text}" : it.kind === "documents" ? "[{chunk_id}] {text}" : "{tool}({args}) = {result}"} onChange={(e) => upd(i, { template: e.target.value })} /></Row>}
            </>
          )}
        </div>
      ))}
      <button onClick={() => set({ items: [...items, { kind: "template", role: "user", template: "", field: "", header: "" }] })}>Add message / source</button>
      <div className="hint">Every character of the rendered messages is linked to its source (template, memory record, retrieved chunk, tool result) in the context inspector.</div>
    </div>
  );
}

function StructuredForm({ cfg, set, spec }: { cfg: Cfg; set: Patch; spec: AgentSpec }) {
  const fs: Cfg[] = cfg.schema_fields ?? [];
  const upd = (i: number, p: Cfg) => set({ schema_fields: fs.map((a, j) => (j === i ? { ...a, ...p } : a)) });
  return (
    <div>
      <ModelForm model={cfg.model} onChange={(m) => set({ model: m })} />
      <Row label="Messages from"><FieldSel spec={spec} label="messages field" value={cfg.messages_field} types={["list"]} onChange={(v) => set({ messages_field: v })} /></Row>
      <Row label="Parsed object to"><FieldSel spec={spec} label="output field" value={cfg.output_field} types={["object"]} onChange={(v) => set({ output_field: v })} /></Row>
      <Row label="Error text to"><FieldSel spec={spec} label="error field" allowNone value={cfg.error_field} types={["text"]} onChange={(v) => set({ error_field: v })} /></Row>
      <h4>Output schema</h4>
      {fs.map((f, i) => (
        <div key={i} className="arow2">
          <input aria-label={`schema name ${i + 1}`} value={f.name} size={10} onChange={(e) => upd(i, { name: e.target.value })} />
          <Sel label={`schema type ${i + 1}`} value={f.type} options={["text", "integer", "number", "boolean", "enum", "list_of_text"]} onChange={(t) => upd(i, { type: t })} />
          {f.type === "enum" && <input aria-label={`choices ${i + 1}`} value={(f.choices ?? []).join(", ")} placeholder="a, b, c" onChange={(e) => upd(i, { choices: e.target.value.split(",").map((s: string) => s.trim()).filter(Boolean) })} />}
          <label className="small"><input type="checkbox" aria-label={`required ${i + 1}`} checked={f.required} onChange={(e) => upd(i, { required: e.target.checked })} /> required</label>
          <input aria-label={`description ${i + 1}`} value={f.description} placeholder="description (sent to the model)" onChange={(e) => upd(i, { description: e.target.value })} />
          <button className="danger small" aria-label={`remove schema field ${i + 1}`} onClick={() => set({ schema_fields: fs.filter((_, j) => j !== i) })}>×</button>
        </div>
      ))}
      <button onClick={() => set({ schema_fields: [...fs, { name: `field${fs.length + 1}`, type: "text", description: "", required: true, choices: [] }] })}>Add schema field</button>
      <h4>Validation behaviour</h4>
      <Row label="Max retries"><Num label="max retries" integer min={0} max={10} value={cfg.retry?.maxRetries ?? 2} onChange={(n) => set({ retry: { ...cfg.retry, maxRetries: n ?? 0 } })} /></Row>
      <Row label="Feed the error back" hint="the rejected reply and the validation errors are added to the retry request"><Check label="feedback" value={cfg.retry?.feedback ?? true} onChange={(b) => set({ retry: { ...cfg.retry, feedback: b } })} /></Row>
      <Row label="When retries are exhausted"><Sel label="on failure" value={cfg.on_failure} options={[{ value: "route", label: "write {} + the error, let a route decide" }, { value: "fail", label: "fail the node" }]} onChange={(v) => set({ on_failure: v })} /></Row>
    </div>
  );
}

function ToolForm({ cfg, set, spec, cat }: { cfg: Cfg; set: Patch; spec: AgentSpec; cat: Catalog | null }) {
  const tool = cat?.tools.find((t) => t.name === cfg.tool);
  return (
    <div>
      <Row label="Tool"><Sel label="tool" value={cfg.tool} options={(cat?.tools.map((t) => t.name) ?? [cfg.tool]) as string[]} onChange={(t) => { const ti = cat?.tools.find((x) => x.name === t); set({ tool: t, args: Object.fromEntries(Object.keys(ti?.args ?? {}).map((k) => [k, ""])), require_approval: true }); }} /></Row>
      {tool && (
        <div className={tool.external ? "warn" : "notrec"}>
          <b>{tool.description}</b>
          <div className="small">Declared effects: {tool.effects.length ? tool.effects.join(", ") : "none"}. {tool.external ? "EXTERNAL: runs only after an explicit approval interrupt, and at most once per thread/node/arguments (effect ledger)." : "No effect outside the process: no approval needed."}</div>
          {Object.keys(tool.limits).length > 0 && <div className="small">Limits: {JSON.stringify(tool.limits)}</div>}
        </div>
      )}
      {tool && Object.keys(tool.args).map((a) => (
        <Row key={a} label={`arg ${a}`}><input aria-label={`argument ${a}`} value={cfg.args?.[a] ?? ""} placeholder="text with {state_field}" onChange={(e) => set({ args: { ...cfg.args, [a]: e.target.value } })} /></Row>
      ))}
      {tool && Object.values(cfg.args ?? {}).map((t, i) => <VarChips key={i} text={String(t)} spec={spec} />)}
      {(cfg.tool === "read_text_file" || cfg.tool === "write_note") && <Row label="Allowed directory" hint={cfg.tool === "write_note" ? "empty = the workbench outbox" : "the only directory this tool may read"}><input aria-label="allowed directory" value={cfg.allowed_dir} onChange={(e) => set({ allowed_dir: e.target.value })} /></Row>}
      <Row label="Require approval" hint={tool?.external ? "cannot be switched off for an external effect" : "only external effects need it"}><Check label="require approval" value={tool?.external ? true : cfg.require_approval} disabled={!!tool?.external} onChange={(b) => set({ require_approval: b })} /></Row>
      <Row label="Result to"><FieldSel spec={spec} label="tool output field" value={cfg.output_field} types={["object"]} onChange={(v) => set({ output_field: v })} /></Row>
      <Row label="Also append to" hint="a list field read by a prompt 'tool results' item"><FieldSel spec={spec} label="tool append field" allowNone value={cfg.append_to} types={["list"]} onChange={(v) => set({ append_to: v })} /></Row>
    </div>
  );
}

function MemoryWriteForm({ cfg, set, spec }: { cfg: Cfg; set: Patch; spec: AgentSpec }) {
  const md = Object.entries((cfg.metadata ?? {}) as Record<string, string>);
  return (
    <div>
      <Row label="Target"><Sel label="memory target" value={cfg.target} options={[{ value: "long_term", label: "long-term store" }, { value: "short_term", label: "short-term (this thread's messages)" }]} onChange={(t) => set({ target: t })} /></Row>
      <Row label="Text"><Txt label="memory text" value={cfg.text} onChange={(t) => set({ text: t })} /></Row>
      <VarChips text={cfg.text} spec={spec} />
      {cfg.target === "short_term" ? (
        <>
          <Row label="Messages field"><FieldSel spec={spec} label="short-term field" value={cfg.field} types={["messages", "records"]} onChange={(v) => set({ field: v })} /></Row>
          <Row label="Role"><Sel label="role" value={cfg.role} options={["user", "assistant", "system"]} onChange={(r) => set({ role: r })} /></Row>
        </>
      ) : (
        <>
          <Row label="Namespace"><input aria-label="namespace" value={cfg.namespace} onChange={(e) => set({ namespace: e.target.value })} /></Row>
          <Row label="Scope" hint="template, e.g. user:{user_id}"><input aria-label="scope" value={cfg.scope} onChange={(e) => set({ scope: e.target.value })} /></Row>
          <Row label="Kind"><Sel label="kind" value={cfg.kind} options={["semantic", "episodic", "procedural", "summary"]} onChange={(k) => set({ kind: k })} /></Row>
          <Row label="Importance (0-1)"><Num label="importance" min={0} max={1} value={cfg.importance} onChange={(n) => set({ importance: n ?? 0.5 })} /></Row>
          <Row label="Generated" hint="a model-generated assertion is stored as generated; storing it does not make it true"><Check label="generated" value={cfg.generated} onChange={(b) => set({ generated: b })} /></Row>
          <Row label="Evidence" hint="template naming the source of the fact"><input aria-label="evidence" value={cfg.evidence} onChange={(e) => set({ evidence: e.target.value })} /></Row>
          <Row label="Decision stage"><Sel label="write mode" value={cfg.mode} options={[{ value: "direct", label: "write directly" }, { value: "approve", label: "propose; a person accepts, rejects or edits" }]} onChange={(m) => set({ mode: m })} /></Row>
          <Row label="Skip duplicates"><Check label="skip duplicates" value={cfg.skip_duplicates} onChange={(b) => set({ skip_duplicates: b })} /></Row>
          <Row label="Length limits"><Num label="min chars" integer min={0} value={cfg.min_chars} onChange={(n) => set({ min_chars: n ?? 0 })} /> – <Num label="max chars" integer min={1} value={cfg.max_chars} onChange={(n) => set({ max_chars: n ?? 1000 })} /></Row>
          <h4>Metadata</h4>
          {md.map(([k, v]) => (
            <div key={k} className="arow2"><input aria-label={`metadata key ${k}`} defaultValue={k} size={8} onBlur={(e) => { if (e.target.value && e.target.value !== k) { const o = { ...cfg.metadata }; delete o[k]; o[e.target.value] = v; set({ metadata: o }); } }} />
              <input aria-label={`metadata value ${k}`} value={v} onChange={(e) => set({ metadata: { ...cfg.metadata, [k]: e.target.value } })} />
              <button className="danger small" aria-label={`remove metadata ${k}`} onClick={() => { const o = { ...cfg.metadata }; delete o[k]; set({ metadata: o }); }}>×</button></div>
          ))}
          <button onClick={() => set({ metadata: { ...cfg.metadata, [`key${md.length + 1}`]: "" } })}>Add metadata</button>
        </>
      )}
    </div>
  );
}

function SimpleFields({ rows, cfg, set, spec }: { rows: { key: string; label: string; types?: string[]; none?: boolean; hint?: string }[]; cfg: Cfg; set: Patch; spec: AgentSpec }) {
  return <>{rows.map((r) => <Row key={r.key} label={r.label} hint={r.hint}><FieldSel spec={spec} label={r.label} value={cfg[r.key] ?? ""} types={r.types} allowNone={r.none} onChange={(v) => set({ [r.key]: v })} /></Row>)}</>;
}

/** The configuration form of one node. Every block has a specific visual form; none asks for Python. */
export function NodeForm({ type, cfg, spec, onConfig, runValues }: { type: string; cfg: Cfg; spec: AgentSpec; onConfig: Patch; runValues?: Record<string, any> }) {
  const cat = useCatalog();
  switch (type) {
    case "agent.set_state": return <SetStateForm cfg={cfg} set={onConfig} spec={spec} />;
    case "agent.prompt": return <PromptForm cfg={cfg} set={onConfig} spec={spec} values={runValues} />;
    case "agent.chat_model": return (
      <div>
        <ModelForm model={cfg.model} onChange={(m) => onConfig({ model: m })} />
        <SimpleFields cfg={cfg} set={onConfig} spec={spec} rows={[{ key: "messages_field", label: "Messages from", types: ["list"] }, { key: "output_field", label: "Reply text to", types: ["text"] }, { key: "append_to", label: "Also append reply to", types: ["messages", "records", "list"], none: true, hint: "e.g. the thread's conversation" }]} />
      </div>
    );
    case "agent.structured_output": return <StructuredForm cfg={cfg} set={onConfig} spec={spec} />;
    case "agent.embed_text": return (
      <div>
        <SimpleFields cfg={cfg} set={onConfig} spec={spec} rows={[{ key: "from_field", label: "Text from", types: ["text"] }, { key: "output_field", label: "Embedding info to", types: ["object"] }]} />
        <Row label="Provider"><Sel label="embedding provider" value={cfg.provider} options={[{ value: "local_hash", label: "local_hash (lexical, NOT semantic)" }, { value: "ollama", label: "Ollama embeddings" }]} onChange={(p) => onConfig({ provider: p })} /></Row>
        {cfg.provider === "ollama" ? <Row label="Model"><input aria-label="embedding model" value={cfg.model} onChange={(e) => onConfig({ model: e.target.value })} /></Row>
          : <Row label="Dimension"><Num label="dimension" integer min={16} value={cfg.dimension} onChange={(n) => onConfig({ dimension: n ?? 256 })} /></Row>}
        <Row label="Normalize (L2)"><Check label="normalize" value={cfg.normalize} onChange={(b) => onConfig({ normalize: b })} /></Row>
      </div>
    );
    case "agent.retrieve": return (
      <div>
        <Row label="Index"><select aria-label="index" value={cfg.index} onChange={(e) => onConfig({ index: e.target.value })}><option value="">(choose)</option>{spec.indexes.map((i) => <option key={i.id} value={i.id}>{i.id}</option>)}</select></Row>
        <Row label="Query"><Txt label="retrieval query" value={cfg.query} onChange={(q) => onConfig({ query: q })} /></Row>
        <VarChips text={cfg.query} spec={spec} />
        <Row label="Top-k"><Num label="k" integer min={1} value={cfg.k} onChange={(n) => onConfig({ k: n ?? 3 })} /></Row>
        <Row label="Score threshold" hint="cosine similarity when the index is normalized; chunks below it are listed as excluded"><Num label="score threshold" nullable value={cfg.score_threshold} onChange={(n) => onConfig({ score_threshold: n })} /></Row>
        <SimpleFields cfg={cfg} set={onConfig} spec={spec} rows={[{ key: "output_field", label: "Chunks to", types: ["documents"] }]} />
      </div>
    );
    case "agent.citations": return <SimpleFields cfg={cfg} set={onConfig} spec={spec} rows={[{ key: "text_field", label: "Generated text", types: ["text"] }, { key: "docs_field", label: "Retrieved chunks", types: ["documents"] }, { key: "output_field", label: "Result to", types: ["object"] }]} />;
    case "agent.tool_call": return <ToolForm cfg={cfg} set={onConfig} spec={spec} cat={cat} />;
    case "agent.human_interrupt": return (
      <div>
        <Row label="Prompt"><Txt label="interrupt prompt" value={cfg.prompt} onChange={(t) => onConfig({ prompt: t })} /></Row>
        <SimpleFields cfg={cfg} set={onConfig} spec={spec} rows={[{ key: "show_field", label: "Show field", none: true }, { key: "edit_field", label: "Edit writes to", none: true }, { key: "decision_field", label: "Decision to", types: ["text"] }]} />
        <Row label="Allowed actions">{(["approve", "reject", "edit"] as const).map((a) => (
          <label key={a} className="small"><input type="checkbox" aria-label={`action ${a}`} checked={(cfg.actions ?? []).includes(a)} onChange={(e) => onConfig({ actions: e.target.checked ? [...(cfg.actions ?? []), a] : (cfg.actions ?? []).filter((x: string) => x !== a) })} /> {a} </label>
        ))}</Row>
      </div>
    );
    case "agent.memory_select": return (
      <div>
        <Row label="Policy"><select aria-label="policy" value={cfg.policy} onChange={(e) => onConfig({ policy: e.target.value })}><option value="">(choose)</option>{spec.policies.map((p) => <option key={p.id} value={p.id}>{p.id}</option>)}</select></Row>
        <SimpleFields cfg={cfg} set={onConfig} spec={spec} rows={[{ key: "output_field", label: "Selected records to", types: ["records"] }, { key: "short_term_field", label: "Short-term messages", types: ["messages", "records"], none: true, hint: "this thread's conversation; the policy may read it" }]} />
        <div className="hint">Edit the policy's stages in the Memory tab.</div>
      </div>
    );
    case "agent.memory_write": return <MemoryWriteForm cfg={cfg} set={onConfig} spec={spec} />;
    default: return <div className="warn">No form for {type}.</div>;
  }
}
