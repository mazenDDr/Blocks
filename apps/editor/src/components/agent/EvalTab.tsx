import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../../api";
import { usePolling } from "../../hooks";
import type { AnyRun, Graph, UiDoc } from "../../types";

interface CheckResult { field: string; kind: string; expected: unknown; observed: string | null; passed: boolean; note: string | null }
interface CaseResult { case: string; seed?: number | null; childRunId: string | null; requestId?: string; status: string; error: string | null; passed: boolean; checks: CheckResult[]; latencyMs: number; modelCalls: number }
interface EvalView { runId: string; status: string; error: string | null; name: string; caseCount: number; cases: CaseResult[];
  report: null | { cases: number; passed: number; passRate: number; wilson95: [number, number] | null; totalSeconds: number; interpretation: string;
    perSeed?: { seed: number; passed: number; cases: number; passRate: number; wilson95: [number, number] | null }[] | null;
    stability?: { alwaysPass: string[]; neverPass: string[]; varies: string[] } | null } }

const uid = () => crypto.randomUUID();
const EXAMPLE = [{ id: "c1", input: { question: "SYNTHETIC: what is 2 + 3? Reply with the number only." }, checks: [{ field: "answer", kind: "number_close", value: 5, tolerance: 0 }] }];
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** Cases from a .json array or .jsonl file (one case per line). */
export function parseCases(text: string): unknown[] {
  const t = text.trim();
  if (t.startsWith("[")) return JSON.parse(t);
  return t.split(/\r?\n/).filter((l) => l.trim()).map((l, i) => { try { return JSON.parse(l); } catch { throw new Error(`line ${i + 1} is not valid JSON`); } });
}

export function CasesEditor({ text, setText }: { text: string; setText: (t: string) => void }) {
  const [fileErr, setFileErr] = useState<string | null>(null);
  return <>
    <label style={{ display: "block" }}>Cases (JSON: id, input, checks)<textarea aria-label="evaluation cases" className="prod-json" rows={10} value={text} onChange={(e) => setText(e.target.value)} /></label>
    <label className="small">Load cases from a file (.json array or .jsonl) <input type="file" aria-label="evaluation cases file" accept=".json,.jsonl,application/json" onChange={async (e) => {
      const f = e.target.files?.[0]; if (!f) return;
      try { setText(JSON.stringify(parseCases(await f.text()), null, 2)); setFileErr(null); } catch (err) { setFileErr(`${f.name}: ${(err as Error).message}`); }
    }} /></label>
    {fileErr && <div className="error small">{fileErr}</div>}
  </>;
}

interface Comparison { nameA: string; nameB: string; shared: number; onlyA: number; onlyB: number; passedA: number; passedB: number; fixed: number; regressed: number;
  mcnemarExactP: number; interpretation: string; rows: { case: string; seed: number | null; a: boolean; b: boolean; change: string }[] }

/** Two finished evaluations side by side (ADR 0081): fixed / regressed cases and an exact McNemar p-value. */
export function EvalCompare({ a, b }: { a: string; b: string }) {
  const cmp = usePolling<Comparison>(`/api/agent/evals/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`, 0, [a, b]);
  if (cmp.error) return <div className="error small">{cmp.error}</div>;
  const c = cmp.data;
  if (!c) return null;
  const changed = c.rows.filter((r) => r.change !== "same");
  return <section aria-label="evaluation comparison">
    <p className="eval-compare"><b>{c.passedA}</b> → <b>{c.passedB}</b> of {c.shared} shared runs · {c.fixed} fixed · {c.regressed} regressed · exact McNemar p = {c.mcnemarExactP}
      {(c.onlyA || c.onlyB) ? <span className="muted"> · not compared: {c.onlyA} only in A, {c.onlyB} only in B</span> : null}</p>
    <p className="small muted">{c.interpretation}</p>
    {changed.length > 0 && <table className="dtable" aria-label="changed cases"><thead><tr><th>case</th><th>seed</th><th>{c.nameA}</th><th>{c.nameB}</th><th>change</th></tr></thead>
      <tbody>{changed.map((r) => <tr key={`${r.case}:${r.seed ?? ""}`}><td>{r.case}</td><td>{r.seed ?? ""}</td><td>{r.a ? "pass" : "fail"}</td><td>{r.b ? "pass" : "fail"}</td><td>{r.change}</td></tr>)}</tbody></table>}
  </section>;
}

/** One evaluation's score and per-case results; polls while it runs. */
export function EvalResult({ evalId, live, openRun, onStatus }: { evalId: string; live: boolean; openRun?: (id: string) => void; onStatus?: (s: string) => void }) {
  const view = usePolling<EvalView>(`/api/agent/evals/${evalId}`, live ? 1000 : 0, [live]);
  const v = view.data, rep = v?.report;
  useEffect(() => { if (v?.status) onStatus?.(v.status); }, [v?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!v) return view.error ? <div className="error small">{view.error}</div> : null;
  return <section aria-label="evaluation result">
    <p className="provenance">run {v.runId} · {v.name} · {v.status}{v.error ? ` · ${v.error}` : ""} · {v.cases.length} runs done</p>
    {rep && <p className="eval-score"><b>{rep.passed} / {rep.cases}</b> passed ({pct(rep.passRate)}){rep.wilson95 && <> · 95% interval {pct(rep.wilson95[0])}–{pct(rep.wilson95[1])}</>} · {rep.totalSeconds} s</p>}
    {rep?.perSeed && <table className="dtable" aria-label="per-seed pass rates"><thead><tr><th>seed</th><th>passed</th><th>rate</th><th>95% interval</th></tr></thead>
      <tbody>{rep.perSeed.map((p) => <tr key={p.seed}><td>{p.seed}</td><td>{p.passed} / {p.cases}</td><td>{pct(p.passRate)}</td><td>{p.wilson95 ? `${pct(p.wilson95[0])}–${pct(p.wilson95[1])}` : ""}</td></tr>)}</tbody></table>}
    {rep?.stability && <p className="small">Cases that vary with the seed: {rep.stability.varies.join(", ") || "none"} · never pass: {rep.stability.neverPass.join(", ") || "none"}</p>}
    {rep && <p className="small muted">{rep.interpretation}</p>}
    <div className="tablewrap tall"><table className="dtable" aria-label="evaluation cases result"><thead><tr><th>case</th>{rep?.perSeed && <th>seed</th>}<th>result</th><th>failed checks (expected → observed)</th><th>latency</th><th></th></tr></thead>
      <tbody>{v.cases.map((c) => <tr key={`${c.case}:${c.seed ?? ""}`} className={c.passed ? "" : "fail"}>
        <td>{c.case}</td>{rep?.perSeed && <td>{c.seed}</td>}<td>{c.passed ? "pass" : c.status === "completed" ? "fail" : c.status}</td>
        <td>{c.checks.filter((k) => !k.passed).map((k, i) => <div key={i}><code>{k.field}</code> {k.kind} {JSON.stringify(k.expected)} → {k.observed === null ? <i>{k.note}</i> : <code>{k.observed}</code>}</div>)}{c.error && <div className="error small">{c.error}</div>}</td>
        <td className="num">{Math.round(c.latencyMs)} ms</td>
        <td>{c.childRunId && openRun ? <button onClick={() => openRun(c.childRunId!)} aria-label={`open run of case ${c.case}${c.seed != null ? ` seed ${c.seed}` : ""}`}>Open run</button> : c.requestId ? <code className="small">{c.requestId}</code> : null}</td></tr>)}</tbody></table></div>
  </section>;
}

/** Evaluation runs (ADR 0077): this graph over a labelled case set; each case is a real recorded run, scored by literal checks. */
export function EvalTab({ projectId, graph, ui, allRuns, reloadRuns, ensureSaved, openRun }: {
  projectId: string; graph: Graph; ui: UiDoc; allRuns: AnyRun[]; reloadRuns: () => void; ensureSaved?: () => Promise<void>; openRun: (id: string) => void;
}) {
  const initial = (ui as unknown as { evaluationCases?: unknown[] }).evaluationCases ?? EXAMPLE;
  const [text, setText] = useState(() => JSON.stringify(initial, null, 2));
  const [name, setName] = useState("evaluation");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [seedText, setSeedText] = useState("");
  const seeds = seedText.split(",").map((x) => x.trim()).filter(Boolean).map(Number);
  const seedError = seeds.some((x) => !Number.isInteger(x)) ? "Seeds are comma-separated integers." : null;
  const evals = useMemo(() => allRuns.filter((r) => (r as unknown as { kind: string }).kind === "agent_eval" && (r as unknown as { config: { project_id?: string } }).config?.project_id === projectId), [allRuns, projectId]);
  const [evalId, setEvalId] = useState<string | null>(null);
  useEffect(() => { if (!evalId && evals.length) setEvalId(evals[evals.length - 1].id); }, [evals, evalId]);
  const current = evals.find((r) => r.id === evalId);
  const [compareId, setCompareId] = useState("");
  const finished = evals.filter((r) => r.status === "completed" && r.id !== evalId);
  const live = !!current && ["queued", "preparing", "running", "cancelling"].includes(current.status);
  let parsed: unknown = null, parseError: string | null = null;
  try { parsed = parseCases(text); if (!Array.isArray(parsed) || parsed.length === 0) parseError = "Cases must be a non-empty JSON array."; }
  catch (e) { parseError = `Not valid JSON: ${(e as Error).message}`; }
  async function start() {
    setBusy(true); setErr(null);
    try {
      await ensureSaved?.();
      const r = await api.post<{ runId: string }>("/api/runs", { graph, config: { kind: "agent_eval", project_id: projectId, name, cases: parsed, seeds } }, { "Idempotency-Key": uid() });
      setEvalId(r.runId); reloadRuns();
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  return (
    <div className="scroll pad agent-eval" aria-label="agent evaluation">
      <h3>Evaluate this graph on labelled cases</h3>
      <p className="small muted">Each case runs as a real, recorded agent run on a fresh thread. Checks compare fields of the final state literally (equals, contains, not_contains, regex, one_of, number_close); nothing is graded by a model. A pass rate describes these cases only.</p>
      <div><label>Name <input aria-label="evaluation name" value={name} onChange={(e) => setName(e.target.value)} /></label></div>
      <CasesEditor text={text} setText={setText} />
      <div><label>Seeds <input aria-label="evaluation seeds" value={seedText} placeholder="e.g. 1, 2, 3 (empty = the graph's own)" onChange={(e) => setSeedText(e.target.value)} /></label>
        <span className="small muted"> each seed is set on every model block; one run per case and seed</span></div>
      {parseError && <div className="error small">{parseError}</div>}{seedError && <div className="error small">{seedError}</div>}
      <div className="row"><button className="primary" disabled={busy || !!parseError || !!seedError || live} onClick={start}>Run evaluation</button>
        <label className="small">evaluation <select aria-label="evaluation run" value={evalId ?? ""} onChange={(e) => setEvalId(e.target.value || null)}>
          <option value="">(none)</option>{[...evals].reverse().map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status}</option>)}</select></label></div>
      {err && <div className="error small pre">{err}</div>}
      {evalId && current?.status === "completed" && finished.length > 0 && <div><label className="small">compare with <select aria-label="compare evaluation" value={compareId} onChange={(e) => setCompareId(e.target.value)}>
        <option value="">(none)</option>{[...finished].reverse().map((r) => <option key={r.id} value={r.id}>{r.id}</option>)}</select></label></div>}
      {evalId && compareId && current?.status === "completed" && <EvalCompare a={compareId} b={evalId} />}
      {evalId && <EvalResult key={evalId} evalId={evalId} live={live} openRun={openRun} onStatus={(st) => { if (st !== current?.status) reloadRuns(); }} />}
    </div>
  );
}
