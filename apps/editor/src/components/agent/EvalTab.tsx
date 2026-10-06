import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../../api";
import { usePolling } from "../../hooks";
import type { AnyRun, Graph, UiDoc } from "../../types";

interface CheckResult { field: string; kind: string; expected: unknown; observed: string | null; passed: boolean; note: string | null }
interface CaseResult { case: string; childRunId: string; status: string; error: string | null; passed: boolean; checks: CheckResult[]; latencyMs: number; modelCalls: number }
interface EvalView { runId: string; status: string; error: string | null; name: string; caseCount: number; cases: CaseResult[];
  report: null | { cases: number; passed: number; passRate: number; wilson95: [number, number] | null; totalSeconds: number; interpretation: string } }

const uid = () => crypto.randomUUID();
const EXAMPLE = [{ id: "c1", input: { question: "SYNTHETIC: what is 2 + 3? Reply with the number only." }, checks: [{ field: "answer", kind: "number_close", value: 5, tolerance: 0 }] }];
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** Evaluation runs (ADR 0077): this graph over a labelled case set; each case is a real recorded run, scored by literal checks. */
export function EvalTab({ projectId, graph, ui, allRuns, reloadRuns, ensureSaved, openRun }: {
  projectId: string; graph: Graph; ui: UiDoc; allRuns: AnyRun[]; reloadRuns: () => void; ensureSaved?: () => Promise<void>; openRun: (id: string) => void;
}) {
  const initial = (ui as unknown as { evaluationCases?: unknown[] }).evaluationCases ?? EXAMPLE;
  const [text, setText] = useState(() => JSON.stringify(initial, null, 2));
  const [name, setName] = useState("evaluation");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const evals = useMemo(() => allRuns.filter((r) => (r as unknown as { kind: string }).kind === "agent_eval" && (r as unknown as { config: { project_id?: string } }).config?.project_id === projectId), [allRuns, projectId]);
  const [evalId, setEvalId] = useState<string | null>(null);
  useEffect(() => { if (!evalId && evals.length) setEvalId(evals[evals.length - 1].id); }, [evals, evalId]);
  const current = evals.find((r) => r.id === evalId);
  const live = !!current && ["queued", "preparing", "running", "cancelling"].includes(current.status);
  const view = usePolling<EvalView>(evalId ? `/api/agent/evals/${evalId}` : null, live ? 1000 : 0, [current?.status]);
  useEffect(() => { if (view.data && view.data.status !== current?.status) reloadRuns(); }, [view.data?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  let parsed: unknown = null, parseError: string | null = null;
  try { parsed = JSON.parse(text); if (!Array.isArray(parsed) || parsed.length === 0) parseError = "Cases must be a non-empty JSON array."; }
  catch (e) { parseError = `Not valid JSON: ${(e as Error).message}`; }
  async function start() {
    setBusy(true); setErr(null);
    try {
      await ensureSaved?.();
      const r = await api.post<{ runId: string }>("/api/runs", { graph, config: { kind: "agent_eval", project_id: projectId, name, cases: parsed } }, { "Idempotency-Key": uid() });
      setEvalId(r.runId); reloadRuns();
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  const v = view.data, rep = v?.report;
  return (
    <div className="scroll pad agent-eval" aria-label="agent evaluation">
      <h3>Evaluate this graph on labelled cases</h3>
      <p className="small muted">Each case runs as a real, recorded agent run on a fresh thread. Checks compare fields of the final state literally (equals, contains, not_contains, regex, one_of, number_close); nothing is graded by a model. A pass rate describes these cases only.</p>
      <div><label>Name <input aria-label="evaluation name" value={name} onChange={(e) => setName(e.target.value)} /></label></div>
      <label style={{ display: "block" }}>Cases (JSON: id, input, checks)<textarea aria-label="evaluation cases" className="prod-json" rows={12} value={text} onChange={(e) => setText(e.target.value)} /></label>
      {parseError && <div className="error small">{parseError}</div>}
      <div className="row"><button className="primary" disabled={busy || !!parseError || live} onClick={start}>Run evaluation</button>
        <label className="small">evaluation <select aria-label="evaluation run" value={evalId ?? ""} onChange={(e) => setEvalId(e.target.value || null)}>
          <option value="">(none)</option>{[...evals].reverse().map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status}</option>)}</select></label></div>
      {err && <div className="error small pre">{err}</div>}
      {v && <section aria-label="evaluation result">
        <p className="provenance">run {v.runId} · {v.name} · {v.status}{v.error ? ` · ${v.error}` : ""} · {v.cases.length} / {v.caseCount} cases done</p>
        {rep && <p className="eval-score"><b>{rep.passed} / {rep.cases}</b> passed ({pct(rep.passRate)}){rep.wilson95 && <> · 95% interval {pct(rep.wilson95[0])}–{pct(rep.wilson95[1])}</>} · {rep.totalSeconds} s</p>}
        {rep && <p className="small muted">{rep.interpretation}</p>}
        <div className="tablewrap tall"><table className="dtable" aria-label="evaluation cases result"><thead><tr><th>case</th><th>result</th><th>failed checks (expected → observed)</th><th>latency</th><th></th></tr></thead>
          <tbody>{v.cases.map((c) => <tr key={c.case} className={c.passed ? "" : "fail"}>
            <td>{c.case}</td><td>{c.passed ? "pass" : c.status === "completed" ? "fail" : c.status}</td>
            <td>{c.checks.filter((k) => !k.passed).map((k, i) => <div key={i}><code>{k.field}</code> {k.kind} {JSON.stringify(k.expected)} → {k.observed === null ? <i>{k.note}</i> : <code>{k.observed}</code>}</div>)}{c.error && <div className="error small">{c.error}</div>}</td>
            <td className="num">{Math.round(c.latencyMs)} ms</td>
            <td><button onClick={() => openRun(c.childRunId)} aria-label={`open run of case ${c.case}`}>Open run</button></td></tr>)}</tbody></table></div>
      </section>}
    </div>
  );
}
