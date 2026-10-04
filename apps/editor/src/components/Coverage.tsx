import { useEffect, useState } from "react";
import { api, errorText } from "../api";
import type { Ledger } from "../types";

const COLORS: Record<string, string> = { tested: "ok", experimental: "old", unsupported: "bad", expanded: "" };

/** The public coverage ledger (VISION 14.3), straight from GET /api/coverage: generated from the registry, the adapters and tests/, never typed by hand. */
export function CoverageView() {
  const [L, setL] = useState<Ledger | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [onlyPortable, setOnlyPortable] = useState(false);
  useEffect(() => { api.get<Ledger>("/api/coverage").then(setL).catch((e) => setErr(errorText(e))); }, []);
  if (err) return <div className="fullws"><div className="error pre">{err}</div></div>;
  if (!L) return <div className="fullws muted pad">Loading the coverage ledger…</div>;
  const B = ["pytorch", "keras", "jax"];
  const rows = L.model.filter((r) => r.type.includes(q) && (!onlyPortable || (r.execution.keras === "tested" && r.execution.jax === "tested")));
  const cell = (t: { cases: string[]; workloads: string[]; refusalCases: string[] }) => [t.cases.length && `${t.cases.length} conformance`, t.refusalCases.length && `${t.refusalCases.length} refusal`, t.workloads.length && `${t.workloads.length} workload`].filter(Boolean).join(", ") || "-";
  return (
    <div className="fullws coveragews">
      <h3>Coverage ledger</h3>
      <div className="bknote small">Generated from the operation registry and the backend adapters (<code>docs/COVERAGE.md</code> is the same data; a test fails when it is stale). Statuses: <b>tested</b> (a test that runs it exists), <b>experimental</b>
        (implemented, no covering test found), <b>unsupported</b> (refused before execution with a stable code), <b>expanded</b> (structural block, expanded into flat nodes first). {L.conformanceCases} single-operation conformance cases + workloads {L.workloads.join(", ")}.</div>
      <div className="small">
        {L.backends.map((b) => <span key={b.id} style={{ marginRight: 14 }}><b>{b.title}</b> {Object.entries(b.pinned).map(([k, v]) => `${k} ${v}`).join(", ")}</span>)}
      </div>
      <div style={{ margin: "6px 0" }}>
        <input placeholder="filter operations…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="filter operations" />{" "}
        <label><input type="checkbox" checked={onlyPortable} onChange={(e) => setOnlyPortable(e.target.checked)} /> portable subset only</label>
      </div>
      <table className="compat">
        <thead>
          <tr><th rowSpan={2}>Operation</th><th colSpan={3}>Execution</th><th colSpan={2}>Visual depth</th><th colSpan={3}>Runtime inspection</th><th colSpan={3}>Tests</th><th rowSpan={2}>Restrictions</th></tr>
          <tr>{B.map((b) => <th key={"e" + b}>{b}</th>)}<th>explain</th><th>architecture</th>{B.map((b) => <th key={"i" + b}>{b}</th>)}{B.map((b) => <th key={"t" + b}>{b}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.type}>
              <td><code>{r.type}</code></td>
              {B.map((b) => <td key={b}><span className={`badge ${COLORS[r.execution[b]] ?? ""}`}>{r.execution[b]}</span></td>)}
              <td>{r.explain ? "yes" : "no"}</td><td>{r.architecture}</td>
              {B.map((b) => <td key={b} className="small">{r.inspection[b]}</td>)}
              {B.map((b) => <td key={b} className="small">{cell(r.tests[b])}</td>)}
              <td className="small">{B.flatMap((b) => r.restrictions[b].map((x) => `${b}: ${x}`)).join("; ") || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h4>Known restrictions (refused before execution)</h4>
      <table className="compat"><thead><tr><th>Operation</th><th>Case</th><th>Backend</th><th>Code</th></tr></thead>
        <tbody>{L.refusals.map((x, i) => <tr key={i}><td><code>{x.op}</code></td><td>{x.case}</td><td>{x.backend}</td><td><code>{x.code}</code></td></tr>)}</tbody></table>
      <h4>Other graph kinds ({L.other.length} operations; own native libraries)</h4>
      <details><summary>Show</summary>
        <table className="compat"><thead><tr><th>Operation</th><th>Graph kind</th><th>Native backend</th><th>Explain</th><th>Dedicated view</th><th>Test files</th></tr></thead>
          <tbody>{L.other.map((r) => <tr key={r.type}><td><code>{r.type}</code></td><td>{r.graphKind}</td><td>{r.backend}</td><td>{r.explain ? "yes" : "no"}</td><td>{r.view ?? "-"}</td><td>{r.testFiles.length}</td></tr>)}</tbody></table>
      </details>
      <div className="small muted">{L.templateNote}</div>
    </div>
  );
}
