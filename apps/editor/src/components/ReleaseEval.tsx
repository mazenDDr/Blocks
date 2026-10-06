import { useState } from "react";
import { api, errorText } from "../api";
import { CasesEditor, EvalResult, parseCases } from "./agent/EvalTab";

const EXAMPLE = [{ id: "c1", input: { question: "SYNTHETIC question" }, checks: [{ field: "output", kind: "contains", value: "SYNTHETIC" }] }];

/** Evaluate the deployed release on labelled cases (ADR 0078): each case is a real recorded serving request as its own user. */
export function ReleaseEval({ releaseId, deployed, inputFields }: { releaseId: string; deployed: boolean; inputFields: string[] }) {
  const [text, setText] = useState(() => JSON.stringify(inputFields.length ? [{ ...EXAMPLE[0], input: Object.fromEntries(inputFields.map((f) => [f, "SYNTHETIC input"])) }] : EXAMPLE, null, 2));
  const [evalId, setEvalId] = useState<string | null>(null);
  const [status, setStatus] = useState("queued");
  const [err, setErr] = useState<string | null>(null);
  let cases: unknown[] | null = null, parseError: string | null = null;
  try { cases = parseCases(text); } catch (e) { parseError = (e as Error).message; }
  const live = !!evalId && !["completed", "failed", "cancelled"].includes(status);
  return (
    <section className="release-eval" aria-label="release evaluation">
      <h4>Evaluate this deployed release</h4>
      <p className="small muted">Each case is sent as a real request through the release's route, as a user and session unique to this evaluation and case (no conversation or memory is shared with real users). Checks read <code>output</code> (or <code>output.key</code> for JSON).</p>
      <CasesEditor text={text} setText={setText} />
      {parseError && <div className="error small">{parseError}</div>}
      <button disabled={!deployed || !!parseError || live} title={deployed ? "" : "Deploy the release first"} onClick={async () => {
        setErr(null);
        try { const r = await api.post<{ runId: string }>(`/api/production/releases/${releaseId}/evaluate`, { name: "release evaluation", cases }); setStatus("queued"); setEvalId(r.runId); }
        catch (e) { setErr(errorText(e)); }
      }}>Evaluate release</button>
      {err && <div className="error small pre">{err}</div>}
      {evalId && <EvalResult key={evalId} evalId={evalId} live={live} onStatus={setStatus} />}
    </section>
  );
}
