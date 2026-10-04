import { useEffect, useState } from "react";
import { fmt, useGet } from "./common";
import type { TraceOverview, TransitionTrace } from "./types";

export function TraceTab({ runId, tid, setTid }: { runId: string | null; tid: number | null; setTid: (t: number | null) => void }) {
  const [text, setText] = useState(tid == null ? "" : String(tid));
  useEffect(() => { setText(tid == null ? "" : String(tid)); }, [tid]);
  const ov = useGet<TraceOverview>(runId ? `/api/rl/runs/${runId}/trace` : null);
  const tr = useGet<TransitionTrace>(runId && tid != null ? `/api/rl/runs/${runId}/transitions/${tid}` : null);
  const [sel, setSel] = useState(0);
  useEffect(() => setSel(0), [tid]);
  if (!runId) return <div className="empty pad">No run selected. A transition trace is read from the bounded capture of a finished run.</div>;
  const t = tr.data;
  const use = t?.uses?.[sel];
  return (
    <div className="scroll pad rltrace">
      <h3>Transition → update trace</h3>
      <div className="small muted">{ov.data?.capturePolicy.statement ?? "BOUNDED CAPTURE: only declared transitions are tracked."}{ov.data && ` Dropped by caps: ${JSON.stringify(ov.data.dropped)}.`}</div>
      <div className="arow2">transition id <input aria-label="transition id" type="number" value={text} onChange={(e) => setText(e.target.value)} />
        <button onClick={() => setTid(text === "" ? null : Number(text))}>Trace</button>
        {ov.data && <select aria-label="tracked transitions" value="" onChange={(e) => e.target.value && setTid(Number(e.target.value))}>
          <option value="">tracked transitions ({ov.data.tracked.length})…</option>
          {ov.data.tracked.map((r) => <option key={r.tid} value={r.tid}>id {r.tid} · ep {r.episodeId} step {r.episodeStep} · {r.uses} use(s){r.terminated ? " · terminated" : r.truncated ? " · truncated" : ""}</option>)}</select>}</div>
      {tr.error && <div className="error pre">{tr.error}</div>}
      {t && !t.captured && <div className="warn" aria-label="not captured"><b>Not captured.</b> {t.reason}{t.finalBuffer && <div className="small">Final buffer record: {JSON.stringify(t.finalBuffer)}</div>}</div>}
      {t?.captured && t.transition && (
        <>
          <ol className="tracechain" aria-label="trace chain">{t.chain!.map((c) => <li key={c.stage}><b>{c.stage}</b><div>{c.text}</div></li>)}</ol>
          <div className="rlcols">
            <section>
              <h4>1 · The transition</h4>
              <table className="kv"><tbody>
                <tr><td>id / episode / step</td><td>{t.transition.tid} / {t.transition.episodeId} / {t.transition.episodeStep} (environment {t.transition.envIndex})</td></tr>
                <tr><td>observation s</td><td className="small"><code>[{t.transition.obs.map((x) => fmt(x, 3)).join(", ")}]</code></td></tr>
                <tr><td>action a</td><td>{t.transition.action} <span className="muted">({t.transition.actionSource}, ε {fmt(t.transition.epsilon, 3)}, policy v{t.transition.policyVersion}){t.transition.qValues ? ` · Q at acting time ${JSON.stringify(t.transition.qValues.map((v) => Number(v.toFixed(3))))}` : ""}</span></td></tr>
                <tr><td>reward r</td><td>{fmt(t.transition.reward)} = {Object.entries(t.transition.components).map(([k, v]) => `${k} ${fmt(v)}`).join(" + ")}</td></tr>
                <tr><td>next observation s′</td><td className="small"><code>[{t.transition.nextObs.map((x) => fmt(x, 3)).join(", ")}]</code> <span className="muted">(true next state)</span></td></tr>
                <tr><td>flags</td><td>terminated <b>{String(t.transition.terminated)}</b> · truncated <b>{String(t.transition.truncated)}</b>{t.transition.truncated && !t.transition.terminated && <span className="muted"> — a time limit, not a terminal state: the target still bootstraps</span>}</td></tr>
              </tbody></table>
              <h4>2 · Buffer insertion</h4>
              <table className="kv"><tbody>
                <tr><td>slot</td><td>{t.transition.slot} (ring buffer)</td></tr><tr><td>inserted at</td><td>environment step {t.transition.insertTick}</td></tr>
                <tr><td>overwritten</td><td>{t.transition.evicted ? `by id ${t.transition.evicted.byTid} at step ${t.transition.evicted.tick} (after update ${t.transition.evicted.afterUpdate})` : "not overwritten before the run ended"}</td></tr>
                <tr><td>sampled in total</td><td>{t.transition.sampleCount} times (last at update {t.transition.lastSampledUpdate}); {t.uses!.length} recorded in detail{t.usesDropped ? ` (${t.usesDropped} not recorded: cap)` : ""}</td></tr>
              </tbody></table>
            </section>
            <section>
              <h4>3 · Minibatches that sampled it</h4>
              {t.uses!.length === 0 && <div className="muted">It was never sampled into a minibatch before the run ended.</div>}
              <table className="dtable" aria-label="uses"><thead><tr><th>update</th><th>pos in batch</th><th>Q(s,a)</th><th>max Q_target(s′)</th><th>mask</th><th>target y</th><th>TD error</th><th>loss part</th><th>policy</th></tr></thead>
                <tbody>{t.uses!.map((u, i) => <tr key={u.update} className={i === sel ? "sel" : ""} onClick={() => setSel(i)} style={{ cursor: "pointer" }}>
                  <td className="num">{u.update}</td><td className="num">{u.batchPosition}/{u.batchSize}</td><td className="num">{fmt(u.qSA)}</td><td className="num">{fmt(u.nextValue)}</td><td className="num">{u.bootstrapMask}</td>
                  <td className="num">{fmt(u.target)}</td><td className="num">{fmt(u.tdError)}</td><td className="num">{fmt(u.lossContribution, 3)}</td><td>v{u.policyVersionBefore}→v{u.policyVersionAfter}</td></tr>)}</tbody></table>
              {use && (
                <div className="usedetail" aria-label="selected use">
                  <h4>4 · Objective and update #{use.update}</h4>
                  <pre>{`y = r + γ · mask · max_a' Q_target(s', a')\n  = ${fmt(t.transition.reward)} + γ · ${use.bootstrapMask} · ${fmt(use.nextValue)}  =  ${fmt(use.target)}\nTD error = Q(s,a) − y = ${fmt(use.qSA)} − ${fmt(use.target)} = ${fmt(use.tdError)}\nloss contribution = ${use.huberRegion ? `huber (${use.huberRegion} region)` : "squared error"} / B = ${fmt(use.lossContribution, 4)}   (batch loss ${fmt(use.batchLoss, 4)})\n∂L/∂Q(s,a) = ${fmt(use.dLossDQ, 4)}   gradient norm of the whole batch ${fmt(use.gradNorm, 4)}`}</pre>
                  <table className="kv"><tbody>
                    <tr><td>sampling probability</td><td>{fmt(use.samplingProbability, 3)} (uniform, batch {use.batchSize} of {use.minibatch?.bufferSize ?? "?"} stored)</td></tr>
                    <tr><td>target network</td><td>copy of the online network from update {use.targetVersion}</td></tr>
                    <tr><td>resulting policy version</td><td><b>v{use.policyVersionAfter}</b> {use.paramSha256After && <span className="muted small">parameters sha256 <code>{use.paramSha256After.slice(0, 16)}…</code></span>}</td></tr>
                    {use.minibatch && <tr><td>minibatch</td><td className="small">{use.minibatch.size} ids; tracked ones: {use.minibatch.tracked.join(", ")}{use.minibatch.targetSynced ? " · the target network was re-synced after this update" : ""}</td></tr>}
                  </tbody></table>
                  {use.minibatch && <details><summary className="small">all transition ids in the minibatch</summary><div className="small">{use.minibatch.batchTids.map((x) => <span key={x} className={x === t.tid ? "hit" : use.minibatch!.tracked.includes(x) ? "trk" : ""}>{x} </span>)}</div></details>}
                </div>
              )}
            </section>
          </div>
          <div className="small muted">{t.capturePolicy}</div>
        </>
      )}
    </div>
  );
}
