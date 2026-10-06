import { LineChart } from "../LineChart";
import { fmtInt } from "../../util";
import { fmt, useGet } from "./common";
import type { EvalReport, MeanCI } from "./types";

const ci = (m: MeanCI | undefined) => (m && m.mean != null ? `${fmt(m.mean)}${m.ci ? ` [${fmt(m.ci[0])}, ${fmt(m.ci[1])}]` : ""}` : "—");

export function EvalTab({ runId, status }: { runId: string | null; status: string | undefined }) {
  const st = useGet<EvalReport>(runId && status === "completed" ? `/api/rl/runs/${runId}/eval` : null, [status]);
  if (!runId) return <div className="empty pad">No run selected.</div>;
  if (status !== "completed") return <div className="empty pad">The evaluation report is written when the run completes (status: {status ?? "?"}). Intermediate greedy evaluations appear on the Run tab.</div>;
  if (st.error) return <div className="error pad pre">{st.error}</div>;
  const r = st.data;
  if (!r) return <div className="muted pad">loading…</div>;
  const f = r.final;
  const comps = Object.keys(f.componentReturns);
  return (
    <div className="scroll pad rleval">
      <h3>Evaluation report <span className="badge">run {r.runId.slice(0, 8)} · training seed {r.seed}</span></h3>
      <div className="small muted">Protocol: {f.episodes.length} {f.policy ?? `greedy episodes (ε=${f.epsilon})`}{f.policy ? " episodes" : ""} on separate environment instances, one per declared seed {JSON.stringify(r.evaluationSeeds)}; evaluation interactions never enter the replay buffer or the step count. {f.rewardBasis}.</div>
      <table className="kv" aria-label="evaluation summary"><tbody>
        <tr><td>task return (default reward)</td><td><b>{ci(f.taskReturn)}</b> <span className="muted">mean [95% t-interval over {f.taskReturn.n} episodes]</span></td></tr>
        <tr><td>return (training reward)</td><td>{ci(f.return)}</td></tr>
        <tr><td>episode length</td><td>{ci(f.length)}</td></tr>
        <tr><td>terminated / truncated</td><td>{f.terminatedCount} / {f.truncatedCount} of {f.episodes.length}</td></tr>
        <tr><td>task success</td><td>{f.successRate == null ? "no success rule" : `${(f.successRate * 100).toFixed(0)}%`} <span className="muted">({f.successRule})</span></td></tr>
        <tr><td>interactions / updates</td><td>policy version {f.policyVersion} after {f.update} gradient updates at environment step {fmtInt(f.tick)} · evaluated in {fmt(f.elapsedSec, 3)} s since start</td></tr>
      </tbody></table>
      <h4>Per-component return (mean over evaluation episodes)</h4>
      <table className="dtable"><thead><tr><th>component</th><th>weighted (training reward)</th><th>raw (unweighted)</th></tr></thead>
        <tbody>{comps.map((c) => <tr key={c}><td><b>{c}</b></td><td className="num">{ci(f.componentReturns[c])}</td><td className="num">{ci(f.rawComponentReturns[c])}</td></tr>)}</tbody></table>
      <div className="small muted">A high reward can come from a flawed reward definition: read the components and the success rate next to the return.</div>
      <h4>Evaluations during the run (unsmoothed; the first point is the untrained policy)</h4>
      <LineChart series={[{ id: "t", label: "task return mean", color: "#2a9d4b", points: r.history.filter((e) => e.taskReturn.mean != null).map((e) => [e.tick, e.taskReturn.mean as number]) },
        { id: "l", label: "episode length mean", color: "#8aa4d6", points: r.history.map((e) => [e.tick, e.length.mean as number]), dashed: true }]} xLabel="environment steps" yLabel="mean over evaluation seeds" height={170} />
      <h4>Episodes</h4>
      <div className="tablewrap"><table className="dtable"><thead><tr><th>seed</th><th>length</th><th>task return</th><th>training return</th><th>end</th><th>success</th></tr></thead>
        <tbody>{f.episodes.map((e) => <tr key={e.seed}><td className="num">{e.seed}</td><td className="num">{e.length}</td><td className="num">{fmt(e.taskReturn)}</td><td className="num">{fmt(e.return)}</td><td>{e.terminated ? "terminated" : "truncated"}</td><td>{e.success == null ? "—" : e.success ? "yes" : "no"}</td></tr>)}</tbody></table></div>
      <div className="small muted">Intervals describe the episodes of this single training run. For claims about a method, repeat the run with several seeds (Variants tab): the unit of replication is the run.</div>
    </div>
  );
}
