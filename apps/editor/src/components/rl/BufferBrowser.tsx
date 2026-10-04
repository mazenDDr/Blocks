import { useState } from "react";
import { fmtInt } from "../../util";
import { fmt, useGet } from "./common";
import type { BufferPage } from "./types";

export function BufferTab({ runId, onTrace }: { runId: string | null; onTrace: (tid: number) => void }) {
  const [offset, setOffset] = useState(0);
  const [sort, setSort] = useState("tid");
  const [order, setOrder] = useState("asc");
  const [flag, setFlag] = useState("any");
  const [source, setSource] = useState("any");
  const [episode, setEpisode] = useState("");
  const q = `offset=${offset}&limit=25&sort=${sort}&order=${order}&flag=${flag}&source=${source}${episode ? `&episode=${episode}` : ""}`;
  const st = useGet<BufferPage>(runId ? `/api/rl/runs/${runId}/buffer?${q}` : null);
  if (!runId) return <div className="empty pad">No run selected. The buffer browser reads the replay buffer recorded at the end of a run.</div>;
  if (st.error) return <div className="error pad pre">{st.error}</div>;
  const b = st.data;
  if (!b) return <div className="muted pad">{st.loading ? "loading…" : ""}</div>;
  const s = b.buffer;
  const comps = b.provenance.components;
  return (
    <div className="scroll pad rlbuf">
      <h3>Replay buffer <span className="badge">end of run</span></h3>
      <div className="rlstats">
        <span><b>{fmtInt(s.size)}</b> / {fmtInt(s.capacity)} stored</span><span><b>{fmtInt(s.transitionsInserted)}</b> inserted, {fmtInt(s.evicted)} overwritten</span>
        <span>policy versions {s.policyVersionRange[0]}–{s.policyVersionRange[1]} (final v{s.finalPolicyVersion})</span><span>age {s.ageRange[0]}–{s.ageRange[1]} env steps</span>
        <span>{s.terminated} terminated · {s.truncated} truncated</span><span>sampled {s.sampleCounts.min}–{s.sampleCounts.max}× (mean {fmt(s.sampleCounts.mean, 3)}); {s.sampleCounts.neverSampled} never</span>
      </div>
      <div className="small muted">Sampling: {s.samplingWeight}. {b.provenance.source}</div>
      <div className="arow2">sort <select aria-label="buffer sort" value={sort} onChange={(e) => { setSort(e.target.value); setOffset(0); }}>{["tid", "sampleCount", "age", "reward", "policyVersion", "episode"].map((x) => <option key={x}>{x}</option>)}</select>
        <select aria-label="buffer order" value={order} onChange={(e) => setOrder(e.target.value)}><option>asc</option><option>desc</option></select>
        ending <select aria-label="buffer flag" value={flag} onChange={(e) => { setFlag(e.target.value); setOffset(0); }}>{["any", "ending", "terminated", "truncated"].map((x) => <option key={x}>{x}</option>)}</select>
        action source <select aria-label="buffer source" value={source} onChange={(e) => { setSource(e.target.value); setOffset(0); }}>{["any", "random", "greedy"].map((x) => <option key={x}>{x}</option>)}</select>
        episode <input aria-label="buffer episode" type="number" value={episode} onChange={(e) => { setEpisode(e.target.value); setOffset(0); }} /></div>
      <div className="tablewrap tall"><table className="dtable" aria-label="buffer rows"><thead><tr><th>id</th><th>env</th><th>episode</th><th>step</th><th>policy v</th><th>age</th><th>source</th><th>action</th><th>reward</th>
        {comps.map((c) => <th key={c}>{c}</th>)}<th>term</th><th>trunc</th><th>sampled</th><th>last upd</th><th /></tr></thead>
        <tbody>{b.rows.map((r) => (
          <tr key={r.tid}><td className="num"><b>{r.tid}</b></td><td>{r.envIndex}</td><td className="num">{r.episodeId}</td><td className="num">{r.episodeStep}</td><td className="num">{r.policyVersion}</td><td className="num">{fmtInt(r.age)}</td>
            <td>{r.actionSource}{r.actionSource === "random" ? ` (ε ${fmt(r.epsilon, 2)})` : ""}</td><td className="num">{r.action}</td><td className="num">{fmt(r.reward)}</td>
            {comps.map((c) => <td key={c} className="num">{fmt(r.components[c])}</td>)}
            <td>{r.terminated ? "yes" : ""}</td><td>{r.truncated ? "yes" : ""}</td><td className="num">{r.sampleCount}</td><td className="num">{r.lastSampledUpdate >= 0 ? r.lastSampledUpdate : "—"}</td>
            <td><button onClick={() => onTrace(r.tid)}>Trace</button></td></tr>))}</tbody></table></div>
      <div className="pager"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 25))}>Previous</button> rows {b.total ? offset + 1 : 0}–{offset + b.rows.length} of {fmtInt(b.total)}
        <button disabled={offset + b.rows.length >= b.total} onClick={() => setOffset(offset + 25)}>Next</button></div>
    </div>
  );
}
