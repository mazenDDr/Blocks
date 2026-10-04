import { useMemo, useState } from "react";
import { LineChart, type Series } from "../LineChart";
import { fmtInt } from "../../util";
import { Check } from "../agent/common";
import { fmt } from "./common";
import type { Curves } from "./types";

function movingAvg(pts: [number, number][], w: number): [number, number][] {
  return pts.map((p, i) => { const a = pts.slice(Math.max(0, i - w + 1), i + 1); return [p[0], a.reduce((s, x) => s + x[1], 0) / a.length] as [number, number]; });
}

export function RunTab({ curves, status, summary }: { curves: Curves | null; status: string | undefined; summary: { envSteps: number; updates: number; totalSteps: number | null } | undefined }) {
  const [smooth, setSmooth] = useState(false);
  const [byTaskReturn, setByTask] = useState(false);
  const eps = curves?.episodes ?? [];
  const raw: [number, number][] = useMemo(() => eps.map((e) => [e.tick, byTaskReturn ? e.taskReturn : e.return]), [eps, byTaskReturn]);
  const series: Series[] = [{ id: "raw", label: `${byTaskReturn ? "task return" : "training return"} per episode (raw, unsmoothed)`, color: "#8aa4d6", points: raw }];
  if (smooth) series.push({ id: "ma", label: "moving average of 10 episodes (a display aid, not data)", color: "#c62828", points: movingAvg(raw, 10) });
  const evals: Series = { id: "eval", label: "greedy evaluation, task return (mean over evaluation seeds)", color: "#2a9d4b", points: (curves?.evals ?? []).filter((e) => e.taskReturn.mean != null).map((e) => [e.tick, e.taskReturn.mean as number]) };
  const ups = curves?.updates ?? [];
  const live = !!status && ["queued", "preparing", "running", "cancelling"].includes(status);
  const lastEval = curves?.evals[curves.evals.length - 1];
  return (
    <div className="scroll pad rlrun">
      {!curves && <div className="empty pad">No run selected. Press Run to train; everything below is read from events the worker records (no simulated curves).</div>}
      {curves && (
        <>
          <div className="rlstats" aria-label="run counters">
            <span><b>{fmtInt(summary?.envSteps ?? eps[eps.length - 1]?.tick ?? 0)}</b>{curves.totalSteps ? ` / ${fmtInt(curves.totalSteps)}` : ""} environment steps</span>
            <span><b>{fmtInt(summary?.updates ?? ups[ups.length - 1]?.update ?? 0)}</b> gradient updates</span>
            <span><b>{curves.targetSyncs}</b> target syncs</span>
            <span><b>{eps.length}</b> episodes</span>
            <span>seed <b>{curves.seed}</b></span>
            {live && <span className="badge">live</span>}
          </div>
          {curves.setup && <div className="small muted">autoreset mode {curves.setup.declaredAutoresetMode} (Gymnasium declares {curves.setup.autoresetMode}) · effective reward weights {JSON.stringify(curves.setup.effectiveWeights)} · policy network {fmtInt(curves.setup.networkParams)} parameters</div>}
          <h4>Training episode return (x: environment steps)</h4>
          <div className="small"><label><Check label="smooth" value={smooth} onChange={setSmooth} /> overlay a 10-episode moving average (raw points stay visible)</label>{" "}
            <label><Check label="task return" value={byTaskReturn} onChange={setByTask} /> show the task return (environment default reward) instead of the training reward</label></div>
          <LineChart series={[...series, ...(evals.points.length ? [evals] : [])]} xLabel="environment steps" yLabel="episode return" />
          <div className="small muted">Training returns come from the ε-greedy behavior policy; the green line is the separate greedy evaluation (own environments and seeds). They are different quantities.{lastEval && ` Latest evaluation: task return ${fmt(lastEval.taskReturn.mean)} (n=${lastEval.taskReturn.n}) at step ${fmtInt(lastEval.tick)}.`}</div>
          <div className="rlgrid">
            <div><h4>TD loss (every {curves.updates.length > 1 ? ups[1].update - ups[0].update : "log interval"} gradient steps)</h4>
              <LineChart series={[{ id: "loss", label: "batch TD loss", color: "#c62828", points: ups.map((u) => [u.update, u.loss]) }]} xLabel="gradient update" yLabel="loss" height={150} /></div>
            <div><h4>Mean Q(s,a) of the batch and mean TD target</h4>
              <LineChart series={[{ id: "q", label: "mean Q(s,a)", color: "#1f6feb", points: ups.map((u) => [u.update, u.meanQ]) }, { id: "t", label: "mean target y", color: "#d9730d", points: ups.map((u) => [u.update, u.meanTarget]), dashed: true }]} xLabel="gradient update" yLabel="value" height={150} /></div>
            <div><h4>ε and gradient norm</h4>
              <LineChart series={[{ id: "eps", label: "epsilon", color: "#2a9d4b", points: ups.map((u) => [u.update, u.epsilon]) }]} xLabel="gradient update" yLabel="epsilon" height={150} />
              <LineChart series={[{ id: "gn", label: "gradient norm (before clipping)", color: "#6b21a8", points: ups.map((u) => [u.update, u.gradNorm]) }]} xLabel="gradient update" yLabel="norm" height={150} /></div>
            <div><h4>Episode length and termination vs truncation</h4>
              <LineChart series={[{ id: "len", label: "episode length", color: "#334", points: eps.map((e) => [e.tick, e.length]) }]} xLabel="environment steps" yLabel="steps" height={150} />
              <div className="small">{eps.filter((e) => e.terminated).length} terminated · {eps.filter((e) => e.truncated).length} truncated (time limit) · every episode ends in exactly one of the two</div></div>
          </div>
          <details><summary className="small">Last episodes (raw rows)</summary>
            <table className="dtable"><thead><tr><th>step</th><th>env</th><th>episode</th><th>length</th><th>return</th><th>task return</th><th>end</th><th>policy v</th><th>ε</th></tr></thead>
              <tbody>{eps.slice(-12).reverse().map((e) => <tr key={e.seq}><td className="num">{e.tick}</td><td>{e.envIndex}</td><td>{e.episodeId}</td><td className="num">{e.length}</td><td className="num">{fmt(e.return)}</td><td className="num">{fmt(e.taskReturn)}</td>
                <td>{e.terminated ? "terminated" : "truncated"}</td><td className="num">{e.policyVersion}</td><td className="num">{fmt(e.epsilon, 3)}</td></tr>)}</tbody></table></details>
          <div className="prov">Provenance: run {curves.runId} · seed {curves.seed} · {curves.provenance.source}</div>
        </>
      )}
    </div>
  );
}
