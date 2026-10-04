import { useEffect, useState } from "react";
import { api } from "../api";
import { useInspect } from "../hooks";
import type { Graph, ProcedureRunSummary } from "../types";
import { css, fmtNum, viridis } from "../util";
import { ColorBar, Heatmap } from "./Heatmap";
import { NotRecorded } from "./Provenance";

function Bars({ values, mask, label, signed = true, highlight }: { values: (number | null)[]; mask?: boolean[]; label: string; signed?: boolean; highlight?: number }) {
  const finite = values.filter((v): v is number => v !== null && Math.abs(v) < 1e8);
  const m = Math.max(1e-9, ...finite.map((v) => Math.abs(v)));
  return (
    <div className="bars" role="img" aria-label={label}>
      {values.map((v, i) => {
        const big = v === null || Math.abs(v) >= 1e8;
        const h = big ? 0 : (Math.abs(v!) / m) * 34;
        return (
          <div key={i} className={`bar ${mask && mask[i] === false ? "masked" : ""} ${highlight === i ? "hl" : ""}`} title={`${i}: ${v === null ? "non-finite" : big ? `${v} (masked fill)` : fmtNum(v)}${mask && mask[i] === false ? " · masked key" : ""}`}>
            <div className="pos" style={{ height: signed && v !== null && v < 0 ? 0 : h }} />
            {signed && <div className="neg" style={{ height: v !== null && v < 0 ? h : 0 }} />}
          </div>
        );
      })}
    </div>
  );
}

export function AttentionWorkspace({ graph, runs, setMessage }: { graph: Graph; runs: ProcedureRunSummary[]; setMessage: (m: string) => void }) {
  const [insts, setInsts] = useState<{ path: string; module: string; roles: string[]; missing: string[] }[]>([]);
  const [instance, setInstance] = useState("");
  const [runId, setRunId] = useState("");
  const [sample, setSample] = useState(0);
  const [head, setHead] = useState(0);
  const [token, setToken] = useState(0);
  useEffect(() => {
    const ctl = new AbortController();
    api.post<{ instances: typeof insts }>("/api/attention/instances", { graph }, undefined, ctl.signal).then((r) => { setInsts(r.instances); if (!r.instances.some((i) => i.path === instance)) setInstance(r.instances[0]?.path ?? ""); }).catch(() => {});
    return () => ctl.abort();
  }, [JSON.stringify(graph.modules), JSON.stringify(graph.nodes)]); // eslint-disable-line react-hooks/exhaustive-deps
  const req = instance ? { graph, instance, sample, head, token, runId: runId || undefined } : null;
  const r = useInspect<any>(req ? "/api/attention/inspect" : null, req);
  const d = r.data;
  const T: number = d?.seqLen ?? 0;
  const w = d?.weights;
  const toks: number[] | undefined = d?.tokens;
  void setMessage;
  return (
    <div className="attention">
      <div className="actions">
        <label>Attention block <select aria-label="attention instance" value={instance} onChange={(e) => setInstance(e.target.value)}>{insts.length === 0 && <option value="">none in this graph</option>}{insts.map((i) => <option key={i.path} value={i.path}>{i.path} ({i.module})</option>)}</select></label>
        <label>Weights <select aria-label="attention run" value={runId} onChange={(e) => setRunId(e.target.value)}><option value="">initial (not trained)</option>{[...runs].reverse().map((x) => <option key={x.id} value={x.id}>run {x.id} · latest checkpoint</option>)}</select></label>
        <label>Example <input type="number" aria-label="sample" min={0} value={sample} onChange={(e) => setSample(Math.max(0, Number(e.target.value) || 0))} /></label>
        <label>Head <input type="number" aria-label="head" min={0} max={(d?.heads ?? 1) - 1} value={head} onChange={(e) => setHead(Math.max(0, Number(e.target.value) || 0))} /></label>
        <label>Token <input type="number" aria-label="token" min={0} max={Math.max(0, T - 1)} value={token} onChange={(e) => setToken(Math.max(0, Number(e.target.value) || 0))} /></label>
      </div>
      {!insts.length && <NotRecorded message="This graph has no attention block the inspector understands (an instance of a module with inner nodes 'weights' and 'scores', like multi_head_attention). Nothing is shown rather than a guessed structure." />}
      {r.loading && <div className="muted">running the instrumented forward pass…</div>}
      {r.error && <div className="error pre">{r.error}</div>}
      {d && !d.available && <div className="notrec">{d.message}</div>}
      {d?.available && (
        <>
          <div className="notice">{d.weightsNote}. {d.dataNote}. {d.mode}. <b>Attention weights are a visualisation of an operation, not an explanation of why the output changed</b>; a causal claim needs an intervention (use the debugger's sandbox).</div>
          <div className="toks" aria-label="tokens">
            {Array.from({ length: T }, (_, i) => {
              const wt = w?.row?.[i] ?? 0;
              const allowed = d.mask?.keysAllowed?.[i] !== false;
              return <button key={i} className={`tok ${i === token ? "sel" : ""} ${allowed ? "" : "pad"}`} style={{ background: allowed ? css(viridis(Math.min(1, wt * 2))) : undefined, color: allowed && wt > 0.35 ? "#fff" : undefined }} onClick={() => setToken(i)} title={`token ${i}${toks ? ` (id ${toks[i]})` : ""}: weight from token ${token} = ${fmtNum(wt)}${allowed ? "" : " · masked (padding)"}`}>{toks ? (toks[i] === 0 ? "pad" : toks[i]) : i}</button>;
            })}
          </div>
          <div className="muted small">Colour = attention weight of the selected token (query {token}) on each token (key), head {head}. Select a token to move through the views.</div>
          <div className="attgrid">
            <div>
              <h4>Weights, head {head} (rows: query token, columns: key token)</h4>
              {w && <><Heatmap values={w.matrix} scale={{ mode: "sequential", min: 0, max: 1 }} cell={Math.max(10, Math.min(28, Math.floor(260 / T)))} title="attention weights" /><ColorBar scale={{ mode: "sequential", min: 0, max: 1 }} /></>}
              <div className="small">row {token} sums to {w ? fmtNum(w.rowSum, 7) : "n/a"} {d.mask && <>· keys allowed: {d.mask.keysAllowed.map((a: boolean) => (a ? "✓" : "✗")).join("")}</>}</div>
            </div>
            <div>
              <h4>For query token {token} (head {head})</h4>
              {d.input && <div><b>input</b> (from {d.input.producer}, {d.input.row.length} values) <Bars values={d.input.row} label="input row" /></div>}
              {d.projections && <>
                <div><b>Q</b> (d_head {d.headDim}) <Bars values={d.projections.q} label="query" /></div>
                <div><b>K</b> of every token <Heatmap values={d.projections.k} scale={{ mode: "diverging", min: -1, max: 1 }} cell={8} title="keys: rows are tokens" /></div>
                <div><b>V</b> of every token <Heatmap values={d.projections.v} scale={{ mode: "diverging", min: -1, max: 1 }} cell={8} title="values: rows are tokens" /></div>
              </>}
              {d.scores && <div><b>scores</b> q·k/√d_head (before the mask) <Bars values={d.scores.row} mask={d.mask?.keysAllowed} label="scores" /></div>}
              {d.mask && <div><b>mask</b> {d.mask.meaning}</div>}
              {d.maskedScores && <div><b>masked scores</b> (masked keys carry the fill value) <Bars values={d.maskedScores.row} mask={d.mask?.keysAllowed} label="masked scores" /></div>}
              {w && <div><b>weights</b> softmax over keys <Bars values={w.row} mask={d.mask?.keysAllowed} signed={false} label="weights" /></div>}
              {d.context && <div><b>context</b> = Σ weight·V <Bars values={d.context.row} label="context" /><span className="muted small"> recomputed from the captured weights and values: max |diff| {d.context.maxAbsDiffToCaptured != null ? d.context.maxAbsDiffToCaptured.toExponential(1) : "n/a"}</span></div>}
              {d.merged && <div><b>all heads merged</b> <Bars values={d.merged.row} label="merged heads" /></div>}
              {d.output && <div><b>attention output</b> (after W_o) <Bars values={d.output.row} label="attention output" /></div>}
            </div>
          </div>
          <h4>Not available</h4>
          <ul className="small">
            {Object.entries(d.unavailable ?? {}).map(([k, v]) => <li key={k}><b>{k}</b>: {String(v)}</li>)}
            <li><b>key/value cache</b>: {d.cacheOrKvState.reason}</li>
            {Object.keys(d.unavailable ?? {}).length === 0 && <li>Every internal this inspector looks for was exposed by this module and captured.</li>}
          </ul>
          <div className="muted small">{d.provenance.kind} — {d.provenance.captured}. {d.residual?.note ?? ""}</div>
        </>
      )}
    </div>
  );
}
