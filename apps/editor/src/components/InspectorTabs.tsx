import { useEffect, useState } from "react";
import { useInspect } from "../hooks";
import type { ActivationsResult, Explain, GNode, NodeView, OpInfo, Unavailable, WeightsResult } from "../types";
import { fmtInt, fmtNum, fmtShape, pair } from "../util";
import { ColorBar, Heatmap, type ScaleSpec } from "./Heatmap";
import { NotRecorded, ProvLine } from "./Provenance";

export interface Ctx { runId: string | null; step: number | null; sample: number | null }

const NO_RUN = "No run yet — nothing recorded. Train the model from the run panel, then pick the run in the inspection context above.";

function Result<T extends { available: true }>({ state, children }: { state: { data: (T | Unavailable) | null; loading: boolean; error: string | null }; children: (d: T) => React.ReactNode }) {
  if (state.error) return <div className="error">{state.error}</div>;
  if (!state.data) return <div className="muted">{state.loading ? "loading…" : ""}</div>;
  if (!state.data.available) {
    const u = state.data as Unavailable;
    return <><NotRecorded message={u.message} /><ProvLine p={u.provenance} /></>;
  }
  return <>{children(state.data as T)}{state.loading && <span className="muted"> refreshing…</span>}</>;
}

// ---------------------------------------------------------------------------------------- Weights
const clampCell = (cols: number, max = 12, width = 330) => Math.max(2, Math.min(max, Math.floor(width / Math.max(1, cols))));

function WeightView({ w, name }: { w: WeightsResult["params"][string]; name: string }) {
  const sc: ScaleSpec = { mode: "diverging", min: w.stats.min, max: w.stats.max };
  const v = w.values;
  let body: React.ReactNode;
  if (w.shape.length === 4) {
    const kw = w.shape[3];
    body = (
      <div className="filters">
        {(v as number[][][][]).map((filt, i) => (
          <div className="filter" key={i}>
            <small>filter {w.slice.offset + i}</small>
            <div className="slices">{filt.map((k, c) => <Heatmap key={c} values={k} scale={sc} cell={clampCell(kw * 1.6, 14, 60)} title={`${name}[${w.slice.offset + i}, ${c}]`} />)}</div>
          </div>
        ))}
      </div>
    );
  } else if (w.shape.length === 2) {
    body = <Heatmap values={v as number[][]} scale={sc} cell={clampCell(w.shape[1], 10)} title={`${name} rows ${w.slice.offset}..`} />;
  } else if (w.shape.length === 1) {
    body = <Heatmap values={[v as number[]]} scale={sc} cell={clampCell(w.shape[0], 14)} />;
  } else {
    body = <code>{JSON.stringify(v)}</code>;
  }
  return (
    <div className="valuepanel">
      <h4>{name} <small>shape [{w.shape.join(", ")}] · rows {w.slice.offset}–{w.slice.offset + w.slice.limit - 1} of {w.slice.of}</small></h4>
      {body}
      <ColorBar scale={sc} />
      <div className="muted small">Shared diverging scale over the whole tensor: range [{fmtNum(w.stats.min)}, {fmtNum(w.stats.max)}], mean {fmtNum(w.stats.mean)}, std {fmtNum(w.stats.std)}, n={fmtInt(w.stats.count)}. White = 0.</div>
    </div>
  );
}

export function WeightsTab({ ctx, node, view }: { ctx: Ctx; node: GNode; view?: NodeView }) {
  const [offset, setOffset] = useState(0);
  useEffect(() => setOffset(0), [node.id, ctx.runId, ctx.step]);
  const hasParams = (view?.params ?? 0) > 0;
  const state = useInspect<WeightsResult | Unavailable>(ctx.runId && hasParams ? `/api/runs/${ctx.runId}/inspect` : null,
    { kind: "weights", node: node.id, checkpointStep: ctx.step, offset, limit: node.type === "pytorch.nn.conv2d" ? 6 : undefined });
  if (!hasParams) return <div className="muted">This block has no learnable parameters, so there are no weights to show.</div>;
  if (!ctx.runId) return <NotRecorded message={NO_RUN} />;
  return (
    <Result state={state}>
      {(d: WeightsResult) => {
        const first = Object.values(d.params)[0];
        return (
          <>
            {Object.entries(d.params).map(([n, w]) => <WeightView key={n} name={n} w={w} />)}
            {first.slice.of > first.slice.limit && (
              <div className="pager">
                <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - first.slice.limit))}>Previous</button>
                <button disabled={first.slice.offset + first.slice.limit >= first.slice.of} onClick={() => setOffset(first.slice.offset + first.slice.limit)}>Next</button>
              </div>
            )}
            <ProvLine p={d.provenance} />
          </>
        );
      }}
    </Result>
  );
}

// ---------------------------------------------------------------------------------------- Activations
export function FeatureMaps({ d }: { d: ActivationsResult }) {
  if (d.layout === "vector") {
    const sc: ScaleSpec = { mode: "diverging", min: d.stats.min, max: d.stats.max };
    return <><Heatmap values={[d.values as number[]]} scale={sc} cell={clampCell(d.values.length, 14)} /><ColorBar scale={sc} /></>;
  }
  const sc: ScaleSpec = { mode: "sequential", min: d.stats.min, max: d.stats.max };
  const maps = d.values as number[][][];
  const cell = Math.max(1, Math.floor(76 / maps[0][0].length));
  return (
    <>
      <div className="fmaps">
        {maps.map((m, i) => (
          <figure key={i}><Heatmap values={m} scale={sc} cell={cell} title={`channel ${d.slice.offset + i}`} /><figcaption>ch {d.slice.offset + i}</figcaption></figure>
        ))}
      </div>
      <ColorBar scale={sc} />
    </>
  );
}

export function ActivationsTab({ ctx, node }: { ctx: Ctx; node: GNode }) {
  const [offset, setOffset] = useState(0);
  useEffect(() => setOffset(0), [node.id, ctx.runId, ctx.step, ctx.sample]);
  const ready = ctx.runId != null && ctx.sample != null;
  const state = useInspect<ActivationsResult | Unavailable>(ready ? `/api/runs/${ctx.runId}/inspect` : null,
    { kind: "activations", node: node.id, checkpointStep: ctx.step, sample: ctx.sample, offset, limit: 12 });
  if (!ctx.runId) return <NotRecorded message={NO_RUN} />;
  if (ctx.sample == null) return <NotRecorded message="Pick a validation sample in the inspection context above." />;
  return (
    <Result state={state}>
      {(d: ActivationsResult) => (
        <div className="valuepanel">
          <div className="muted small">Output of this node for one validation sample (a feature map is an <i>output</i> for this input, not a learned filter). Shape [{d.shape.join(", ")}].</div>
          <FeatureMaps d={d} />
          <div className="muted small">Shared sequential scale over the whole tensor: range [{fmtNum(d.stats.min)}, {fmtNum(d.stats.max)}], mean {fmtNum(d.stats.mean)}, std {fmtNum(d.stats.std)}. Showing {d.layout === "feature_maps" ? "channels" : "values"} {d.slice.offset}–{d.slice.offset + d.slice.limit - 1} of {d.slice.of}.</div>
          {d.slice.of > d.slice.limit && (
            <div className="pager">
              <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - d.slice.limit))}>Previous</button>
              <button disabled={d.slice.offset + d.slice.limit >= d.slice.of} onClick={() => setOffset(d.slice.offset + d.slice.limit)}>Next</button>
            </div>
          )}
          <ProvLine p={d.provenance} />
        </div>
      )}
    </Result>
  );
}

// ---------------------------------------------------------------------------------------- Explain
export function ExplainTab({ op, view }: { op: OpInfo; view?: NodeView }) {
  const e: Explain | undefined = view?.explain;
  if (!view?.typed || !e) return <div className="muted">Explanation needs a valid, typed node (fix the errors on the card first).</div>;
  if (e.error) return <div className="error">{e.error}</div>;
  return (
    <div className="explain">
      <p>{op.purpose}</p>
      {e.equation && <><h4>Equation</h4><pre>{e.equation}</pre></>}
      {e.shapeRule && <><h4>Shape rule</h4><pre>{e.shapeRule}</pre></>}
      {e.reduction && <><h4>Reduction</h4><p>{e.reduction}</p></>}
      {e.note && <p>{e.note}</p>}
      <h4>Parameters</h4>
      <pre>{e.parameters?.formula}</pre>
      {e.parameters && e.parameters.terms.length > 0 && (
        <table><tbody>
          {e.parameters.terms.map((t) => <tr key={t.name}><td>{t.name}</td><td>[{t.shape.join(", ")}]</td><td className="num">{fmtInt(t.count)}</td></tr>)}
          <tr><td colSpan={2}><b>total</b></td><td className="num"><b>{fmtInt(e.parameters.total)}</b></td></tr>
        </tbody></table>
      )}
      <div className="muted small">Computed from this node's current configuration and inferred shapes (draft graph).</div>
    </div>
  );
}

// ---------------------------------------------------------------------------------------- Architecture (schematic)
function Stack({ n, label, active, onPick }: { n: number; label: string; active?: number | null; onPick?: (i: number) => void }) {
  const shown = Math.min(n, 32);
  return (
    <div className="stack">
      <div className="stackcells">
        {Array.from({ length: shown }, (_, i) => (
          <button key={i} className={`cell ${active === i ? "on" : ""}`} onClick={() => onPick?.(i)} disabled={!onPick} title={`${label} ${i}`} aria-label={`${label} ${i}`} />
        ))}
      </div>
      <small>{n} {label}{n > shown ? ` (first ${shown} shown)` : ""}</small>
    </div>
  );
}

export function ArchitectureTab({ node, view }: { node: GNode; view?: NodeView }) {
  const [pick, setPick] = useState<number | null>(null);
  useEffect(() => setPick(null), [node.id]);
  const rc = (view?.resolvedConfig ?? {}) as Record<string, any>;
  const inT = view?.inputShapes ? Object.values(view.inputShapes)[0] : undefined;
  const outT = view?.outputShapes ? Object.values(view.outputShapes)[0] : undefined;
  return (
    <div className="arch">
      <div className="schematic-flag">Schematic — structure only. No learned or computed values are drawn here.</div>
      {!view?.typed && <div className="muted">Shapes are unknown until the node validates.</div>}
      {node.type === "pytorch.nn.conv2d" && view?.typed && (() => {
        const [kh, kw] = pair(rc.kernel_size);
        const cin = Number(rc.in_channels), cout = Number(rc.out_channels), g = Number(rc.groups);
        return (
          <>
            <div className="archrow">
              <Stack n={cin} label="input channels" />
              <div className="arrow">{"→"} {cout} filters, each {cin / g}×{kh}×{kw} {"→"}</div>
              <Stack n={cout} label="output channels (one per filter)" active={pick} onPick={setPick} />
            </div>
            <div className="muted small">Input {fmtShape(inT)} {"→"} output {fmtShape(outT)}. Click an output channel to expand its filter.</div>
            {pick !== null && (
              <div className="expand">
                <b>Filter {pick}</b>: {cin / g} kernel slices of {kh}×{kw} (one per input channel{g > 1 ? " in its group" : ""}) + {rc.bias ? "1 bias" : "no bias"} = {(cin / g) * kh * kw + (rc.bias ? 1 : 0)} parameters.
                <div className="slices">
                  {Array.from({ length: Math.min(cin / g, 16) }, (_, c) => <div key={c} className="blank" style={{ width: kw * 10, height: kh * 10 }} title={`W[${pick}, ${c}, :, :]`}><small>c{c}</small></div>)}
                </div>
                <div className="muted small">Boxes are empty on purpose: weights of this filter appear in the Weights tab once a run exists. A filter is learned; a feature map is what it produces for one input.</div>
              </div>
            )}
          </>
        );
      })()}
      {node.type === "pytorch.nn.linear" && view?.typed && (
        <div className="archrow"><Stack n={Number(rc.in_features)} label="input features" /><div className="arrow">{"→"} W [{rc.out_features}×{rc.in_features}] {"→"}</div><Stack n={Number(rc.out_features)} label="outputs" /></div>
      )}
      {node.type !== "pytorch.nn.conv2d" && node.type !== "pytorch.nn.linear" && view?.typed && (
        <div className="archrow"><div className="box">in {fmtShape(inT)}</div><div className="arrow">{"→"} {node.type.split(".").pop()} {"→"}</div><div className="box">out {fmtShape(outT)}</div></div>
      )}
    </div>
  );
}
