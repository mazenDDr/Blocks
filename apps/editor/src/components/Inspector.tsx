import { useEffect, useState, type ReactNode } from "react";
import { usePolling, useInspect } from "../hooks";
import type { ActivationsResult, Checkpoint, Diagnostic, GEdge, GNode, Graph, NodeView, OpInfo, AnyRun, Sample, Unavailable, Validation } from "../types";
import { dtypeOf, fmtNum, fmtShape, runLabel, shortHash } from "../util";
import { ConfigForm } from "./ConfigForm";
import { ConnectorConfigEditor, isConnectorOp } from "./Connectors";
import { ActivationsTab, ArchitectureTab, type Ctx, ExplainTab, FeatureMaps, WeightsTab } from "./InspectorTabs";
import { CompositeForm, RepeatForm, SelectForm, SharingRow } from "./StructuralForms";
import { CodeNodeForm } from "./CodeNodeForm";
import { NotRecorded, ProvLine } from "./Provenance";

/** Run / checkpoint / validation-sample selection used by the Weights, Activations and wire views. */
export function useInspectionData(runId: string | null) {
  const cks = usePolling<{ checkpoints: Checkpoint[] }>(runId ? `/api/runs/${runId}/checkpoints` : null, 3000);
  const [haveSamples, setHave] = useState(false);
  useEffect(() => setHave(false), [runId]);
  const smp = usePolling<{ available: boolean; samples: Sample[]; classes: string[] | null }>(runId ? `/api/runs/${runId}/samples` : null, haveSamples ? 0 : 3000);
  useEffect(() => { if (smp.data?.available) setHave(true); }, [smp.data]);
  return { checkpoints: cks.data?.checkpoints ?? [], samples: smp.data?.samples ?? [] };
}

export function InspectionBar({ runs, ctx, setCtx, currentHash, checkpoints, samples }: {
  runs: AnyRun[]; ctx: Ctx; setCtx: (c: Ctx) => void; currentHash?: string; checkpoints: Checkpoint[]; samples: Sample[];
}) {
  const run = runs.find((r) => r.id === ctx.runId);
  return (
    <div className="ctxbar">
      <div className="ctxrow">
        <label>Run
          <select value={ctx.runId ?? ""} onChange={(e) => setCtx({ runId: e.target.value || null, step: null, sample: ctx.sample })}>
            <option value="">(none)</option>
            {[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{runLabel(r)}</option>)}
          </select>
        </label>
        <label>Checkpoint
          <select value={ctx.step ?? ""} disabled={!checkpoints.length} onChange={(e) => setCtx({ ...ctx, step: e.target.value === "" ? null : Number(e.target.value) })}>
            <option value="">latest</option>
            {checkpoints.map((c) => <option key={c.sha256} value={c.step}>step {c.step}{c.epoch != null ? ` (epoch ${c.epoch})` : ""} {c.status}</option>)}
          </select>
        </label>
        <label>Val sample
          <select value={ctx.sample ?? ""} disabled={!samples.length} onChange={(e) => setCtx({ ...ctx, sample: e.target.value === "" ? null : Number(e.target.value) })}>
            <option value="">(none)</option>
            {samples.map((s) => <option key={s.index} value={s.index}>#{s.index} {s.id}</option>)}
          </select>
        </label>
      </div>
      {run && currentHash && run.graphHash !== currentHash && (
        <div className="warn">This run used graph {shortHash(run.graphHash)}; the draft is {shortHash(currentHash)}. Values below belong to the old architecture and are not mapped onto the draft.</div>
      )}
      {!run && <div className="muted small">No run selected: value tabs show "not recorded".</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- wire
export function WireInspector({ edge, graph, validation, ctx, ops }: { edge: GEdge; graph: Graph; validation: Validation | null; ctx: Ctx; ops: Record<string, OpInfo> }) {
  const src = graph.nodes.find((n) => n.id === edge.from.node), dst = graph.nodes.find((n) => n.id === edge.to.node);
  const t = validation?.nodes[edge.from.node]?.outputShapes?.[edge.from.port];
  const dstDiag: Diagnostic[] = (validation?.nodes[edge.to.node]?.diagnostics ?? []).filter((d) => d.port === edge.to.port);
  const ready = ctx.runId != null && ctx.sample != null && !!t;
  const state = useInspect<ActivationsResult | Unavailable>(ready ? `/api/runs/${ctx.runId}/inspect` : null,
    { kind: "activations", node: edge.from.node, checkpointStep: ctx.step, sample: ctx.sample, limit: 4 });
  return (
    <div className="wire">
      <h3>Wire {edge.id}</h3>
      <table><tbody>
        <tr><td>Producer</td><td>{src ? `${ops[src.type]?.displayName ?? src.type} ${src.id}` : edge.from.node}.{edge.from.port}</td></tr>
        <tr><td>Consumer</td><td>{dst ? `${ops[dst.type]?.displayName ?? dst.type} ${dst.id}` : edge.to.node}.{edge.to.port}</td></tr>
        <tr><td>Shape</td><td>{t ? `${fmtShape(t)} ${dtypeOf(t)}` : "unknown (producer has errors)"}</td></tr>
        <tr><td>Draft graph</td><td>{shortHash(validation?.graphHash)}</td></tr>
      </tbody></table>
      {dstDiag.map((d, i) => <div key={i} className="errbadge"><b>{d.code}</b> {d.message}</div>)}
      <h4>Value crossing this wire</h4>
      {!ctx.runId && <NotRecorded message="No run selected. Values are only shown from a recorded run; nothing is simulated." />}
      {ctx.runId && ctx.sample == null && <NotRecorded message="Pick a validation sample in the inspection context to see the real tensor for that sample." />}
      {ready && state.error && <div className="error">{state.error}</div>}
      {ready && state.data && !state.data.available && <><NotRecorded message={(state.data as Unavailable).message} /><ProvLine p={(state.data as Unavailable).provenance} /></>}
      {ready && state.data?.available && (
        <div className="valuepanel">
          <table><tbody>
            <tr><td>min / max</td><td>{fmtNum(state.data.stats.min)} / {fmtNum(state.data.stats.max)}</td></tr>
            <tr><td>mean / std</td><td>{fmtNum(state.data.stats.mean)} / {fmtNum(state.data.stats.std)}</td></tr>
            <tr><td>elements</td><td>{state.data.stats.count}</td></tr>
          </tbody></table>
          <div className="muted small">First {state.data.slice.limit} of {state.data.slice.of} {state.data.layout === "feature_maps" ? "channels" : "values"}:</div>
          <FeatureMaps d={state.data} />
          <ProvLine p={state.data.provenance} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- shell
const TABS = ["Config", "Architecture", "Weights", "Activations", "Explain"] as const;

export interface InnerActions {
  /** the selected node lives inside an expanded instance: edits go to the module definition */
  inner?: { module: string; version: string; instances: number; local: string } | null;
  allViews?: Record<string, NodeView>;
  onOpenModule?: () => void; onToggleExpand?: () => void; expanded?: boolean; onShare?: (target: string | null) => void; onEditCode?: (id: string, version: string) => void;
  moduleEditing?: boolean;
}

export function NodeInspector({ node, op, ops, view, graph, ctx, onConfig, onConnect, onDelete, onRename, tabSet, extra }: {
  extra?: InnerActions;
  node: GNode; op?: OpInfo; ops: Record<string, OpInfo>; view?: NodeView; graph: Graph; ctx: Ctx;
  /** Replaces the model-graph tabs (Architecture, Weights, ...) with graph-kind specific ones; "Config" is always first. */
  tabSet?: { names: string[]; render: (tab: string) => ReactNode };
  onConfig: (patch: Record<string, unknown>) => void;
  onConnect: (toNode: string, toPort: string, from: { node: string; port: string } | null) => void;
  onDelete: () => void; onRename: (newId: string) => string | null;
}) {
  const [tab, setTab] = useState<string>("Config");
  const [name, setName] = useState(node.id);
  const [nameErr, setNameErr] = useState<string | null>(null);
  useEffect(() => { setName(node.id); setNameErr(null); }, [node.id]);
  const diags = view?.diagnostics ?? [];
  const inner = extra?.inner ?? null;
  const structural = ["core.composite", "core.repeat", "core.select"].includes(node.type);
  const inPorts = view?.inputPorts ?? op?.inputs ?? [];
  const portsOf = (id: string, type: string) => extra?.allViews?.[id]?.outputPorts ?? ops[type]?.outputs ?? [];
  return (
    <div className="node-inspector">
      <div className="ni-head">
        <div><b>{op?.displayName ?? node.type}</b> <span className="badge">{op?.backend ?? "unresolved"}</span></div>
        {!inner && <button className="danger" onClick={onDelete}>Delete node</button>}
      </div>
      {inner && <div className="notice">Inner node <code>{node.id}</code> of module <b>{inner.module}</b> v{inner.version} (<code>{inner.local}</code>). Edits here change the module definition, so they apply to all {inner.instances} instance(s) of it.</div>}
      {(view?.sharedWith || node.sharedWith) && <div className="notice shared">{"\u{1F517}"} Shares parameters with <b>{view?.sharedWith ?? node.sharedWith}</b>: the very same tensors, not a copy.</div>}
      {(node.type === "diag.probe" || node.type === "diag.histogram" || node.type === "diag.timer") && <div className="muted small"><b>Observation-only</b>: passes the value through unchanged; records only while a capture covers it.</div>}
      {node.type === "diag.assert" && <div className="warn"><b>Execution-changing</b>: stops the run when the check fails (only when assertions are enabled for the run).</div>}
      <div className="row"><label className="lbl">Node id</label>
        <input value={name} aria-label="node id" disabled={!!inner} onChange={(e) => setName(e.target.value)}
          onBlur={() => { if (name !== node.id) { const err = onRename(name); setNameErr(err); if (err) setName(node.id); } }}
          onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }} />
      </div>
      {nameErr && <div className="error">{nameErr}</div>}
      <p className="muted small">{op?.purpose ?? "This operation is not available in the registry; the node is preserved but blocks execution."}</p>
      {diags.map((d, i) => (
        <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}>
          <b>{d.code}</b> {d.message}{d.port ? ` (port ${d.port})` : ""}
          {d.fixes.filter((f) => f.key).map((f, j) => <div key={j}><button onClick={() => onConfig({ [f.key!]: f.value })}>{f.label}</button></div>)}
        </div>
      ))}
      <div className="tabs" role="tablist">
        {(tabSet ? ["Config", ...tabSet.names] : [...TABS]).map((t) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}
      </div>
      <div className="tabbody">
        {tab === "Config" && op && (
          <>
            {structural && (
              <div className="actions">
                {extra?.onOpenModule && <button className="primary" onClick={extra.onOpenModule}>Open module</button>}
                {extra?.onToggleExpand && node.type !== "core.select" && <button onClick={extra.onToggleExpand}>{extra.expanded ? "Collapse" : "Expand in place"}</button>}
              </div>
            )}
            {node.type === "core.composite" ? <CompositeForm node={node} graph={graph} onConfig={onConfig} />
              : node.type === "core.repeat" ? <RepeatForm node={node} graph={graph} onConfig={onConfig} />
                : node.type === "core.select" ? <SelectForm node={node} graph={graph} onConfig={onConfig} />
                  : node.type === "code.block" ? <CodeNodeForm node={node} graph={graph} view={view} onConfig={onConfig} onEdit={extra?.onEditCode} />
                    : isConnectorOp(node.type)
                      ? <ConnectorConfigEditor node={node} onChange={onConfig} />
                      : <ConfigForm op={op} node={node} resolved={view?.resolvedConfig} onChange={onConfig} />}
            {!inner && !structural && node.type !== "code.block" && (view?.params ?? 0) > 0 || node.sharedWith ? <SharingRow node={node} graph={graph} onShare={(t) => extra?.onShare?.(t)} /> : null}
            {node.type === "tabular.train_validation_split" && <div className="muted small">Set both n_folds and fold for a k-fold partition (validation_fraction is then ignored).</div>}
            {inPorts.length > 0 && !inner && (
              <>
                <h4>Connections</h4>
                {inPorts.map((p) => {
                  const cur = graph.edges.find((e) => e.to.node === node.id && e.to.port === p);
                  const val = cur ? `${cur.from.node}.${cur.from.port}` : "";
                  return (
                    <div className="row" key={p}>
                      <label className="lbl">input {p}{op.inputKinds?.[p] && op.inputKinds[p] !== "tensor" ? <small> ({op.inputKinds[p]})</small> : null}</label>
                      <select aria-label={`source for ${p}`} value={val} onChange={(e) => {
                        if (!e.target.value) return onConnect(node.id, p, null);
                        const [n, ...rest] = e.target.value.split("."); onConnect(node.id, p, { node: n, port: rest.join(".") });
                      }}>
                        <option value="">(not connected)</option>
                        {graph.nodes.filter((n) => n.id !== node.id).flatMap((n) => portsOf(n.id, n.type).filter((po) => (ops[n.type]?.outputKinds?.[po] ?? "tensor") === (op.inputKinds?.[p] ?? "tensor")).map((po) => `${n.id}.${po}`)).map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    </div>
                  );
                })}
              </>
            )}
          </>
        )}
        {tab !== "Config" && tabSet && tabSet.render(tab)}
        {!tabSet && tab === "Architecture" && <ArchitectureTab node={node} view={view} />}
        {!tabSet && tab === "Weights" && <WeightsTab ctx={ctx} node={node} view={view} />}
        {!tabSet && tab === "Activations" && <ActivationsTab ctx={ctx} node={node} />}
        {!tabSet && tab === "Explain" && op && <ExplainTab op={op} view={view} />}
      </div>
    </div>
  );
}
