import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, errorText } from "../api";
import { isTensorType, type Diagnostic, type GEdge, GNode, Graph, NodeView, OpInfo, TabularRunSummary, Validation } from "../types";
import { fmtShape, shortHash, uid } from "../util";
import { NotRecorded } from "./Provenance";
import { NodeResultView, SchemaList, TabProv, TabularExplain, TablePreview, VIEW_LABEL } from "./Tabular";
import { DomainResultView } from "./DomainViews";
import { ConnectorSourceView, JoinView } from "./Connectors";
import { CacheRetentionPolicy } from "./CacheRetentionPolicy";

const ACTIVE = ["queued", "preparing", "running", "cancelling"];

/** Tabs of the node inspector for a tabular node: the table it outputs, the view that matches its operation, and the explanation. */
export function tabularTabs(op: OpInfo | undefined, node: GNode, view: NodeView | undefined, runId: string | null, onConfig?: (patch: Record<string, unknown>) => void, refreshKey?: number): { names: string[]; render: (tab: string) => ReactNode } {
  const tablePorts = op ? op.outputs.filter((p) => op.outputKinds[p] === "table") : [];
  const kind = op?.summaryKind ?? "step";
  const resultTab = VIEW_LABEL[kind] ?? "Result";
  const names = [...(tablePorts.length ? ["Table"] : []), resultTab, "Explain"];
  return {
    names,
    render: (tab) => {
      if (tab === "Table") return <TableTab runId={runId} node={node.id} ports={tablePorts} />;
      if (tab === "Explain") return <TabularExplain explain={view?.explain} purpose={op?.purpose ?? ""} typed={!!view?.typed} />;
      if (kind === "connector_source") return <ConnectorSourceView runId={runId} node={node.id} pinned={(node.config.pin as string | null) ?? null} onPin={onConfig ? (id) => onConfig({ pin: id }) : undefined} />;
      if (kind === "join") return <JoinView runId={runId} node={node.id} />;
      if (op?.graphKind === "domain") return <DomainResultView key={`${runId}/${node.id}/${refreshKey}`} runId={runId} node={node.id} />;
      return <NodeResultView kind={kind} runId={runId} node={node.id} />;
    },
  };
}

function TableTab({ runId, node, ports }: { runId: string | null; node: string; ports: string[] }) {
  const [port, setPort] = useState(ports[0]);
  return (
    <div>
      {ports.length > 1 && <div className="tabs">{ports.map((p) => <button key={p} className={p === port ? "on" : ""} onClick={() => setPort(p)}>{p}</button>)}</div>}
      <TablePreview runId={runId} node={node} port={port} />
    </div>
  );
}

export function TabularWireInspector({ edge, graph, validation, runId, ops }: { edge: GEdge; graph: Graph; validation: Validation | null; runId: string | null; ops: Record<string, OpInfo> }) {
  const src = graph.nodes.find((n) => n.id === edge.from.node), dst = graph.nodes.find((n) => n.id === edge.to.node);
  const t = validation?.nodes[edge.from.node]?.outputShapes?.[edge.from.port];
  const dstDiag: Diagnostic[] = (validation?.nodes[edge.to.node]?.diagnostics ?? []).filter((d) => d.port === edge.to.port);
  const sop = src ? ops[src.type] : undefined;
  return (
    <div className="wire">
      <h3>Wire {edge.id}</h3>
      <table><tbody>
        <tr><td>Producer</td><td>{src ? `${sop?.displayName ?? src.type} ${src.id}` : edge.from.node}.{edge.from.port}</td></tr>
        <tr><td>Consumer</td><td>{dst ? `${ops[dst.type]?.displayName ?? dst.type} ${dst.id}` : edge.to.node}.{edge.to.port}</td></tr>
        <tr><td>Wire kind</td><td><span className="badge">{edge.kind}</span> <span className="muted small">a {edge.kind} wire never means a tensor or another kind</span></td></tr>
        <tr><td>Static type</td><td>{t ? fmtShape(t) : "unknown (producer has errors)"}{t && !isTensorType(t) && t.partition ? <span className={`badge part-${t.partition}`}>{t.partition}</span> : null}</td></tr>
      </tbody></table>
      {t && <SchemaList t={t} max={12} />}
      {dstDiag.map((d, i) => <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}><b>{d.code}</b> {d.message}</div>)}
      <h4>Value crossing this wire</h4>
      {edge.kind === "table"
        ? <TablePreview runId={runId} node={edge.from.node} port={edge.from.port} />
        : runId ? (sop?.graphKind === "domain" ? <DomainResultView runId={runId} node={edge.from.node} /> : <NodeResultView kind={sop?.summaryKind ?? "step"} runId={runId} node={edge.from.node} />) : <NotRecorded message="No run selected. Values are read from a recorded run, never simulated." />}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- run bar (right pane top)
export function TabularRunBar({ runs, runId, setRunId, currentHash }: { runs: TabularRunSummary[]; runId: string | null; setRunId: (id: string | null) => void; currentHash?: string }) {
  const run = runs.find((r) => r.id === runId);
  return (
    <div className="ctxbar">
      <div className="ctxrow">
        <label>Run
          <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value || null)} aria-label="inspected run">
            <option value="">(none)</option>
            {[...runs].reverse().map((r) => <option key={r.id} value={r.id}>{r.id} · {r.status} · {r.progress.nodesDone}/{r.progress.nodes} nodes</option>)}
          </select>
        </label>
      </div>
      {run && currentHash && run.graphHash !== currentHash && <div className="warn">This run used graph {shortHash(run.graphHash)}; the draft is {shortHash(currentHash)}. Values below belong to that earlier graph.</div>}
      {!run && <div className="muted small">No run selected: value tabs show “not recorded”.</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------------------- node cache retention
interface CacheSummary { entries: number; bytes: number; projects: { projectId: string | null; entries: number; bytes: number; nodes: number }[] }
const fmtBytes = (n: number) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.ceil(n / 1024)} KB`);

function CacheControls({ projectId, refresh }: { projectId: string; refresh: string }) {
  const [sum, setSum] = useState<CacheSummary | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const load = () => api.get<CacheSummary>("/api/cache/nodes").then(setSum).catch(() => setSum(null));
  useEffect(() => { load(); }, [projectId, refresh]);
  const mine = sum?.projects.find((p) => p.projectId === projectId);
  const prune = async (keepLatestPerNode: number | null) => {
    try {
      const preview = await api.post<{ removed: number; indexBytes: number }>("/api/cache/nodes/prune", { projectId, keepLatestPerNode, dryRun: true });
      if (!preview.removed) { setMsg("Nothing to remove."); return; }
      if (!window.confirm(`Remove ${preview.removed} cached node results (${fmtBytes(preview.indexBytes)}) for ${projectId}? Recorded runs are not affected; removed results are recomputed when next needed.`)) return;
      const r = await api.post<{ removed: number; bytesFreed: number }>("/api/cache/nodes/prune", { projectId, keepLatestPerNode, dryRun: false });
      setMsg(`Removed ${r.removed} entries, freed ${fmtBytes(r.bytesFreed)}.`); load();
    } catch (e) { setMsg(errorText(e)); }
  };
  return (
    <>{mine && <div className="small cachectl">
      Cached results for this project: {mine.entries} ({fmtBytes(mine.bytes)}, {mine.nodes} nodes){" "}
      <button className="link" onClick={() => prune(1)}>keep only the latest per node</button> · <button className="link" onClick={() => prune(null)}>clear</button>
      {msg && <span className="muted"> {msg}</span>}
    </div>}<CacheRetentionPolicy key={projectId} projectId={projectId} /></>
  );
}

// ---------------------------------------------------------------------------------------- node cache explanation
function CacheTable({ run, onSelectNode }: { run: TabularRunSummary; onSelectNode: (id: string) => void }) {
  const done = run.nodes.filter((n) => n.cache);
  const count = (s: string) => done.filter((n) => n.cache!.status === s).length;
  return (
    <details className="cachetable">
      <summary>Node cache: {count("hit")} reused · {count("miss")} executed and recorded · {count("bypass")} always run</summary>
      <table><thead><tr><th>Node</th><th>Result</th><th>Why</th></tr></thead><tbody>
        {done.map((n) => (
          <tr key={n.node}>
            <td><button className="link" onClick={() => onSelectNode(n.node)}>{n.node}</button></td>
            <td>{n.cache!.status === "hit" ? <>reused from run <code>{n.cache!.fromRun}</code></> : n.cache!.status === "miss" ? "executed" : "always runs"}</td>
            <td className="small">{n.cache!.reason}{n.cache!.stored ? ` (${n.cache!.stored})` : ""}</td>
          </tr>
        ))}
      </tbody></table>
    </details>
  );
}

// ---------------------------------------------------------------------------------------- bottom panel
export function TabularRunPanel({ projectId, graph, validation, runs, reloadRuns, runId, setRunId, ensureSaved, onSelectNode }: {
  projectId: string; graph: Graph; validation: Validation | null; runs: TabularRunSummary[]; reloadRuns: () => void;
  runId: string | null; setRunId: (id: string | null) => void; ensureSaved: () => Promise<void>; onSelectNode: (id: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [reuse, setReuse] = useState(false);
  const key = useRef<{ sig: string; id: string }>({ sig: "", id: "" });
  const run = runs.find((r) => r.id === runId) ?? null;
  const active = !!run && ACTIVE.includes(run.status);
  const blocked = !validation?.ok;
  const diags = validation?.diagnostics ?? [];

  async function start() {
    setBusy(true); setErr(null);
    try {
      await ensureSaved();
      const body = { projectId, config: reuse && graph.graphKind === "tabular" ? { cache: "reuse" } : {} };
      const sig = JSON.stringify(body) + validation?.graphHash;
      if (key.current.sig !== sig) key.current = { sig, id: uid() };
      const r = await api.post<{ runId: string }>("/api/runs", body, { "Idempotency-Key": key.current.id });
      key.current = { sig: "", id: "" };
      setRunId(r.runId); reloadRuns();
    } catch (e) { setErr(errorText(e)); } finally { setBusy(false); }
  }
  async function cancel() {
    if (!runId) return;
    try { await api.post(`/api/runs/${runId}/cancel`, {}); reloadRuns(); } catch (e) { setErr(errorText(e)); }
  }
  const exports = graph.nodes.filter((n) => n.type === "tabular.predictions_export");

  return (
    <div className="runpanel tabrun">
      <div className="tabrun-grid">
        <section>
          <h3>Run</h3>
          <div className="actions">
            <button className="primary" disabled={busy || blocked} onClick={start} title={blocked ? "Fix the graph errors first" : "Save the project and execute the graph in a worker process"}>Run graph</button>
            <button disabled={!active || run?.status === "cancelling"} onClick={cancel}>Cancel</button>
          </div>
          {graph.graphKind === "tabular" && (
            <label className="small" title="Reuse a recorded node result when its operation, settings, inputs, implementation and environment are unchanged. Sources are always re-read.">
              <input type="checkbox" checked={reuse} onChange={(e) => setReuse(e.target.checked)} /> Reuse unchanged node results (cache)
            </label>
          )}
          {graph.graphKind === "tabular" && <CacheControls projectId={projectId} refresh={runs.map((r) => r.id + r.status).join(",")} />}
          {err && <div className="error pre">{err}</div>}
          <h4>Problems ({diags.filter((d) => d.severity === "error").length} errors, {diags.filter((d) => d.severity === "warning").length} warnings)</h4>
          {diags.length === 0 && <div className="muted small">{graph.graphKind === "domain" ? "No contract violations. Training and evaluation use separate recorded partitions." : "No problems. Fit nodes read only the training partition."}</div>}
          {diags.map((d, i) => (
            <div key={i} className={`problem ${d.severity}`} onClick={() => d.nodeId && onSelectNode(d.nodeId)} role="button" tabIndex={0}>
              <b>{d.code}</b> <span className="muted">{d.nodeId}{d.port ? `.${d.port}` : ""}</span> <code className="small">{d.path}</code>
              <div>{d.message}</div>
              {d.fixes.map((f, j) => <div key={j} className="fix small">Fix: {f.label}</div>)}
            </div>
          ))}
        </section>
        <section>
          <h3>Runs</h3>
          {runs.length === 0 && <div className="muted">No runs yet for this project.</div>}
          <table><tbody>
            {[...runs].reverse().map((r) => (
              <tr key={r.id} className={r.id === runId ? "sel" : ""} onClick={() => setRunId(r.id)} style={{ cursor: "pointer" }}>
                <td>{r.id}</td><td><span className={`status ${r.status}`}>{r.status}</span></td><td>{r.progress.nodesDone}/{r.progress.nodes} nodes</td>
                <td>graph {shortHash(r.graphHash)}{validation && r.graphHash !== validation.graphHash && <span className="badge old">older graph</span>}</td>
              </tr>
            ))}
          </tbody></table>
        </section>
        <section>
          <h3>Run record</h3>
          {!run ? <NotRecorded message="No run selected." /> : (
            <div>
              <div><b>{run.id}</b> <span className={`status ${run.status}`}>{run.status}</span> · {run.progress.nodesDone}/{run.progress.nodes} nodes
                {run.libraries && <span className="muted small"> · {Object.entries(run.libraries).map(([k, v]) => `${k} ${v}`).join(", ")}</span>}</div>
              {run.failure && <div className="errbadge"><b>{run.failure.code}</b> at {run.failure.node}: {run.failure.message}</div>}
              {run.sources.map((s) => <div key={s.node} className="small"><b>data</b> {s.node}: {s.path} · {s.rows} rows · sha256 <code>{s.sha256.slice(0, 16)}…</code></div>)}
              {(run.snapshots ?? []).map((s) => <div key={s.node} className="small"><b>source</b> {s.node}: {s.connector} · {s.mode} · snapshot <code>{s.snapshotId.slice(0, 12)}…</code> · {s.rows} rows{s.reproducibility?.limited ? <span className="badge old">reproducibility limited</span> : null}</div>)}
              {run.config.seed != null && <div className="small"><b>run seed</b> {run.config.seed} (replaces the seed of every node that has one)</div>}
              {run.config.trial && <div className="small"><b>study trial</b> {String((run.config.trial as any).studyId)} / {String((run.config.trial as any).trialId)} · attempt {String((run.config.trial as any).attempt)} · seed {String((run.config.trial as any).seed ?? "—")} · fold {String((run.config.trial as any).fold ?? "—")}</div>}
              {run.splits.map((s) => <div key={s.node} className="small"><b>split</b> {s.node}: seed {s.seed}, validation fraction {s.validationFraction}, {s.nTrain} train / {s.nValidation} validation{s.stratifyBy ? `, stratified by ${s.stratifyBy}` : ""}{s.groupBy ? `, grouped by ${s.groupBy}` : ""}</div>)}
              <div className="nodes small">{run.nodes.map((n) => <span key={n.node} className={`runmark ${n.status}`} onClick={() => onSelectNode(n.node)} title={n.cache ? `${n.status} · cache ${n.cache.status}: ${n.cache.reason}` : n.status}>{n.node}{n.cache && n.cache.status !== "bypass" ? <span className={`badge cache-${n.cache.status}`}>{n.cache.status === "hit" ? "reused" : "ran"}</span> : null}</span>)}</div>
              {run.cache?.mode === "reuse" && <CacheTable run={run} onSelectNode={onSelectNode} />}
              {run.status === "completed" && exports.map((n) => (
                <div key={n.id}><a href={`/api/runs/${run.id}/tables/${n.id}/predictions.csv`} download>Download predictions CSV ({n.id})</a></div>
              ))}
              <TabProv p={{ runId: run.id, graphHash: run.graphHash }} label="events and artifacts are stored in the workbench" />
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
