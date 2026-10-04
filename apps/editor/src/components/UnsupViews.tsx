import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { useInspect } from "../hooks";
import type { Graph, TablePage } from "../types";
import { fmtInt, fmtNum } from "../util";
import { LineChart } from "./LineChart";
import { NotRecorded } from "./Provenance";
import { Scatter, type Pt } from "./Scatter";
import { TabProv, useNodeResult } from "./Tabular";
import { useGet } from "./rl/common";

/** The draft graph, so a view can offer the other clustering nodes of the same graph as colour sources. */
export const GraphContext = createContext<Graph | null>(null);

function KV({ rows }: { rows: [string, ReactNode][] }) {
  return <table className="kv"><tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>;
}

interface PointsRes { points: Pt[]; total: number; shown: number; downsampled: boolean; colorBy: { column: string; from: string } | null; provenance: { runId: string; graphHash: string; source: string } }

function PointsView({ runId, node, port, x, y, color, colorNode, colorPort, xLabel, yLabel, caption }: {
  runId: string; node: string; port: string; x: string; y: string; color?: string; colorNode?: string; colorPort?: string; xLabel?: string; yLabel?: string; caption?: ReactNode;
}) {
  const q = `node=${node}&port=${port}&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}${color ? `&color=${encodeURIComponent(color)}` : ""}${colorNode ? `&color_node=${colorNode}&color_port=${colorPort ?? "assignments"}` : ""}`;
  const st = useGet<PointsRes>(`/api/runs/${runId}/points?${q}`);
  if (st.error) return <div className="error pre">{st.error}</div>;
  if (!st.data) return <div className="muted">{st.loading ? "loading…" : ""}</div>;
  return (
    <div>
      <Scatter points={st.data.points} xLabel={xLabel ?? x} yLabel={yLabel ?? y} colorLabel={color ? `${color}${colorNode ? ` of ${colorNode}` : ""}` : undefined} />
      <div className="small muted">{fmtInt(st.data.shown)} of {fmtInt(st.data.total)} recorded rows{st.data.downsampled ? " (every k-th row)" : ""}. {caption}</div>
    </div>
  );
}

/** Column names of a recorded table (one-row page). */
function useColumns(runId: string | null, node: string, port: string): { name: string; dtype: string }[] {
  const st = useInspect<TablePage | { available: false }>(runId ? `/api/runs/${runId}/inspect` : null, runId ? { kind: "table", node, port, offset: 0, limit: 1 } : null);
  return st.data && st.data.available ? st.data.columns : [];
}

function Gate({ runId, st, children }: { runId: string | null; st: ReturnType<typeof useNodeResult>; children: (d: any, prov: any) => ReactNode }) {
  if (!runId) return <NotRecorded message="No run selected. Values here are read from a recorded run; nothing is simulated." />;
  if (st.error) return <div className="error pre">{st.error}</div>;
  if (!st.data) return <div className="muted">{st.loading ? "loading…" : ""}</div>;
  if (!st.data.available) return <><NotRecorded message={st.data.message} /><TabProv p={st.data.provenance} /></>;
  return <>{children(st.data.data, st.data.provenance)}</>;
}

// ------------------------------------------------------------------------------------------------ clustering (k-means, Gaussian mixture, DBSCAN)
export function ClusteringView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  const g = useContext(GraphContext);
  const [fx, setFx] = useState(0), [fy, setFy] = useState(1);
  return <Gate runId={runId} st={st}>{(d, prov) => {
    const feats: string[] = d.features;
    const isK = d.method === "kmeans", isG = d.method === "gmm", isD = d.method === "dbscan";
    void g;
    return (
      <div className="unsup">
        <div className="badge">{d.estimator} · scikit-learn {d.sklearn}</div>
        <KV rows={[
          ["rows × features", `${fmtInt(d.nRows)} × ${feats.length} (${d.scale === "standardize" ? "standardized on these rows" : "not scaled"})`],
          ...(isK ? [["inertia (objective)", <b key="i">{fmtNum(d.inertia, 6)}</b>] as [string, ReactNode], ["iterations", `${d.nIter} (converged: ${String(d.converged)}; best of ${d.params.n_init} initialisations: #${d.selectedInit})`] as [string, ReactNode]] : []),
          ...(isG ? [["log-likelihood", fmtNum(d.logLikelihood, 6)] as [string, ReactNode], ["BIC / AIC", `${fmtNum(d.bic, 6)} / ${fmtNum(d.aic, 6)}`] as [string, ReactNode], ["EM", `${d.nIter} iterations, converged ${String(d.converged)}`] as [string, ReactNode]] : []),
          ...(isD ? [["clusters / noise", `${d.nClusters} clusters, ${d.nNoise} noise points (${(d.noiseFraction * 100).toFixed(1)}%) — noise is not a cluster`] as [string, ReactNode], ["core / border", `${d.nCore} / ${d.nBorder}`] as [string, ReactNode]] : []),
          ["can assign new points", d.canPredict ? "yes (predict)" : <span key="p"><b>no</b> — {d.predictNote}</span>],
        ]} />
        {isK && <>
          <h4>Clusters (centres in original units)</h4>
          <table className="dtable" aria-label="clusters"><thead><tr><th>cluster</th><th>size</th><th>inertia share</th>{feats.map((f) => <th key={f}>{f}</th>)}</tr></thead>
            <tbody>{d.clusters.map((c: any) => <tr key={c.cluster}><td><b>{c.cluster}</b></td><td className="num">{c.size}</td><td className="num">{(c.inertiaShare * 100).toFixed(1)}%</td>{feats.map((f) => <td key={f} className="num">{fmtNum(c.center[f], 4)}</td>)}</tr>)}</tbody></table>
          <h4>Objective progression (replay of the selected initialisation)</h4>
          <LineChart series={[{ id: "in", label: "inertia after each Lloyd iteration", color: "#1f6feb", points: d.iterations.map((i: any) => [i.iteration, i.inertia] as [number, number]) }]} xLabel="iteration" yLabel="inertia" height={150} />
          <table className="dtable"><thead><tr><th>iteration</th><th>inertia</th><th>max centre shift</th><th>points reassigned</th></tr></thead>
            <tbody>{d.iterations.map((i: any) => <tr key={i.iteration}><td className="num">{i.iteration}</td><td className="num">{fmtNum(i.inertia, 6)}</td><td className="num">{i.maxCentroidShift == null ? "—" : fmtNum(i.maxCentroidShift, 4)}</td><td className="num">{i.reassigned ?? "—"}</td></tr>)}</tbody></table>
          <div className="small muted">{d.iterationNote} {d.emptyClusterNote}</div>
        </>}
        {isG && <>
          <h4>Components</h4>
          <table className="dtable"><thead><tr><th>component</th><th>weight</th><th>hard size</th>{feats.map((f) => <th key={f}>{f}</th>)}</tr></thead>
            <tbody>{d.components.map((c: any) => <tr key={c.component}><td><b>{c.component}</b></td><td className="num">{fmtNum(c.weight, 4)}</td><td className="num">{c.size}</td>{feats.map((f) => <td key={f} className="num">{fmtNum(c.mean[f], 4)}</td>)}</tr>)}</tbody></table>
          <div className="small muted">{d.assignmentRule}. {d.densityNote}</div>
        </>}
        {isD && <>
          <h4>k-distance curve (to choose eps)</h4>
          <LineChart series={[{ id: "kd", label: `distance to the ${d.params.min_samples}-th neighbour, sorted descending`, color: "#d9730d", points: d.kDistance.sortedDescending.map((v: number, i: number) => [i, v] as [number, number]) }]} xLabel="point (sorted)" yLabel="distance" height={150} />
          <div className="small muted">{d.statusRule}. {d.connectivity}</div>
        </>}
        <h4>Assignments ({fmtInt(d.nRows)} rows; the full table is on the Table tab)</h4>
        {runId && feats.length >= 2 ? (
          <>
            <div className="small">x <select aria-label="scatter x feature" value={fx} onChange={(e) => setFx(Number(e.target.value))}>{feats.map((f, i) => <option key={f} value={i}>{f}</option>)}</select>
              {" "}y <select aria-label="scatter y feature" value={fy} onChange={(e) => setFy(Number(e.target.value))}>{feats.map((f, i) => <option key={f} value={i}>{f}</option>)}</select></div>
            <PointsView runId={runId} node={node} port="assignments" x={feats[fx]} y={feats[fy]} color="cluster"
              caption="Two of the clustering's features, original units — a slice of the feature space, not a projection. Colours are cluster ids (arbitrary labels)." />
          </>
        ) : <div className="muted small">A scatter needs at least two features.</div>}
        <div className="small muted">{d.limits}</div>
        <TabProv p={prov} />
      </div>
    );
  }}</Gate>;
}

// ------------------------------------------------------------------------------------------------ PCA
export function PcaView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  const cols = useColumns(runId, node, "components");
  const [color, setColor] = useState("");
  return <Gate runId={runId} st={st}>{(d, prov) => {
    const ratio: number[] = d.scree.ratio, cum: number[] = d.scree.cumulative;
    const k = d.nComponentsKept;
    const others = cols.filter((c) => !c.name.startsWith("PC"));
    return (
      <div className="unsup">
        <KV rows={[["rows × features", `${fmtInt(d.nRows)} × ${d.features.length} (${d.scale === "standardize" ? "standardized" : "not scaled"}, then centered)`], ["components kept", `${k}${d.params.whiten ? " (whitened)" : ""}`],
          ["variance explained by kept", <b key="v">{(d.cumulativeRatio[k - 1] * 100).toFixed(2)}%</b>],
          ["components for 80 / 90 / 95 / 99%", Object.values(d.componentsNeeded).join(" / ")], ["can transform new points", "yes (the fitted components)"]]} />
        <h4>Scree (explained variance ratio per component and cumulative)</h4>
        <LineChart series={[{ id: "r", label: "explained variance ratio", color: "#1f6feb", points: ratio.map((v, i) => [i + 1, v] as [number, number]) }, { id: "c", label: "cumulative", color: "#2a9d4b", points: cum.map((v, i) => [i + 1, v] as [number, number]), dashed: true }]} xLabel="component" yLabel="ratio" height={160} yMin={0} />
        <h4>Component directions (loadings on the {d.scale === "standardize" ? "standardized " : ""}features)</h4>
        <table className="dtable" aria-label="loadings"><thead><tr><th>component</th><th>variance ratio</th>{d.features.map((f: string) => <th key={f}>{f}</th>)}</tr></thead>
          <tbody>{d.loadings.map((l: any) => <tr key={l.pc}><td><b>{l.pc}</b></td><td className="num">{(l.ratio * 100).toFixed(2)}%</td>{d.features.map((f: string) => <td key={f} className="num">{fmtNum(l.weights[f], 3)}</td>)}</tr>)}</tbody></table>
        <h4>Reconstruction error by number of components (mean squared error per entry)</h4>
        <LineChart series={[{ id: "mse", label: "reconstruction MSE (exact PCA inverse transform)", color: "#c62828", points: d.reconstruction.byComponents.map((r: any) => [r.components, r.mse] as [number, number]) }]} xLabel="components kept" yLabel="MSE" height={140} yMin={0} />
        <div className="small muted">{d.reconstruction.note}. With {k} components: {d.reconstruction.mseKept == null ? "not computed" : fmtNum(d.reconstruction.mseKept, 5)}. {d.centeringNote}</div>
        {runId && k >= 2 && <>
          <h4>Projection onto PC1 / PC2</h4>
          <div className="small">colour by <select aria-label="pca colour column" value={color} onChange={(e) => setColor(e.target.value)}><option value="">(none)</option>{others.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}</select>
            {color && <span className="muted"> — a recorded column of the input table; it was not used by PCA</span>}</div>
          <PointsView runId={runId} node={node} port="components" x="PC1" y="PC2" color={color || undefined} caption="PCA is a linear projection: distances between points are preserved only along the kept directions; variance explained is stated above." />
        </>}
        <div className="small muted">{d.limits}</div>
        <TabProv p={prov} />
      </div>
    );
  }}</Gate>;
}

// ------------------------------------------------------------------------------------------------ projection (t-SNE)
export function ProjectionView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  const g = useContext(GraphContext);
  const cols = useColumns(runId, node, "embedding");
  const [color, setColor] = useState("");
  const clusterNodes = useMemo(() => (g?.nodes ?? []).filter((n) => ["sklearn.kmeans", "sklearn.gaussian_mixture", "sklearn.dbscan"].includes(n.type)), [g]);
  return <Gate runId={runId} st={st}>{(d, prov) => {
    const others = cols.filter((c) => c.name !== "x" && c.name !== "y");
    const [src, col] = color.includes("|") ? color.split("|") : ["", color];
    return (
      <div className="unsup">
        <div className="projbanner" role="note"><b>{d.projectionStatus}.</b> Distances, cluster sizes and the gaps between groups in this plot are not evidence of structure in the original space. It is a view, not a result.</div>
        <KV rows={[["method", `${d.estimator} · perplexity ${d.params.perplexity} · PCA initialisation · seed ${d.params.seed}`], ["fitted data", `${fmtInt(d.nRows)} rows × ${d.features.length} features (${d.scale === "standardize" ? "standardized" : "not scaled"})`],
          ["KL divergence", fmtNum(d.klDivergence, 5)], ["trustworthiness@5", <b key="t">{fmtNum(d.trustworthiness5, 4)}</b>], ["can place new points", <b key="n">no</b>]]} />
        <div className="small muted">{d.trustworthinessNote} {d.transformNote}</div>
        {runId && <>
          <div className="small">colour by <select aria-label="projection colour" value={color} onChange={(e) => setColor(e.target.value)}><option value="">(none)</option>
            {others.map((c) => <option key={c.name} value={c.name}>{c.name} (input column)</option>)}
            {clusterNodes.map((n) => <option key={n.id} value={`${n.id}|cluster`}>cluster of {n.id} (fitted separately in the original space)</option>)}</select></div>
          <PointsView runId={runId} node={node} port="embedding" x="x" y="y" color={col || undefined} colorNode={src || undefined} colorPort="assignments" xLabel="t-SNE 1 (arbitrary axis)" yLabel="t-SNE 2 (arbitrary axis)"
            caption={<b>Non-metric projection: do not read distances or separation off this plot.</b>} />
        </>}
        <ul className="assump">{d.limits.map((l: string, i: number) => <li key={i}>{l}</li>)}</ul>
        <TabProv p={prov} />
      </div>
    );
  }}</Gate>;
}

// ------------------------------------------------------------------------------------------------ diagnostics report
const KIND_BADGE: Record<string, string> = { internal: "internal", external: "EXTERNAL (labels not used for fitting)", "model-based": "model-based", stability: "stability", descriptive: "descriptive" };

export function ClusterReportView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  return <Gate runId={runId} st={st}>{(d, prov) => {
    const internal = d.internal;
    return (
      <div className="unsup">
        <div className="policybanner" role="note">{d.policy}</div>
        <KV rows={[["method", `${d.method} · scikit-learn ${d.sklearn}`], ["evaluated on", `${fmtInt(d.n)} rows of the ${d.evaluatedOn.partition} partition${d.evaluatedOn.sameRowsAsFit ? " (the rows the model was fitted on)" : " (NEW rows: predictions, not the fit)"}`]]} />
        <h4>Metrics for this method</h4>
        <table className="dtable" aria-label="metrics"><thead><tr><th>metric</th><th>value</th><th>kind</th><th>reading</th></tr></thead><tbody>
          {Object.entries(d.values).map(([k, v]) => {
            const info = internal?.[k];
            return <tr key={k}><td><b>{k}</b></td><td className="num">{fmtNum(v as number, 6)}</td><td>{KIND_BADGE[d.valueKinds[k]] ?? d.valueKinds[k]}</td><td className="small">{info ? `${info.better} — ${info.assumption}` : ""}</td></tr>;
          })}
          {Object.entries(d.unavailable ?? {}).map(([k, why]) => <tr key={k} className="muted"><td><b>{k}</b></td><td>not applicable</td><td /><td className="small">{String(why)}</td></tr>)}</tbody></table>
        {internal?.notes?.length > 0 && <div className="small muted">{internal.notes.join("; ")}</div>}
        {d.clusterSizes && <div className="small">cluster sizes: {Object.entries(d.clusterSizes).map(([c, n]) => `${c === "-1" ? "noise" : c}: ${n}`).join(" · ")}</div>}
        {d.sweep && <>
          <h4>{d.method === "kmeans" ? "Elbow and silhouette over k" : "BIC / AIC over the number of components"}</h4>
          <LineChart series={d.method === "kmeans"
            ? [{ id: "in", label: "inertia", color: "#1f6feb", points: d.sweep.rows.map((r: any) => [r.k, r.inertia] as [number, number]) }]
            : [{ id: "b", label: "BIC (lower is better)", color: "#1f6feb", points: d.sweep.rows.map((r: any) => [r.k, r.bic] as [number, number]) }, { id: "a", label: "AIC (lower is better)", color: "#d9730d", points: d.sweep.rows.map((r: any) => [r.k, r.aic] as [number, number]), dashed: true }]}
            xLabel={d.method === "kmeans" ? "k" : "components"} yLabel={d.method === "kmeans" ? "inertia" : "criterion"} height={160} />
          <LineChart series={[{ id: "s", label: "silhouette (higher is better)", color: "#2a9d4b", points: d.sweep.rows.filter((r: any) => r.silhouette != null).map((r: any) => [r.k, r.silhouette] as [number, number]) },
            { id: "db", label: "Davies-Bouldin (lower is better)", color: "#c62828", points: d.sweep.rows.filter((r: any) => r.daviesBouldin != null).map((r: any) => [r.k, r.daviesBouldin] as [number, number]), dashed: true }]} xLabel="k" yLabel="score" height={150} />
          <div className="small muted">{d.sweep.note}</div>
        </>}
        {d.stability && d.method === "pca" && <>
          <h4>Stability of the components</h4>
          <table className="dtable"><thead><tr><th>component</th><th>mean |cos|</th><th>min |cos|</th></tr></thead><tbody>{d.stability.perComponent.map((p: any) => <tr key={p.pc}><td>{p.pc}</td><td className="num">{fmtNum(p.meanAbsCosine, 4)}</td><td className="num">{fmtNum(p.minAbsCosine, 4)}</td></tr>)}</tbody></table>
          <div className="small muted">{d.stability.statement}</div>
        </>}
        {d.stability && d.method !== "pca" && <>
          <h4>Stability (adjusted Rand index between partitions)</h4>
          {d.stability.seeds?.applicable === false ? <div className="small muted">Seeds: {d.stability.seeds.reason}</div> : d.stability.seeds && (
            <KV rows={[["seed refits", `${d.stability.seeds.runs} runs · pairwise ARI mean ${fmtNum(d.stability.seeds.pairwise.mean, 4)} · min ${fmtNum(d.stability.seeds.pairwise.min, 4)}`], ["", <span key="s" className="small muted">{d.stability.seeds.statement}</span>]]} />)}
          <KV rows={[["resampling", d.stability.resampling.scheme], ["ARI vs reference", `mean ${fmtNum(d.stability.resampling.mean, 4)} · min ${fmtNum(d.stability.resampling.min, 4)} · max ${fmtNum(d.stability.resampling.max, 4)} over ${d.stability.resampling.runs} runs`]]} />
          <LineChart series={[{ id: "ari", label: "ARI of each resample vs the reference partition", color: "#1f6feb", points: d.stability.resampling.ari.map((v: number, i: number) => [i + 1, v] as [number, number]) }]} xLabel="resample" yLabel="ARI" height={130} yMin={Math.min(0, ...d.stability.resampling.ari)} />
          <div className="small muted">{d.stability.resampling.statement} {d.stability.caveat}</div>
        </>}
        {d.explainedVariance && <KV rows={[["variance explained by kept components", `${(d.values.explainedVarianceRatioKept * 100).toFixed(2)}%`], ["components for 90%", d.explainedVariance.componentsNeeded["0.9"]]]} />}
        {d.projection && <div className="projbanner">{d.projection.statement}</div>}
        {d.kDistance && <div className="small muted">k-distance curve: see the DBSCAN node's result.</div>}
        <h4>External agreement</h4>
        {d.external?.available === false ? <div className="muted small">{d.external.reason}</div> : d.external && (
          <div>
            <div className="externalbanner" role="note"><b>EXTERNAL.</b> {d.external.statement} Label column: <code>{d.external.labelsColumn}</code>.</div>
            <KV rows={[["adjusted Rand", fmtNum(d.external.adjustedRand, 4)], ["normalized mutual information", fmtNum(d.external.normalizedMutualInfo, 4)], ["homogeneity / completeness / V", `${fmtNum(d.external.homogeneity, 4)} / ${fmtNum(d.external.completeness, 4)} / ${fmtNum(d.external.vMeasure, 4)}`]]} />
            <table className="dtable" aria-label="contingency"><thead><tr><th>label \ cluster</th>{d.external.clusterLabels.map((c: number) => <th key={c}>{c === -1 ? "noise" : c}</th>)}</tr></thead>
              <tbody>{d.external.contingency.map((row: number[], i: number) => <tr key={i}><td><b>{d.external.truthClasses[i]}</b></td>{row.map((v, j) => <td key={j} className="num">{v}</td>)}</tr>)}</tbody></table>
            {d.external.noteNoise && <div className="small muted">{d.external.noteNoise}</div>}
          </div>)}
        <TabProv p={prov} />
      </div>
    );
  }}</Gate>;
}
