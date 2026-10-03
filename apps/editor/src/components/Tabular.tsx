import { useEffect, useState, type ReactNode } from "react";
import { useInspect } from "../hooks";
import { isTensorType, type DensityPlot, type Explain, SummaryResult, TabProvenance, TabUnavailable, TablePage, WireType } from "../types";
import { fmtInt, fmtNum, fmtP, shortHash } from "../util";
import { NotRecorded } from "./Provenance";
import { DensityOnly, DensityTailPlot, Histogram, ScatterPlot, StripPlot } from "./StatsPlot";

// ---------------------------------------------------------------------------------------- shared
export function TabProv({ p, label }: { p?: TabProvenance | null; label?: string }) {
  if (!p) return null;
  const sp = p.split;
  const bits = [
    p.runId ? `run ${p.runId}` : "no run", p.graphHash ? `graph ${shortHash(p.graphHash)}` : null,
    p.nodeId ? `node ${p.nodeId}${p.port ? "." + p.port : ""}` : null, p.partition ? `partition ${p.partition}` : null,
    p.sourceSha256 ? `data sha256 ${p.sourceSha256.slice(0, 12)}…` : null,
    sp ? `split seed ${sp.seed} (${sp.nTrain} train / ${sp.nValidation} validation)` : null,
    p.fittedOn ? `fitted on ${p.fittedOn.partition} rows of ${p.fittedOn.node ?? "?"} (${p.fittedOn.rows} rows, ids ${shortHash(p.fittedOn.rowIdsSha256)})` : null,
    label ?? null,
  ].filter(Boolean);
  return <div className="prov" title={JSON.stringify(p)}>Provenance: {bits.join(" · ")}</div>;
}

type Res<T> = SummaryResult<T> | TabUnavailable;

/** Ask the control service for one recorded result of a node. No run => nothing is shown (and nothing is invented). */
export function useNodeResult<T = any>(runId: string | null, kind: string, node: string) {
  const st = useInspect<Res<T>>(runId ? `/api/runs/${runId}/inspect` : null, runId ? { kind, node } : null);
  return st;
}

function Gate<T>({ runId, st, children }: { runId: string | null; st: { data: Res<T> | null; loading: boolean; error: string | null }; children: (r: SummaryResult<T>) => ReactNode }) {
  if (!runId) return <NotRecorded message="No run selected. Values here are read from a recorded run; nothing is simulated." />;
  if (st.error) return <div className="error pre">{st.error}</div>;
  if (!st.data) return <div className="muted">{st.loading ? "loading…" : ""}</div>;
  if (!st.data.available) return <><NotRecorded message={st.data.message} /><TabProv p={st.data.provenance} /></>;
  return <>{children(st.data)}</>;
}

const fmtCell = (v: unknown) => (v === null || v === undefined ? <span className="missing">∅</span> : typeof v === "number" ? (Number.isInteger(v) ? String(v) : fmtNum(v, 6)) : String(v));

export function SmallTable({ sample }: { sample: { columns: string[]; rowIds: number[]; rows: unknown[][] } | undefined }) {
  if (!sample) return null;
  return (
    <div className="tablewrap"><table className="dtable"><thead><tr><th>row</th>{sample.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
      <tbody>{sample.rows.map((r, i) => <tr key={i}><td className="rid">{sample.rowIds[i]}</td>{r.map((v, j) => <td key={j} className="num">{fmtCell(v)}</td>)}</tr>)}</tbody></table></div>
  );
}

// ---------------------------------------------------------------------------------------- table preview
const PAGE = 25;
export function TablePreview({ runId, node, port }: { runId: string | null; node: string; port?: string }) {
  const [offset, setOffset] = useState(0);
  useEffect(() => setOffset(0), [runId, node, port]);
  const st = useInspect<TablePage | TabUnavailable>(runId ? `/api/runs/${runId}/inspect` : null, runId ? { kind: "table", node, port, offset, limit: PAGE } : null);
  if (!runId) return <NotRecorded message="No run selected. The table preview shows rows recorded by a run, never a guess." />;
  if (st.error) return <div className="error pre">{st.error}</div>;
  const d = st.data;
  if (!d) return <div className="muted">loading…</div>;
  if (!d.available) return <><NotRecorded message={d.message} /><TabProv p={d.provenance} /></>;
  return (
    <div className="tablepreview">
      <div className="small">
        <b>{d.node}.{d.port}</b> <span className={`badge part-${d.partition}`}>{d.partition}</span>{" "}
        rows {fmtInt(d.offset + 1)}–{fmtInt(d.offset + d.rows.length)} of {fmtInt(d.total)} · {d.columns.length} columns <span className="muted">(bounded preview, {PAGE} rows per page)</span>
      </div>
      <div className="tablewrap"><table className="dtable">
        <thead><tr><th>row id</th>{d.columns.map((c) => <th key={c.name}>{c.name}<small>{c.dtype}</small></th>)}</tr></thead>
        <tbody>{d.rows.map((r, i) => <tr key={i}><td className="rid">{d.rowIds[i]}</td>{r.map((v, j) => <td key={j} className="num">{fmtCell(v)}</td>)}</tr>)}</tbody>
      </table></div>
      <div className="pager"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
        <button disabled={!d.truncated} onClick={() => setOffset(offset + PAGE)}>Next</button></div>
      <TabProv p={d.provenance} />
    </div>
  );
}

// ---------------------------------------------------------------------------------------- views per summary kind
function KV({ rows }: { rows: [string, ReactNode][] }) {
  return <table className="kv"><tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>;
}

export function SourceView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    return <><KV rows={[["file", d.path], ["rows × columns", `${fmtInt(d.rows)} × ${d.columns}`], ["bytes", fmtInt(d.bytes)], ["SHA-256", <code key="h">{d.sha256}</code>]]} />
      <h4>Schema</h4><table><tbody>{d.schema.map((c: any) => <tr key={c.name}><td>{c.name}</td><td>{c.dtype}</td></tr>)}</tbody></table>
      <div className="muted small">The source file is read-only; its hash identifies exactly which data this run used.</div><TabProv p={r.provenance} /></>;
  }}</Gate>;
}

export function ProfileView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "profile", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    return (
      <div className="profile">
        <KV rows={[["rows", fmtInt(d.rows)], ["columns", d.columnCount], ["method", d.method + (d.exact ? " (exact)" : " (sampled)")],
          ["exact duplicate rows", `${d.duplicateRows} repeats in ${d.rowsInDuplicateGroups} rows`]]} />
        <div className="tablewrap"><table className="dtable"><thead><tr><th>column</th><th>type</th><th>missing</th><th>unique</th><th>summary</th><th>flags</th></tr></thead><tbody>
          {d.columns.map((c: any) => (
            <tr key={c.name}><td><b>{c.name}</b></td><td>{c.dtype}</td>
              <td className={`num ${c.missing ? "bad" : ""}`}>{c.missing}{c.missing ? ` (${(c.missingFraction * 100).toFixed(1)}%)` : ""}</td><td className="num">{c.unique}</td>
              <td>{c.stats ? `mean ${fmtNum(c.stats.mean, 4)} · sd ${fmtNum(c.stats.std, 4)} · min ${fmtNum(c.stats.min, 4)} · median ${fmtNum(c.stats.median, 4)} · max ${fmtNum(c.stats.max, 4)}`
                : (c.top ?? []).map((t: any) => `${t.value} (${t.count})`).join(", ")}</td>
              <td>{(c.flags ?? []).join("; ")}</td></tr>
          ))}</tbody></table></div>
        <TabProv p={r.provenance} />
      </div>
    );
  }}</Gate>;
}

export function PartitionView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    return <><KV rows={[["method", d.method], ["seed", d.seed], ["validation fraction", d.validationFraction], ["stratified by", d.stratifyBy ?? "—"], ["grouped by", d.groupBy ?? "—"],
      ["train rows", fmtInt(d.nTrain)], ["validation rows", fmtInt(d.nValidation)], ["overlapping rows", d.overlapRows],
      ...(d.groupBy ? [["groups in both partitions", d.groupOverlap] as [string, ReactNode]] : []),
      ["train row ids sha256", <code key="a">{d.trainRowIdsSha256.slice(0, 16)}…</code>], ["validation row ids sha256", <code key="b">{d.validationRowIdsSha256.slice(0, 16)}…</code>]]} />
      <div className="muted small">Fit nodes may only read the <b>train</b> output. Statistics learned from it are applied to the validation output by an “Apply fitted transform” node.</div><TabProv p={r.provenance} /></>;
  }}</Gate>;
}

export function FitStateView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "fit_state", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const own = d.fittedOn;
    return (
      <div>
        <div className="ownership"><b>Owned by the {own.partition} partition</b> of <code>{own.node}</code> · {fmtInt(own.rows)} rows · row ids {shortHash(own.rowIdsSha256)}</div>
        {d.transform === "standardize" && <>
          <table><thead><tr><th>column</th><th>mean</th><th>scale (std, ddof 0)</th><th>variance</th></tr></thead>
            <tbody>{d.columns.map((c: string) => <tr key={c}><td>{c}</td><td className="num">{fmtNum(d.mean?.[c], 6)}</td><td className="num">{fmtNum(d.scale?.[c], 6)}</td><td className="num">{fmtNum(d.variance?.[c], 6)}</td></tr>)}</tbody></table>
          <div className="muted small">z = (x − mean) / scale · samples seen: {JSON.stringify(d.nSamplesSeen)}</div></>}
        {d.transform === "onehot" && <>
          {d.columns.map((c: string) => <div key={c}><b>{c}</b>: {d.categories[c].map((k: unknown) => <span key={String(k)} className="chip">{String(k)}</span>)}</div>)}
          <div className="muted small">Output columns: {d.featureNamesOut.join(", ")} · unknown categories: {d.params.handle_unknown}</div></>}
        {d.transform === "impute" && <>
          <table><thead><tr><th>column</th><th>{d.strategy} (fitted)</th><th>missing in training</th></tr></thead>
            <tbody>{d.columns.map((c: string) => <tr key={c}><td>{c}</td><td className="num">{fmtNum(d.statistics[c], 6)}</td><td className="num">{d.missingInTraining[c]}</td></tr>)}</tbody></table></>}
        <TabProv p={r.provenance} />
      </div>
    );
  }}</Gate>;
}

export function CoefficientsView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "coefficients", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const bars = (rows: { feature: string; value: number }[], intercept: number | null) => {
      const mx = Math.max(...rows.map((c) => Math.abs(c.value)), 1e-12);
      return <table className="coef"><tbody>
        {rows.map((c) => <tr key={c.feature}><td>{c.feature}</td><td className="num">{fmtNum(c.value, 6)}</td>
          <td className="barcell"><div className="signed"><i style={{ width: `${(Math.abs(c.value) / mx) * 100}%`, background: c.value >= 0 ? "#1f6feb" : "#c62828", marginLeft: c.value >= 0 ? "50%" : `${50 - (Math.abs(c.value) / mx) * 50}%` }} /></div></td></tr>)}
        {intercept != null && <tr><td><i>intercept</i></td><td className="num">{fmtNum(intercept, 6)}</td><td /></tr>}</tbody></table>;
    };
    return (
      <div>
        <div className="small"><b>{d.estimator}</b> · target <code>{d.target}</code> · {fmtInt(d.nTrain)} training rows · scikit-learn {d.sklearn}</div>
        {d.task === "regression" ? <>{bars(d.coefficients, d.intercept)}
          <div className="muted small">{d.preprocessingContext}</div>
          <div className="small">rank {d.rank} of {d.features.length} features{d.rank < d.features.length ? " — collinear columns!" : ""}</div></> : <>
          {d.perClass.map((pc: any) => <div key={String(pc.class)}><h4>class {String(pc.class)}{pc.note ? ` (${pc.note})` : ""}</h4>{bars(pc.coefficients, pc.intercept)}</div>)}
          <div className="small">iterations {JSON.stringify(d.nIter)} · {d.converged ? "converged" : <b className="bad">did NOT converge — raise max_iter or scale features</b>}</div></>}
        <details><summary className="small">estimator parameters</summary><pre>{JSON.stringify(d.params, null, 1)}</pre></details>
        <TabProv p={r.provenance} />
      </div>
    );
  }}</Gate>;
}

const ORDER = ["mse", "rmse", "mae", "r2", "accuracy", "log_loss", "roc_auc"];
export function MetricsView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "metrics", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const pts: { observed: number; predicted: number; rowId: number }[] = d.points ?? [];
    const names: Record<string, string> = { mse: "MSE", rmse: "RMSE", mae: "MAE", r2: "R²", accuracy: "accuracy", log_loss: "log-loss", roc_auc: "ROC-AUC" };
    return (
      <div>
        <div className={d.evaluatedPartition === "validation" ? "ownership" : "warn"}>Evaluated on the <b>{d.evaluatedPartition}</b> partition · {fmtInt(d.n)} rows{d.warning ? ` — ${d.warning}` : ""}</div>
        <table><tbody>{Object.entries(d.values).sort(([a], [b]) => (ORDER.indexOf(a) - ORDER.indexOf(b))).map(([k, v]) => <tr key={k}><td>{names[k] ?? k}</td><td className="num"><b>{fmtNum(v as number, 6)}</b></td></tr>)}
          {Object.entries(d.unavailable ?? {}).map(([k, v]) => <tr key={k}><td>{names[k] ?? k}</td><td className="muted">unavailable: {String(v)}</td></tr>)}</tbody></table>
        {pts.length > 0 && <div className="plots"><div><h4>predicted vs observed</h4><ScatterPlot points={pts} /></div>
          <div><h4>residuals (observed − predicted)</h4><Histogram values={pts.map((p) => p.observed - p.predicted)} />
            <div className="small muted">mean {fmtNum(d.residuals.mean, 4)} · sd {fmtNum(d.residuals.std, 4)} · range {fmtNum(d.residuals.min, 4)} .. {fmtNum(d.residuals.max, 4)}</div></div></div>}
        {d.confusion && <div className="confusion"><h4>confusion matrix <small>({d.confusion.axes})</small></h4>
          <table><thead><tr><th /><>{d.confusion.labels.map((l: unknown) => <th key={String(l)}>{String(l)}</th>)}</></tr></thead>
            <tbody>{d.confusion.matrix.map((row: number[], i: number) => <tr key={i}><th>{String(d.confusion.labels[i])}</th>{row.map((v, j) => <td key={j} className={i === j ? "diag" : ""}>{v}</td>)}</tr>)}</tbody></table>
          {d.positiveClass !== undefined && <div className="small muted">ROC-AUC and log-loss use P(class = {String(d.positiveClass)}).</div>}</div>}
        <TabProv p={r.provenance} />
      </div>
    );
  }}</Gate>;
}

export function DistributionView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "distribution", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const plot: DensityPlot = { x: d.curve.x, density: d.curve.density, family: d.family };
    return <><DensityOnly plot={plot} />
      <KV rows={[["family", d.family], ["parametrization", d.parametrization], ["shape k", fmtNum(d.shape)], ["scale θ", fmtNum(d.scale)], ["rate β = 1/θ", fmtNum(d.rate)], ["location", fmtNum(d.loc)],
        ["mean / variance", `${fmtNum(d.mean, 6)} / ${fmtNum(d.variance, 6)}`], ["SciPy", <code key="s">{d.scipy}</code>]]} />
      <h4>Quantiles</h4><table><tbody>{d.quantiles.map((q: any) => <tr key={q.p}><td>P(X ≤ x) = {q.p}</td><td className="num">x = {fmtNum(q.x, 6)}</td></tr>)}</tbody></table>
      <div className="muted small">{d.note}</div><TabProv p={r.provenance} /></>;
  }}</Gate>;
}

export function TailView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "tail", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const plot: DensityPlot = { ...d.plot, tail: d.tail, observed: d.observed, family: d.reference.family, statisticName: "T" };
    return <><DensityTailPlot plot={plot} pValue={d.pValue} statName="T" />
      <KV rows={[["reference", `${d.reference.family} (shape ${d.reference.shape}, scale ${d.reference.scale}, loc ${d.reference.loc})`], ["observed", fmtNum(d.observed)], ["tail", d.tail],
        ["formula", d.formula], ["p-value (SciPy)", <b key="p">{fmtP(d.pValue)}</b>], ["area by numerical integration", `${fmtNum(d.areaNumerical, 10)} (± ${fmtNum(d.areaNumericalError, 2)})`],
        ...(d.closedForm ? [["closed form", `${d.closedForm.expression} = ${fmtNum(d.closedForm.value, 10)}`] as [string, ReactNode]] : []),
        ["density at observed (not a probability)", fmtNum(d.densityAtObserved, 8)]]} />
      <div className="muted small">{d.note}</div><TabProv p={r.provenance} /></>;
  }}</Gate>;
}

export function NumberView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "number", node);
  return <Gate runId={runId} st={st}>{(r) => <><KV rows={[["x", fmtNum(r.data.x)], ["Γ(x)", fmtNum(r.data.value, 12)], ["ln Γ(x)", fmtNum(r.data.logGamma, 12)], ["identity", r.data.identity], ["SciPy", r.data.scipy]]} /><TabProv p={r.provenance} /></>}</Gate>;
}

export function TestResultView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "test_result", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const reject = d.decision === "reject_h0";
    const md = d.meanDifference;
    return (
      <div className="testresult">
        {d.teachingFixture && <div className="warn"><b>Teaching fixture.</b> The reference distribution and the observed statistic are specified by the example, not measured from data.</div>}
        {d.plot && <DensityTailPlot plot={d.plot} pValue={d.pValue} alpha={d.alpha} statName={d.statistic.name} />}
        {d.observations && d.groups && <><h4>Individual observations{d.design?.pairing === "paired" ? " (paired by unit)" : ""}</h4><StripPlot groups={d.groups} obs={d.observations} />
          {d.observations.truncated && <div className="small muted">first 500 per group shown</div>}</>}
        <KV rows={[
          ["method", d.methodLabel ?? d.method], ["null (H0)", d.null], ["alternative", d.alternative],
          ["statistic", `${d.statistic.name} = ${fmtNum(d.statistic.value, 6)}${d.df != null ? ` · df = ${fmtNum(d.df, 6)}` : ""}`],
          ...(d.reference ? [["reference distribution", `${d.reference.family} (shape ${d.reference.shape}, scale ${d.reference.scale}, loc ${d.reference.loc})`] as [string, ReactNode]] : []),
          ...(d.tail ? [["tail rule", d.tail] as [string, ReactNode]] : []),
          ["p-value", <b key="p">{fmtP(d.pValue)}</b>], ["alpha", d.alpha],
          ["decision", <b key="d" className={reject ? "bad" : "good"}>{d.decisionText}</b>], ["decision rule", d.decisionRule],
          ...(d.criticalValue != null ? [["critical value", fmtNum(d.criticalValue, 6)] as [string, ReactNode]] : []),
        ]} />
        {d.pValueFormula && <div className="small muted">p = {d.pValueFormula}{d.closedForm ? ` = ${d.closedForm.expression} = ${fmtNum(d.closedForm.value, 10)}` : ""}</div>}
        {md && <><h4>Effect and uncertainty</h4>
          <KV rows={[[`mean difference (${md.definition})`, <b key="m">{fmtNum(md.value, 6)}</b>], [`${(md.level * 100).toFixed(0)}% confidence interval`, `[${fmtNum(md.ciLow, 6)}, ${fmtNum(md.ciHigh, 6)}]${md.alternative !== "two-sided" ? " (one-sided bound)" : ""}`],
            ...d.effects.map((e: any) => [e.label, fmtNum(e.value, 5)] as [string, ReactNode])]} /></>}
        {!md && d.effects && <><h4>Effect</h4><KV rows={d.effects.map((e: any) => [e.label, fmtNum(e.value, 5)] as [string, ReactNode])} />{d.uncertainty && <div className="small muted">{d.uncertainty}</div>}</>}
        {d.groups && <><h4>Groups</h4><table><thead><tr><th>group</th><th>n</th><th>mean</th><th>sd</th><th>median</th><th>min</th><th>max</th></tr></thead>
          <tbody>{d.groups.map((g: any) => <tr key={g.label}><td>{g.label}</td><td className="num">{g.n}</td><td className="num">{fmtNum(g.mean, 5)}</td><td className="num">{g.sd == null ? "—" : fmtNum(g.sd, 5)}</td><td className="num">{fmtNum(g.median, 5)}</td><td className="num">{fmtNum(g.min, 5)}</td><td className="num">{fmtNum(g.max, 5)}</td></tr>)}</tbody></table></>}
        {d.design && <><h4>Design</h4><KV rows={[["sample unit", d.design.sampleUnit], ["pairing", d.design.pairing], ["unit column", d.design.unitColumn ?? "— (rows assumed independent)"],
          ["independence", typeof d.design.independence === "string" ? d.design.independence : d.design.independence?.checked ? `checked: ${d.design.independence.nUnits} units, ${d.design.independence.repeatedUnits} repeated` : d.design.independence?.note],
          ...(d.design.nPairs != null ? [["pairs", d.design.nPairs] as [string, ReactNode]] : []), ["missing values", `${d.design.missingPolicy}: ${d.design.missingDropped} dropped`]]} /></>}
        <h4>Assumptions</h4><ul className="assump">{d.assumptions.map((a: string, i: number) => <li key={i}>{a}</li>)}</ul>
        {d.diagnostics && Object.keys(d.diagnostics).length > 0 && <><h4>Diagnostics (shown, not decisive)</h4>
          <table><tbody>{Object.entries(d.diagnostics).map(([k, v]: [string, any]) => <tr key={k}><td>{k}</td><td className="num">stat {fmtNum(v.statistic, 4)} · p {fmtNum(v.pValue, 4)}</td></tr>)}</tbody></table></>}
        {d.scipy && <div className="small"><code>{d.scipy}</code></div>}
        <div className="small muted">statistic source: {d.statisticSource}. {d.caveat}</div>
        <TabProv p={r.provenance} label={d.teachingFixture ? "TEACHING FIXTURE" : undefined} />
      </div>
    );
  }}</Gate>;
}

export function StepView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult(runId, "summary", node);
  return <Gate runId={runId} st={st}>{(r) => {
    const d = r.data;
    const scalars = Object.entries(d).filter(([k, v]) => !["before", "after", "preview", "trainRowIds", "validationRowIds"].includes(k) && (v === null || ["string", "number", "boolean"].includes(typeof v)));
    const objs = Object.entries(d).filter(([k, v]) => !["before", "after", "preview", "trainRowIds", "validationRowIds"].includes(k) && v && typeof v === "object");
    return (
      <div>
        <KV rows={scalars.map(([k, v]) => [k, String(v)])} />
        {objs.map(([k, v]) => <details key={k}><summary className="small">{k}</summary><pre>{JSON.stringify(v, null, 1)}</pre></details>)}
        {d.before && <><h4>Before (first rows)</h4><SmallTable sample={d.before} /></>}
        {d.after && <><h4>After (first rows)</h4><SmallTable sample={d.after} /></>}
        <TabProv p={r.provenance} />
      </div>
    );
  }}</Gate>;
}

export const VIEW_LABEL: Record<string, string> = {
  source: "Source", profile: "Profile", split: "Partition", fit_state: "Fitted state", coefficients: "Coefficients", metrics: "Metrics",
  test_result: "Test", distribution: "Distribution", tail: "Tail probability", number: "Value", step: "Changes",
};

export function NodeResultView({ kind, runId, node }: { kind: string; runId: string | null; node: string }) {
  switch (kind) {
    case "source": return <SourceView runId={runId} node={node} />;
    case "profile": return <ProfileView runId={runId} node={node} />;
    case "split": return <PartitionView runId={runId} node={node} />;
    case "fit_state": return <FitStateView runId={runId} node={node} />;
    case "coefficients": return <CoefficientsView runId={runId} node={node} />;
    case "metrics": return <MetricsView runId={runId} node={node} />;
    case "test_result": return <TestResultView runId={runId} node={node} />;
    case "distribution": return <DistributionView runId={runId} node={node} />;
    case "tail": return <TailView runId={runId} node={node} />;
    case "number": return <NumberView runId={runId} node={node} />;
    default: return <StepView runId={runId} node={node} />;
  }
}

export function TabularExplain({ explain, purpose, typed }: { explain?: Explain; purpose: string; typed: boolean }) {
  if (!typed || !explain) return <div className="muted">Explanation needs a valid, typed node (fix the errors on the card first).</div>;
  if (explain.error) return <div className="error">{explain.error}</div>;
  return (
    <div className="explain">
      <p>{purpose}</p>
      {explain.equation && <><h4>Equation</h4><pre>{explain.equation}</pre></>}
      {explain.rule && <><h4>Rule</h4><p>{explain.rule}</p></>}
      {explain.note && <p>{explain.note}</p>}
      <div className="muted small">Computed from this node's current configuration (draft graph).</div>
    </div>
  );
}

/** Compact schema listing used on node cards and the wire inspector. */
export function SchemaList({ t, max = 5 }: { t: WireType; max?: number }) {
  if (isTensorType(t) || t.kind !== "table") return null;
  const cols = t.columns ?? [];
  return (
    <div className="schema">
      {cols.slice(0, max).map((c) => <div key={c.name}><span>{c.name}</span><em>{c.dtype}</em></div>)}
      {cols.length > max && <div className="muted">+{cols.length - max} more…</div>}
      {(t.pending ?? []).map((p) => <div key={p} className="muted">+ {p}</div>)}
    </div>
  );
}
