import type { DensityPlot } from "../types";
import { fmtNum } from "../util";

const W = 380, H = 210, M = { l: 38, r: 12, t: 16, b: 30 };

/** Density curve with the shaded tail region(s), the observed statistic and the critical boundary(ies). Every value is plotted from the
 *  arrays the worker computed with SciPy (no drawing of an idealised curve). */
export function DensityTailPlot({ plot, pValue, alpha, statName }: { plot: DensityPlot; pValue?: number | null; alpha?: number | null; statName?: string }) {
  const xs = plot.x, ys = plot.density;
  if (!xs?.length) return <div className="muted">No curve recorded.</div>;
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const ymax = Math.max(...ys.filter(Number.isFinite)) * 1.08 || 1;
  const sx = (x: number) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - (Math.min(y, ymax) / ymax) * (H - M.t - M.b);
  const line = xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(" ");
  const region = (lo: number, hi: number) => {
    const pts = xs.map((x, i) => [x, ys[i]] as const).filter(([x]) => x >= lo && x <= hi);
    if (pts.length < 2) return "";
    return `M${sx(pts[0][0])},${sy(0)} ` + pts.map(([x, y]) => `L${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ") + ` L${sx(pts[pts.length - 1][0])},${sy(0)} Z`;
  };
  const obs = plot.observed;
  const regions: string[] = [];
  if (obs != null) {
    if (plot.tail === "upper") regions.push(region(obs, x1));
    else if (plot.tail === "lower") regions.push(region(x0, obs));
    else if (plot.tail === "two_sided") regions.push(region(Math.abs(obs), x1), region(x0, -Math.abs(obs)));
  }
  const crit = [plot.criticalLow, plot.criticalHigh ?? plot.criticalValue].filter((v): v is number => v != null && Number.isFinite(v) && v >= x0 && v <= x1);
  const ticks = Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * i) / 4);
  return (
    <figure className="statplot">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="density with shaded tail">
        <line x1={M.l} x2={W - M.r} y1={sy(0)} y2={sy(0)} stroke="#889" />
        <line x1={M.l} x2={M.l} y1={M.t} y2={sy(0)} stroke="#889" />
        {regions.map((d, i) => d && <path key={i} d={d} fill="#d9730d" fillOpacity="0.45" stroke="none" className="tail-region" />)}
        <path d={line} fill="none" stroke="#1f6feb" strokeWidth="1.8" />
        {crit.map((c, i) => (
          <g key={i}><line x1={sx(c)} x2={sx(c)} y1={M.t} y2={sy(0)} stroke="#2a9d4b" strokeDasharray="5 3" strokeWidth="1.4" className="critical-line" />
            <text x={sx(c)} y={M.t - 4} fontSize="9" textAnchor="middle" fill="#2a9d4b">c={fmtNum(c, 4)}</text></g>
        ))}
        {obs != null && obs >= x0 && obs <= x1 && (
          <g><line x1={sx(obs)} x2={sx(obs)} y1={M.t + 8} y2={sy(0)} stroke="#c62828" strokeWidth="1.6" className="observed-line" />
            <text x={sx(obs)} y={M.t + 6} fontSize="10" textAnchor="middle" fill="#c62828">{statName ?? plot.statisticName ?? "t"} = {fmtNum(obs, 4)}</text></g>
        )}
        {ticks.map((t, i) => <g key={i}><line x1={sx(t)} x2={sx(t)} y1={sy(0)} y2={sy(0) + 3} stroke="#889" /><text x={sx(t)} y={sy(0) + 13} fontSize="9" textAnchor="middle" fill="#667">{fmtNum(t, 3)}</text></g>)}
        <text x={4} y={M.t + 4} fontSize="9" fill="#667">density</text>
      </svg>
      <figcaption className="small muted">
        <span className="sw" style={{ background: "#1f6feb" }} /> density (a height, not a probability)
        {obs != null && <>{" · "}<span className="sw" style={{ background: "#d9730d", opacity: 0.55 }} /> shaded tail area = p-value{pValue != null ? ` = ${fmtNum(pValue, 6)}` : ""}</>}
        {crit.length > 0 && <>{" · "}<span className="sw" style={{ background: "#2a9d4b" }} /> critical boundary{alpha != null ? ` at alpha = ${alpha}` : ""}</>}
        {plot.family === "t" && plot.df != null && <> · reference: t distribution, df = {fmtNum(plot.df, 5)}</>}
      </figcaption>
    </figure>
  );
}

/** Plain curve (no tail) for a distribution block. */
export function DensityOnly({ plot }: { plot: DensityPlot }) {
  return <DensityTailPlot plot={{ ...plot, observed: undefined, tail: undefined }} />;
}

/** Individual observations per group with mean and +-1 SD; deterministic jitter by index. */
export function StripPlot({ groups, obs }: { groups: { label: string; mean: number; sd: number | null }[]; obs: { A: number[]; B: number[] } }) {
  const all = [...obs.A, ...obs.B];
  if (!all.length) return null;
  const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.1 || 1;
  const w = 300, h = 170, m = { l: 36, r: 10, t: 10, b: 28 };
  const sy = (v: number) => h - m.b - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (h - m.t - m.b);
  const cx = [m.l + (w - m.l - m.r) * 0.27, m.l + (w - m.l - m.r) * 0.73];
  const sets = [obs.A, obs.B];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" style={{ maxWidth: 340 }} role="img" aria-label="individual observations by group">
      <line x1={m.l} x2={m.l} y1={m.t} y2={h - m.b} stroke="#889" />
      {Array.from({ length: 5 }, (_, i) => lo - pad + ((hi - lo + 2 * pad) * i) / 4).map((t, i) => <g key={i}><line x1={m.l - 3} x2={m.l} y1={sy(t)} y2={sy(t)} stroke="#889" /><text x={m.l - 5} y={sy(t) + 3} fontSize="9" textAnchor="end" fill="#667">{fmtNum(t, 3)}</text></g>)}
      {sets.map((vals, g) => (
        <g key={g}>
          {vals.map((v, i) => <circle key={i} cx={cx[g] + (((i * 37) % 21) - 10) * 2.2} cy={sy(v)} r="3" fill={g ? "#d9730d" : "#1f6feb"} fillOpacity="0.7" />)}
          <line x1={cx[g] - 28} x2={cx[g] + 28} y1={sy(groups[g].mean)} y2={sy(groups[g].mean)} stroke="#111" strokeWidth="2" />
          {groups[g].sd != null && <line x1={cx[g]} x2={cx[g]} y1={sy(groups[g].mean - groups[g].sd!)} y2={sy(groups[g].mean + groups[g].sd!)} stroke="#111" strokeWidth="1" />}
          <text x={cx[g]} y={h - 10} fontSize="10" textAnchor="middle" fill="#334">{groups[g].label} (n={vals.length})</text>
        </g>
      ))}
    </svg>
  );
}

/** Predicted (y) vs observed (x) with the y = x line. */
export function ScatterPlot({ points }: { points: { observed: number; predicted: number }[] }) {
  if (!points.length) return null;
  const vals = points.flatMap((p) => [p.observed, p.predicted]);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.05 || 1;
  const L = 34, R = 192, T = 8, B = 166;
  const fx = (v: number) => L + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (R - L);
  const fy = (v: number) => B - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (B - T);
  return (
    <svg viewBox="0 0 200 190" width="100%" style={{ maxWidth: 230 }} role="img" aria-label="predicted versus observed">
      <rect x={L} y={T} width={R - L} height={B - T} fill="#fff" stroke="#c9ced8" />
      <line x1={fx(lo - pad)} y1={fy(lo - pad)} x2={fx(hi + pad)} y2={fy(hi + pad)} stroke="#2a9d4b" strokeDasharray="4 3" />
      {points.map((p, i) => <circle key={i} cx={fx(p.observed)} cy={fy(p.predicted)} r="2.2" fill="#1f6feb" fillOpacity="0.6" />)}
      <text x={L} y={B + 12} fontSize="9" fill="#556">observed {fmtNum(lo - pad, 4)} .. {fmtNum(hi + pad, 4)}</text>
      <text x={L} y={B + 23} fontSize="9" fill="#556">y axis: predicted · dashed: perfect prediction</text>
    </svg>
  );
}

export function Histogram({ values, bins = 16 }: { values: number[]; bins?: number }) {
  if (!values.length) return null;
  const lo = Math.min(...values), hi = Math.max(...values), w = (hi - lo) / bins || 1;
  const counts = Array.from({ length: bins }, () => 0);
  values.forEach((v) => { counts[Math.min(bins - 1, Math.floor((v - lo) / w))]++; });
  const mx = Math.max(...counts), W2 = 230, H2 = 110;
  return (
    <svg viewBox={`0 0 ${W2} ${H2 + 14}`} width="100%" style={{ maxWidth: 260 }} role="img" aria-label="residual histogram">
      {counts.map((c, i) => <rect key={i} x={6 + (i * (W2 - 12)) / bins} y={H2 - (c / mx) * (H2 - 8)} width={(W2 - 12) / bins - 1} height={(c / mx) * (H2 - 8)} fill="#8a93a6" />)}
      <line x1={6} x2={W2 - 6} y1={H2} y2={H2} stroke="#889" />
      <text x={6} y={H2 + 11} fontSize="9" fill="#556">{fmtNum(lo, 3)}</text><text x={W2 - 6} y={H2 + 11} fontSize="9" textAnchor="end" fill="#556">{fmtNum(hi, 3)}</text>
    </svg>
  );
}
