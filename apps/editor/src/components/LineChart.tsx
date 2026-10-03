import { useState } from "react";

export interface Series { id: string; label: string; color: string; points: [number, number][]; dashed?: boolean }

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toPrecision(10)));
  return out;
}

/** Small hand-rolled SVG line chart. Raw points only (no smoothing). Hover shows the nearest x for every series. */
export function LineChart({ series, xLabel, yLabel, height = 190, yMin }: { series: Series[]; xLabel: string; yLabel: string; height?: number; yMin?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 560, H = height, m = { l: 46, r: 10, t: 20, b: 30 };
  const pts = series.flatMap((s) => s.points);
  if (pts.length === 0) return <div className="empty">No points recorded yet.</div>;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = yMin ?? Math.min(...ys), y1 = Math.max(...ys);
  const px = (x: number) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const py = (y: number) => H - m.b - ((y - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const nearest = (s: Series, x: number) => s.points.reduce((a, b) => (Math.abs(b[0] - x) < Math.abs(a[0] - x) ? b : a), s.points[0]);
  const hx = hover !== null ? x0 + ((hover - m.l) / (W - m.l - m.r)) * (x1 - x0) : null;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${yLabel} versus ${xLabel}`}
        onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); setHover(((e.clientX - r.left) / r.width) * W); }}
        onMouseLeave={() => setHover(null)}>
        {niceTicks(y0, y1).map((t) => (
          <g key={"y" + t}><line x1={m.l} x2={W - m.r} y1={py(t)} y2={py(t)} className="grid" /><text x={m.l - 5} y={py(t) + 3} textAnchor="end" className="tick">{Number(t.toPrecision(3))}</text></g>
        ))}
        {niceTicks(x0, x1).map((t) => (
          <g key={"x" + t}><line x1={px(t)} x2={px(t)} y1={m.t} y2={H - m.b} className="grid" /><text x={px(t)} y={H - m.b + 13} textAnchor="middle" className="tick">{t}</text></g>
        ))}
        <text x={(W + m.l) / 2} y={H - 3} textAnchor="middle" className="axis">{xLabel}</text>
        <text x={4} y={11} className="axis">{yLabel}</text>
        {series.map((s) => (
          <g key={s.id}>
            {s.points.length > 1 && <polyline fill="none" stroke={s.color} strokeWidth={1.5} strokeDasharray={s.dashed ? "5 3" : undefined} points={s.points.map((p) => `${px(p[0])},${py(p[1])}`).join(" ")} />}
            {s.points.length <= 40 && s.points.map((p, i) => <circle key={i} cx={px(p[0])} cy={py(p[1])} r={2.5} fill={s.color} />)}
          </g>
        ))}
        {hx !== null && <line x1={hover!} x2={hover!} y1={m.t} y2={H - m.b} className="cursor" />}
      </svg>
      <div className="legend">
        {series.map((s) => {
          const n = hx !== null && s.points.length ? nearest(s, hx) : null;
          return (
            <span key={s.id}><i style={{ background: s.color, borderTop: s.dashed ? "2px dashed #fff" : undefined }} />{s.label}{n ? ` — ${xLabel} ${n[0]}: ${Number(n[1].toPrecision(4))}` : ""}</span>
          );
        })}
      </div>
    </div>
  );
}
