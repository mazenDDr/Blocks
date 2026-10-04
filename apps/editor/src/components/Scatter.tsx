import { useState } from "react";
import { fmtNum } from "../util";

export interface Pt { id: number; x: number; y: number; c: string | number | null }
const PALETTE = ["#1f6feb", "#d9730d", "#2a9d4b", "#a626aa", "#c62828", "#0e9aa7", "#8c6d1f", "#e83e8c", "#5c6bc0", "#6d4c41", "#00897b", "#7cb342"];

/** Categorical scatter plot of recorded points. Colour is a recorded column (cluster id, a label, ...); -1 (DBSCAN noise) is grey. */
export function Scatter({ points, xLabel, yLabel, colorLabel, height = 300 }: { points: Pt[]; xLabel: string; yLabel: string; colorLabel?: string; height?: number }) {
  const [hover, setHover] = useState<Pt | null>(null);
  const W = 460, H = height, m = { l: 42, r: 10, t: 10, b: 30 };
  if (!points.length) return <div className="muted">No points.</div>;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const padX = (x1 - x0 || 1) * 0.05, padY = (y1 - y0 || 1) * 0.05;
  x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
  const cats = [...new Set(points.map((p) => (p.c == null ? "—" : String(p.c))))].sort((a, b) => (Number(a) - Number(b)) || a.localeCompare(b));
  const color = (c: string | number | null) => { const k = c == null ? "—" : String(c); return k === "-1" || k === "—" ? "#9aa3b2" : PALETTE[cats.indexOf(k) % PALETTE.length]; };
  const sx = (x: number) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
  const sy = (y: number) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  const ticks = (lo: number, hi: number) => Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  return (
    <figure className="scatter">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`scatter of ${yLabel} against ${xLabel}`} data-points={points.length}>
        {ticks(y0, y1).map((t, i) => <g key={"y" + i}><line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke="#e6e9ee" /><text x={m.l - 4} y={sy(t) + 3} fontSize="9" textAnchor="end" fill="#667">{fmtNum(t, 3)}</text></g>)}
        {ticks(x0, x1).map((t, i) => <g key={"x" + i}><line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} stroke="#e6e9ee" /><text x={sx(t)} y={H - m.b + 12} fontSize="9" textAnchor="middle" fill="#667">{fmtNum(t, 3)}</text></g>)}
        <text x={(W + m.l) / 2} y={H - 3} fontSize="10" textAnchor="middle" fill="#445">{xLabel}</text>
        <text x={4} y={m.t + 4} fontSize="10" fill="#445">{yLabel}</text>
        {points.map((p) => <circle key={p.id} cx={sx(p.x)} cy={sy(p.y)} r={hover?.id === p.id ? 5 : 3} fill={color(p.c)} fillOpacity={0.75} stroke={hover?.id === p.id ? "#000" : "none"}
          onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)}><title>row {p.id}: ({fmtNum(p.x, 4)}, {fmtNum(p.y, 4)}){p.c != null ? ` · ${colorLabel ?? "colour"} ${p.c}` : ""}</title></circle>)}
      </svg>
      <figcaption className="small">
        {colorLabel && <>colour: <b>{colorLabel}</b> {cats.map((c) => <span key={c} className="legendchip"><i style={{ background: color(c === "—" ? null : c) }} />{c === "-1" ? "-1 (noise)" : c}</span>)}</>}
        {hover && <span className="muted"> · row {hover.id}: ({fmtNum(hover.x, 4)}, {fmtNum(hover.y, 4)})</span>}
      </figcaption>
    </figure>
  );
}
