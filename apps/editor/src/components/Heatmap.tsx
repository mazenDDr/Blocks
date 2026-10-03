import { useEffect, useRef } from "react";
import { diverging, viridis } from "../util";

export interface ScaleSpec { mode: "diverging" | "sequential"; min: number; max: number }

/** Colour of value v under a shared scale. Diverging is symmetric about 0 using max(|min|,|max|). */
export function colorOf(v: number, s: ScaleSpec): [number, number, number] {
  if (s.mode === "diverging") {
    const m = Math.max(Math.abs(s.min), Math.abs(s.max)) || 1;
    return diverging(v / m);
  }
  const r = s.max - s.min || 1;
  return viridis((v - s.min) / r);
}

/** Canvas heatmap of a 2-D array. `cell` is the pixel size of one value. */
export function Heatmap({ values, scale, cell = 6, title }: { values: number[][]; scale: ScaleSpec; cell?: number; title?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const h = values.length, w = values[0]?.length ?? 0;
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !w || !h) return;
    const ctx = cv.getContext("2d")!;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const [r, g, b] = colorOf(values[y][x], scale);
      const o = (y * w + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values, scale.mode, scale.min, scale.max, w, h]);
  return <canvas ref={ref} width={w} height={h} className="heat" title={title} style={{ width: w * cell, height: h * cell }} />;
}

export function ColorBar({ scale }: { scale: ScaleSpec }) {
  const steps = 32;
  const lo = scale.mode === "diverging" ? -Math.max(Math.abs(scale.min), Math.abs(scale.max)) : scale.min;
  const hi = scale.mode === "diverging" ? -lo : scale.max;
  const stops = Array.from({ length: steps }, (_, i) => {
    const v = lo + ((hi - lo) * i) / (steps - 1);
    const [r, g, b] = colorOf(v, scale);
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  });
  return (
    <div className="colorbar">
      <span>{Number(lo.toPrecision(3))}</span>
      <div style={{ background: `linear-gradient(to right, ${stops.join(",")})` }} />
      <span>{Number(hi.toPrecision(3))}</span>
    </div>
  );
}
