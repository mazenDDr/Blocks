import { useEffect, useState } from "react";

export type TrackPiece = { id: string; from: string; to: string };

/**
 * The signature (design/DIRECTION.md): while a run is working, a blue marble rolls along the graph's real wires in the
 * order data flows, from the input to the output, then rests for a beat and rolls again. The wire under it tints blue and
 * each piece it rolls into rings once, so the run reads as data passing through the blocks. It keeps the same size on
 * screen at any zoom. Reduced motion: the marble rests at the start of the track and only the input piece is ringed.
 * Paused while the tab is hidden.
 */
export function Marble({ track, active }: { track: TrackPiece[]; active: boolean }) {
  const key = track.map((t) => t.id).join("|");
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!active || track.length === 0) return;
    // The wires may not be drawn yet (the effect runs before React Flow renders edges): try again shortly.
    if (!document.querySelector(".react-flow__edge-path")) { const retry = window.setTimeout(() => setTick((t) => t + 1), 250); return () => window.clearTimeout(retry); }
    const edgeEl = (id: string) => document.querySelector<SVGGElement>(`.react-flow__edge[data-id="${CSS.escape(id)}"]`);
    const nodeEl = (id: string) => document.querySelector<HTMLElement>(`.react-flow__node[data-id="${CSS.escape(id)}"]`);
    const live = () => track
      .map((t) => ({ t, path: edgeEl(t.id)?.querySelector<SVGPathElement>(".react-flow__edge-path") ?? null }))
      .filter((x): x is { t: TrackPiece; path: SVGPathElement } => !!x.path && x.path.getTotalLength() > 0);
    const first = live()[0];
    const svg = first?.path.ownerSVGElement;
    if (!svg) return;

    const ns = "http://www.w3.org/2000/svg";
    const marble = document.createElementNS(ns, "g");
    marble.setAttribute("class", "marble");
    marble.setAttribute("aria-hidden", "true");
    const shadow = document.createElementNS(ns, "ellipse"); shadow.setAttribute("class", "marble-shadow");
    shadow.setAttribute("rx", "8"); shadow.setAttribute("ry", "3"); shadow.setAttribute("cy", "9");
    const ball = document.createElementNS(ns, "circle"); ball.setAttribute("r", "9"); ball.setAttribute("class", "marble-ball");
    const shine = document.createElementNS(ns, "circle"); shine.setAttribute("r", "3"); shine.setAttribute("cx", "-3"); shine.setAttribute("cy", "-3"); shine.setAttribute("class", "marble-shine");
    marble.append(shadow, ball, shine);
    svg.appendChild(marble);

    const zoom = () => {
      const m = getComputedStyle(document.querySelector(".react-flow__viewport") ?? document.body).transform;
      const a = m && m !== "none" ? parseFloat(m.slice(m.indexOf("(") + 1)) : 1;
      return a > 0 ? a : 1;
    };
    const place = (p: SVGPathElement, at: number) => {
      const pt = p.getPointAtLength(at);
      marble.setAttribute("transform", `translate(${pt.x} ${pt.y}) scale(${1 / zoom()})`);
    };
    const ring = (id: string) => {
      const el = nodeEl(id);
      if (!el) return;
      el.classList.remove("marble-visit");
      void el.offsetWidth; // restart the animation
      el.classList.add("marble-visit");
    };
    let rolling: SVGGElement | null = null;
    const roll = (id: string | null) => {
      const el = id ? edgeEl(id) : null;
      if (el === rolling) return;
      rolling?.classList.remove("rolling");
      el?.classList.add("rolling");
      rolling = el;
    };
    const cleanup = () => {
      roll(null);
      marble.remove();
      document.querySelectorAll(".marble-visit").forEach((el) => el.classList.remove("marble-visit"));
    };

    if (matchMedia("(prefers-reduced-motion: reduce)").matches) { place(first.path, 0); ring(first.t.from); return cleanup; }

    const SPEED = 240, REST = 900; // flow px per second; ms resting at the end
    let raf = 0, start = performance.now(), lastIndex = -1, ended = false;
    const frame = (now: number) => {
      const pieces = live();
      if (!pieces.length) { raf = requestAnimationFrame(frame); return; }
      const lengths = pieces.map((x) => x.path.getTotalLength());
      const total = lengths.reduce((a, b) => a + b, 0);
      const rollMs = (total / SPEED) * 1000;
      const t = (now - start) % (rollMs + REST);
      const k = Math.min(1, t / rollMs);
      const eased = k * k * (3 - 2 * k); // gathers speed, then slows into the output
      let d = eased * total, i = 0;
      while (i < pieces.length - 1 && d > lengths[i]) { d -= lengths[i]; i++; }
      if (k < 1) {
        if (i !== lastIndex) { if (lastIndex === -1 || i < lastIndex) ring(pieces[0].t.from); else ring(pieces[i].t.from); lastIndex = i; ended = false; }
        roll(pieces[i].t.id);
      } else if (!ended) { ended = true; roll(null); ring(pieces[pieces.length - 1].t.to); lastIndex = -1; }
      const owner = pieces[i].path.ownerSVGElement;
      if (owner && marble.ownerSVGElement !== owner) owner.appendChild(marble);
      place(pieces[i].path, Math.min(d, lengths[i]));
      raf = requestAnimationFrame(frame);
    };
    const visibility = () => { cancelAnimationFrame(raf); if (!document.hidden) { start = performance.now(); lastIndex = -1; raf = requestAnimationFrame(frame); } };
    raf = requestAnimationFrame(frame);
    document.addEventListener("visibilitychange", visibility);
    return () => { cancelAnimationFrame(raf); document.removeEventListener("visibilitychange", visibility); cleanup(); };
  }, [key, active, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

/** Wires in the order data reaches them (Kahn's order of their source nodes). */
export function flowOrder(nodes: { id: string }[], edges: { id: string; from: { node: string }; to: { node: string } }[]): TrackPiece[] {
  const indeg = new Map(nodes.map((n) => [n.id, 0]));
  for (const e of edges) indeg.set(e.to.node, (indeg.get(e.to.node) ?? 0) + 1);
  const queue = nodes.filter((n) => !indeg.get(n.id)).map((n) => n.id);
  const rank = new Map<string, number>();
  while (queue.length) {
    const id = queue.shift()!; rank.set(id, rank.size);
    for (const e of edges) if (e.from.node === id) { const d = (indeg.get(e.to.node) ?? 1) - 1; indeg.set(e.to.node, d); if (d === 0) queue.push(e.to.node); }
  }
  return [...edges].sort((a, b) => (rank.get(a.from.node) ?? 1e9) - (rank.get(b.from.node) ?? 1e9)).map((e) => ({ id: e.id, from: e.from.node, to: e.to.node }));
}
