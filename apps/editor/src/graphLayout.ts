/** Whole-scope layered layout on actual measured card rectangles. Layout only: no graph mutation. */
export interface LayoutCard { id: string; key: string; x: number; y: number; width: number; height: number }
export interface LayoutLink { from: string; to: string }
export const AUTO_LAYOUT_LIMIT = 1000;

/**
 * Left-to-right layers by longest path from sources (cycles broken at back edges in document order),
 * ordered by barycenter sweeps with document order as the tie-break. Cards in a layer are stacked by
 * their measured heights and layers are spaced by their widest card, so no two cards overlap. The
 * current top-left corner of the scope is kept so the graph does not jump.
 */
export function layeredLayout(cards: LayoutCard[], links: LayoutLink[], gapX = 80, gapY = 40): Record<string, { x: number; y: number }> {
  if (!cards.length || cards.length > AUTO_LAYOUT_LIMIT || new Set(cards.map(c => c.id)).size !== cards.length || new Set(cards.map(c => c.key)).size !== cards.length)
    throw Error(`E_LAYOUT_SELECTION: Auto-arrange needs 1–${AUTO_LAYOUT_LIMIT} distinct cards in this scope.`);
  if (cards.some(c => !Number.isFinite(c.x) || !Number.isFinite(c.y))) throw Error("E_LAYOUT_POSITION: A card has an invalid position.");
  if (cards.some(c => !Number.isFinite(c.width) || !Number.isFinite(c.height) || c.width <= 0 || c.height <= 0))
    throw Error("E_LAYOUT_MEASUREMENT: Wait for actual dimensions of every card.");
  const index = new Map(cards.map((c, i) => [c.id, i]));
  const out: number[][] = cards.map(() => []);
  for (const l of links) {
    const a = index.get(l.from), b = index.get(l.to);
    if (a !== undefined && b !== undefined && a !== b && !out[a].includes(b)) out[a].push(b);
  }
  // Break cycles: drop edges to a node still on the depth-first stack.
  const state = cards.map(() => 0), forward: number[][] = cards.map(() => []);
  const visit = (start: number) => {
    const stack: [number, number][] = [[start, 0]]; state[start] = 1;
    while (stack.length) {
      const top = stack[stack.length - 1], [n, i] = top;
      if (i < out[n].length) {
        top[1]++; const m = out[n][i];
        if (state[m] === 1) continue;
        forward[n].push(m);
        if (state[m] === 0) { state[m] = 1; stack.push([m, 0]); }
      } else { state[n] = 2; stack.pop(); }
    }
  };
  for (let i = 0; i < cards.length; i++) if (!state[i]) visit(i);
  // Longest-path layers over the acyclic forward edges (Kahn order).
  const indegree = cards.map(() => 0); forward.forEach(ms => ms.forEach(m => indegree[m]++));
  const layer = cards.map(() => 0), queue = cards.map((_, i) => i).filter(i => !indegree[i]);
  for (let q = 0; q < queue.length; q++) {
    const n = queue[q];
    for (const m of forward[n]) { layer[m] = Math.max(layer[m], layer[n] + 1); if (!--indegree[m]) queue.push(m); }
  }
  const depth = Math.max(...layer) + 1;
  const layers: number[][] = Array.from({ length: depth }, () => []);
  cards.forEach((_, i) => layers[layer[i]].push(i));
  const into: number[][] = cards.map(() => []); forward.forEach((ms, n) => ms.forEach(m => into[m].push(n)));
  const rank = new Map<number, number>();
  const rerank = () => layers.forEach(ns => ns.forEach((n, r) => rank.set(n, r)));
  rerank();
  const sweep = (order: number[], neighbours: number[][]) => {
    for (const l of order) {
      const centre = (n: number) => neighbours[n].length ? neighbours[n].reduce((s, m) => s + rank.get(m)!, 0) / neighbours[n].length : rank.get(n)!;
      layers[l] = layers[l].map(n => [n, centre(n)] as const).sort((a, b) => a[1] - b[1] || a[0] - b[0]).map(([n]) => n);
      layers[l].forEach((n, r) => rank.set(n, r));
    }
  };
  const down = Array.from({ length: depth }, (_, i) => i).slice(1), up = Array.from({ length: depth }, (_, i) => depth - 1 - i).slice(1);
  for (let i = 0; i < 4; i++) { sweep(down, into); sweep(up, forward); }
  const left = Math.min(...cards.map(c => c.x)), top = Math.min(...cards.map(c => c.y));
  const positions: Record<string, { x: number; y: number }> = {};
  let x = left;
  for (const ns of layers) {
    let y = top;
    for (const n of ns) { positions[cards[n].key] = { x, y }; y += cards[n].height + gapY; }
    x += Math.max(...ns.map(n => cards[n].width)) + gapX;
  }
  return positions;
}
