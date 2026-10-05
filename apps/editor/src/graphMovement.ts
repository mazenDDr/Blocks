/** Explicit user-declared layout translation; no graph/runtime values are changed. */
export interface MovePoint { id: string; key: string; x: number; y: number }
export class MovementError extends Error {
  code: string;
  constructor(code: string, message: string) { super(`${code}: ${message}`); this.code = code; }
}
const fail = (code: string, message: string): never => { throw new MovementError(code, message); };
const LIMIT = 1_000_000;
export function moveGraphNodes(points: MovePoint[], dx: number, dy: number) {
  if (!points.length || points.length > 100 || new Set(points.map(p => p.id)).size !== points.length || new Set(points.map(p => p.key)).size !== points.length)
    fail("E_MOVE_SELECTION", "Select 1–100 uniquely identified actual layout cards.");
  if (![dx, dy].every(n => Number.isFinite(n) && Math.abs(n) <= LIMIT)) fail("E_MOVE_DELTA", "Enter finite horizontal and vertical offsets within ±1,000,000 layout units.");
  for (const p of points) {
    if (![p.x, p.y, p.x + dx, p.y + dy].every(n => Number.isFinite(n) && Math.abs(n) <= LIMIT))
      fail("E_MOVE_POSITION", "Current and resulting origins must be finite and within ±1,000,000 layout units.");
  }
  if (dx === 0 && dy === 0) return {};
  return Object.fromEntries(points.map(p => [p.key, { x: p.x + dx, y: p.y + dy }]));
}
