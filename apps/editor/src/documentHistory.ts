/** Immutable draft history only: no run, artifact, checkpoint or external side effect. */
export interface History<T> { current: T; past: T[]; future: T[]; transaction: number | null }
export type HistoryAction<T> =
  | { type: "edit"; transaction: number; update: (current: T) => T }
  | { type: "reset"; value: T }
  | { type: "undo" }
  | { type: "redo" };
export const HISTORY_LIMIT = 100;
export const HISTORY_BYTES = 8 * 1024 * 1024;
export function initialHistory<T>(value: T): History<T> { return { current: value, past: [], future: [], transaction: null }; }
function bounded<T>(values: T[], current: T): T[] {
  let bytes = JSON.stringify(current).length * 2;
  const retained: T[] = [];
  for (let i = values.length - 1; i >= 0 && retained.length < HISTORY_LIMIT; i--) {
    bytes += JSON.stringify(values[i]).length * 2;
    if (bytes > HISTORY_BYTES) break;
    retained.unshift(values[i]);
  }
  return retained;
}
export function reduceHistory<T>(history: History<T>, action: HistoryAction<T>): History<T> {
  if (action.type === "reset") return initialHistory(action.value);
  if (action.type === "undo") {
    if (!history.past.length) return history;
    const current = history.past.at(-1)!;
    return { current, past: history.past.slice(0, -1), future: bounded([...history.future, history.current], current), transaction: null };
  }
  if (action.type === "redo") {
    if (!history.future.length) return history;
    const current = history.future.at(-1)!;
    return { current, past: bounded([...history.past, history.current], current), future: history.future.slice(0, -1), transaction: null };
  }
  const current = action.update(history.current);
  if (current === history.current || JSON.stringify(current) === JSON.stringify(history.current)) return history;
  const past = history.transaction === action.transaction ? history.past : [...history.past, history.current];
  return { current, past: bounded(past, current), future: [], transaction: action.transaction };
}
