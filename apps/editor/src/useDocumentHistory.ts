import { useCallback, useReducer, useRef, type SetStateAction } from "react";
import { initialHistory, reduceHistory } from "./documentHistory";
import type { Graph, UiDoc } from "./types";
interface Draft { graph: Graph; ui: UiDoc }
const resolve = <T,>(value: SetStateAction<T>, old: T): T => typeof value === "function" ? (value as (old: T) => T)(old) : value;

export function useDocumentHistory(graph: Graph, ui: UiDoc) {
  const [history, dispatch] = useReducer(reduceHistory<Draft>, { graph, ui }, initialHistory<Draft>);
  const sequence = useRef(0);
  const pending = useRef<number | null>(null);
  const drag = useRef<number | null>(null);
  // Paired graph/layout changes in one event are a single draft edit, including add/delete.
  const transaction = useCallback(() => {
    if (pending.current === null) {
      pending.current = ++sequence.current;
      queueMicrotask(() => { pending.current = null; });
    }
    return pending.current;
  }, []);
  const setGraph = useCallback((value: SetStateAction<Graph>) => dispatch({ type: "edit", transaction: transaction(), update: old => ({ ...old, graph: resolve(value, old.graph) }) }), [transaction]);
  const setUi = useCallback((value: SetStateAction<UiDoc>, dragging?: boolean) => {
    if (dragging === true && drag.current === null) drag.current = ++sequence.current;
    const token = dragging !== undefined && drag.current !== null ? drag.current : transaction();
    dispatch({ type: "edit", transaction: token, update: old => ({ ...old, ui: resolve(value, old.ui) }) });
    if (dragging === false) drag.current = null;
  }, [transaction]);
  const reset = useCallback((value: Draft) => { pending.current = null; drag.current = null; dispatch({ type: "reset", value }); }, []);
  const move = useCallback((type: "undo" | "redo") => { pending.current = null; drag.current = null; dispatch({ type }); }, []);
  return { graph: history.current.graph, ui: history.current.ui, setGraph, setUi, reset, move,
    canUndo: history.past.length > 0, canRedo: history.future.length > 0 };
}
