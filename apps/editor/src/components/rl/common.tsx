import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, errorText } from "../../api";
import type { GNode, Graph } from "../../types";
import type { Curves } from "./types";

export const nodeOfType = (g: Graph, type: string): GNode | undefined => g.nodes.find((n) => n.type === type);

/** Replace one node's config (fields merged) in the graph. */
export function patchConfig(setGraph: (f: (g: Graph) => Graph) => void, type: string, patch: Record<string, unknown>) {
  setGraph((g) => ({ ...g, nodes: g.nodes.map((n) => (n.type === type ? { ...n, config: { ...n.config, ...patch } } : n)) }));
}

export const fmt = (v: number | null | undefined, d = 4) => (v == null || !Number.isFinite(v) ? "—" : String(Number(v.toPrecision(d))));

/** Accumulates the incremental /curves feed of one run (unsmoothed events), polling while the run is live. */
export function useCurves(runId: string | null, live: boolean): { data: Curves | null; error: string | null } {
  const [data, setData] = useState<Curves | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cursor = useRef(-1);
  const acc = useRef<Curves | null>(null);
  const idRef = useRef<string | null>(null);
  useEffect(() => {
    if (idRef.current !== runId) { idRef.current = runId; cursor.current = -1; acc.current = null; setData(null); setError(null); }
    if (!runId) return;
    let alive = true;
    let busy = false;
    const pull = async () => {
      if (busy) return;
      busy = true;
      try {
        const d = await api.get<Curves>(`/api/rl/runs/${runId}/curves?after=${cursor.current}`);
        if (!alive || idRef.current !== runId) return;
        const prev = acc.current;
        const merged: Curves = prev ? {
          ...d, setup: d.setup ?? prev.setup, episodes: [...prev.episodes, ...d.episodes], updates: [...prev.updates, ...d.updates], evals: [...prev.evals, ...d.evals],
          captured: [...prev.captured, ...d.captured], targetSyncs: prev.targetSyncs + d.targetSyncs,
        } : d;
        cursor.current = d.cursor;
        acc.current = merged;
        setData(merged); setError(null);
      } catch (e) { if (alive) setError(errorText(e)); } finally { busy = false; }
    };
    pull();
    const h = live ? setInterval(pull, 1000) : undefined;
    return () => { alive = false; if (h) clearInterval(h); };
  }, [runId, live]);
  return { data, error };
}

export function Fact({ k, children }: { k: string; children: ReactNode }) {
  return <tr><td>{k}</td><td>{children}</td></tr>;
}

export function Collapsible({ title, children, open }: { title: string; children: ReactNode; open?: boolean }) {
  return <details open={open}><summary className="small">{title}</summary>{children}</details>;
}

/** Tab-local state hook for async JSON GETs (re-fetched when the url changes). */
export function useGet<T>(url: string | null, deps: unknown[] = []) {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: false });
  useEffect(() => {
    if (!url) { setState({ data: null, error: null, loading: false }); return; }
    let alive = true;
    setState((s) => ({ ...s, loading: true }));
    api.get<T>(url).then((d) => alive && setState({ data: d, error: null, loading: false })).catch((e) => alive && setState({ data: null, error: errorText(e), loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, ...deps]);
  return state;
}
