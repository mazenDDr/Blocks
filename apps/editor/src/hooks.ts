import { useEffect, useRef, useState } from "react";
import { api, errorText } from "./api";
import type { EpochEnd, Graph, RunEvent, RunSummary, Validation } from "./types";

/** Debounced validation of the draft graph. Stale responses are aborted, so shapes always match the latest edit. */
export function useValidation(graph: Graph, delay = 250) {
  const [state, setState] = useState<{ data: Validation | null; pending: boolean; error: string | null }>({ data: null, pending: true, error: null });
  useEffect(() => {
    const ctl = new AbortController();
    setState((s) => ({ ...s, pending: true }));
    const t = setTimeout(() => {
      api.validate(graph, ctl.signal)
        .then((data) => setState({ data, pending: false, error: null }))
        .catch((e) => { if (!ctl.signal.aborted) setState((s) => ({ ...s, pending: false, error: errorText(e) })); });
    }, delay);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [graph, delay]);
  return state;
}

export function usePolling<T>(url: string | null, intervalMs: number, deps: unknown[] = []): { data: T | null; error: string | null; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!url) { setData(null); return; }
    let alive = true;
    const run = () => api.get<T>(url).then((d) => { if (alive) { setData(d); setError(null); } }).catch((e) => { if (alive) setError(errorText(e)); });
    run();
    const h = intervalMs > 0 ? setInterval(run, intervalMs) : undefined;
    return () => { alive = false; if (h) clearInterval(h); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, intervalMs, tick, ...deps]);
  return { data, error, reload: () => setTick((t) => t + 1) };
}

export function useRuns(projectId: string | null) {
  const { data, error, reload } = usePolling<{ runs: RunSummary[] }>(projectId ? `/api/runs?project=${encodeURIComponent(projectId)}` : "/api/runs", 2000);
  return { runs: data?.runs ?? [], error, reload };
}

export interface StreamState {
  trainLoss: [number, number][]; epochs: EpochEnd[]; status: string | null; connected: boolean; lastSeq: number;
  events: number; log: { seq: number; type: string; text: string }[];
}
const EMPTY: StreamState = { trainLoss: [], epochs: [], status: null, connected: false, lastSeq: -1, events: 0, log: [] };
const EVENT_TYPES = ["run_queued", "run_preparing", "run_started", "train_step", "epoch_end", "val_detail", "weight_stats", "checkpoint",
  "cancel_acknowledged", "validation_error", "error", "run_finished"];

/** Live run state from the SSE endpoint. EventSource reconnects with Last-Event-ID; events with seq <= lastSeq are ignored. */
export function useRunStream(runId: string | null, onTerminal?: () => void): StreamState {
  const [state, setState] = useState<StreamState>(EMPTY);
  const cb = useRef(onTerminal);
  cb.current = onTerminal;
  useEffect(() => {
    setState(EMPTY);
    if (!runId) return;
    let last = -1;
    let pending: RunEvent[] = [];
    let timer: ReturnType<typeof setTimeout> | null = null;
    const flush = () => {
      timer = null;
      const batch = pending; pending = [];
      if (!batch.length) return;
      setState((s) => {
        const n: StreamState = { ...s, trainLoss: [...s.trainLoss], epochs: [...s.epochs], log: [...s.log], events: s.events + batch.length };
        for (const e of batch) {
          n.lastSeq = e.seq;
          if (e.type === "train_step") n.trainLoss.push([e.data.step, e.data.loss]);
          else if (e.type === "epoch_end") n.epochs.push(e.data);
          else if (e.type === "run_finished") n.status = e.data.status;
          else if (e.type === "run_started") n.status = "running";
          if (e.type !== "train_step" && e.type !== "weight_stats" && e.type !== "val_detail")
            n.log.push({ seq: e.seq, type: e.type, text: e.type === "error" ? e.data.message : e.type === "run_finished" ? `${e.data.status}${e.data.error ? ": " + e.data.error : ""}` : e.type === "epoch_end" ? `epoch ${e.data.epoch} val acc ${(e.data.val_acc * 100).toFixed(1)}%` : "" });
        }
        return n;
      });
    };
    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    const onEv = (m: MessageEvent) => {
      const e: RunEvent = JSON.parse(m.data);
      if (e.seq <= last) return; // replayed after a reconnect
      last = e.seq;
      pending.push(e);
      if (!timer) timer = setTimeout(flush, 120);
    };
    for (const t of EVENT_TYPES) es.addEventListener(t, onEv as EventListener);
    es.addEventListener("end", () => { es.close(); flush(); setState((s) => ({ ...s, connected: false })); cb.current?.(); });
    es.onopen = () => setState((s) => ({ ...s, connected: true }));
    es.onerror = () => setState((s) => ({ ...s, connected: false }));
    return () => { es.close(); if (timer) clearTimeout(timer); };
  }, [runId]);
  return state;
}

/** POST to an inspect/infer endpoint whenever `req` changes. `req === null` means "nothing to ask". */
export function useInspect<T>(url: string | null, body: unknown | null) {
  const [state, setState] = useState<{ data: T | null; loading: boolean; error: string | null }>({ data: null, loading: false, error: null });
  const key = url && body ? url + JSON.stringify(body) : null;
  useEffect(() => {
    if (!key || !url) { setState({ data: null, loading: false, error: null }); return; }
    const ctl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    api.post<T>(url, body, undefined, ctl.signal)
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((e) => { if (!ctl.signal.aborted) setState({ data: null, loading: false, error: errorText(e) }); });
    return () => ctl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
