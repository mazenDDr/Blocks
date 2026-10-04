import type { Graph, UiDoc } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: any) {
    super(typeof detail?.message === "string" ? detail.message : `HTTP ${status}`);
  }
}

async function call<T>(method: string, url: string, body?: unknown, headers: Record<string, string> = {}, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method, signal,
    headers: { ...(body !== undefined ? { "Content-Type": "application/json" } : {}), ...headers },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data);
  return data as T;
}

export const api = {
  get: <T,>(url: string, signal?: AbortSignal) => call<T>("GET", url, undefined, {}, signal),
  post: <T,>(url: string, body: unknown, headers?: Record<string, string>, signal?: AbortSignal) => call<T>("POST", url, body, headers, signal),
  put: <T,>(url: string, body: unknown) => call<T>("PUT", url, body),
  del: <T,>(url: string) => call<T>("DELETE", url),
  validate: (graph: Graph, signal?: AbortSignal) => call<any>("POST", "/api/validate", { graph }, {}, signal),
  saveProject: (id: string, graph: Graph, ui: UiDoc) => call<{ id: string; graphHash: string }>("PUT", `/api/projects/${encodeURIComponent(id)}`, { graph, ui }),
};

export function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    const diags = Array.isArray(e.detail?.diagnostics) ? e.detail.diagnostics : [];
    return [e.message, ...diags.map((d: any) => `${d.code} ${d.nodeId ?? ""}: ${d.message}`)].join("\n");
  }
  return e instanceof Error ? e.message : String(e);
}
