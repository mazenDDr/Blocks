import { useEffect, useState } from "react";
import { api } from "../api";
import type { BackendInfoJson, CompatNode, CompatReport, Graph } from "../types";

export function useBackends() {
  const [list, setList] = useState<BackendInfoJson[]>([]);
  useEffect(() => { api.get<{ backends: BackendInfoJson[] }>("/api/backends").then((r) => setList(r.backends)).catch(() => {}); }, []);
  return list;
}

export function statusBadge(n: CompatNode | undefined) {
  if (!n) return null;
  return <span className={`badge compat-${n.status}`}>{n.status === "converted" ? `converted (${n.conversions.length})` : n.status}</span>;
}

/** The backend of a model project: what each backend is for, whether it is installed here, the per-node compatibility report with every conversion and
 *  every refusal (stable code + reason), the facets VISION 14.2 asks for, and what is NOT implemented (training). */
export function BackendWorkspace({ graph, setBackend, report, pending, error, backends, onExport }: {
  graph: Graph; setBackend: (b: string) => void; report: CompatReport | null; pending: boolean; error: string | null; backends: BackendInfoJson[]; onExport: (b: string) => void;
}) {
  const [filter, setFilter] = useState<"all" | "converted" | "unsupported">("all");
  const sel = backends.find((b) => b.id === graph.backend);
  const rows = report ? Object.entries(report.nodes).filter(([, n]) => filter === "all" || n.status === filter) : [];
  return (
    <div className="fullws backendws">
      <h3>Backend of this model graph</h3>
      <div className="bk-cards">
        {backends.map((b) => (
          <button key={b.id} className={`bk-card ${b.id === graph.backend ? "on" : ""}`} onClick={() => setBackend(b.id)} aria-pressed={b.id === graph.backend}>
            <b>{b.title}</b> {b.id === graph.backend && <span className="badge ok">selected</span>}
            <div className="small muted">{b.role}</div>
            <div className="small">{b.available ? <span className="good">installed: {Object.entries(b.versions).map(([k, v]) => `${k} ${v}`).join(", ")}</span> : <span className="bad" title={b.reason ?? ""}>unavailable: {b.reason}</span>}</div>
            <div className="small muted">pinned: {Object.entries(b.pinned).map(([k, v]) => `${k} ${v ?? "?"}`).join(", ")}</div>
            <div className="small"><b>Training:</b> {b.training}</div>
          </button>
        ))}
      </div>
      <div className="bknote small">
        Switching the backend never rewrites the graph: every setting stays as authored. A node the selected backend cannot run is reported below and in the canvas
        with a stable code and a reason, and the graph is not executable (or exportable) on that backend until it is fixed. Nothing is substituted. PyTorch training runs
        (Run panel) are the only worker runs; {sel && sel.id !== "pytorch" ? `${sel.title} has no training runs here (forward, loss, gradients and one SGD step are verified by tests and the API).` : "other backends execute forward, loss, gradients and one SGD step only."}
      </div>
      <h3>Compatibility report — {report?.title ?? graph.backend} {pending && <small className="muted">(checking…)</small>}</h3>
      {error && <div className="error pre">{error}</div>}
      {report && (
        <>
          <div className={`vsum ${report.ok ? "good" : "bad"}`}>
            {report.ok ? (report.executable ? "Compatible and executable here." : `Compatible, but the backend is unavailable: ${report.availabilityReason}`) : "Not executable on this backend until the unsupported items below are resolved."}
            {" "}<small>{report.counts.supported} supported · {report.counts.converted} with conversions · {report.counts.unsupported} unsupported</small>
          </div>
          {report.structural.map((d, i) => <div key={i} className="error"><b>{d.code}</b> {d.nodeId ?? ""}: {d.message}</div>)}
          {report.sharing.length > 0 && <div className="small">Parameter sharing preserved: {report.sharing.map((s) => `${s.users.join(", ")} use the tensors of ${s.owner}`).join("; ")} (one set of variables on every backend, never silently split).</div>}
          <div className="small">Show: {(["all", "converted", "unsupported"] as const).map((f) => <button key={f} className={`link ${filter === f ? "on" : ""}`} onClick={() => setFilter(f)} style={{ marginRight: 8, fontWeight: filter === f ? 700 : 400 }}>{f}</button>)}</div>
          <table className="compat">
            <thead><tr><th>Node</th><th>Operation</th><th>Status</th><th>Conversions / reason</th></tr></thead>
            <tbody>
              {rows.map(([id, n]) => (
                <tr key={id} className={n.status === "unsupported" ? "bad" : ""}>
                  <td>{id}</td><td><code>{n.type}</code></td><td>{statusBadge(n)}</td>
                  <td>{n.status === "unsupported" ? <><b>{n.code}</b> {n.reason}</> : n.conversions.length ? <ul>{n.conversions.map((c, i) => <li key={i}><span className="badge">{c.kind}</span> {c.detail}</li>)}</ul> : <span className="muted">none: runs as authored</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4>Facets (VISION 14.2)</h4>
          <table className="compat"><tbody>
            {Object.entries(report.facets).map(([k, v]) => <tr key={k}><th>{k}</th><td>{v}</td></tr>)}
            <tr><th>initialization</th><td>{report.init}</td></tr>
            <tr><th>layout</th><td>{report.layout}</td></tr>
          </tbody></table>
        </>
      )}
      <div style={{ marginTop: 10 }}>
        <b>Export native code:</b> {backends.map((b) => <button key={b.id} onClick={() => onExport(b.id)} style={{ marginLeft: 6 }}>{b.title}</button>)}
        <span className="small muted"> deterministic text with “# node:” source-map comments; an incompatible graph shows the refusals instead.</span>
      </div>
    </div>
  );
}
