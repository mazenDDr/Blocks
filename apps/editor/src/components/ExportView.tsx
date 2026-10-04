import { useEffect, useState } from "react";
import { api, errorText } from "../api";
import type { Graph } from "../types";

const BACKENDS: [string, string][] = [["pytorch", "PyTorch"], ["keras", "TensorFlow / Keras 3"], ["jax", "JAX"]];

/** Generated native source for the draft graph on a chosen backend, with the `# node:` source map comments left visible. The graph is sent as it is
 *  (no save needed). An incompatible graph gets the list of unsupported nodes with their stable codes instead of code: nothing is substituted. */
export function ExportView({ projectId, graph, initialBackend, onClose }: { projectId: string; graph: Graph; initialBackend: string; onClose: () => void }) {
  const [backend, setBackend] = useState(BACKENDS.some(([b]) => b === initialBackend) ? initialBackend : "pytorch");
  const [res, setRes] = useState<{ code: string; graphHash: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setRes(null); setErr(null);
    api.post<{ code: string; graphHash: string }>("/api/export", { graph, backend }).then(setRes).catch((e) => setErr(errorText(e)));
  }, [graph, backend]);
  const title = BACKENDS.find(([b]) => b === backend)?.[1] ?? backend;
  return (
    <div className="modal" role="dialog" aria-label={`Exported ${title} code`}>
      <div className="modal-body">
        <div className="modal-head"><b>Native code export</b>
          <span>
            <select value={backend} onChange={(e) => setBackend(e.target.value)} aria-label="export backend">
              {BACKENDS.map(([b, t]) => <option key={b} value={b}>{t}</option>)}
            </select>{" "}
            {res && <button onClick={() => navigator.clipboard?.writeText(res.code)}>Copy</button>}{" "}
            <button onClick={onClose}>Close</button>
          </span>
        </div>
        {err && <div className="error pre" role="alert">{err}</div>}
        {res && (
          <>
            <pre className="code">{res.code.split("\n").map((l, i) => <div key={i}><span className="ln">{i + 1}</span>{l}</div>)}</pre>
            <div className="prov">Provenance: project {projectId} · graph {res.graphHash.slice(0, 8)} · {title} · generated text for reading and export; the app never executes it. Each statement carries a "# node:" comment mapping it to the graph. The graph stays the authoritative, editable form: editing this file does not change the graph.</div>
          </>
        )}
      </div>
    </div>
  );
}
