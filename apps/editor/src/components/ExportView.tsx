import { useEffect, useState } from "react";
import { api, errorText } from "../api";

/** Generated PyTorch for the saved project, with the `# node:` source map comments left visible. */
export function ExportView({ projectId, ensureSaved, onClose }: { projectId: string; ensureSaved: () => Promise<void>; onClose: () => void }) {
  const [res, setRes] = useState<{ code: string; graphHash: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    ensureSaved().then(() => api.get<{ code: string; graphHash: string }>(`/api/projects/${encodeURIComponent(projectId)}/export/pytorch`)).then(setRes).catch((e) => setErr(errorText(e)));
  }, [projectId, ensureSaved]);
  return (
    <div className="modal" role="dialog" aria-label="Exported PyTorch code">
      <div className="modal-body">
        <div className="modal-head"><b>PyTorch export</b>
          <span>
            {res && <button onClick={() => navigator.clipboard?.writeText(res.code)}>Copy</button>}{" "}
            <button onClick={onClose}>Close</button>
          </span>
        </div>
        {err && <div className="error pre">{err}</div>}
        {res && (
          <>
            <pre className="code">{res.code.split("\n").map((l, i) => <div key={i}><span className="ln">{i + 1}</span>{l}</div>)}</pre>
            <div className="prov">Provenance: project {projectId} · graph {res.graphHash.slice(0, 8)} · generated text for reading and export; the app never executes it. Each statement carries a "# node:" comment mapping it to the graph.</div>
          </>
        )}
      </div>
    </div>
  );
}
