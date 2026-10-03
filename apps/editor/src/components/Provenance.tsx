import type { Provenance } from "../types";
import { shortHash } from "../util";

/** The provenance line every numeric panel carries. */
export function ProvLine({ p, extra }: { p: Provenance | null | undefined; extra?: string }) {
  if (!p) return null;
  const parts = [
    p.runId ? `run ${p.runId}` : "no run",
    p.graphHash ? `graph ${shortHash(p.graphHash)}` : null,
    p.checkpointStep != null ? `${p.checkpointStatus ? "checkpoint " : ""}step ${p.checkpointStep}${p.checkpointStatus ? ` (${p.checkpointStatus})` : ""}` : null,
    p.epoch != null && p.checkpointStep == null ? `epoch ${p.epoch}` : null,
    p.sampleId ? `sample ${p.sampleIndex != null ? "#" + p.sampleIndex + " " : ""}${p.sampleId}` : null,
    p.nodeId ? `node ${p.nodeId}` : null,
    p.normalization ? `normalization: ${p.normalization}` : null,
    p.source ?? null,
    extra ?? null,
  ].filter(Boolean);
  return <div className="prov" title={JSON.stringify(p)}>Provenance: {parts.join(" · ")}</div>;
}

export function NotRecorded({ message }: { message: string }) {
  return <div className="notrec"><b>Not recorded.</b> {message}</div>;
}
