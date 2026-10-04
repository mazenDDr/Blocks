import { useState } from "react";
import type { Graph, OpInfo, TabularRunSummary, Validation } from "../types";
import { ConfigForm } from "./ConfigForm";
import { DomainResultView } from "./DomainViews";
import { TabularExplain } from "./Tabular";
import { TabularRunBar, TabularRunPanel } from "./TabularPanels";

export function DomainWorkspace({ projectId, graph, validation, ops, runs, runId, setRunId, reloadRuns, ensureSaved, onConfig }: {
  projectId: string; graph: Graph; validation: Validation | null; ops: Record<string, OpInfo>; runs: TabularRunSummary[];
  runId: string | null; setRunId: (id: string | null) => void; reloadRuns: () => void; ensureSaved: () => Promise<void>;
  onConfig: (node: string, patch: Record<string, unknown>) => void;
}) {
  const [selected, setSelected] = useState(graph.nodes[0]?.id);
  const node = graph.nodes.find((n) => n.id === selected) ?? graph.nodes[0];
  const area = graph.nodes.some((n) => n.type.startsWith("domain.vision")) ? "Vision" : graph.nodes.some((n) => n.type.startsWith("domain.nlp")) ? "NLP" : "Speech";
  const run = runs.find((r) => r.id === runId);
  const op = node && ops[node.type], view = node && validation?.nodes[node.id];
  return <div className="domain-workspace">
    <div className="domain-heading"><h2>{area} workspace</h2><p>Run the graph, then follow its recorded data, transforms and learned predictions. Every view belongs to the selected immutable run.</p></div>
    <TabularRunBar runs={runs} runId={runId} setRunId={setRunId} currentHash={validation?.graphHash} />
    <nav className="tabs" aria-label="domain stages">{graph.nodes.map((n) => <button key={n.id} className={n.id === node?.id ? "on" : ""} onClick={() => setSelected(n.id)}>{ops[n.type]?.displayName ?? n.id}</button>)}</nav>
    {node && <div className="domain-layout"><aside><h3>{node.id}</h3><p>{op?.purpose}</p>
      {view?.diagnostics.map((d, i) => <p key={i} className={d.severity === "error" ? "errbadge" : "warn"}>{d.code}: {d.message}</p>)}
      {op && <ConfigForm op={op} node={node} resolved={view?.resolvedConfig} onChange={(patch) => onConfig(node.id, patch)} />}
      <TabularExplain explain={view?.explain} purpose={op?.purpose ?? ""} typed />
    </aside><section><DomainResultView key={`${runId}/${node.id}/${run?.maxSeq}`} runId={runId} node={node.id} /></section></div>}
    <TabularRunPanel projectId={projectId} graph={graph} validation={validation} runs={runs} reloadRuns={reloadRuns} runId={runId} setRunId={setRunId} ensureSaved={ensureSaved} onSelectNode={setSelected} />
  </div>;
}
