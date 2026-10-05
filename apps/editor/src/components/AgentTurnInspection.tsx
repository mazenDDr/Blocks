import { useState } from "react";

/** Recorded native data only; opening an inspection never invokes the model. */
export function AgentTurnInspection({ evidence }: { evidence: any }) {
  const [selected, setSelected] = useState("");
  const call = evidence.contexts.find((c: any) => c.callId === selected) ?? evidence.contexts[0];
  return <section aria-label="served agent inspection">
    <h4>Native agent turn</h4>
    <p className="provenance">Execution {evidence.executionId} · thread {evidence.threadId} · graph {evidence.graphHash} · source {evidence.sourceRunId}</p>
    <p>{evidence.isolation}</p><p>{evidence.capturePolicy}</p>
    <pre className="prod-recorded">{JSON.stringify({ provider: evidence.provider, modelCalls: evidence.modelCalls, stateSha256: evidence.stateSha256, eventsSha256: evidence.eventsSha256 }, null, 2)}</pre>
    {evidence.contexts.length ? <>
      <label>Recorded model call <select aria-label="served model call" value={call.callId} onChange={(e) => setSelected(e.target.value)}>
        {evidence.contexts.map((c: any) => <option key={c.callId} value={c.callId}>{c.nodeId} · {c.callId}</option>)}
      </select></label>
      <p className="provenance">Context {call.sha256} · provider latency {call.latencyMs} ms</p>
      <pre className="prod-recorded">{JSON.stringify(call.usage, null, 2)}</pre>
      {call.value ? <>
        <h4>Exact sent messages and source segments</h4>
        <pre className="prod-recorded">{JSON.stringify(call.value.messages, null, 2)}</pre>
        <details><summary>Full recorded context / settings / tokens / response / provenance</summary>
          <pre className="prod-recorded">{JSON.stringify(call.value, null, 2)}</pre>
        </details>
      </> : <p>Full context was not captured under this release’s policy.</p>}
    </> : <p>This native workflow made no model calls.</p>}
    {evidence.finalState != null && <details><summary>Actual final state and native state/control events</summary>
      <pre className="prod-recorded">{JSON.stringify({ state: evidence.finalState, events: evidence.events }, null, 2)}</pre>
    </details>}
  </section>;
}
