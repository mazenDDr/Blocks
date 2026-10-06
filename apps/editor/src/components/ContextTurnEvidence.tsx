/** Native recorded retrieval and memory-policy evidence. Inspection makes no calls. */
export function ContextTurnEvidence({ evidence }: { evidence: any }) {
  return <section aria-label="served retrieval and memory policies">
    <h4>Pinned retrieval and short-term policy decisions</h4>
    <p>Index snapshots belong to the registered version. Policy selection uses this turn’s bounded short-term state. Local hash embeddings are lexical hashing.</p>
    <details open><summary>Retrieved chunk IDs, scores and embedding identity</summary>
      <pre className="prod-recorded">{JSON.stringify(evidence.retrievals, null, 2)}</pre>
    </details>
    {(evidence.memorySelections ?? []).map((selection: any, index: number) => <details key={`${selection.applicationSha256}:${index}`} open>
      <summary>Native policy application · {selection.node}</summary>
      <p className="provenance">Application {selection.applicationSha256}</p>
      <pre className="prod-recorded">{JSON.stringify(selection.summary, null, 2)}</pre>
      {selection.value ? <pre className="prod-recorded">{JSON.stringify(selection.value, null, 2)}</pre> : <p>Full policy application was not captured under this release’s policy.</p>}
    </details>)}
  </section>;
}
