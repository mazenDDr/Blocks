import { useState } from "react";
import { api, errorText } from "../api";
import { usePolling } from "../hooks";
import type { Graph } from "../types";
interface Model {
  modelId: string; runId: string; node: string; family: string; epochs: number; available: boolean;
  checkpointSha256: string; error?: { code: string; message: string }; inference: { input?: string; readout?: string; decoding?: string };
}
interface Result { predictions: { maskPng?: string; size?: number[]; text?: string; words?: { text: string; start: number; end: number; label: string }[]; tokenTimes?: number[] }[]; provenance: unknown }
export function DomainModels({ runId, graph, onConfig }: { runId: string | null; graph: Graph; onConfig: (node: string, patch: Record<string, unknown>) => void }) {
  const models = usePolling<{ models: Model[] }>(runId ? `/api/domain/models?runId=${encodeURIComponent(runId)}` : null, 3000);
  const [input, setInput] = useState(""); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null); const [extra, setExtra] = useState(2);
  const model = models.data?.models.find((m) => m.available); const missing = models.data?.models.find((m) => !m.available);
  const resume = () => {
    if (!model) return;
    const node = graph.nodes.find((n) => n.id === model.node);
    if (!node) { setError("The recorded training node is absent from this draft."); return; }
    onConfig(node.id, { resume_model_id: model.modelId, epochs: model.epochs + extra });
    if (model.family === "nlp") {
      const source = graph.edges.find((e) => e.to.node === node.id && e.to.port === "tokens");
      if (source) onConfig(source.from.node, { fitted_model_id: model.modelId });
    }
    setError(""); setResult(null);
  };
  const load = async () => {
    if (!model) return;
    setBusy(true); setError("");
    try { const data = await api.get<{ records: unknown[] }>(`/api/domain/models/${model.modelId}/example`); setInput(JSON.stringify({ records: data.records }, null, 2)); }
    catch (e) { setError(errorText(e)); } finally { setBusy(false); }
  };
  const infer = async () => {
    if (!model) return;
    setBusy(true); setError(""); setResult(null);
    try { setResult(await api.post<Result>(`/api/domain/models/${model.modelId}/predict`, JSON.parse(input))); }
    catch (e) { setError(errorText(e)); } finally { setBusy(false); }
  };
  return <section className="prod-panel domain-models" aria-label="persisted domain model">
    <h3>Saved model and local inference</h3>
    <p>Completed runs save native weights, Adam state, RNG and pinned preprocessing. Inference uses those artifacts without training or reading the dataset. Local CPU, at most four records and two concurrent requests.</p>
    {(error || models.error) && <p role="alert">{error || models.error}</p>}
    {!model && !missing && <p>No completed model checkpoint recorded for this run. Older runs require a new training run.</p>}
    {missing && <p role="alert">{missing.error?.code}: {missing.error?.message}</p>}
    {model && <>
      <p>{model.family} · completed epoch {model.epochs} · run {model.runId} · node {model.node}</p>
      <p>Model {model.modelId.slice(0, 16)} · weights {model.checkpointSha256.slice(0, 16)}. {model.inference.input ?? model.inference.readout ?? model.inference.decoding}</p>
      <a href={`/api/domain/models/${model.modelId}/checkpoint`} download>Download native checkpoint</a>{" · "}
      <a href={`/api/domain/models/${model.modelId}`} target="_blank" rel="noreferrer">Open pinned manifest</a>
      <div className="row"><label>Additional epochs <input aria-label="additional domain epochs" type="number" min={1} max={100} value={extra} onChange={(e) => setExtra(Number(e.target.value))} /></label>
        <button disabled={busy || !Number.isInteger(extra) || extra < 1 || extra > 100} onClick={resume}>Prepare checkpoint continuation</button></div>
      <p>Continuation edits the draft; use Run to create a new run. Data, tokenizer, split, model and optimizer settings must match. Total epochs must remain within the operation's limit; changed inputs are refused.</p>
      <button disabled={busy} onClick={load}>Load recorded validation input</button>
      <label>Inference request JSON <textarea aria-label="domain inference request" rows={6} value={input} onChange={(e) => setInput(e.target.value)} /></label>
      <button disabled={busy || !input.trim()} onClick={infer}>Predict with saved model</button>
      <p>Input contracts: vision uses base64 RGB PNG after explicit geometry; NLP uses original text and the saved tokenizer; audio uses normalized PCM channels and the saved sample rate. The loaded example is labelled SYNTHETIC held-out data.</p>
    </>}
    {result && <div aria-label="saved model predictions">
      {result.predictions.map((p, i) => <article key={i}>
        {p.maskPng && <figure><img src={`data:image/png;base64,${p.maskPng}`} alt={`Predicted segmentation for input ${i}`} width={240} /><figcaption>Saved model's predicted class mask</figcaption></figure>}
        {p.text != null && <p>{model?.family === "nlp" ? "Input text" : "Predicted text"}: <code>{p.text || "(empty greedy prediction)"}</code></p>}
        {p.words && <table><thead><tr><th>Original word</th><th>Character span</th><th>Predicted label</th></tr></thead><tbody>{p.words.map((w) => <tr key={`${w.start}:${w.end}`}><td>{w.text}</td><td>{w.start}–{w.end}</td><td>{w.label}</td></tr>)}</tbody></table>}
        {p.tokenTimes && <p>Greedy token onset seconds: {p.tokenTimes.join(", ") || "no tokens"}; these are not forced alignments.</p>}
      </article>)}
      <details><summary>Complete prediction and provenance</summary><pre className="prod-recorded">{JSON.stringify(result, null, 2)}</pre></details>
    </div>}
  </section>;
}
