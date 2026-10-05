import { useState } from "react";
import { api, ApiError, errorText } from "../api";
import { usePolling } from "../hooks";
import { AgentTurnInspection } from "./AgentTurnInspection";

interface Candidate { runId: string; node: string; pipelineSha256: string; graphHash: string; adapter?: "tabular" | "domain" | "model" | "rl" | "unsup" | "agent" | "conversation"; family?: string }
interface Version { id: string; name: string; runId: string; node: string; owner: string; intendedUse: string; limitations: string; pipelineSha256: string; manifest: any; adapter?: "domain" | "model" | "rl" | "unsup" | "agent" | "conversation"; family?: string; modelId?: string }
interface Config { target: "local" | "staging"; namespace: string; concurrency: number; queueLimit: number; timeoutSeconds: number; maxBatch: number; sessionMode: "stateless" | "counter" | "conversation"; captureInputs: boolean }
interface Release { id: string; versionId: string; config: Config; compatibility: any; resources: any }
interface Overview { candidates: Candidate[]; versions: Version[]; releases: Release[]; routes: { target: string; namespace: string; release: string }[]; aliases: { name: string; version: string }[]; lifecycle: any[]; traffic: any[]; capabilities: any }
type Tab = "Registry" | "Release" | "Requests" | "Traffic" | "Monitoring";
const INITIAL: Config = { target: "local", namespace: "lab", concurrency: 2, queueLimit: 8, timeoutSeconds: 10, maxBatch: 32, sessionMode: "stateless", captureInputs: false };
const short = (s: string) => s.slice(0, 12);
function Recorded({ value }: { value: unknown }) { return <pre className="prod-recorded">{JSON.stringify(value, null, 2)}</pre>; }
function Metrics({ rows }: { rows: [string, unknown][] }) { return <table><tbody>{rows.map(([label, value]) => <tr key={label}><th scope="row">{label}</th><td>{value == null ? "not recorded" : String(value)}</td></tr>)}</tbody></table>; }

export function ProductionWorkspace({ onOpenRun }: { onOpenRun: (id: string) => void }) {
  const overview = usePolling<Overview>("/api/production", 3000);
  const data = overview.data;
  const [tab, setTab] = useState<Tab>("Registry");
  const [candidateKey, setCandidate] = useState("");
  const [versionId, setVersion] = useState("");
  const [releaseId, setRelease] = useState("");
  const [name, setName] = useState("Synthetic sensor classifier");
  const [owner, setOwner] = useState("local scientist");
  const [use, setUse] = useState("Evaluate the labelled SYNTHETIC fixture locally");
  const [limits, setLimits] = useState("Synthetic example; no evidence for real-world sensor accuracy");
  const [alias, setAlias] = useState("candidate");
  const [config, setConfig] = useState<Config>(INITIAL);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [payload, setPayload] = useState("[]");
  const [reference, setReference] = useState<unknown>(null);
  const [user, setUser] = useState("local-user");
  const [session, setSession] = useState("investigation");
  const [requestId, setRequestId] = useState("");
  const [trace, setTrace] = useState<any>(null);
  const [conversation, setConversation] = useState<unknown>(null);
  const [replay, setReplay] = useState<unknown>(null);
  const [labels, setLabels] = useState("[]");
  const [trafficPayloads, setTrafficPayloads] = useState("[]");
  const [pattern, setPattern] = useState("steady");
  const [rate, setRate] = useState(5);
  const [duration, setDuration] = useState(2);
  const [concurrency, setConcurrency] = useState(2);
  const [maxRequests, setMaxRequests] = useState(100);
  const [expected, setExpected] = useState("[200]");
  const [jobId, setJob] = useState<string | null>(null);
  const [since, setSince] = useState("0");
  const candidates = data?.candidates ?? [];
  const candidate = candidates.find((c) => `${c.runId}:${c.node}` === candidateKey) ?? candidates.at(-1);
  const version = data?.versions.find((v) => v.id === versionId);
  const release = data?.releases.find((r) => r.id === releaseId);
  const route = release ? data?.routes.find((r) => r.target === release.config.target && r.namespace === release.config.namespace) : undefined;
  const requests = usePolling<{ requests: any[] }>(tab === "Requests" && release ? `/api/production/requests?release=${release.id}` : null, 1500);
  const job = usePolling<any>(jobId ? `/api/production/traffic/${jobId}` : null, 1000);
  const monitor = usePolling<any>(tab === "Monitoring" && release ? `/api/production/releases/${release.id}/monitor?since=${encodeURIComponent(since)}` : null, 3000);
  const loadResult = job.data?.result ?? job.data;
  const act = async (title: string, fn: () => Promise<void>) => {
    setBusy(title); setError(null); setNotice("");
    try { await fn(); overview.reload(); } catch (e) { setError(e instanceof ApiError && Array.isArray(e.detail) ? e.detail.map((d: any) => `${d.loc?.join(".")}: ${d.msg}`).join("\n") : errorText(e)); } finally { setBusy(null); }
  };
  const update = <K extends keyof Config>(key: K, value: Config[K]) => setConfig((c) => ({ ...c, [key]: value }));
  const selectedVersion = <label>Registered version <select aria-label="registered version" value={versionId} onChange={(e) => { setVersion(e.target.value); setReference(null); const v = data?.versions.find((v) => v.id === e.target.value); if (["agent", "conversation"].includes(v?.adapter ?? "")) setConfig((c) => ({ ...c, maxBatch: 1, sessionMode: v?.adapter === "conversation" ? "conversation" : "stateless" })); }}>
    <option value="">choose…</option>{data?.versions.map((v) => <option key={v.id} value={v.id}>{v.name} · {short(v.id)}</option>)}</select></label>;
  const selectedRelease = <label>Release <select aria-label="production release" value={releaseId} onChange={(e) => { setRelease(e.target.value); setTrace(null); setReplay(null); setConversation(null); }}>
    <option value="">choose…</option>{data?.releases.map((r) => <option key={r.id} value={r.id}>{r.config.target}/{r.config.namespace} · {short(r.id)}</option>)}</select></label>;
  return <div className="fullws"><section className="production-workspace">
    <header className="prod-header"><div><h2>Production investigation</h2><p>Real local serving · staging uses a separate local route · one control process / one replica</p></div>
      <button onClick={overview.reload}>Refresh records</button></header>
    <p className="notice-inline synthetic">The production_sensors example uses labelled SYNTHETIC sensors. Measured local traffic is evidence for this bounded test. Remote deployment, streaming and autoscaling are not implemented.</p>
    <nav className="tabs" aria-label="production tabs">{(["Registry", "Release", "Requests", "Traffic", "Monitoring"] as Tab[]).map((t) => <button key={t} className={tab === t ? "on" : ""} aria-pressed={tab === t} onClick={() => setTab(t)}>{t}</button>)}</nav>
    {(error || overview.error) && <p role="alert" className="error">{error ?? overview.error}</p>}{notice && <p role="status">{notice}</p>}{busy && <p role="status">{busy}…</p>}
    {tab === "Registry" && <div className="prod-grid"><section className="prod-card"><h3>Register a recorded native pipeline</h3>
      {!candidates.length && <p>No compatible models recorded yet. Run production_sensors, a supported tabular regression/classification graph, or a vision/NLP/speech example first. Old runs without a recorded pipeline or domain checkpoint need a rerun; registration never trains.</p>}
      <label>Completed run / estimator <select aria-label="registration candidate" value={candidate ? `${candidate.runId}:${candidate.node}` : ""} onChange={(e) => setCandidate(e.target.value)}>
        {candidates.map((c) => <option key={`${c.runId}:${c.node}`} value={`${c.runId}:${c.node}`}>{c.runId} / {c.node} · {c.adapter === "conversation" ? "native agent conversation (LangGraph / local Ollama)" : c.adapter === "agent" ? "isolated agent turn (LangGraph / local Ollama)" : c.adapter === "domain" ? `${c.family} model (PyTorch)` : c.adapter === "model" ? "image classifier graph (PyTorch)" : c.adapter === "rl" ? "greedy DQN policy (PyTorch)" : c.adapter === "unsup" ? `${c.family} (scikit-learn, unsupervised)` : "tabular pipeline (scikit-learn)"}</option>)}</select></label>
      <label>Name <input value={name} onChange={(e) => setName(e.target.value)} /></label><label>Owner <input value={owner} onChange={(e) => setOwner(e.target.value)} /></label>
      <label>Intended use <textarea value={use} onChange={(e) => setUse(e.target.value)} /></label><label>Limitations <textarea value={limits} onChange={(e) => setLimits(e.target.value)} /></label>
      <button disabled={!!busy || !candidate} onClick={() => act("Registering version", async () => { const v = await api.post<Version>("/api/production/versions", { runId: candidate!.runId, node: candidate!.node, name, owner, intendedUse: use, limitations: limits }); setVersion(v.id); if (["agent", "conversation"].includes(v.adapter ?? "")) setConfig((c) => ({ ...c, maxBatch: 1, sessionMode: v.adapter === "conversation" ? "conversation" : "stateless" })); setNotice(`Registered immutable version ${v.id}`); })}>Register version</button>
      {candidate && <p className="provenance">run {candidate.runId} · graph {candidate.graphHash} · pipeline {candidate.pipelineSha256 ?? "pinned at registration"}</p>}
    </section><section className="prod-card"><h3>Pinned pipeline and evidence</h3>{selectedVersion}
      {version ? <><p>{version.owner} · {version.intendedUse}</p><p>{version.limitations}</p><p className="provenance">version {version.id} · run {version.runId} · node {version.node}</p>
        <button onClick={() => onOpenRun(version.runId)}>Open source run</button>
        {["agent", "conversation"].includes(version.adapter ?? "") ? <><h4>Pinned native agent: text input → LangGraph turn → text output</h4>
          <Metrics rows={[["Input fields", version.manifest.inputFields.join(", ")], ["Output field", version.manifest.outputField], ["Mode", version.adapter === "conversation" ? "persistent native conversation; no tool/memory effects" : "isolated single turn; no tool/memory effects"]]} />
          <Recorded value={{ provider: version.manifest.provider, limits: version.manifest.limits, environment: version.manifest.environment }} /></>
        : version.adapter === "unsup" ? <><h4>Pinned {version.family}: raw features → fitted scaler → native estimator</h4>
          <Metrics rows={[["Features", version.manifest.features.join(", ")], ["Scaled", version.manifest.scaled ? "fitted StandardScaler" : "no"], ["Estimator parameters", JSON.stringify(version.manifest.params)],
            ["Fitted on", `${version.manifest.fittedOn?.rows ?? "?"} rows`], ["Frozen reference", `${version.manifest.referenceRows} fitted rows (in-sample)`]]} />
          <p>Clusters are not classes. Label-based evidence is external agreement (ARI/NMI) with labels you supply; PCA reports reconstruction error.</p></>
        : version.adapter === "rl" ? <><h4>Pinned policy: observation → final Q-network → greedy action</h4>
          <Metrics rows={[["Environment", version.manifest.envId], ["Input contract", version.manifest.inputContract], ["Policy", version.manifest.policy],
            ["Final Q-network", `${short(version.manifest.checkpointSha256)} (policy version ${version.manifest.policyVersion})`], ["Actions", version.manifest.actionSpace.n],
            ["Frozen reference", `${version.manifest.referenceRows} replay-buffer observations`]]} />
          <Recorded value={{ evaluation: version.manifest.evaluation, observationSpace: version.manifest.observationSpace, environment: version.manifest.environment }} /></>
        : version.adapter === "model" ? <><h4>Pinned model graph: image → training preprocessing → lowered PyTorch graph → softmax</h4>
          <Metrics rows={[["Input contract", version.manifest.inputContract], ["Preprocessing", version.manifest.preprocessing], ["Checkpoint", `${short(version.manifest.checkpointSha256)} (step ${version.manifest.checkpointStep})`],
            ["Classes", version.manifest.outputSchema.classes.join(", ")], ["Frozen reference", `${version.manifest.referenceRows} held-out images (${short(version.manifest.referenceSha256)})`]]} />
          <Recorded value={{ environment: version.manifest.environment, source: version.manifest.source, evaluation: version.manifest.evaluation }} /></>
        : version.adapter === "domain" ? <><h4>Pinned {version.family} model: request → native preprocessing → PyTorch forward → decoding</h4>
          <Metrics rows={[["Input contract", version.manifest.inputContract], ["Model manifest", short(version.modelId ?? "")], ["Checkpoint (native state dict)", short(version.manifest.checkpointSha256)],
            ["Completed epochs", version.manifest.epochs], ["Tokenizer", version.manifest.tokenizer], ["Labels / classes", (version.manifest.outputSchema.classes ?? []).join(", ") || null]]} />
          <Recorded value={{ environment: version.manifest.environment, source: version.manifest.source }} /></>
        : <><h4>Input → fitted transforms → estimator → output</h4>
        <Recorded value={version.manifest.inputSchema} /><table><thead><tr><th>Node</th><th>Native operation</th><th>Fitted identity</th></tr></thead><tbody>
          {version.manifest.steps.map((s: any) => <tr key={s.node}><td>{s.node}</td><td>{s.type}</td><td>{s.fitNode ? short(version.manifest.fitArtifacts[s.fitNode]) : "declared column selection"}</td></tr>)}</tbody></table>
        <Recorded value={{ featureOrder: version.manifest.featureOrder, outputSchema: version.manifest.outputSchema, environment: version.manifest.environment, modelSha256: version.manifest.modelSha256 }} /></>}
        <label>Movable alias <input aria-label="version alias" value={alias} onChange={(e) => setAlias(e.target.value)} /></label><button disabled={!!busy} onClick={() => act("Updating alias", async () => { await api.put(`/api/production/aliases/${encodeURIComponent(alias)}`, { versionId }); setNotice("Alias updated; existing releases retain their pinned version."); })}>Point alias to selected version</button>
        <details><summary>Full immutable manifest / source / evaluations</summary><Recorded value={version.manifest} /></details></> : <p>Select a version to inspect its recorded pipeline.</p>}
      <Recorded value={data?.aliases ?? []} />
    </section></div>}
    {tab === "Release" && <div className="prod-grid"><section className="prod-card"><h3>Build an immutable release candidate</h3>{selectedVersion}
      <label>Target <select value={config.target} onChange={(e) => update("target", e.target.value as Config["target"])}><option>local</option><option>staging</option></select></label>
      <label>Namespace <input value={config.namespace} onChange={(e) => update("namespace", e.target.value)} /></label>
      {(["concurrency", "queueLimit", "timeoutSeconds", "maxBatch"] as const).map((k) => <label key={k}>{k}<input aria-label={`serving ${k}`} type="number" value={config[k]} onChange={(e) => update(k, Number(e.target.value))} /></label>)}
      <label>Session state <select aria-label="serving session mode" value={config.sessionMode} onChange={(e) => update("sessionMode", e.target.value as Config["sessionMode"])}><option value="stateless" disabled={version?.adapter === "conversation"}>Stateless</option><option value="counter" disabled={["agent", "conversation"].includes(version?.adapter ?? "")}>Durable request counter</option><option value="conversation" disabled={version?.adapter !== "conversation"}>Native conversation checkpoint</option></select></label>
      <p>Successful counter/checkpoint updates are serialized and isolated by release/user/session. Conversation mode persists native state/history even with trace capture off; failed/cancelled turns are discarded. Caller-declared user IDs do not provide authentication. No native estimator or training state is updated.</p>
      <label><input type="checkbox" checked={config.captureInputs} onChange={(e) => update("captureInputs", e.target.checked)} /> Capture inputs, bounded transformed values and agent contexts/state/events for replay/drift</label>
      <button disabled={!!busy || !version} onClick={() => act("Checking and warming candidate", async () => { const r = await api.post<Release>("/api/production/releases", { versionId, config: version?.adapter === "conversation" ? { ...config, maxBatch: 1, sessionMode: "conversation" } : version?.adapter === "agent" ? { ...config, maxBatch: 1, sessionMode: "stateless" } : version?.adapter === "domain" ? { ...config, maxBatch: Math.min(config.maxBatch, 4) } : config }); setRelease(r.id); setNotice(`Candidate ${r.id} ready for explicit deployment.`); })}>Preview release candidate</button>{["agent", "conversation"].includes(version?.adapter ?? "") && <p>Agent releases use maxBatch=1 and the registered graph’s stateless/conversation mode. New conversations start at declared defaults; research history is never copied. Warmup does not create a live conversation. Warmup invokes the real local model. Capture also retains exact sent contexts and state/control events. Ollama manages the model device; placement and cost are not measured.</p>}{version?.adapter === "domain" && <p>Domain models take 1–4 records per request; maxBatch is capped at 4.</p>}
    </section><section className="prod-card"><h3>Review and deploy</h3>{selectedRelease}
      {release && <><p className="provenance">release {release.id} · version {release.versionId}</p><Recorded value={{ config: release.config, compatibility: release.compatibility, resources: release.resources }} />
        <p>Current route: {route ? short(route.release) : "not deployed"}</p>
        <button disabled={!!busy || route?.release === release.id} onClick={() => act("Deploying local endpoint", async () => { await api.post(`/api/production/releases/${release.id}/deploy`, { expectedCurrent: route?.release ?? null }); setNotice("Local endpoint routes this exact release."); })}>Deploy selected release</button>{" "}
        <button disabled={!!busy || !route || route.release === release.id || !data?.lifecycle.some((e) => ["deployed", "rolled_back"].includes(e.type) && e.data.releaseId === release.id)} onClick={() => act("Rolling back route", async () => { await api.post(`/api/production/releases/${release.id}/rollback`, { expectedCurrent: route?.release ?? null }); setNotice("Known prior release restored; in-flight requests keep their original version."); })}>Rollback to selected release</button>
        <p>Endpoint: <code>/api/serve/{release.config.target}/{release.config.namespace}/predict</code></p></>}
      <details open><summary>Recorded lifecycle and routing</summary><Recorded value={data?.routes ?? []} /><Recorded value={data?.lifecycle ?? []} /></details>
    </section></div>}
    {tab === "Requests" && <div className="prod-grid"><section className="prod-card"><h3>Send a bounded real request</h3>{selectedVersion}{selectedRelease}
      <button disabled={!!busy || !version} onClick={() => act("Reading recorded reference inputs", async () => { const r = await api.get<any>(`/api/production/versions/${versionId}/reference-input`); setPayload(JSON.stringify(r.records, null, 2)); setTrafficPayloads(JSON.stringify([r.records], null, 2)); setLabels(JSON.stringify(r.observedLabels ?? [])); setReference({ ...r.provenance, labelNote: r.labelNote, inputContract: r.inputContract }); })}>{["agent", "conversation"].includes(version?.adapter ?? "") ? "Load recorded source-turn input" : version?.adapter === "domain" ? "Load recorded held-out example" : version?.adapter === "model" ? "Load held-out validation images" : version?.adapter === "rl" ? "Load replay-buffer observations" : version?.adapter === "unsup" ? "Load fitted rows" : "Load recorded training inputs"}</button>
      <label>Records (JSON array matching the release schema)<textarea className="prod-json" aria-label="prediction records" value={payload} onChange={(e) => setPayload(e.target.value)} /></label>
      {reference != null && <details><summary>Payload provenance</summary><Recorded value={reference} /></details>}
      <label>User namespace <input disabled={!!busy} value={user} onChange={(e) => { setUser(e.target.value); setConversation(null); }} /></label><label>Session <input disabled={!!busy} value={session} onChange={(e) => { setSession(e.target.value); setConversation(null); }} /></label>
      <button disabled={!!busy || !release || route?.release !== release.id} onClick={() => act("Running pinned inference", async () => { const id = crypto.randomUUID(); setRequestId(id); setReplay(null); const req = { requestId: id, records: JSON.parse(payload), user, session: ["counter", "conversation"].includes(release?.config.sessionMode ?? "") ? session : null, expectedRelease: release?.id }; try { const t = await api.post<any>(`/api/serve/${release!.config.target}/${release!.config.namespace}/predict`, req); setTrace(t); } catch (e) { const t = await api.get<any>(`/api/production/requests/${id}?user=${encodeURIComponent(user)}`); setTrace(t); throw e; } requests.reload(); })}>Send prediction request</button>
      <p className="provenance">Current request {requestId || "not sent"}</p>
      <button disabled={!requestId || trace?.requestId === requestId} onClick={() => api.post(`/api/production/requests/${requestId}/cancel`, { user }).then(() => setNotice("Cancellation requested; state commit will be refused.")).catch((e) => setError(errorText(e)))}>Cancel in-flight request</button>
      {release?.config.sessionMode === "conversation" && <>
        <button disabled={!!busy} onClick={() => act("Reading native conversation checkpoint", async () => setConversation(await api.get(`/api/production/releases/${release.id}/conversation?user=${encodeURIComponent(user)}&session=${encodeURIComponent(session)}`)))}>Inspect conversation checkpoint</button>
        {conversation != null && <Recorded value={conversation} />}
      </>}
      <h4>Recorded requests for this release</h4><div className="prod-list">{requests.data?.requests.map((t) => <button key={`${t.user}:${t.requestId}`} onClick={() => { setTrace(t); setReplay(null); }}>{t.status} · {short(t.requestId)} · {t.totalMs == null ? "latency not recorded" : `${t.totalMs.toFixed(2)} ms`} · {t.user}</button>)}</div>
    </section><section className="prod-card"><h3>Prediction → release → run → source → preprocessing</h3>
      {trace ? <><p className="provenance">request {trace.requestId} · release {trace.releaseId} · version {trace.versionId} · trace {trace.traceSha256}</p>
        <Recorded value={{ result: trace.result?.agent ? { predictions: trace.result.predictions, family: trace.result.family } : trace.result, sessionState: trace.sessionState, conversationParent: trace.conversationParent, conversationState: trace.conversationState, status: trace.status, error: trace.error, queueMs: trace.queueMs, totalMs: trace.totalMs, timings: trace.timings }} />
        {trace.result?.agent && <AgentTurnInspection key={trace.result.agent.executionId} evidence={trace.result.agent} />}{trace.lineage && <><button onClick={() => onOpenRun(trace.lineage.runId)}>Open source run</button><Recorded value={trace.lineage} /></>}
        <p>{trace.capturePolicy}</p><button disabled={!!busy || !trace.records} onClick={() => act("Replaying isolated pinned forward pass", async () => setReplay(await api.post(`/api/production/requests/${trace.requestId}/replay`, { user: trace.user })))}>Replay captured inputs in isolation</button>
        {replay != null && <Recorded value={replay} />}
        <label>Observed ground-truth labels (JSON array)<textarea aria-label="ground truth labels" value={labels} onChange={(e) => setLabels(e.target.value)} /></label>
        {["agent", "conversation"].includes(version?.adapter ?? "") ? <p>One reference string per turn. Quality is literal string agreement with text you supply, not semantic correctness. Replay makes a new native model call and its response may differ.</p> : version?.adapter === "unsup" ? <p>{version.family === "pca" ? "PCA takes no labels; its reconstruction error is monitored instead." : "Optional external labels (strings or integers), one per row. Agreement is reported as ARI/NMI, never accuracy."}</p>
          : version?.adapter === "rl" ? <p>One reference action (integer) per observation, supplied by you. Recorded training actions came from the ε-greedy behaviour policy and are not offered as ground truth. The metric is action agreement, not environment return.</p>
          : version?.adapter === "model" ? <p>One class name per image. Loaded validation images come with their recorded held-out labels.</p>
          : version?.adapter === "domain"
          ? <p>One label per predicted record: {version.family === "nlp" ? "a list of IOB2 labels, one per original word" : version.family === "speech" ? "the reference transcript string" : "an H×W integer class mask (0 = background)"}. Quality uses the family's native metric ({version.family === "nlp" ? "word accuracy and seqeval span F1" : version.family === "speech" ? "corpus CER/WER" : "pixel accuracy, mean IoU/Dice"}).</p>
          : <p>When using loaded training inputs, labels are their recorded training-reference labels. That quality is in-sample evidence, not a held-out production benchmark.</p>}
        <button disabled={!!busy || trace.status !== 200} onClick={() => act("Recording ground truth", async () => { await api.post(`/api/production/requests/${trace.requestId}/labels`, { user: trace.user, labels: JSON.parse(labels) }); setNotice("Ground truth recorded with label delay; conflicting overwrites are refused."); monitor.reload(); })}>Record ground truth</button>
      </> : <p>Select or send a request. Nothing is fabricated when there is no recorded request.</p>}
    </section></div>}
    {tab === "Traffic" && <div className="prod-grid"><section className="prod-card"><h3>Bounded HTTP traffic builder</h3>{selectedRelease}
      <label>Representative payload batches (JSON array of arrays)<textarea className="prod-json" aria-label="traffic payloads" value={trafficPayloads} onChange={(e) => setTrafficPayloads(e.target.value)} /></label>
      <p>Payload batches rotate in order, allowing measured request-size distributions. Use Requests → Load recorded training inputs to start from real recorded rows.</p>
      <label>Arrival pattern <select aria-label="traffic pattern" value={pattern} onChange={(e) => setPattern(e.target.value)}>{["steady", "ramp", "burst", "closed_loop"].map((p) => <option key={p}>{p}</option>)}</select></label>
      <label>Rate (requests/s; response-driven mode ignores this target)<input aria-label="traffic rate" type="number" value={rate} onChange={(e) => setRate(Number(e.target.value))} /></label>
      <label>Duration (seconds)<input aria-label="traffic duration" type="number" value={duration} onChange={(e) => setDuration(Number(e.target.value))} /></label>
      <label>Generator concurrency<input aria-label="traffic concurrency" type="number" value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} /></label>
      <label>Maximum arrivals<input type="number" value={maxRequests} onChange={(e) => setMaxRequests(Number(e.target.value))} /></label>
      <label>Expected HTTP statuses<input value={expected} onChange={(e) => setExpected(e.target.value)} /></label>
      <button disabled={!!busy || !release || route?.release !== release.id || job.data?.state === "running"} onClick={() => act("Starting measured HTTP traffic", async () => { const j = await api.post<any>("/api/production/traffic", { releaseId, payloads: JSON.parse(trafficPayloads), pattern, rate, durationSeconds: duration, concurrency, maxRequests, expectedStatuses: JSON.parse(expected) }); setJob(j.id); })}>Run bounded load test</button>{" "}
      <button disabled={!jobId || job.data?.state !== "running"} onClick={() => api.post(`/api/production/traffic/${jobId}/cancel`, {}).then(() => job.reload()).catch((e) => setError(errorText(e)))}>Stop new arrivals</button>
      <h4>Previous measured tests</h4>{data?.traffic.map((t) => <button className="prod-history" key={t.id} onClick={() => setJob(t.id)}>{t.state} · {short(t.id)} · {t.spec.pattern}</button>)}
    </section><section className="prod-card"><h3>Observed load and generator limits</h3>{job.error && <p className="error">{job.error}</p>}
      {job.data ? <><p role="status">Load test {job.data.state}</p><p className="provenance">job {jobId} · result {job.data.resultId ?? job.data.id}</p>
        {loadResult?.observed && <><Metrics rows={[["Successful requests", loadResult.observed.successfulRequests], ["Errors", loadResult.observed.errors],
          ["Achieved throughput (successful requests/s, including drain)", loadResult.observed.achievedRps], ["Offered arrivals/s", loadResult.generator.offeredRps],
          ["Sent requests/s", loadResult.generator.sentRps], ["Dropped at generator", loadResult.generator.droppedAtGenerator],
          ["p50 latency (ms)", loadResult.observed.p50Ms], ["p95 latency (ms)", loadResult.observed.p95Ms], ["p99 latency (ms)", loadResult.observed.p99Ms],
          ["Mean queue delay (ms)", loadResult.observed.meanQueueMs], ["Timeouts", loadResult.observed.timeouts], ["Unexpected statuses", loadResult.observed.unexpectedStatuses],
          ["Process CPU seconds", loadResult.resources.cpuSeconds], ["Peak sampled RSS (bytes)", loadResult.resources.peakSampledRssBytes]]} />
          <p>{loadResult.resources.scope}. Cost: {loadResult.resources.cost}.</p><p>{loadResult.generator.mode}: {loadResult.generator.limits}</p></>}
        <details><summary>Full measured result / configuration / request evidence</summary><Recorded value={loadResult} /></details></> : <p>No measured load result selected.</p>}
    </section></div>}
    {tab === "Monitoring" && <section className="prod-card">{selectedRelease}<label>Request-window start (Unix seconds)<input value={since} onChange={(e) => setSince(e.target.value)} /></label>
      <h3>Observed drift and separately measured task quality</h3><p>Drift alone does not establish an accuracy drop. Quality is unavailable until aligned ground-truth labels are supplied. No automatic retraining or rollback occurs.</p>
      {monitor.error && <p className="error">{monitor.error}</p>}{monitor.data ? <><p className="provenance">release {monitor.data.releaseId} · version {monitor.data.versionId} · reference {monitor.data.reference.sha256}</p>
        <Metrics rows={[["Observed requests", monitor.data.health.requests], ["Errors", monitor.data.health.errors], ["Schema errors", monitor.data.health.schemaErrors],
          ["p95 request handling (ms, excludes persistence/HTTP encoding)", monitor.data.health.p95Ms], ["Rows / records with supplied labels", monitor.data.labelBasedQuality.labelledRows ?? monitor.data.labelBasedQuality.labelledRecords],
          ["Mean label delay (seconds)", monitor.data.labelBasedQuality.meanLabelDelaySeconds]]} />
        <h4>Input changes against the recorded reference</h4><table><thead><tr><th>Field</th><th>KS statistic / total variation</th><th>Current missing fraction</th><th>Evidence</th></tr></thead><tbody>
          {Object.entries(monitor.data.inputDrift).map(([field, d]: [string, any]) => <tr key={field}><td>{field}</td><td>{d.ksStatistic ?? d.totalVariation ?? "not recorded"}</td><td>{d.currentMissingFraction ?? "not recorded"}</td><td>{d.available ? `${d.referenceN} reference / ${d.currentN} current observations` : d.reason}</td></tr>)}</tbody></table>
        <h4>Prediction distribution change</h4><Recorded value={monitor.data.predictionDrift} /><h4>Quality measured only from supplied labels</h4>
        {monitor.data.labelBasedQuality.available ? <Metrics rows={Object.entries(monitor.data.labelBasedQuality.values)} /> : <p>{monitor.data.labelBasedQuality.reason}</p>}
        <details><summary>Full monitoring window / methods / label and alert evidence</summary><Recorded value={monitor.data} /></details></> : <p>Select a release to inspect its request evidence.</p>}
    </section>}
  </section></div>;
}
