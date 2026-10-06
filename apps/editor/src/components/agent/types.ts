// Types for the agent graph kind (Milestone 4): mirrors python/agent/spec.py and the /api/agent/* responses.
import type { Diagnostic, GNode, Graph } from "../../types";

export type FieldType = "text" | "integer" | "number" | "boolean" | "list" | "object" | "messages" | "documents" | "records" | "json";
export type ReducerKind = "replace" | "append" | "append_unique" | "add" | "max" | "min" | "merge" | "keep_last_n";
export interface StateField {
  name: string; type: FieldType; reducer: { kind: ReducerKind; n?: number | null }; default?: unknown; description?: string;
  properties?: Record<string, string>; scope?: "turn" | "thread";
}
export type Predicate = { always?: boolean; all?: Predicate[]; any?: Predicate[]; not?: Predicate; field?: string; op?: string; value?: unknown; other?: string; fn?: string };
export interface Case { id: string; label: string; when: Predicate; to: string }
export interface Route { id: string; from: string; cases: Case[]; default: string; defaultLabel?: string }
export interface Join { node: string; waitFor: string[] }
export interface Limits { maxSteps: number; maxModelCalls?: number | null; maxTokens?: number | null; maxSeconds?: number | null; maxToolCalls?: number | null }
export interface EmbeddingsSpec { provider: "ollama" | "local_hash"; model: string; dimension: number; normalize: boolean; batchSize?: number }
export interface IndexSpec {
  id: string; description?: string; loader: { directory: string; glob: string; encoding?: string };
  splitter: { strategy: "recursive" | "paragraph" | "fixed"; chunkSize: number; chunkOverlap: number }; embeddings: EmbeddingsSpec; store?: "faiss";
}
export interface PolicyStage { id: string; op: "retrieve" | "filter" | "rank" | "dedupe" | "budget" | "summarize"; config: Record<string, any> }
export interface Policy { id: string; name?: string; description?: string; query: string; embeddings: EmbeddingsSpec; stages: PolicyStage[] }
export interface AgentSpec { state: StateField[]; routes: Route[]; joins: Join[]; limits: Limits; indexes: IndexSpec[]; policies: Policy[] }
export const emptySpec = (): AgentSpec => ({ state: [], routes: [], joins: [], limits: { maxSteps: 25 }, indexes: [], policies: [] });
export const specOf = (g: Graph): AgentSpec => ({ ...emptySpec(), ...((g.agent as Partial<AgentSpec>) ?? {}) });

export interface ModelSpec {
  provider: "ollama" | "anthropic" | "openai_compatible" | "fixture"; model: string; temperature: number | null; max_tokens: number | null; seed: number | null; timeout_s: number;
  think: boolean; base_url?: string | null; api_key?: { kind: "env" | "file"; name?: string; path?: string; key?: string } | null; fixture: Record<string, any>;
  reasoning_effort?: "none" | "low" | "medium" | "high" | null;
}

export interface AgentNodeView {
  known: boolean; diagnostics: Diagnostic[]; type: string; typed: boolean; resolvedConfig?: Record<string, any>; reads?: string[]; writes?: string[]; effects?: string[];
  explain?: { summary?: string; tool?: ToolInfo; schema?: unknown }; fixtureModel?: boolean;
}
export interface AgentAnalysis {
  loops: { nodes: string[]; hasConditionalExit: boolean; boundedBy: string }[]; parallel: { from: string; branches: string[] }[]; joins: Join[]; policies: string[];
  limits: Limits; usesFixtureModel: boolean; inputFields: string[]; state: StateField[];
}

export interface ToolInfo { name: string; description: string; args: Record<string, string>; effects: string[]; external: boolean; requiresApproval: boolean; limits: Record<string, unknown> }
export interface Catalog {
  providers: Record<string, { label: string; settings: string[]; structured: string; usage: string; unsupported?: Record<string, string>; ollama?: { reachable: boolean; models: string[] } | null }>;
  defaults: { anthropicModel: string; ollamaModel: string };
  reducers: { kind: ReducerKind; doc: string; appliesTo: FieldType[] }[]; fieldTypes: FieldType[];
  predicate: { operators: { op: string; label: string; unary: boolean }[]; functions: string[] };
  tools: ToolInfo[]; effects: { external: string[]; policy: string };
  stages: { defaults: Record<string, Record<string, any>>; schemas: Record<string, any>; order: string[] };
  memoryKinds: string[]; embeddings: { provider: string; label: string; available?: boolean }[]; fixtureNote: string;
}

// ---- runs, trace
export interface AgentRunSummary {
  kind: "agent"; id: string; status: string; error: string | null; graphHash: string; createdAt: number; updatedAt: number; threadId: string; input: Record<string, any>;
  rerunOf: string | null; stoppedBy: string | null; pendingInterrupt: { node: string; interruptId: string; payload: any; step: number } | null; modelCalls: number; fixtureCalls: number; totalLatencyMs: number; maxSeq: number;
  config: { project_id: string | null };
}
export interface StateChange { field: string; reducer: string; written: unknown; before: unknown; after: unknown; changed: boolean }
export interface RouteEval { case: string; label: string; to: string; predicate: string; values: Record<string, unknown>; result: boolean }
export interface TraceModelCall {
  callId: string; purpose: string; attempt: number; provider: string; model: string; fixture: boolean; latencyMs: number; usage: { inputTokens: number | null; outputTokens: number | null; source: string };
  cost: { amount: number | null; basis: string }; resolved: Record<string, unknown>; ignored: Record<string, string>; tokensEstimate: number; validationErrors: string[] | null; response: string; seq: number;
}
export interface TraceStep {
  seq: number; step: number | null; node: string; type: string; replay: boolean; status: string; durationMs?: number; changes: StateChange[]; reads: Record<string, unknown>;
  route: { route: string; evaluated: RouteEval[]; taken: string; label: string; to: string; via: string } | null;
  modelCalls: TraceModelCall[]; toolCalls: any[]; retrievals: any[]; events: any[];
}
export interface Trace {
  runId: string; status: string; run: { threadId: string; libraries?: Record<string, string>; limits?: Limits; finished?: string; stoppedBy?: string; error?: string; resumes?: { seq: number; value: unknown }[]; graphHash?: string };
  steps: TraceStep[]; totals: { modelCalls: number; toolCalls: number; latencyMs: number; inputTokens: number; outputTokens: number; providerReportedCalls: number; unreportedCalls: number; fixtureCalls: number };
  interrupts: { seq: number; node: string; interruptId: string; payload: any; step: number }[]; pendingInterrupt: { node: string; interruptId: string; payload: any } | null;
  budget: { kind: string; limit: number; used: number; message: string } | null; failures: { seq: number; type: string; node: string | null; message: string; code?: string }[];
  provenance: { runId: string; source: string; graphHash?: string };
}

// ---- context inspector
export interface Segment { start: number; end: number; source: Record<string, any>; tokensEstimate: number }
export interface CtxMessage { index: number; role: string; content: string; segments: Segment[]; tokensEstimate: number }
export interface ExcludedItem { kind: "memory_record" | "retrieved_chunk"; id: string; status: string; stage: string; op: string; reason: string; text: string; scores: Record<string, number>; tokensEstimate: number; applicationId?: string; retrievalId?: string; store?: string }
export interface CallContext {
  callId: string; runId: string; threadId: string; node: string; purpose: string; attempt: number; provider: string; model: string; fixture: boolean; resolved: Record<string, unknown>; ignored: Record<string, string>;
  messages: CtxMessage[]; providerRequest: { role: string; content: string }[]; response: string; usage: { inputTokens: number | null; outputTokens: number | null; source: string }; latencyMs: number;
  tokens: { estimateTotal: number; estimateBasis: string; providerInput: number | null; providerOutput: number | null; providerReported: boolean; modelLimit: number | null; modelLimitSource: string; reservedOutput: number | null };
  excluded: ExcludedItem[]; applications: Record<string, { policy: string; node: string; universe: number } | null>; retrievals: Record<string, unknown>; boundary: string;
  provenance: { runId: string; graphHash: string; node: string; artifactSha256: string; source: string };
}

// ---- memory
export interface MemRecord {
  id: string; store: string; namespace: string; scope: string; kind: string; text: string; metadata: Record<string, any>; importance: number; created_at: number; updated_at: number;
  expires_at: number | null; generated: boolean; version: number; source: Record<string, any>; deleted_at: number | null;
}
export interface Decision { id: string; action: string; reason: string; rank?: number }
export interface AppStage { id: string; op: string; in: number; out: number; note: string; decisions: Decision[]; skipped?: string }
export interface AppRecord { id: string; store: string; kind: string; text: string; status: string; excludedAt: { stage: string; op: string; reason: string } | null; scores: Record<string, number>; tokens: number; trail: { stage: string; op: string; action: string; reason: string }[] }
export interface Application { id: string; policyId: string; stages: AppStage[]; final: string[]; records: Record<string, AppRecord>; tokensEstimate: number; asOf: number; query: string; embeddings: string }
export interface PreviewResult {
  noModelCall: boolean; noStateMutation: boolean; embeddingCalls: boolean; callId: string; runId: string;
  before: { messages: CtxMessage[]; tokensEstimate: number }; after: { messages: CtxMessage[]; tokensEstimate: number };
  segments: { key: string; status: "added" | "removed" | "unchanged" | "changed"; role: string; source: Record<string, any>; text: string; tokensEstimate: number }[];
  recordChanges: { id: string; before: string; after: string; beforeReason: string | null; afterReason: string | null; afterScores: Record<string, number> }[];
  applications: { applicationId: string; policy: string; stages: AppStage[]; final: string[]; records: Record<string, AppRecord>; tokensEstimate: number; skippedStages: string[] }[]; note: string;
}
export interface TraceRecord {
  recordId: string; stored: boolean; storedRecord: MemRecord | null; caution: string;
  applications: { applicationId: string; policyId: string; node: string; present: boolean; status?: string; usedInContext?: boolean; trail?: { stage: string; op: string; action: string; reason: string }[];
    firstDisappearedAt?: { stage: string; op: string; reason: string } | null; scores?: Record<string, number>; tokensEstimate?: number; verdict: string }[];
}
export interface IndexManifest {
  id: string; identity: string; builtAt: number; buildSeconds: number; documents: { doc_id: string; path: string; sha256: string; bytes: number }[]; chunks: number; dimension: number;
  embedding: { identity: string; label: string; normalized: boolean }; embeddingsReused: number; embeddingsComputed: number; scoreInterpretation: string; action?: string; store: string;
}
export type { GNode, Graph };
