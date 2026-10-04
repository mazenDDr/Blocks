// Hand-mirrored from packages/graph-schema/schema.json and the control service responses.
export type Dim = number | string;
export interface TensorType { shape: Dim[]; dtype: string }
/** Static type of a wire in a tabular graph: a table (schema, rows, partition), a fitted state, a model, a distribution, ... */
export interface ValueType {
  kind: string; columns?: { name: string; dtype: string }[]; rows?: number | null; rowsExact?: boolean; partition?: string;
  columnsComplete?: boolean; pending?: string[]; [k: string]: unknown;
}
export type WireType = TensorType | ValueType;
export const isTensorType = (t: WireType): t is TensorType => Array.isArray((t as TensorType).shape);

export interface Endpoint { node: string; port: string }
export interface GNode { id: string; type: string; version: string; config: Record<string, unknown>; stateRef?: string | null; sharedWith?: string | null }
export interface GEdge { id: string; kind: string; from: Endpoint; to: Endpoint }
export interface PortSpec { name: string; dtype: string; shape: (number | string | null)[] | null; description?: string }
export interface ModuleOutput { name: string; from: Endpoint; dtype?: string | null; shape?: (number | string | null)[] | null; description?: string }
export interface ModuleParam { name: string; default: unknown; type?: string | null; description?: string }
export interface ModuleDef {
  id: string; version: string; description: string; inputs: PortSpec[]; outputs: ModuleOutput[]; params: ModuleParam[]; nodes: GNode[]; edges: GEdge[]; reduction?: Record<string, any> | null;
}
export interface CodeIO { name: string; dtype: string; shape?: (number | string | null)[] | null; same_as?: string | null; description?: string }
export interface CodeBlockDef {
  id: string; version: string; description: string; inputs: CodeIO[]; outputs: CodeIO[]; config: { name: string; type: string; default: unknown; description?: string }[];
  state: { name: string; shape: number[]; dtype: string; init: number }[]; effects: string[]; randomness: string; differentiable: boolean; dependencies: string[];
  source: string; fixtures: Record<string, any>[]; limits: Record<string, number>;
}
export interface Graph {
  schemaVersion: string; graphKind: string; backend: string; nodes: GNode[]; edges: GEdge[];
  modules?: ModuleDef[]; codeBlocks?: CodeBlockDef[]; training?: Record<string, any> | null;
  /** agent graphs: state schema, routes, joins, limits, indexes, memory policies (see components/agent/types.ts) */
  agent?: Record<string, unknown> | null;
}
export interface UiDoc {
  schemaVersion: string; positions: Record<string, { x: number; y: number }>; pinnedBaseline?: string | null;
  description?: string; synthetic?: boolean; defaultInput?: Record<string, unknown>; seedExample?: string;
}

export interface OpInfo {
  type: string; version: string; backend: string; graphKind: string; summaryKind: string | null; displayName: string; category: string; purpose: string;
  inputs: string[]; outputs: string[]; inputKinds: Record<string, string>; outputKinds: Record<string, string>;
  configSchema: JSchema; defaults: Record<string, unknown>;
}
export interface JSchema {
  type?: string; title?: string; properties?: Record<string, JSchema>; default?: unknown; enum?: unknown[]; const?: unknown;
  anyOf?: JSchema[]; prefixItems?: JSchema[]; items?: JSchema; minimum?: number; exclusiveMinimum?: number;
  $ref?: string; $defs?: Record<string, JSchema>; description?: string;
}

export interface Fix { label: string; node: string | null; key: string | null; value: unknown }
export interface Diagnostic { code: string; severity: "error" | "warning"; nodeId: string | null; port: string | null; path: string; message: string; fixes: Fix[] }
export interface ExplainStep { id: string; local: string; type: string; equation: string; params: number; config?: Record<string, unknown>; outputShape?: Dim[]; sharesParameters?: boolean }
export interface NodeView {
  known: boolean; typed: boolean; diagnostics: Diagnostic[];
  inputShapes?: Record<string, WireType>; outputShapes?: Record<string, WireType>; params?: number;
  resolvedConfig?: Record<string, unknown>; explain?: Explain;
  type?: string; nodeVersion?: string;
  inputPorts?: string[]; outputPorts?: string[]; sharedWith?: string | null; sharesParameters?: boolean; observationOnly?: boolean;
  // structural instances (composite / repeat / select)
  structural?: boolean; kind?: string; module?: string; version?: string; members?: string[]; children?: string[]; note?: string; iterations?: number | null; termination?: string | null;
}
export interface InstanceBoundary { kind: "in" | "out"; port: string; node?: string; nodePort?: string }
export interface InstanceInfo {
  path: string; kind: string; module: string | null; version: string | null; inputs: string[]; outputs: string[]; members: string[]; children: string[]; note: string;
  sharedWith: string | null; iterations: number | null; termination: string | null; typed: boolean; params: number;
  inputShapes: Record<string, WireType>; outputShapes: Record<string, WireType>; diagnostics: Diagnostic[]; explain: Explain & { steps?: ExplainStep[] };
  innerEdges?: { id: string; from: Endpoint; to: Endpoint }[]; boundary?: InstanceBoundary[];
}
export interface ModuleSummary {
  id: string; version: string; description: string; inputs: PortSpec[]; outputs: ModuleOutput[]; params: ModuleParam[]; nodes: number; contentHash: string; usedBy: string[]; reduction?: Record<string, any> | null;
}
export interface ProcedureCheck { valid: boolean; diagnostics: Diagnostic[]; stages: string[] }
export interface Explain {
  equation?: string; shapeRule?: string; reduction?: string; note?: string; error?: string; rule?: string; executionEffect?: string;
  broadcast?: { rule: string; alignment: Record<string, unknown>[] }; axes?: { in: string[] | null; out: string[] | null };
  steps?: ExplainStep[]; composite?: { instances: string[]; iterations: number | null };
  parameters?: { formula: string; terms: { name: string; shape: number[]; count: number }[]; total: number };
}
export interface Validation {
  ok: boolean; graphHash: string; graphKind?: string; totalParams: number; diagnostics: Diagnostic[]; nodes: Record<string, NodeView>;
  flat?: Record<string, NodeView>; instances?: Record<string, InstanceInfo>; modules?: ModuleSummary[]; procedure?: ProcedureCheck;
  /** agent graphs */
  agent?: import("./components/agent/types").AgentAnalysis; order?: string[];
}

export interface RunSummary {
  kind?: "model"; id: string; status: string; error: string | null; graphHash: string; createdAt: number; updatedAt: number;
  config: RunConfig; classes: string[] | null; totalParams: number | null;
  split: { seed: number; valFraction: number; nTrain: number; nVal: number; datasetSha256: string } | null;
  progress: { step: number; epochsDone: number; epochs: number; stepsPerEpoch: number | null };
  final: EpochEnd | null; maxSeq: number;
}
export interface RunConfig {
  data: string; epochs: number; batch_size: number; lr: number; optimizer: "sgd" | "adam"; momentum: number; seed: number;
  val_fraction: number; split_seed: number | null; project_id: string | null;
}
export interface EpochEnd { epoch: number; step: number; train_loss: number; val_loss: number; val_acc: number }
export interface RunMetrics {
  summary: RunSummary; graph: Graph | null; trainLoss: [number, number][]; epochs: EpochEnd[];
  provenance: { runId: string; graphHash: string; source: string };
}
export interface Checkpoint { step: number; epoch: number | null; status: string; sha256: string; size: number; graphHash: string }
export interface Sample { index: number; id: string; label: number; labelName: string }

export interface Provenance {
  runId: string | null; graphHash?: string; checkpointStep?: number | null; checkpointStatus?: string | null; checkpointSha256?: string | null;
  nodeId?: string; sampleIndex?: number | null; sampleId?: string; source?: string; normalization?: string; epoch?: number;
  preprocessing?: string; trueLabel?: string | null;
}
export interface Unavailable { available: false; kind: string; reason: string; message: string; provenance: Provenance }
export interface Stats { min: number; max: number; mean: number; std: number; count: number }
export interface Slice { offset: number; limit: number; of: number; truncated: boolean }
export interface WeightsResult {
  available: true; nodeId: string; provenance: Provenance;
  params: Record<string, { shape: number[]; stats: Stats; slice: Slice; values: any }>;
}
export interface ActivationsResult {
  available: true; nodeId: string; layout: "feature_maps" | "vector"; shape: number[]; stats: Stats; slice: Slice;
  channelStats: Stats[]; values: any; provenance: Provenance;
}
export interface SampleLossResult {
  available: true; nVal: number; provenance: Provenance;
  worst: { index: number; id: string; label: number; labelName: string; pred: number; predName: string; loss: number }[];
}
export interface ConfusionResult { available: true; classes: string[]; matrix: number[][]; axes: string; provenance: Provenance }
export interface InferResult {
  available: true; classes: string[]; logits: number[]; probabilities: number[]; predicted: number; predictedName: string; provenance: Provenance;
}

export interface RunEvent { run_id: string; seq: number; ts: number; type: string; graph_hash: string; node_id: string | null; data: any }

// ---- tabular graph kind -------------------------------------------------------------------------------
export interface TabularNodeStatus { node: string; type?: string; status: "pending" | "running" | "finished" | "failed"; rows?: Record<string, number> }
export interface TabularRunSummary {
  kind: "tabular"; id: string; status: string; error: string | null; graphHash: string; createdAt: number; updatedAt: number; maxSeq: number;
  config: { kind: string; project_id: string | null; seed?: number | null; source_pins?: Record<string, string>; trial?: Record<string, unknown> | null };
  snapshots?: { node: string; connector: string; mode: string; snapshotId: string; kind: string; contentSha256: string | null; rows: number; reproducibility: { level: string; limited: boolean } }[];
  nodes: TabularNodeStatus[]; progress: { nodesDone: number; nodes: number };
  sources: { node: string; path: string; sha256: string; bytes: number; rows: number }[];
  splits: { node: string; seed: number; validationFraction: number; stratifyBy: string | null; groupBy: string | null; nTrain: number; nValidation: number; trainRowIdsSha256: string; validationRowIdsSha256: string }[];
  failure: { node: string | null; code: string; message: string } | null;
  libraries: Record<string, string> | null;
}
export interface ProcedureRunSummary {
  kind: "procedure"; id: string; status: string; error: string | null; graphHash: string; createdAt: number; updatedAt: number; maxSeq: number; config: Record<string, any>;
  totalParams: number | null; synthetic: boolean | null; dataNote: string | null; resumedFrom: { run_id: string; step: number } | null; rerunOf: string | null;
  progress: { step: number; epochsDone: number; epochs: number | null; stepsPerEpoch: number | null };
  last: { step: number; loss: number; lr: number; grad_norm: number | null } | null; validation: Record<string, number> | null; stoppedBy: string | null; checkpoints: number; captures: number[];
}
export interface SandboxRunSummary { kind: "sandbox"; id: string; status: string; parent: string; step: number; createdAt: number; updatedAt: number; graphHash: string; config: Record<string, any> }
export type AnyRun = RunSummary | TabularRunSummary | ProcedureRunSummary | SandboxRunSummary | import("./components/agent/types").AgentRunSummary;
export const isTabularRun = (r: AnyRun): r is TabularRunSummary => r.kind === "tabular";
export const isProcedureRun = (r: AnyRun): r is ProcedureRunSummary => r.kind === "procedure";
export const isSandboxRun = (r: AnyRun): r is SandboxRunSummary => r.kind === "sandbox";
export const isModelRun = (r: AnyRun): r is RunSummary => r.kind === undefined || r.kind === "model";
export const isAgentRun = (r: AnyRun): r is import("./components/agent/types").AgentRunSummary => r.kind === "agent";

export interface TabProvenance {
  runId: string | null; graphHash?: string; nodeId?: string | null; port?: string | null; partition?: string; source?: unknown; sourceSha256?: string;
  split?: { node: string; seed: number; validationFraction: number; nTrain: number; nValidation: number }; fittedOn?: any; artifactSha256?: string;
}
export interface TablePage {
  available: true; kind: "table"; node: string; port: string; columns: { name: string; dtype: string }[]; partition: string | null;
  rowIds: number[]; rows: unknown[][]; total: number; offset: number; limit: number; truncated: boolean; provenance: TabProvenance;
}
export interface SummaryResult<T = any> { available: true; kind: string; node: string; nodeType: string; summaryKind: string; data: T; provenance: TabProvenance }
export interface TabUnavailable { available: false; kind: string; reason: string; message: string; provenance: TabProvenance }
export interface DensityPlot {
  x: number[]; density: number[]; cdf?: number[]; survival?: number[]; tail?: "upper" | "lower" | "two_sided"; observed?: number;
  criticalValue?: number; criticalLow?: number | null; criticalHigh?: number | null; family?: string; df?: number; statisticName?: string;
}

// ---- connections, snapshots, studies (Milestone 2b) ----------------------------------------------------
export type ConnType = "postgres" | "s3" | "dvc";
export interface SecretRef { kind: "env" | "file"; name?: string; key?: string; path?: string }
export interface SecretStatus { kind: "env" | "file"; name?: string; key?: string; resolvable: boolean; warning?: string }
export interface SourceErrorInfo { code: string; message: string; recoverable: boolean; resource?: string | null; connectionId?: string | null; hint?: string }
export interface Connection {
  id: string; name: string; type: ConnType; settings: Record<string, any>; secrets: Record<string, SecretStatus>;
  lastTest: { ok: boolean; testedAt?: number; error?: SourceErrorInfo; serverVersion?: string; versioning?: string; head?: string; note?: string | null; latencyMs?: number } | null;
}
export interface DiscTable { schema: string; name: string; kind: string; estimatedRows: number | null; selectable: boolean }
export interface DiscColumn { name: string; type: string; dtype: string; nullable: boolean; selectable: boolean; primaryKey: boolean }
export interface PreviewResult {
  columns: { name: string; dtype: string; pgType?: string }[] | null; rows: unknown[][] | null; truncated?: boolean; rowCount?: number; elapsedMs?: number;
  bounds?: Record<string, unknown>; query?: { text: string; params: unknown[] }; note?: string | null; provenance?: { source: string };
  size?: number; etag?: string; versionId?: string | null; commit?: string; dvcMd5?: string | null; format?: string;
}
export interface S3Object { key: string; size?: number; etag?: string; lastModified?: string; versionId?: string | null; versionCount?: number | null; format: string }
export interface Compiled { sql: string; params: unknown[]; paramTypes: string[]; columns: string[]; ordered: boolean; warnings: string[]; placeholderStyle: string }

export interface TrialAttempt { attempt: number; kind: string; runId: string | null; status: string; error: string | null }
export interface MetricValue {
  available: boolean; value?: number; metric: string; step?: number | null; epoch?: number | null; aggregation?: string; partition?: string | null; n?: number; reason?: string;
  provenance?: { runId: string; node?: string }; definition?: string;
}
export interface StudyTrial {
  trialId: string; index: number; group: number; groupKey: string; isBaseline: boolean; label: string; assignments: { target: { scope: string; node?: string; field: string }; value: unknown }[];
  seed: number | null; fold: number | null; status: string; attemptCount: number; retried: boolean; attempts: TrialAttempt[]; runId: string | null;
  metric: MetricValue | null; value: number | null; delta: number | null; better: boolean | null; changedFields: string[]; attributionWarning: string | null; error: string | null;
  diagnostics: { code: string; message: string }[];
}
export interface StudyGroup {
  group: number; label: string; isBaseline: boolean; trials: number; completed: number; failed: number; cancelled: number; invalid: number; retried: number;
  n: number; mean: number | null; std: number | null; min: number | null; max: number | null; delta: number | null; better: boolean | null; rank?: number;
  repeats: { trialId: string; seed: number | null; fold: number | null; value: number | null; status: string }[];
}
export interface StudyView {
  id: string; name: string; hypothesis: string; notes: string; state: string; graphKind: string; graphHash: string; createdAt: number;
  objective: { metric: { name: string; node?: string | null; select: string }; direction: string; semantics: string };
  limits: { max_trials: number; max_concurrency: number; max_attempts: number }; plan: { total: number; groups: number; seeds: number; folds: number; invalid: number };
  baseline: { mode: string | null; value: number | null; runId: string | null; n: number; note?: string };
  counts: Record<string, number>; trials: StudyTrial[]; groups: StudyGroup[]; best: { label: string; mean: number; n: number } | null; notice: string; schedulerError?: string | null;
}
export interface StudyListItem { id: string; name: string; state: string; graphKind: string; trials: number; counts: Record<string, number>; objective: { direction: string; metric: { name: string } } }
export interface RunDiff {
  a: string; b: string; changeCount: number; warning: string | null; identity: { a: unknown; b: unknown };
  graph: { available: boolean; changes: Record<string, any>[] }; runConfig: { field: string; from: unknown; to: unknown }[]; data: { node: string; from: unknown; to: unknown }[];
}
