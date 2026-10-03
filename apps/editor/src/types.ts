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
export interface GNode { id: string; type: string; version: string; config: Record<string, unknown>; stateRef?: string | null }
export interface GEdge { id: string; kind: string; from: Endpoint; to: Endpoint }
export interface Graph { schemaVersion: string; graphKind: string; backend: string; nodes: GNode[]; edges: GEdge[] }
export interface UiDoc {
  schemaVersion: string; positions: Record<string, { x: number; y: number }>; pinnedBaseline?: string | null;
  description?: string; synthetic?: boolean;
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
export interface NodeView {
  known: boolean; typed: boolean; diagnostics: Diagnostic[];
  inputShapes?: Record<string, WireType>; outputShapes?: Record<string, WireType>; params?: number;
  resolvedConfig?: Record<string, unknown>; explain?: Explain;
}
export interface Explain {
  equation?: string; shapeRule?: string; reduction?: string; note?: string; error?: string; rule?: string;
  parameters?: { formula: string; terms: { name: string; shape: number[]; count: number }[]; total: number };
}
export interface Validation {
  ok: boolean; graphHash: string; graphKind?: string; totalParams: number; diagnostics: Diagnostic[]; nodes: Record<string, NodeView>;
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
  config: { kind: string; project_id: string | null };
  nodes: TabularNodeStatus[]; progress: { nodesDone: number; nodes: number };
  sources: { node: string; path: string; sha256: string; bytes: number; rows: number }[];
  splits: { node: string; seed: number; validationFraction: number; stratifyBy: string | null; groupBy: string | null; nTrain: number; nValidation: number; trainRowIdsSha256: string; validationRowIdsSha256: string }[];
  failure: { node: string | null; code: string; message: string } | null;
  libraries: Record<string, string> | null;
}
export type AnyRun = RunSummary | TabularRunSummary;
export const isTabularRun = (r: AnyRun): r is TabularRunSummary => r.kind === "tabular";

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
