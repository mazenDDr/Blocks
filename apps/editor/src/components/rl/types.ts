// Shapes of the /api/rl/* responses (mirrors services/control/rl_api.py).
import type { Diagnostic } from "../../types";

export interface SpaceInfo { type: string; n?: number; start?: number; shape?: number[]; dtype?: string; low?: (number | string)[]; high?: (number | string)[]; bounded?: boolean; nvec?: number[]; repr?: string }
export interface WrapperInfo { name: string; effect?: string; weights?: Record<string, number>; maxEpisodeSteps?: number; range?: number[]; base?: boolean }
export interface CatalogEntry {
  id: string; title: string; family: string; status: string; summary: string; version: number | null; gymnasium: string; rewardComponents: string[]; defaultWeights: Record<string, number>;
  rewardDefinition: string; termination: string; truncation: string; successRule: string | null; successText: string; renderModes: string[]; license: string; dependencies: string[];
  resources: string; seeding: string; snapshot: string; observationSpace: SpaceInfo; actionSpace: SpaceInfo; timeLimit: number | null; wrapperChain: WrapperInfo[]; autoresetModes: string[];
}
export interface Catalog { gymnasium: string; entries: CatalogEntry[]; note: string }

export interface DqnEquation {
  algorithm: string; targetEquation: string; lossEquation: string; update: string; targetUpdate: string; exploration: string; boundaries: string; updateToDataRatio: number; learningStarts: number;
}
export interface RLValidation {
  environment: null | { spec: Record<string, any>; observationSpace: SpaceInfo; actionSpace: SpaceInfo; timeLimit: number | null; wrapperChain: WrapperInfo[]; components: string[]; weights: Record<string, number>;
    defaultWeights: Record<string, number>; rewardModified: boolean; autoresetMode: string; catalog: Record<string, any> };
  network: null | { obsDim: number; nActions: number; params: number; outputNode: string; shapes: Record<string, Record<string, { shape: (number | string)[]; dtype: string }>> };
  learner: null | { algorithm: string; config: Record<string, any>; equation: DqnEquation };
  buffer: null | { capacity: number };
  catalogIds: string[]; ready: boolean; gridSchematic?: string[][];
}

export interface Setup {
  wrapperChain: WrapperInfo[]; autoresetMode: string; declaredAutoresetMode: string; components: string[]; effectiveWeights: Record<string, number>; obsDim: number; nActions: number; equation: DqnEquation; networkParams: number;
}
export interface EpisodeEnd { seq: number; tick: number; envIndex: number; episodeId: number; length: number; return: number; taskReturn: number; components: Record<string, number>; terminated: boolean; truncated: boolean; policyVersion: number; epsilon: number }
export interface UpdateEv { seq: number; update: number; tick: number; loss: number; gradNorm: number; meanQ: number; epsilon: number; bufferSize: number; policyVersion: number; targetVersion: number; meanTarget: number }
export interface MeanCI { n: number; mean: number | null; std: number | null; ci: [number, number] | null; level?: number; method?: string; warning?: string; note?: string }
export interface EvalEv {
  seq: number; tick: number; update: number; policyVersion: number; final: boolean; return: MeanCI; taskReturn: MeanCI; length: MeanCI; terminatedCount: number; truncatedCount: number; successRate: number | null; episodeReturns: number[];
}
export interface CapturedEv { seq: number; episodeId: number; sha256: string; length: number | null; capturedSteps: number; startTick: number; unfinished?: boolean }
export interface Curves {
  runId: string; status: string; maxSeq: number; cursor: number; setup: Setup | null; episodes: EpisodeEnd[]; updates: UpdateEv[]; evals: EvalEv[]; captured: CapturedEv[]; targetSyncs: number; totalSteps: number | null; seed: number | null;
  provenance: { runId: string; graphHash: string; source: string };
}

export interface CapturedStep {
  k: number; tid: number; qValues: number[]; action: number; actionSource: string; epsilon: number; policyVersion: number; reward: number; components: Record<string, number>; rawComponents: Record<string, number>;
  terminated: boolean; truncated: boolean; obs: number[]; nextObs: number[];
}
export interface CapturedEpisode {
  episodeId: number; envIndex: number; kind: string; startTick: number; steps: CapturedStep[]; frames: string[]; length: number | null; capturedSteps: number; return?: number; taskReturn?: number;
  terminated?: boolean; truncated?: boolean; components?: Record<string, number>; boundedCapture: boolean; frameNote: string; unfinished?: boolean; framesCapped: boolean;
}
export interface EvalEpisodeRow { seed: number; length: number; return: number; taskReturn: number; terminated: boolean; truncated: boolean; success: boolean | null; components: Record<string, number>; rawComponents: Record<string, number> }
export interface EvalCaptured { seed: number; frames: string[]; qValues: number[][]; actions: number[]; length: number; taskReturn: number; terminated: boolean; truncated: boolean; framesCapped: boolean }
export interface EvalEntry {
  tick: number; update: number; policyVersion: number; final: boolean; seeds: number[]; episodes: EvalEpisodeRow[]; epsilon: number; policy?: string; return: MeanCI; taskReturn: MeanCI; length: MeanCI;
  terminatedCount: number; truncatedCount: number; successRate: number | null; successRule: string; componentReturns: Record<string, MeanCI>; rawComponentReturns: Record<string, MeanCI>;
  rewardBasis: string; capturedEpisodes?: EvalCaptured[]; elapsedSec: number;
}
export interface EvalReport { runId: string; final: EvalEntry; history: EvalEntry[]; seed: number; evaluationSeeds: number[] }

export interface BufferRow {
  tid: number; envIndex: number; episodeId: number; episodeStep: number; policyVersion: number; insertTick: number; age: number; epsilon: number; actionSource: string; action: number; reward: number;
  components: Record<string, number>; terminated: boolean; truncated: boolean; sampleCount: number; lastSampledUpdate: number; obs: number[]; nextObs: number[];
}
export interface BufferPage {
  runId: string; rows: BufferRow[]; total: number; offset: number; limit: number;
  buffer: { size: number; capacity: number; transitionsInserted: number; evicted: number; tick: number; finalPolicyVersion: number; samplingWeight: string;
    sampleCounts: { min: number; max: number; mean: number; neverSampled: number }; policyVersionRange: number[]; terminated: number; truncated: number; ageRange: number[]; sampleCountHistogram: number[] };
  provenance: { runId: string; source: string; components: string[] };
}
export interface TraceOverview {
  runId: string; capturePolicy: { statement: string } & Record<string, any>; dropped: Record<string, number>; updateRecords: number; finalPolicyVersion: number; envSteps: number;
  tracked: { tid: number; episodeId: number; episodeStep: number; uses: number; evicted: boolean; policyVersion: number; terminated: boolean; truncated: boolean; reward: number }[];
}
export interface TraceUse {
  update: number; batchPosition: number; batchSize: number; slot: number; samplingProbability: number; tick: number; qSA: number; nextValue: number; bootstrapMask: number; target: number; tdError: number;
  lossContribution: number; dLossDQ: number; huberRegion: string | null; batchLoss: number; gradNorm: number; policyVersionBefore: number; policyVersionAfter: number; targetVersion: number; paramSha256After: string | null;
  minibatch: null | { tracked: number[]; size: number; batchTids: number[]; meanQ: number; bufferSize: number; targetSynced: boolean };
}
export interface TransitionTrace {
  captured: boolean; tid: number; reason?: string; capturePolicy: string; finalBuffer?: Record<string, any>;
  transition?: BufferRowLike; uses?: TraceUse[]; chain?: { stage: string; text: string }[]; components?: string[]; usesDropped?: number;
}
export interface BufferRowLike {
  tid: number; slot: number; envIndex: number; episodeId: number; episodeStep: number; policyVersion: number; insertTick: number; epsilon: number; actionSource: string; obs: number[]; action: number; reward: number;
  components: Record<string, number>; rawComponents: Record<string, number>; nextObs: number[]; terminated: boolean; truncated: boolean; sampleCount: number; lastSampledUpdate: number;
  evicted: null | { byTid: number; tick: number; afterUpdate: number }; qValues?: number[]; frameIndex?: number;
}

export interface MeanCIAgg extends MeanCI {}
export interface VariantRun { runId: string; status: string; seed: number | null; available: boolean; taskReturn?: number; trainReturn?: number; length?: number; terminated?: number; truncated?: number; successRate?: number | null; weights?: Record<string, number>; envSteps?: number; updates?: number; reason?: string }
export interface Variant {
  label: string; isBaseline?: boolean; assignments: unknown[]; runs: VariantRun[]; nRuns: number; nCompleted: number; taskReturn: MeanCI; trainReturn: MeanCI; length: MeanCI; successRate: MeanCI;
  terminated: number; truncated: number; episodes: number; components: Record<string, MeanCI>; rawComponents: Record<string, MeanCI>; initialTaskReturn: MeanCI; envSteps: number | null; updates: number | null; weights: Record<string, number> | null;
}
export interface Comparison { studyId?: string; name?: string; variants: Variant[]; evaluationSeedSets: number[][]; notes: string[] }

export type RLDiagnostic = Diagnostic;
