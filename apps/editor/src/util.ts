import { isTensorType, type AnyRun, type Dim, type GNode, type OpInfo, type WireType } from "./types";

export const fmtDim = (d: Dim) => String(d);
export function fmtShape(t?: WireType | null): string {
  if (!t) return "unknown";
  if (isTensorType(t)) return t.shape.map(fmtDim).join(" × ");
  if (t.kind === "table") return `${t.rows == null ? "? rows" : `${fmtInt(t.rows)} rows`} × ${t.columns?.length ?? 0}${t.columnsComplete === false ? "+" : ""} cols`;
  return t.kind;
}
export const dtypeOf = (t?: WireType | null) => (t && isTensorType(t) ? t.dtype : "");
export const fmtInt = (n: number) => n.toLocaleString("en-US");
export const fmtNum = (v: number, digits = 4) => (Number.isFinite(v) ? Number(v.toPrecision(digits)).toString() : String(v));
/** p-values: fixed decimals when moderate (0.0404276820), exponent when tiny. */
export const fmtP = (p: number | null | undefined) => (p == null || !Number.isFinite(p) ? "n/a" : p >= 0.001 ? p.toFixed(10) : p.toExponential(6));
export const shortHash = (h?: string | null) => (h ? h.slice(0, 8) : "n/a");

export const pair = (v: unknown): [number, number] => (Array.isArray(v) ? [Number(v[0]), Number(v[1])] : [Number(v), Number(v)]);

/** One-line summary of the settings that matter, per VISION 8.3 (e.g. "32 filters · 3x3 kernel · stride 1 · padding 1"). */
export function summarize(node: GNode, resolved?: Record<string, unknown>): string {
  if (node.type === "__module_input") return `${(node.config as any).name}: ${(node.config as any).dtype}${(node.config as any).shape ? " [" + (node.config as any).shape.map((d: any) => d ?? "?").join(", ") + "]" : ""}`;
  if (node.type === "__module_output") return `output ${(node.config as any).name}`;
  if (node.type === "tensor.dense") { const c = { ...node.config, ...(resolved ?? {}) } as any; return `${c.in_features ?? "?"} → ${c.out_features ?? 8}${c.bias === false ? " · no bias" : ""}`; }
  const c = { ...node.config, ...(resolved ?? {}) } as Record<string, any>;
  const k = (v: any) => pair(v).join("×");
  switch (node.type) {
    case "pytorch.nn.conv2d": {
      const pad = Array.isArray(c.padding) ? pair(c.padding)[0] : c.padding ?? 0;
      return `${c.out_channels ?? "?"} filters · ${k(c.kernel_size ?? [3, 3])} kernel · stride ${pair(c.stride ?? 1)[0]} · padding ${pad}`;
    }
    case "pytorch.nn.max_pool2d": return `${k(c.kernel_size ?? [2, 2])} window · stride ${c.stride ? pair(c.stride)[0] : "= window"}`;
    case "pytorch.nn.adaptive_avg_pool2d": return `output ${k(c.output_size ?? [1, 1])}`;
    case "pytorch.nn.linear": return `${c.out_features ?? "?"} outputs${c.bias === false ? " · no bias" : ""}`;
    case "pytorch.nn.flatten": return `dims ${c.start_dim ?? 1}..${c.end_dim ?? -1}`;
    case "pytorch.loss.cross_entropy": return `reduction ${c.reduction ?? "mean"}`;
    case "core.tensor_input": return `${(c.shape as Dim[] | undefined)?.map(fmtDim).join("×") ?? "?"} ${c.dtype ?? ""}`;
    case "tabular.csv_source": return String(c.path ?? "").split("/").pop() || "no file chosen";
    case "postgres.query": return `${c.connection || "(no connection)"} · ${c.mode === "sql" ? "raw SQL" : c.query ? `${c.query.base?.name}${c.query.joins?.length ? ` + ${c.query.joins.length} join` : ""}` : "no table"}${c.pin ? " · pinned" : ""}`;
    case "s3.csv_source": return `${c.connection || "(no connection)"} · ${c.key || "no object"}${c.pin ? " · pinned" : c.version_id ? " · version set" : ""}`;
    case "s3.object_listing": return `${c.connection || "(no connection)"} · prefix ${c.prefix || "/"}${c.pin ? " · pinned" : ""}`;
    case "dvc.csv_source": return `${c.connection || "(no connection)"} · ${c.path || "no path"} @ ${c.rev || "default rev"}${c.pin ? " · pinned" : ""}`;
    case "tabular.join": return `${c.how ?? "inner"} on ${(c.left_on as string[] | undefined)?.join(",") || "?"} = ${(c.right_on as string[] | undefined)?.join(",") || "?"}${c.expect && c.expect !== "any" ? ` · expect ${c.expect}` : ""}`;
    case "tabular.select_columns": return `${(c.columns as unknown[] | undefined)?.length ?? 0} typed columns`;
    case "tabular.train_validation_split": return `validation ${c.validation_fraction ?? 0.25} · seed ${c.seed ?? 0}${c.stratify_by ? ` · stratified by ${c.stratify_by}` : ""}${c.group_by ? ` · grouped by ${c.group_by}` : ""}`;
    case "tabular.fit_standardize": case "tabular.fit_onehot": return `fit on train · ${(c.columns as string[] | undefined)?.length ? (c.columns as string[]).join(", ") : "all matching columns"}`;
    case "tabular.fit_impute": return `${c.strategy ?? "median"} · fit on train · ${(c.columns as string[] | undefined)?.length ? (c.columns as string[]).join(", ") : "all numeric"}`;
    case "tabular.apply_transform": return "apply fitted state (no refit)";
    case "sklearn.linear_regression": case "sklearn.logistic_regression": return `target ${c.target || "(not set)"}${c.C !== undefined ? ` · C ${c.C}` : ""}`;
    case "scipy.gamma_distribution": return c.parametrization === "shape_rate" ? `shape ${c.shape} · rate ${c.rate} · loc ${c.loc}` : `shape ${c.shape} · scale ${c.scale} · loc ${c.loc}`;
    case "scipy.tail_probability": return `${c.tail ?? "upper"} tail at ${c.observed}`;
    case "scipy.hypothesis_test": return `alpha ${c.alpha ?? 0.05}${c.teaching_fixture ? " · teaching fixture" : ""}`;
    case "scipy.two_group_comparison": return `${c.method ?? "welch"} · ${c.value_column || "?"} by ${c.group_column || "?"} (${c.group_a || "?"} vs ${c.group_b || "?"})`;
    default: {
      const e = Object.entries(node.config).slice(0, 3).map(([a, b]) => `${a}=${JSON.stringify(b)}`);
      return e.join(" · ");
    }
  }
}

export function shortName(op: OpInfo): string {
  return op.type.split(".").pop()!;
}

export function nextId(base: string, taken: Set<string>): string {
  let i = 1;
  while (taken.has(`${base}_${i}`)) i++;
  return `${base}_${i}`;
}

export function runLabel(r: AnyRun): string {
  if (r.kind === "procedure") return `${r.id} · ${r.status} · step ${r.progress.step}${r.validation?.val_loss != null ? ` · val loss ${r.validation.val_loss.toFixed(3)}` : ""}${r.rerunOf ? " · rerun" : ""}`;
  if (r.kind === "sandbox") return `${r.id} · sandbox of ${r.parent} step ${r.step}`;
  if (r.kind === "tabular") return `${r.id} · ${r.status} · ${r.progress.nodesDone}/${r.progress.nodes} nodes`;
  if (r.kind === "rl") return `${r.id} · ${r.status} · ${fmtInt(r.envSteps)}${r.totalSteps ? `/${fmtInt(r.totalSteps)}` : ""} steps · seed ${r.seed}`;
  if (r.kind === "agent") return `${r.id} · ${r.status} · thread ${r.threadId}${r.fixtureCalls ? " · FIXTURE" : ""}`;
  return `${r.id} · ${r.status}${r.final ? ` · val acc ${(r.final.val_acc * 100).toFixed(0)}%` : ""}`;
}

export const uid = () => (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36));

// ---- colour scales ---------------------------------------------------------------------------------
type RGB = [number, number, number];
const lerp = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Diverging blue-white-red, symmetric about 0: t in [-1, 1]. */
export function diverging(t: number): RGB {
  const c = Math.max(-1, Math.min(1, t));
  const white: RGB = [255, 255, 255];
  return c < 0 ? lerp(white, [33, 102, 172], -c) : lerp(white, [178, 24, 43], c);
}

const VIRIDIS: RGB[] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
/** Sequential viridis: t in [0, 1]. */
export function viridis(t: number): RGB {
  const c = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(c));
  return lerp(VIRIDIS[i], VIRIDIS[i + 1], c - i);
}
export const css = (c: RGB) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;
