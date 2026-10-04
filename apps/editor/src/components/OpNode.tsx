import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { isTensorType, type GNode, type NodeView, type OpInfo, type TabularNodeStatus } from "../types";
import { fmtInt, fmtShape, summarize } from "../util";
import { SchemaList } from "./Tabular";

export interface CardData extends Record<string, unknown> {
  gnode: GNode; op?: OpInfo; view?: NodeView; pending: boolean; runStatus?: TabularNodeStatus;
  /** composite / repeat / select: open the module for editing, expand or collapse the inner nodes in place */
  onOpen?: () => void; onToggle?: () => void; expanded?: boolean; inner?: boolean; localName?: string;
}
export type CardNode = Node<CardData, "card">;

const portTop = (i: number, n: number) => `${((i + 1) * 100) / (n + 1)}%`;

/** VISION 8.3 card. Model graphs show tensor shapes and parameter counts; tabular graphs show the schema (columns/types), row count,
 *  partition and the kind of each wire instead. */
export function OpNodeCard({ data, selected }: NodeProps<CardNode>) {
  const { gnode, op, view, pending, runStatus, onOpen, onToggle, expanded, inner, localName } = data;
  const errors = view?.diagnostics.filter((d) => d.severity === "error") ?? [];
  const warnings = view?.diagnostics.filter((d) => d.severity === "warning") ?? [];
  const inputs = view?.inputPorts ?? op?.inputs ?? [], outputs = view?.outputPorts ?? op?.outputs ?? [];
  const structural = !!view?.structural || ["core.composite", "core.repeat", "core.select"].includes(gnode.type);
  const shared = view?.sharedWith ?? gnode.sharedWith ?? null;
  const diagKind = gnode.type.startsWith("diag.") ? (gnode.type === "diag.assert" ? "execution-changing" : "observation-only") : null;
  const multiIn = inputs.length > 1, multiOut = outputs.length > 1;
  const tabular = op?.graphKind === "tabular";
  return (
    <div className={`card ${tabular ? "tab" : ""} ${selected ? "sel" : ""} ${errors.length ? "err" : ""} ${op ? "" : "unknown"} ${structural ? "structural" : ""} ${shared ? "shared" : ""} ${inner ? "inner" : ""} ${diagKind ?? ""}`}>
      {inputs.map((p, i) => <Handle key={p} id={p} type="target" position={Position.Left} style={{ top: portTop(i, inputs.length) }} title={`input ${p}${op?.inputKinds?.[p] ? ` (${op.inputKinds[p]})` : ""}`} />)}
      {outputs.map((p, i) => <Handle key={p} id={p} type="source" position={Position.Right} style={{ top: portTop(i, outputs.length) }} title={`output ${p}${op?.outputKinds?.[p] ? ` (${op.outputKinds[p]})` : ""}`} />)}
      <div className="card-head">
        <b>{structural ? (view?.kind ? `${view.kind.replace("core.", "")}` : op?.displayName) : (op?.displayName ?? gnode.type)}</b>
        <span className="badge">{structural ? "module" : (op?.backend ?? "unresolved")}</span>
        {shared && <span className="badge shared-badge" title={`This call site uses the parameter tensors of ${shared}: the very same tensors, not a copy.`}>{"\u{1F517}"} shared</span>}
        {diagKind && <span className={`badge diag-${diagKind}`} title={diagKind === "observation-only" ? "Passes the value through unchanged; records only while a capture covers it." : "Stops the run when its check fails. Only active when assertions are enabled."}>{diagKind}</span>}
      </div>
      <div className="card-id">{inner ? localName : gnode.id}{runStatus && <span className={`runmark ${runStatus.status}`}>{runStatus.status === "finished" ? "ran" : runStatus.status}</span>}</div>
      {structural ? (
        <div className="card-sum">
          {view?.module ?? (gnode.config as any).module ?? "(no module)"} {view?.version ? `v${view.version}` : ""}
          {view?.iterations ? ` · ${view.iterations} iterations (${view.termination ?? "fixed_count"})` : ""}
          {shared ? <div className="muted">shares parameters with {shared}</div> : (view?.kind === "core.composite" ? <div className="muted">own parameters (cloned)</div> : null)}
          <div className="card-actions">
            {onOpen && <button className="nodrag" onClick={(e) => { e.stopPropagation(); onOpen(); }}>Open</button>}
            {onToggle && <button className="nodrag" onClick={(e) => { e.stopPropagation(); onToggle(); }}>{expanded ? "Collapse" : "Expand"}</button>}
          </div>
        </div>
      ) : op ? <div className="card-sum">{summarize(gnode, view?.resolvedConfig)}</div> : <div className="card-sum">Operation not available: preserved, cannot run.</div>}
      {view?.typed ? (
        <div className="card-shapes">
          {tabular ? (
            <>
              {inputs.map((p) => { const t = view.inputShapes?.[p]; return t && !isTensorType(t) && t.kind !== "table" ? <div key={"i" + p}>In {p}: {t.kind.replace("_", " ")}</div> : null; })}
              {outputs.map((p) => {
                const t = view.outputShapes?.[p];
                if (!t || isTensorType(t)) return null;
                return (
                  <div key={"o" + p} className="outport">
                    <div><b>{multiOut ? `${p}: ` : "Out: "}</b>{t.kind === "table" ? <>{fmtShape(t)} {t.partition && t.partition !== "full" && <span className={`badge part-${t.partition}`}>{t.partition}</span>}</> : t.kind.replace("_", " ")}</div>
                    {t.kind === "table" && <SchemaList t={t} />}
                    {t.kind === "table" && t.rowsExact === false && t.rows == null && <div className="muted">row count depends on the data (shown after a run)</div>}
                  </div>
                );
              })}
            </>
          ) : (
            <>
              {Object.entries(view.inputShapes ?? {}).map(([p, t]) => <div key={"i" + p}>In{multiIn ? ` ${p}` : ""}: {fmtShape(t)}</div>)}
              {Object.entries(view.outputShapes ?? {}).map(([p, t]) => <div key={"o" + p}>Out{multiOut ? ` ${p}` : ""}: {fmtShape(t)}</div>)}
              {(view.params ?? 0) > 0 && <div className="params">{fmtInt(view.params!)} trainable parameters</div>}
            </>
          )}
        </div>
      ) : (
        <div className="card-shapes muted">{pending ? "validating…" : "schema unknown"}</div>
      )}
      {errors.length > 0 && (
        <div className="errbadge" title={errors.map((e) => `${e.code}: ${e.message}`).join("\n")}>
          <b>{errors[0].code}</b> {errors[0].message}
        </div>
      )}
      {errors.length === 0 && warnings.length > 0 && <div className="warn small" title={warnings.map((e) => e.message).join("\n")}><b>{warnings[0].code}</b></div>}
    </div>
  );
}
