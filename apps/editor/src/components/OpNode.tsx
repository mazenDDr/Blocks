import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { isTensorType, type GNode, type NodeView, type OpInfo, type TabularNodeStatus } from "../types";
import { fmtInt, fmtShape, summarize } from "../util";
import { SchemaList } from "./Tabular";

export interface CardData extends Record<string, unknown> {
  gnode: GNode; op?: OpInfo; view?: NodeView; pending: boolean; runStatus?: TabularNodeStatus;
}
export type CardNode = Node<CardData, "card">;

const portTop = (i: number, n: number) => `${((i + 1) * 100) / (n + 1)}%`;

/** VISION 8.3 card. Model graphs show tensor shapes and parameter counts; tabular graphs show the schema (columns/types), row count,
 *  partition and the kind of each wire instead. */
export function OpNodeCard({ data, selected }: NodeProps<CardNode>) {
  const { gnode, op, view, pending, runStatus } = data;
  const errors = view?.diagnostics.filter((d) => d.severity === "error") ?? [];
  const warnings = view?.diagnostics.filter((d) => d.severity === "warning") ?? [];
  const inputs = op?.inputs ?? [], outputs = op?.outputs ?? [];
  const multiIn = inputs.length > 1, multiOut = outputs.length > 1;
  const tabular = op?.graphKind === "tabular";
  return (
    <div className={`card ${tabular ? "tab" : ""} ${selected ? "sel" : ""} ${errors.length ? "err" : ""} ${op ? "" : "unknown"}`}>
      {inputs.map((p, i) => <Handle key={p} id={p} type="target" position={Position.Left} style={{ top: portTop(i, inputs.length) }} title={`input ${p}${op?.inputKinds?.[p] ? ` (${op.inputKinds[p]})` : ""}`} />)}
      {outputs.map((p, i) => <Handle key={p} id={p} type="source" position={Position.Right} style={{ top: portTop(i, outputs.length) }} title={`output ${p}${op?.outputKinds?.[p] ? ` (${op.outputKinds[p]})` : ""}`} />)}
      <div className="card-head">
        <b>{op?.displayName ?? gnode.type}</b>
        <span className="badge">{op?.backend ?? "unresolved"}</span>
      </div>
      <div className="card-id">{gnode.id}{runStatus && <span className={`runmark ${runStatus.status}`}>{runStatus.status === "finished" ? "ran" : runStatus.status}</span>}</div>
      {op ? <div className="card-sum">{summarize(gnode, view?.resolvedConfig)}</div> : <div className="card-sum">Operation not available: preserved, cannot run.</div>}
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
