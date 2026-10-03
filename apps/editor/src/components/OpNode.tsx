import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import type { GNode, NodeView, OpInfo } from "../types";
import { fmtInt, fmtShape, summarize } from "../util";

export interface CardData extends Record<string, unknown> {
  gnode: GNode; op?: OpInfo; view?: NodeView; pending: boolean;
}
export type CardNode = Node<CardData, "card">;

const portTop = (i: number, n: number) => `${((i + 1) * 100) / (n + 1)}%`;

/** VISION 8.3 card: op name, backend badge, key settings, live input/output shapes, parameter count, error badge. */
export function OpNodeCard({ data, selected }: NodeProps<CardNode>) {
  const { gnode, op, view, pending } = data;
  const errors = view?.diagnostics.filter((d) => d.severity === "error") ?? [];
  const inputs = op?.inputs ?? [], outputs = op?.outputs ?? [];
  const multiIn = inputs.length > 1, multiOut = outputs.length > 1;
  return (
    <div className={`card ${selected ? "sel" : ""} ${errors.length ? "err" : ""} ${op ? "" : "unknown"}`}>
      {inputs.map((p, i) => <Handle key={p} id={p} type="target" position={Position.Left} style={{ top: portTop(i, inputs.length) }} title={`input ${p}`} />)}
      {outputs.map((p, i) => <Handle key={p} id={p} type="source" position={Position.Right} style={{ top: portTop(i, outputs.length) }} title={`output ${p}`} />)}
      <div className="card-head">
        <b>{op?.displayName ?? gnode.type}</b>
        <span className="badge">{op?.backend ?? "unresolved"}</span>
      </div>
      <div className="card-id">{gnode.id}</div>
      {op ? <div className="card-sum">{summarize(gnode, view?.resolvedConfig)}</div> : <div className="card-sum">Operation not available: preserved, cannot run.</div>}
      {view?.typed ? (
        <div className="card-shapes">
          {Object.entries(view.inputShapes ?? {}).map(([p, t]) => <div key={"i" + p}>In{multiIn ? ` ${p}` : ""}: {fmtShape(t)}</div>)}
          {Object.entries(view.outputShapes ?? {}).map(([p, t]) => <div key={"o" + p}>Out{multiOut ? ` ${p}` : ""}: {fmtShape(t)}</div>)}
          {(view.params ?? 0) > 0 && <div className="params">{fmtInt(view.params!)} trainable parameters</div>}
        </div>
      ) : (
        <div className="card-shapes muted">{pending ? "validating…" : "shape unknown"}</div>
      )}
      {errors.length > 0 && (
        <div className="errbadge" title={errors.map((e) => `${e.code}: ${e.message}`).join("\n")}>
          <b>{errors[0].code}</b> {errors[0].message}
        </div>
      )}
    </div>
  );
}
