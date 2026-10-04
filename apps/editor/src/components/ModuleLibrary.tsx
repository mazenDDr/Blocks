import { useEffect, useState } from "react";
import { api, errorText } from "../api";
import type { CodeBlockDef, Graph, ModuleDef } from "../types";

interface Props {
  graph: Graph; inModule: boolean;
  onAddInstance: (m: ModuleDef) => void; onOpen: (m: ModuleDef) => void; onNew: () => void; onImport: (m: ModuleDef) => void; onRemove: (m: ModuleDef) => void;
  onAddCodeNode: (d: CodeBlockDef) => void; onEditCode: (d: CodeBlockDef) => void; onNewCode: () => void; onImportCode: (d: CodeBlockDef) => void;
  usedModules: Set<string>; setMessage: (m: string) => void;
}
const sig = (m: ModuleDef) => `${m.inputs.map((p) => p.name).join(", ") || "-"} → ${m.outputs.map((o) => o.name).join(", ") || "-"}`;

/** Reusable visual functions and code blocks: the project's own definitions, starter modules, and published versions ("My modules"). */
export function ModuleLibrary(p: Props) {
  const [starters, setStarters] = useState<ModuleDef[]>([]);
  const [mine, setMine] = useState<{ id: string; version: string; description: string; inputs: string[]; outputs: string[] }[]>([]);
  const [myCode, setMyCode] = useState<{ id: string; version: string; description: string }[]>([]);
  useEffect(() => {
    api.get<{ modules: ModuleDef[] }>("/api/modules/starters").then((r) => setStarters(r.modules)).catch(() => {});
    api.get<{ modules: typeof mine }>("/api/modules").then((r) => setMine(r.modules)).catch(() => {});
    api.get<{ codeBlocks: typeof myCode }>("/api/codeblocks").then((r) => setMyCode(r.codeBlocks)).catch(() => {});
  }, [p.graph.modules?.length, p.graph.codeBlocks?.length]);
  const have = (id: string, v: string) => (p.graph.modules ?? []).some((m) => m.id === id && m.version === v);
  const importPublished = async (id: string, version: string) => {
    try { const r = await api.get<{ definition: ModuleDef }>(`/api/modules/${id}?version=${version}`); p.onImport(r.definition); } catch (e) { p.setMessage(errorText(e)); }
  };
  const importCode = async (id: string, version: string) => {
    try { const r = await api.get<{ definition: CodeBlockDef }>(`/api/codeblocks/${id}?version=${version}`); p.onImportCode(r.definition); } catch (e) { p.setMessage(errorText(e)); }
  };
  return (
    <div className="library" aria-label="Modules and code blocks">
      <h3>Modules</h3>
      <div className="hint">A module is a reusable subgraph with named, typed ports and arguments. Instances clone parameters unless you share them.</div>
      <button onClick={p.onNew}>New module…</button>
      <h4>In this project</h4>
      {(p.graph.modules ?? []).length === 0 && <div className="empty">No modules yet.</div>}
      {(p.graph.modules ?? []).map((m) => (
        <div className="lib-item static" key={`${m.id}@${m.version}`}>
          <b>{m.id}</b><span className="badge">v{m.version}</span><small>{sig(m)} · {m.nodes.length} nodes</small>
          <div className="actions"><button onClick={() => p.onAddInstance(m)}>Add instance</button><button onClick={() => p.onOpen(m)}>Open</button>
            <button className="danger" disabled={p.usedModules.has(`${m.id}@${m.version}`)} title={p.usedModules.has(`${m.id}@${m.version}`) ? "In use by an instance" : "Remove this definition"} onClick={() => p.onRemove(m)}>Remove</button></div>
        </div>
      ))}
      <h4>Starter modules (built from primitives)</h4>
      {starters.map((m) => (
        <div className="lib-item static" key={m.id}>
          <b>{m.id}</b><span className="badge">v{m.version}</span><small>{sig(m)}</small>
          <div className="small muted">{m.description.slice(0, 110)}{m.description.length > 110 ? "…" : ""}</div>
          <button disabled={have(m.id, m.version)} onClick={() => p.onImport(m)}>{have(m.id, m.version) ? "In project" : "Import into project"}</button>
        </div>
      ))}
      <h4>My modules (published, immutable versions)</h4>
      {mine.length === 0 && <div className="empty">Nothing published yet. Open a module, then Versions ▸ Publish.</div>}
      {mine.map((m) => (
        <div className="lib-item static" key={`${m.id}@${m.version}`}>
          <b>{m.id}</b><span className="badge">v{m.version}</span><small>{m.inputs.join(", ") || "-"} → {m.outputs.join(", ") || "-"}</small>
          <button disabled={have(m.id, m.version)} onClick={() => importPublished(m.id, m.version)}>{have(m.id, m.version) ? "In project" : "Import"}</button>
        </div>
      ))}
      <h3>Python code blocks <small className="muted">optional</small></h3>
      <div className="hint">Typed interface, tested against fixtures, run in an isolated subprocess. Prefer visual blocks when they exist.</div>
      <button onClick={p.onNewCode}>New code block…</button>
      {(p.graph.codeBlocks ?? []).map((d) => (
        <div className="lib-item static" key={`${d.id}@${d.version}`}>
          <b>{d.id}</b><span className="badge">v{d.version}</span><small>{d.inputs.map((i) => i.name).join(", ")} → {d.outputs.map((o) => o.name).join(", ")} · {d.differentiable ? "differentiable" : "gradient boundary"}</small>
          <div className="actions"><button disabled={p.inModule} title={p.inModule ? "Go back to the project to add a node" : ""} onClick={() => p.onAddCodeNode(d)}>Add node</button><button onClick={() => p.onEditCode(d)}>Edit code</button></div>
        </div>
      ))}
      {myCode.length > 0 && <h4>My code blocks</h4>}
      {myCode.map((d) => (
        <div className="lib-item static" key={`${d.id}@${d.version}`}><b>{d.id}</b><span className="badge">v{d.version}</span><small>{d.description}</small>
          <button disabled={(p.graph.codeBlocks ?? []).some((x) => x.id === d.id && x.version === d.version)} onClick={() => importCode(d.id, d.version)}>Import</button></div>
      ))}
    </div>
  );
}
