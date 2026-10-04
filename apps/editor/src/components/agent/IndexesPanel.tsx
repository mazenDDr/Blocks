import { useState } from "react";
import { api } from "../../api";
import type { Graph } from "../../types";
import { Check, ErrorLine, Num, Row, Sel, fmtTime, useAction } from "./common";
import { specOf, type IndexManifest, type IndexSpec } from "./types";

interface Hit { chunk_id: string; doc_id: string; score: number; rank: number; text: string; reason?: string }

function IndexCard({ ix, graph, onChange, onRemove }: { ix: IndexSpec; graph: Graph; onChange: (i: IndexSpec) => void; onRemove: () => void }) {
  const { busy, error, run } = useAction();
  const [man, setMan] = useState<IndexManifest | null>(null);
  const [chunks, setChunks] = useState<{ chunk_id: string; text: string; start: number }[] | null>(null);
  const [q, setQ] = useState(""); const [k, setK] = useState(3);
  const [res, setRes] = useState<{ documents: Hit[]; excluded: Hit[]; scoreInterpretation: string; embedding: { identity: string } } | null>(null);
  const body = { graph, indexId: ix.id };
  const info = () => run(async () => { const r = await api.post<{ built: boolean; manifest: IndexManifest | null }>("/api/agent/indexes/info", body); setMan(r.manifest); });
  const set = (p: Partial<IndexSpec>) => onChange({ ...ix, ...p });
  return (
    <div className="indexcard" aria-label={`index ${ix.id}`}>
      <h4>{ix.id}</h4>
      <Row label="Id"><input aria-label="index id" value={ix.id} onChange={(e) => set({ id: e.target.value })} /></Row>
      <Row label="Document folder" hint="local text files; relative to the repository or absolute"><input aria-label="index directory" value={ix.loader.directory} onChange={(e) => set({ loader: { ...ix.loader, directory: e.target.value } })} /></Row>
      <Row label="File pattern"><input aria-label="index glob" value={ix.loader.glob} onChange={(e) => set({ loader: { ...ix.loader, glob: e.target.value } })} /></Row>
      <Row label="Splitter"><Sel label="splitter strategy" value={ix.splitter.strategy} options={["recursive", "paragraph", "fixed"]} onChange={(s) => set({ splitter: { ...ix.splitter, strategy: s } })} />
        size <Num label="chunk size" integer min={20} value={ix.splitter.chunkSize} onChange={(n) => set({ splitter: { ...ix.splitter, chunkSize: n ?? 400 } })} />
        overlap <Num label="chunk overlap" integer min={0} value={ix.splitter.chunkOverlap} onChange={(n) => set({ splitter: { ...ix.splitter, chunkOverlap: n ?? 0 } })} /></Row>
      <Row label="Embeddings"><Sel label="index embedding provider" value={ix.embeddings.provider} options={[{ value: "local_hash", label: "local_hash (lexical, NOT semantic)" }, { value: "ollama", label: "Ollama embeddings" }]} onChange={(p) => set({ embeddings: { ...ix.embeddings, provider: p } })} />
        {ix.embeddings.provider === "ollama" ? <input aria-label="index embedding model" value={ix.embeddings.model} onChange={(e) => set({ embeddings: { ...ix.embeddings, model: e.target.value } })} />
          : <Num label="index embedding dimension" integer min={16} value={ix.embeddings.dimension} onChange={(n) => set({ embeddings: { ...ix.embeddings, dimension: n ?? 256 } })} />}
        <label className="small"><Check label="index normalize" value={ix.embeddings.normalize} onChange={(b) => set({ embeddings: { ...ix.embeddings, normalize: b } })} /> L2-normalize</label></Row>
      <div className="small muted">Vector store: FAISS (flat inner product), persisted under the workbench. Corpus indexing is separate from per-question runs; unchanged chunk embeddings are reused from a cache.</div>
      <div className="rcasehead">
        <button disabled={busy} onClick={() => run(async () => { const m = await api.post<IndexManifest>("/api/agent/indexes/build", body); setMan(m); setChunks(null); })}>Build / refresh index</button>
        <button disabled={busy} onClick={info}>Show status</button>
        <button disabled={busy || !man} onClick={() => run(async () => setChunks((await api.post<{ chunks: { chunk_id: string; text: string; start: number }[] }>("/api/agent/indexes/chunks", body)).chunks))}>Browse chunks</button>
        <button className="danger" onClick={onRemove}>Remove index</button>
      </div>
      <ErrorLine text={error} />
      {man && (
        <div className="manifest" aria-label="index manifest">
          <div><b>{man.action ?? "status"}</b>: {man.documents.length} documents → {man.chunks} chunks · dimension {man.dimension} · built {fmtTime(man.builtAt)} in {man.buildSeconds}s · identity <code>{man.identity.slice(0, 12)}</code></div>
          <div>Embeddings: {man.embedding.label} <span className="muted">({man.embedding.identity})</span> — computed {man.embeddingsComputed}, reused from cache {man.embeddingsReused}</div>
          <div className="small muted">Score meaning: {man.scoreInterpretation}</div>
          <div className="small">{man.documents.map((d) => `${d.doc_id} (${d.bytes} B, sha ${d.sha256.slice(0, 8)})`).join(" · ")}</div>
        </div>
      )}
      {chunks && <div className="tablewrap short"><table><tbody>{chunks.map((c) => <tr key={c.chunk_id}><td><code>{c.chunk_id}</code></td><td className="num">@{c.start}</td><td>{c.text.slice(0, 160)}</td></tr>)}</tbody></table></div>}
      <div className="rcasehead">Test a query <input aria-label="index query" value={q} size={36} onChange={(e) => setQ(e.target.value)} /> k <Num label="index k" integer min={1} value={k} onChange={(n) => setK(n ?? 3)} />
        <button disabled={busy || !q.trim()} onClick={() => run(async () => setRes(await api.post("/api/agent/indexes/search", { ...body, query: q, k })))}>Search</button></div>
      {res && (
        <table aria-label="search results"><thead><tr><th>rank</th><th>chunk</th><th>score</th><th /></tr></thead><tbody>
          {res.documents.map((d) => <tr key={d.chunk_id}><td>{d.rank}</td><td><code>{d.chunk_id}</code><div className="small muted">{d.text.slice(0, 100)}</div></td><td className="num">{d.score}</td><td>included</td></tr>)}
          {res.excluded.slice(0, 5).map((d) => <tr key={d.chunk_id} className="muted"><td>{d.rank}</td><td><code>{d.chunk_id}</code></td><td className="num">{d.score}</td><td>{d.reason}</td></tr>)}
        </tbody></table>
      )}
    </div>
  );
}

export function IndexesTab({ graph, setGraph }: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void }) {
  const spec = specOf(graph);
  const set = (fn: (l: IndexSpec[]) => IndexSpec[]) => setGraph((g) => ({ ...g, agent: { ...(g.agent ?? {}), indexes: fn(specOf(g).indexes) } }));
  return (
    <div className="aindexes">
      <h3>Document indexes <small className="muted">loader → splitter → embeddings → vector store; retrieval blocks refer to an index by id</small></h3>
      {spec.indexes.length === 0 && <div className="empty">No indexes. Add one to retrieve from local text files.</div>}
      {spec.indexes.map((ix, i) => <IndexCard key={i} ix={ix} graph={graph} onChange={(n) => set((l) => l.map((x, j) => (j === i ? n : x)))} onRemove={() => set((l) => l.filter((_, j) => j !== i))} />)}
      <button onClick={() => set((l) => [...l, { id: `index_${l.length + 1}`, description: "", loader: { directory: "examples/agent_docs", glob: "*.txt" }, splitter: { strategy: "recursive", chunkSize: 400, chunkOverlap: 40 }, embeddings: { provider: "local_hash", model: "nomic-embed-text", dimension: 256, normalize: true }, store: "faiss" }])}>Add index</button>
    </div>
  );
}
