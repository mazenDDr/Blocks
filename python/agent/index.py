"""Corpus index: local text loader + splitter + embeddings + FAISS vector store, persisted under `<workbench>/agent/indexes/<id>/`.

Index construction is separate from per-question execution (VISION 12.3): `ensure_index` builds only when the corpus files, splitter or
embedding identity changed; unchanged chunk embeddings are reused from a cache keyed by (embedding identity, text hash)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from storage.schema import EMBEDDINGS, open_database
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from .models import Embedder
from .spec import IndexSpec

REPO = Path(__file__).resolve().parents[2]


def resolve_dir(d: str) -> Path:
    p = Path(d).expanduser()
    if p.is_absolute():
        return p
    return p.resolve() if p.resolve().exists() else REPO / d


def load_documents(spec: IndexSpec) -> list[dict[str, Any]]:
    root = resolve_dir(spec.loader.directory)
    if not root.is_dir():
        raise FileNotFoundError(f"document directory '{spec.loader.directory}' does not exist")
    docs = []
    for p in sorted(root.glob(spec.loader.glob)):
        if p.is_file():
            data = p.read_bytes()
            docs.append({"doc_id": p.name, "path": str(p), "text": data.decode(spec.loader.encoding), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return docs


def split_documents(docs: list[dict[str, Any]], spec: IndexSpec) -> list[dict[str, Any]]:
    from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

    s = spec.splitter
    kw = {"chunk_size": s.chunkSize, "chunk_overlap": min(s.chunkOverlap, s.chunkSize - 1), "add_start_index": True}
    if s.strategy == "recursive":
        sp = RecursiveCharacterTextSplitter(**kw)
    elif s.strategy == "paragraph":
        sp = CharacterTextSplitter(separator="\n\n", **kw)
    else:
        sp = CharacterTextSplitter(separator="", **kw)
    chunks = []
    for d in docs:
        for i, part in enumerate(sp.create_documents([d["text"]])):
            chunks.append({"chunk_id": f"{d['doc_id']}#{i}", "doc_id": d["doc_id"], "index": i, "text": part.page_content, "start": part.metadata.get("start_index"),
                           "source_path": d["path"], "source_sha256": d["sha256"]})
    return chunks


class EmbedCache:
    def __init__(self, path: Path):
        self.path = path
        with closing(open_database(path, "embeddings", EMBEDDINGS)) as db:
            # Explicit versioned migration already created/validated the cache.
            db.commit()

    def get_many(self, ident: str, texts: list[str]) -> dict[str, list[float]]:
        keys = {hashlib.sha256((ident + "\0" + t).encode()).hexdigest(): t for t in texts}
        out: dict[str, list[float]] = {}
        with closing(open_database(self.path, "embeddings", EMBEDDINGS)) as db:
            for k, t in keys.items():
                row = db.execute("SELECT vec FROM emb WHERE key=?", (k,)).fetchone()
                if row:
                    out[t] = json.loads(row[0])
        return out

    def put_many(self, ident: str, items: dict[str, list[float]]) -> None:
        with closing(open_database(self.path, "embeddings", EMBEDDINGS)) as db:
            for t, v in items.items():
                db.execute("INSERT OR REPLACE INTO emb VALUES (?,?)", (hashlib.sha256((ident + "\0" + t).encode()).hexdigest(), json.dumps(v)))
            db.commit()


def embed_with_cache(texts: list[str], emb: Embedder, cache: EmbedCache) -> tuple[list[list[float]], int, int]:
    have = cache.get_many(emb.identity, texts)
    need = [t for t in dict.fromkeys(texts) if t not in have]
    if need:
        fresh = dict(zip(need, emb.embed(need)))
        cache.put_many(emb.identity, fresh)
        have.update(fresh)
    return [have[t] for t in texts], len(set(texts)) - len(need), len(need)


class IndexStore:
    def __init__(self, workbench: str | Path):
        self.root = Path(workbench) / "agent" / "indexes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = EmbedCache(self.root.parent / "embed_cache.sqlite")

    def dir(self, iid: str) -> Path:
        return self.root / iid

    def manifest(self, iid: str) -> dict[str, Any] | None:
        p = self.dir(iid) / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else None

    def identity(self, spec: IndexSpec, docs: list[dict[str, Any]]) -> str:
        e = Embedder(spec.embeddings)
        blob = json.dumps({"files": [(d["doc_id"], d["sha256"]) for d in docs], "splitter": spec.splitter.model_dump(), "embedding": e.identity, "store": spec.store}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def ensure(self, spec: IndexSpec) -> dict[str, Any]:
        """Build the index when missing or stale; otherwise reuse it. Returns the manifest plus `action` ('built' | 'reused')."""
        import faiss
        import numpy as np

        docs = load_documents(spec)
        ident = self.identity(spec, docs)
        m = self.manifest(spec.id)
        if m and m.get("identity") == ident and (self.dir(spec.id) / "faiss.index").exists():
            return {**m, "action": "reused"}
        t0 = time.perf_counter()
        chunks = split_documents(docs, spec)
        if not chunks:
            raise ValueError("the corpus produced no chunks")
        emb = Embedder(spec.embeddings)
        vecs, reused, computed = embed_with_cache([c["text"] for c in chunks], emb, self.cache)
        arr = np.array(vecs, dtype="float32")
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        d = self.dir(spec.id)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(d / "faiss.index.tmp"))
        (d / "faiss.index.tmp").replace(d / "faiss.index")
        (d / "chunks.json").write_text(json.dumps(chunks, indent=1))
        m = {"id": spec.id, "identity": ident, "builtAt": time.time(), "buildSeconds": round(time.perf_counter() - t0, 3), "documents": [{k: v for k, v in x.items() if k != "text"} for x in docs],
             "chunks": len(chunks), "dimension": int(arr.shape[1]), "embedding": {"identity": emb.identity, "label": emb.label, "normalized": spec.embeddings.normalize},
             "embeddingsReused": reused, "embeddingsComputed": computed, "splitter": spec.splitter.model_dump(), "loader": spec.loader.model_dump(), "store": "faiss (IndexFlatIP)",
             "scoreInterpretation": ("cosine similarity of L2-normalized vectors: 1.0 = same direction, higher is more similar" if spec.embeddings.normalize
                                     else "raw inner product of unnormalized vectors: not bounded, higher is more similar")}
        (d / "manifest.json").write_text(json.dumps(m, indent=1))
        return {**m, "action": "built"}

    def chunks(self, iid: str) -> list[dict[str, Any]]:
        p = self.dir(iid) / "chunks.json"
        return json.loads(p.read_text()) if p.exists() else []

    def search(self, spec: IndexSpec, query: str, k: int, threshold: float | None, consider: int = 50) -> dict[str, Any]:
        """Top-k chunks with scores; chunks that were scored but cut (below the threshold or beyond k) are returned as excluded with the reason."""
        import faiss
        import numpy as np

        m = self.manifest(spec.id)
        if not m:
            raise FileNotFoundError(f"index '{spec.id}' has not been built")
        chunks = self.chunks(spec.id)
        index = faiss.read_index(str(self.dir(spec.id) / "faiss.index"))
        emb = Embedder(spec.embeddings)
        if emb.identity != m["embedding"]["identity"]:
            raise ValueError(f"index was built with {m['embedding']['identity']} but the query embedder is {emb.identity}")
        q = np.array(emb.embed([query]), dtype="float32")
        scores, ids = index.search(q, min(consider, len(chunks)))
        included, excluded = [], []
        for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
            if i < 0:
                continue
            c = {**chunks[int(i)], "score": round(float(s), 5), "rank": rank}
            if threshold is not None and c["score"] < threshold:
                excluded.append({**c, "reason": f"score {c['score']:.3f} is below the score threshold {threshold}"})
            elif len(included) >= k:
                excluded.append({**c, "reason": f"ranked below the selected limit: rank {rank}, k={k}"})
            else:
                included.append(c)
        return {"documents": included, "excluded": excluded, "embedding": m["embedding"], "scoreInterpretation": m["scoreInterpretation"], "indexIdentity": m["identity"]}
