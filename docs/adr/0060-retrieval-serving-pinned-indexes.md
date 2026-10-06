# ADR0060: retrieval serving from pinned index snapshots

Status: accepted on Mac arm64 with real local Ollama (2026-10-06); hosted verification recorded separately in HANDOFF§57.

Agent graphs with `agent.retrieve` nodes could not be served: every serving adapter
refused indexes. In research, an index is rebuilt whenever its document folder,
splitter or embedding model changes, so serving straight from the research index
would make a released version's answers change whenever someone edited a file.

Decision: an `agent_retrieval` adapter (node `__agent_retrieval_graph__`, module
`production/retrieval_agent_adapter.py`) for stateless turns that retrieve from
index snapshots pinned at registration.

- Contract: prompt, set_state, retrieve and at most one chat_model; 1–2 declared
  indexes, each used by a retrieve node with k ≤ 8; no memory policies, tools or
  interrupts; all state turn-scoped; the agent adapter's limit, template, input,
  output and local-Ollama provider bounds. Model-free retrieval workflows (one
  written text output) are allowed.
- Registration copies each index's exact `faiss.index`, `chunks.json` and
  `manifest.json` into the CAS. The live index identity must start with the
  identity prefix the source run recorded in its `index_ready` event; otherwise
  registration refuses (E_AGENT_INDEX), so the snapshot is the index the run used.
  The embedding provider identity is pinned too: the installed Ollama model digest
  and runtime, or the deterministic local_hash parameters. Files are bounded to
  32 MiB each.
- Each turn restores the hash-verified snapshot into its private temporary
  workbench; a runtime subclass serves `ensure_index` from that snapshot (action
  `pinned`) and never reads the document folder or rebuilds. Query embedding uses
  the pinned embedding model; a changed digest refuses (E_AGENT_MODEL_CHANGED).
  Retrieved chunk ids and scores are always recorded in the trace evidence; full
  state, events and contexts follow the release capture policy.
- The family is treated like the other agent families by the runtime, API,
  monitoring, streaming and the editor.

Not provided: conversations or JSON output combined with retrieval, memory policies
or long-term memory, tools, re-indexing a release (register a new version from a
new source run), index sizes beyond the bounds, or retrieval quality benchmarks.

Verification: 3 offline tests with deterministic local_hash embeddings (snapshot
contents and identity; served retrieval; after rewriting a document, adding one and
deleting the live index, served retrievals are identical and nothing is rebuilt;
re-registration then refuses; contract refusals) and a live test with real
nomic-embed-text and qwen3.5:2b: after the source fact file is rewritten and the
live index deleted, the served turn retrieves the pinned chunk, the model is sent
the pinned fact rather than the new text, and answers with it.
