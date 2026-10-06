# ADR0064: pinned retrieval and bounded short-term policies in native text/JSON turns

Status: accepted; Mac native/live/editor/recovery verified, 2026-10-06.

HANDOFF§59 item3 combines ADR0060 retrieval snapshots with native END conversations
and ADR0045 JSON schema validation. Add `context_agent_adapter.py`; preserve prior
execution modules and native LangGraph/compiler/block/policy implementations.

- Four candidates: `agent_context` / `__agent_context_graph__`,
  `conversation_context` / `__agent_context_conversation__`, `agent_context_json` /
  `__agent_context_json__`, and `conversation_context_json` /
  `__agent_context_json_conversation__`. Thread fields select conversation mode;
  structured_output selects JSON. Require retrieval or memory_select. Native
  prompt/set_state/retrieve/memory_select/direct short-term memory_write, at most
  one chat OR structured-output node. No tools, interrupts, external effects,
  long-term reads/writes or research memory/history copied into a release.
- Reuse exact source-used index snapshots and embedding identity checks from
  ADR0060: 0–2 declared, used indexes; k<=8 per retrieval; local hash or installed
  local Ollama embeddings. Serving copies pinned FAISS/chunk/manifest bytes into
  its private runtime and never loads documents or rebuilds a research index.
- At most two used policies, each 1–6 native stages starting with exactly one
  retrieve. Retrieve sources are exactly short_term and bounded k<=8. Policy
  input is a messages field with keep_last_n<=8. Writes append directly only to
  these fields. Local hash <=512 dimensions is lexical hashing, not semantic
  embedding. Extractive summaries <=2000 characters only; no hidden model call.
  Token-estimate budgets <=4096. Native per-stage/per-record decisions persist
  in captured traces; no alternative policy interpreter or global memory store.
- Each prediction owns a private ArtifactStore, MemoryStore and InMemorySaver.
  Stateless turns start at defaults. Conversation turns restore the previous
  successful native END checkpoint through native serde. The checkpoint carries
  bounded history; policy applications belong to the current private execution.
  Admission, cancellation/deadline budgets, session serialization, atomic head +
  trace commit, idempotency, optimistic conflicts, reset/fork/history and isolated
  parent-checkpoint replay reuse existing END conversation machinery.
- ADR0023 graph/input/provider bounds apply: <=16 nodes / 32 fields / 32KiB graph,
  <=8KiB defaults/input, <=25 steps / 30 seconds / 2 model calls / 4096 tokens;
  1–4 turn-scoped text inputs <=2000 characters each. Submitted records are
  immutable trace data; native nodes retain their declared state-write semantics. State/context/checkpoint
  limits are 64/16/128KiB. Each recorded policy application is <=128KiB. Local
  Ollama only, pinned digest, think=false, output<=128 tokens. JSON is a separate
  turn replace object, flat 1–12 field schema <=4KiB, on_failure=fail, retries
  within model budget, only structured_output may write the object. Native JSON
  validation/attempt evidence precedes END checkpoint commit.
- Retrieval IDs/scores/embedding identity and policy decision summaries are
  retained. Full state/events/model context/policy applications require
  captureInputs=true; otherwise only hashes/usage/summaries. Private checkpoint
  state remains persistent independent of trace capture, as in ADR0024.
  Monitoring uses existing string/JSON agreement and committed model-call counts;
  it does not invoke a provider. Editor separates recorded retrieval/policy
  inspection into `ContextTurnEvidence`, preserving model-context inspection.
- No new dependency/schema. Old plain/conversation/JSON-conversation/retrieval/
  calculator/approval/non-agent execution source identities remain unchanged.
  Stateless agent_json pins shared integration files and needs re-registration.

Verification and retained failures are in HANDOFF§62. Focused native covers six
text mode combinations, restart, independent native-state agreement, scope/user
isolation, source/doc/index deletion, concurrent turns, capture-off persistence,
HTTP parent replay/labels/monitor and refusals. Four actual installed Ollama live
cases exercise stateless/conversation × text/JSON, exact sent retrieved chunks
and policy provenance/history, independent JSON labels and replay isolation.
Owned browser journeys cover model-free and real Ollama JSON conversations;
recovery checks exact version/release/source/trace/head/monitor before continuation.

Long-term memory changes require a separately bounded, durable ownership/write
contract. No cross-release memory migration, effect ledger, model summaries,
remote/distributed provider serving or answer-quality benchmark is established.

Final Mac acceptance: native1338 passed/1 skipped/25 deselected525.19s; actual live25
passed; all20 editor runners plus explicit actual Ollama JSON context browser;
integrated sealed/GC recovery207 files/11DBs. Hosted outcome is recorded separately
in HANDOFF§62/63; no hosted acceptance is implied by local acceptance.
